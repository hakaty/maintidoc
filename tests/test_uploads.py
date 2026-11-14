import io

import pytest
from PIL import Image as PILImage

from ui.components.uploads import (
    UploadValidationError,
    generated_pdf_filename,
    strip_exif_and_save,
    validate_image_bytes,
    validate_pdf_bytes,
)


def _png_bytes(size: tuple[int, int] = (16, 16)) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", size, color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_validate_pdf_bytes_accepts_real_pdf_magic() -> None:
    validate_pdf_bytes(b"%PDF-1.4\n...")  # no exception


def test_validate_pdf_bytes_rejects_non_pdf_content() -> None:
    with pytest.raises(UploadValidationError, match="PDF valide"):
        validate_pdf_bytes(b"not a pdf at all")


def test_validate_pdf_bytes_rejects_oversized_file() -> None:
    oversized = b"%PDF-1.4" + b"0" * (51 * 1024 * 1024)
    with pytest.raises(UploadValidationError, match="taille maximale"):
        validate_pdf_bytes(oversized)


def test_validate_image_bytes_accepts_real_png() -> None:
    assert validate_image_bytes(_png_bytes()) == "png"


def test_validate_image_bytes_rejects_non_image_magic_bytes() -> None:
    with pytest.raises(UploadValidationError, match="image"):
        validate_image_bytes(b"this is not an image")


def test_validate_image_bytes_rejects_file_claiming_to_be_png_but_isnt() -> None:
    # Correct PNG magic bytes, garbage after -- magic-byte check alone passes,
    # but PIL fails to actually decode it.
    fake_png = b"\x89PNG\r\n\x1a\n" + b"garbage, not a real PNG stream"
    with pytest.raises(UploadValidationError, match="lue"):
        validate_image_bytes(fake_png)


def test_validate_image_bytes_rejects_oversized_file() -> None:
    oversized = b"\x89PNG\r\n\x1a\n" + b"0" * (11 * 1024 * 1024)
    with pytest.raises(UploadValidationError, match="taille maximale"):
        validate_image_bytes(oversized)


def test_validate_image_bytes_rejects_too_many_megapixels() -> None:
    huge = _png_bytes(size=(6000, 6000))  # 36 MP > 25 MP cap
    with pytest.raises(UploadValidationError, match="trop grande"):
        validate_image_bytes(huge)


def test_strip_exif_and_save_uses_generated_filename_not_original(tmp_path) -> None:
    data = _png_bytes()
    saved_path = strip_exif_and_save(data, tmp_path)

    assert saved_path.parent == tmp_path
    assert saved_path.suffix == ".png"
    assert "original" not in saved_path.name
    assert saved_path.is_file()


def test_strip_exif_and_save_produces_a_different_file_each_time(tmp_path) -> None:
    data = _png_bytes()
    first = strip_exif_and_save(data, tmp_path)
    second = strip_exif_and_save(data, tmp_path)

    assert first != second


def test_generated_pdf_filename_sanitises_and_never_reuses_original_name() -> None:
    malicious_name = "../../etc/passwd; rm -rf /.pdf"
    generated = generated_pdf_filename(malicious_name)

    assert generated != malicious_name
    assert "/" not in generated
    assert ".." not in generated
    assert generated.endswith(".pdf")


def test_generated_pdf_filename_falls_back_when_stem_has_no_safe_characters() -> None:
    generated = generated_pdf_filename("???.pdf")
    assert generated.startswith("document_")
