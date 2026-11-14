"""Upload validation for the Image search and Document tabs.

Defence in depth against malicious/oversized uploads: magic-byte sniffing
(never trust the browser-supplied filename/extension), size caps, EXIF
stripping for images, and generated (never user-supplied) filenames on disk.
"""

import io
import uuid
from pathlib import Path

from PIL import Image as PILImage

PDF_MAGIC = b"%PDF"
MAX_PDF_SIZE_BYTES = 50 * 1024 * 1024
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_MEGAPIXELS = 25

IMAGE_MAGIC_BYTES: dict[str, bytes] = {
    "jpeg": b"\xff\xd8\xff",
    "png": b"\x89PNG\r\n\x1a\n",
    "webp": b"RIFF",  # followed by "WEBP" at offset 8, checked separately
}


class UploadValidationError(ValueError):
    """Raised with a short, user-facing French message -- never a traceback."""


def validate_pdf_bytes(data: bytes) -> None:
    if len(data) > MAX_PDF_SIZE_BYTES:
        max_mb = MAX_PDF_SIZE_BYTES // (1024 * 1024)
        raise UploadValidationError(
            f"Le fichier dépasse la taille maximale autorisée ({max_mb} Mo)."
        )
    if not data.startswith(PDF_MAGIC):
        raise UploadValidationError("Le fichier ne semble pas être un PDF valide.")


def _sniff_image_format(data: bytes) -> str | None:
    if data.startswith(IMAGE_MAGIC_BYTES["jpeg"]):
        return "jpeg"
    if data.startswith(IMAGE_MAGIC_BYTES["png"]):
        return "png"
    if data.startswith(IMAGE_MAGIC_BYTES["webp"]) and data[8:12] == b"WEBP":
        return "webp"
    return None


def validate_image_bytes(data: bytes) -> str:
    """Returns the sniffed image format, or raises UploadValidationError."""
    if len(data) > MAX_IMAGE_SIZE_BYTES:
        max_mb = MAX_IMAGE_SIZE_BYTES // (1024 * 1024)
        raise UploadValidationError(
            f"L'image dépasse la taille maximale autorisée ({max_mb} Mo)."
        )
    image_format = _sniff_image_format(data)
    if image_format is None:
        raise UploadValidationError(
            "Le fichier ne semble pas être une image JPG, PNG ou WebP valide."
        )
    try:
        with PILImage.open(io.BytesIO(data)) as img:
            width, height = img.size
    except Exception as exc:
        raise UploadValidationError("L'image n'a pas pu être lue.") from exc
    megapixels = (width * height) / 1_000_000
    if megapixels > MAX_IMAGE_MEGAPIXELS:
        raise UploadValidationError(
            f"L'image est trop grande ({megapixels:.1f} MP, maximum {MAX_IMAGE_MEGAPIXELS} MP)."
        )
    return image_format


def strip_exif_and_save(data: bytes, destination_dir: Path) -> Path:
    """Re-encodes the image without EXIF metadata, under a generated name
    (never the user's original filename) inside destination_dir."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    with PILImage.open(io.BytesIO(data)) as img:
        img = img.convert("RGB") if img.mode not in ("RGB", "L") else img
        clean = PILImage.frombytes(img.mode, img.size, img.tobytes())
        generated_name = f"{uuid.uuid4().hex}.png"
        destination_path = destination_dir / generated_name
        clean.save(destination_path, format="PNG")
    return destination_path


def generated_pdf_filename(original_name: str) -> str:
    """Never trust the user-supplied filename: generate a new one, keep only
    a sanitised hint of the original stem for readability."""
    stem = Path(original_name).stem
    safe_hint = "".join(c for c in stem if c.isalnum() or c in ("-", "_"))[:40] or "document"
    return f"{safe_hint}_{uuid.uuid4().hex[:8]}.pdf"
