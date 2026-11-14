from pathlib import Path

import pymupdf
import pytest

from app.ingestion.pipeline import CorruptedPdfError, node_from_line, run_extraction


def test_pipeline_outputs_and_counts(sample_pdf: Path, tmp_path: Path) -> None:
    result = run_extraction(sample_pdf, tmp_path / "processed", tmp_path / "media")

    manifest = result.manifest
    assert manifest["total_pages"] == 8
    assert manifest["counts"]["tables"] == 2
    assert manifest["counts"]["images_kept"] == 5  # 4 embedded + 1 vector_render
    assert manifest["counts"]["images_by_source"] == {"embedded": 4, "vector_render": 1}
    assert manifest["counts"]["images_by_kind"] == {"figure": 4, "icon": 1}
    assert manifest["counts"]["text_chunks"] == 8
    assert manifest["scanned_pages"] == []
    assert manifest["repeated_lines_removed"]
    assert any("MDX-100" in line for line in manifest["repeated_lines_removed"])
    assert result.unchanged is False
    assert (result.processed_dir / "manifest.json").is_file()
    assert (result.processed_dir / "nodes.jsonl").is_file()


def test_nodes_jsonl_reloadable(sample_pdf: Path, tmp_path: Path) -> None:
    result = run_extraction(sample_pdf, tmp_path / "processed", tmp_path / "media")
    lines = (result.processed_dir / "nodes.jsonl").read_text(encoding="utf-8").splitlines()

    nodes = [node_from_line(line) for line in lines if line.strip()]
    counted_keys = ("text_chunks", "tables", "images_kept")
    total_expected = sum(result.manifest["counts"][key] for key in counted_keys)
    assert len(nodes) == total_expected
    content_types = {node.metadata["content_type"] for node in nodes}
    assert content_types == {"text", "table", "image"}


def test_second_run_is_unchanged(sample_pdf: Path, tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    media_dir = tmp_path / "media"

    first = run_extraction(sample_pdf, processed_root, media_dir)
    second = run_extraction(sample_pdf, processed_root, media_dir)

    assert first.unchanged is False
    assert second.unchanged is True
    assert second.manifest["pdf_sha256"] == first.manifest["pdf_sha256"]


def test_force_rewrites(sample_pdf: Path, tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    media_dir = tmp_path / "media"

    first = run_extraction(sample_pdf, processed_root, media_dir)
    forced = run_extraction(sample_pdf, processed_root, media_dir, force=True)

    assert first.unchanged is False
    assert forced.unchanged is False
    assert forced.manifest["extraction_date"] != first.manifest["extraction_date"]


def test_corrupted_pdf_raises(tmp_path: Path) -> None:
    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_bytes(b"this is not a pdf at all")

    with pytest.raises(CorruptedPdfError):
        run_extraction(bad_pdf, tmp_path / "processed", tmp_path / "media")


def test_encrypted_pdf_raises(tmp_path: Path) -> None:
    encrypted_pdf = tmp_path / "encrypted.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(
        str(encrypted_pdf),
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="secret123",
        owner_pw="owner123",
    )
    doc.close()

    with pytest.raises(CorruptedPdfError):
        run_extraction(encrypted_pdf, tmp_path / "processed", tmp_path / "media")


def test_encrypted_pdf_cli_exit_code_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.ingestion.__main__ as cli

    monkeypatch.chdir(tmp_path)
    encrypted_pdf = tmp_path / "encrypted.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(
        str(encrypted_pdf),
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="secret123",
        owner_pw="owner123",
    )
    doc.close()

    assert cli.main(["extract", str(encrypted_pdf)]) == 1


def test_missing_pdf_raises(tmp_path: Path) -> None:
    with pytest.raises(CorruptedPdfError):
        run_extraction(tmp_path / "missing.pdf", tmp_path / "processed", tmp_path / "media")


def test_cli_exit_codes(sample_pdf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.ingestion.__main__ as cli

    monkeypatch.chdir(tmp_path)
    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_bytes(b"not a pdf")

    assert cli.main(["extract", str(bad_pdf)]) == 1
