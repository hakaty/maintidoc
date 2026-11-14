from pathlib import Path

from app.ingestion.extract_text import extract_text, normalize_text


def test_pages_and_metadata_complete(sample_pdf: Path, sample_pdf_sha256: str) -> None:
    documents = extract_text(sample_pdf, sample_pdf_sha256)

    assert len(documents) == 8
    for index, document in enumerate(documents):
        assert document.metadata["page"] == index + 1
        assert document.metadata["total_pages"] == 8
        assert document.metadata["source_file"] == str(sample_pdf)
        assert document.metadata["pdf_sha256"] == sample_pdf_sha256
        assert document.metadata["scanned"] is False


def test_table_text_excluded_from_page_text(sample_pdf: Path, sample_pdf_sha256: str) -> None:
    documents = extract_text(sample_pdf, sample_pdf_sha256)

    pannes_page = documents[2]  # page 3: "Tableau 1 : Tableau des pannes"
    assert "Tableau des pannes" in pannes_page.text
    # Row content lives inside the table's bbox and must not be duplicated here.
    assert "Filtre a air colmate" not in pannes_page.text
    assert "Remplacer le filtre" not in pannes_page.text

    caracteristiques_page = documents[5]  # page 6
    assert "Caracteristiques techniques" in caracteristiques_page.text
    assert "bar" not in caracteristiques_page.text
    assert "degC" not in caracteristiques_page.text


def test_running_header_present_before_pipeline_strips_it(
    sample_pdf: Path, sample_pdf_sha256: str
) -> None:
    """extract_text() itself does not remove repeated headers/footers -- that
    is app.ingestion.repeated_lines, applied at the pipeline level."""
    documents = extract_text(sample_pdf, sample_pdf_sha256)

    assert all("MDX-100 - Manuel de maintenance" in doc.text for doc in documents)
    assert all(f"Page {doc.metadata['page']}" in doc.text for doc in documents)


def test_normalize_text_strips_control_characters() -> None:
    assert normalize_text("Hello\x00World\x07") == "HelloWorld"
    assert normalize_text("line one\nline\x00two") == "line one\nlinetwo"
    assert normalize_text("  padded  ") == "padded"
