from pathlib import Path

from llama_index.core.schema import Document as LlamaDocument

from app.ingestion.chunking import chunk_documents
from app.ingestion.extract_images import extract_images
from app.ingestion.extract_tables import extract_tables
from app.ingestion.extract_text import extract_text


def test_chunk_never_spans_two_pages() -> None:
    documents = [
        LlamaDocument(
            text="PAGE-ONE-MARKER. " * 30,
            metadata={"page": 1, "scanned": False},
        ),
        LlamaDocument(
            text="PAGE-TWO-MARKER. " * 30,
            metadata={"page": 2, "scanned": False},
        ),
    ]

    nodes = chunk_documents(
        documents,
        pdf_sha256="deadbeef" * 8,
        source_file="dummy.pdf",
        image_ids_by_page={},
        table_ids_by_page={},
        chunk_size=20,
        chunk_overlap=5,
    )

    assert len(nodes) > 2  # small chunk_size forces multiple chunks per page
    for node in nodes:
        assert not ("PAGE-ONE-MARKER" in node.text and "PAGE-TWO-MARKER" in node.text)
        if node.metadata["page"] == 1:
            assert "PAGE-TWO-MARKER" not in node.text
        else:
            assert "PAGE-ONE-MARKER" not in node.text


def test_linking_metadata_matches_page_content(
    sample_pdf: Path, sample_pdf_sha256: str, tmp_path: Path
) -> None:
    documents = extract_text(sample_pdf, sample_pdf_sha256)
    tables = extract_tables(sample_pdf, sample_pdf_sha256)
    images, _skipped = extract_images(sample_pdf, sample_pdf_sha256, tmp_path)

    image_ids_by_page: dict[int, list[str]] = {}
    for image in images:
        image_ids_by_page.setdefault(image.page, []).append(image.image_id)
    table_ids_by_page: dict[int, list[str]] = {}
    for table in tables:
        table_ids_by_page.setdefault(table.page, []).append(table.table_id)

    nodes = chunk_documents(
        documents, sample_pdf_sha256, str(sample_pdf), image_ids_by_page, table_ids_by_page
    )
    by_page = {node.metadata["page"]: node for node in nodes}

    assert by_page[2].metadata["related_image_ids"] == [images[0].image_id]
    assert by_page[2].metadata["related_table_ids"] == []

    assert by_page[3].metadata["related_table_ids"] == [tables[0].table_id]
    assert by_page[3].metadata["related_image_ids"] == []

    assert by_page[1].metadata["related_image_ids"] == []
    assert by_page[1].metadata["related_table_ids"] == []

    for node in nodes:
        assert node.metadata["content_type"] == "text"
        assert node.metadata["pdf_sha256"] == sample_pdf_sha256
        assert node.metadata["source_file"] == str(sample_pdf)
        assert isinstance(node.metadata["chunk_index"], int)
