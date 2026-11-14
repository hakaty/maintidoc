from pathlib import Path

from app.ingestion.extract_tables import _clean_cell_text, extract_tables, table_to_node


def test_table_count_and_pages(sample_pdf: Path, sample_pdf_sha256: str) -> None:
    tables = extract_tables(sample_pdf, sample_pdf_sha256)

    assert len(tables) == 2
    assert [table.page for table in tables] == [3, 6]


def test_table_markdown_has_header_and_rows(sample_pdf: Path, sample_pdf_sha256: str) -> None:
    tables = extract_tables(sample_pdf, sample_pdf_sha256)
    pannes = next(table for table in tables if table.page == 3)

    lines = pannes.markdown.splitlines()
    assert lines[0].startswith("|")
    assert set(lines[1].strip("|").replace("|", "")) <= {"-"}  # markdown header separator
    assert pannes.row_count == 4  # header + 3 rows
    assert pannes.col_count == 3
    assert "Surchauffe moteur" in pannes.markdown


def test_table_captions_found(sample_pdf: Path, sample_pdf_sha256: str) -> None:
    tables = extract_tables(sample_pdf, sample_pdf_sha256)
    captions = {table.page: table.caption for table in tables}

    assert captions[3] == "Tableau 1 : Tableau des pannes"
    assert captions[6] == "Tableau 2 : Caracteristiques techniques"


def test_table_to_node_metadata(sample_pdf: Path, sample_pdf_sha256: str) -> None:
    tables = extract_tables(sample_pdf, sample_pdf_sha256)
    node = table_to_node(tables[0], sample_pdf_sha256, str(sample_pdf))

    assert node.metadata["content_type"] == "table"
    assert node.metadata["table_id"] == tables[0].table_id
    assert node.text == tables[0].markdown


def test_wrapped_header_and_cell_keep_their_spaces(
    sample_pdf: Path, sample_pdf_sha256: str
) -> None:
    """Regression test: table.to_markdown() joins wrapped lines within a cell
    with a literal "<br>" (or, for table.extract(), drops the space
    entirely) -- the fixture's "pannes" table has a header ("Cause probable")
    and a data cell ("Roulement use") narrow enough to wrap onto two lines,
    which must still read with a single space and no "<br>"."""
    tables = extract_tables(sample_pdf, sample_pdf_sha256)
    pannes = next(table for table in tables if table.page == 3)

    assert "Cause probable" in pannes.markdown
    assert "Roulement use" in pannes.markdown
    assert "<br>" not in pannes.markdown
    assert "Causeprobable" not in pannes.markdown
    assert "Rouleme nt" not in pannes.markdown


def test_clean_cell_text_collapses_and_escapes() -> None:
    assert _clean_cell_text("Cause \nprobable \n") == "Cause probable"
    assert _clean_cell_text("  multiple   spaces  ") == "multiple spaces"
    assert _clean_cell_text(None) == ""
    assert _clean_cell_text("") == ""
    assert _clean_cell_text("a|b") == "a\\|b"
