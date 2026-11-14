from pathlib import Path

import pymupdf
import pytest

from app.ingestion.extract_images import extract_images
from app.ingestion.extract_tables import extract_tables
from app.ingestion.extract_vector_figures import (
    detect_vector_figure_regions,
    extract_vector_figures,
)


@pytest.fixture(scope="module")
def extraction(sample_pdf: Path, sample_pdf_sha256: str, tmp_path_factory: pytest.TempPathFactory):
    media_dir = tmp_path_factory.mktemp("media")
    tables = extract_tables(sample_pdf, sample_pdf_sha256)
    images, _skipped = extract_images(sample_pdf, sample_pdf_sha256, media_dir)

    table_bboxes_by_page: dict[int, list[pymupdf.Rect]] = {}
    for table in tables:
        table_bboxes_by_page.setdefault(table.page, []).append(pymupdf.Rect(table.bbox))
    embedded_bboxes_by_page: dict[int, list[pymupdf.Rect]] = {}
    for image in images:
        embedded_bboxes_by_page.setdefault(image.page, []).append(pymupdf.Rect(image.bbox))

    figures = extract_vector_figures(
        sample_pdf, sample_pdf_sha256, media_dir, table_bboxes_by_page, embedded_bboxes_by_page
    )
    return figures


def test_vector_diagram_detected_on_its_own_page(extraction) -> None:
    figures = extraction

    assert len(figures) == 1
    figure = figures[0]
    assert figure.page == 7
    assert figure.source == "vector_render"
    assert len(figure.sha256) == 64  # a real hash of the rendered PNG, not the "" placeholder
    assert figure.kind == "figure"


def test_vector_diagram_caption_found(extraction) -> None:
    figure = extraction[0]
    assert figure.caption == "Figure 4 : Schema de principe pompe-vanne"


def test_vector_diagram_rendered_to_png(extraction) -> None:
    figure = extraction[0]
    assert figure.image_path.is_file()
    assert figure.image_path.suffix == ".png"
    assert figure.width > 0
    assert figure.height > 0


def test_no_false_positives_on_other_pages(extraction) -> None:
    pages_with_figures = {figure.page for figure in extraction}
    assert pages_with_figures == {7}


def test_full_page_frame_is_recognized(sample_pdf: Path) -> None:
    """Decorative page borders (>= 90% of both page dimensions) are excluded."""
    from app.ingestion.extract_vector_figures import _is_full_page_frame

    with pymupdf.open(sample_pdf) as doc:
        page_rect = doc[0].rect

    full_page_frame = pymupdf.Rect(2, 2, page_rect.width - 2, page_rect.height - 2)
    small_shape = pymupdf.Rect(10, 10, 60, 60)

    assert _is_full_page_frame(full_page_frame, page_rect) is True
    assert _is_full_page_frame(small_shape, page_rect) is False


def test_near_white_strokes_are_filtered() -> None:
    """Diagram line-art needs to be legible in print; near-white decorative
    strokes (e.g. a pastel cover-page swoosh) are never diagram content."""
    from app.ingestion.extract_vector_figures import _is_too_light

    assert _is_too_light({"color": (0.86, 0.86, 0.86), "fill": None}) is True
    assert _is_too_light({"color": (0.0, 0.0, 0.0), "fill": None}) is False
    assert _is_too_light({"color": None, "fill": None}) is False


def test_single_isolated_stroke_is_not_a_diagram(sample_pdf: Path) -> None:
    """A lone ruling line (or any single isolated shape) never forms a
    region on its own: real diagrams cluster multiple shapes together."""
    with pymupdf.open(sample_pdf) as doc:
        page = doc[0]
        ruling_line_as_exclude = pymupdf.Rect(10, 10, 400, 10.5)
        assert detect_vector_figure_regions(page, [], [ruling_line_as_exclude]) == []
