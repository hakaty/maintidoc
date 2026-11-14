"""Vector-drawn figure detection.

Technical manuals sometimes draw schematics as vector paths rather than
embedding a raster image. This clusters page.get_drawings() paths into
candidate figure regions, excluding table areas, full-page frames/borders,
and regions already covered by a kept/skipped embedded image, then renders
survivors to PNG (same MEDIA_DIR layout as embedded images).
"""

import hashlib
from pathlib import Path

import pymupdf

from app.ingestion._captions import FIGURE_CAPTION_PATTERN, find_caption, page_text_blocks
from app.ingestion.extract_images import ExtractedImage

DEFAULT_MIN_AREA_PT2 = 2000.0  # filters ruling lines, underlines, small decorations
DEFAULT_CLUSTER_MARGIN_PT = 10.0  # merge drawings within this distance into one region
DEFAULT_RENDER_DPI = 150
DEFAULT_MIN_SHAPES_PER_CLUSTER = 2  # a lone decorative stroke is not a diagram
FULL_PAGE_COVERAGE_RATIO = 0.9  # excludes decorative page borders/frames
MIN_SHAPE_SIDE_PT = 5.0  # below this, a path is a ruling line/underline, not a shape

# A region this low relative to its width (e.g. a text-column-wide divider
# band made of 2 thin bars) reads as a decorative rule, not a 2D diagram.
MAX_BAND_HEIGHT_PT = 20.0
MAX_BAND_WIDTH_RATIO = 0.7

# Diagram line-art needs to be legible in print; near-white strokes/fills
# (e.g. a pastel cover-page "swoosh") are decorative, never diagram content.
MAX_LIGHTNESS = 0.75


def _is_too_light(drawing: dict) -> bool:
    color = drawing.get("color") or drawing.get("fill")
    if not color:
        return False
    return (sum(color) / len(color)) >= MAX_LIGHTNESS


def _is_full_page_frame(rect: pymupdf.Rect, page_rect: pymupdf.Rect) -> bool:
    if page_rect.width <= 0 or page_rect.height <= 0:
        return False
    return (
        rect.width / page_rect.width >= FULL_PAGE_COVERAGE_RATIO
        and rect.height / page_rect.height >= FULL_PAGE_COVERAGE_RATIO
    )


def _cluster_rects(rects: list[pymupdf.Rect], margin: float) -> list[tuple[pymupdf.Rect, int]]:
    """Repeatedly merge rects within `margin` of each other; track shapes per cluster."""
    clusters = [(rect, 1) for rect in rects]
    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            rect_i, count_i = clusters[i]
            expanded = pymupdf.Rect(rect_i)
            expanded += (-margin, -margin, margin, margin)
            for j in range(i + 1, len(clusters)):
                rect_j, count_j = clusters[j]
                if expanded.intersects(rect_j):
                    clusters[i] = (rect_i | rect_j, count_i + count_j)
                    clusters.pop(j)
                    merged = True
                    break
            if merged:
                break
    return clusters


def detect_vector_figure_regions(
    page: pymupdf.Page,
    table_bboxes: list[pymupdf.Rect],
    exclude_bboxes: list[pymupdf.Rect],
    min_area: float = DEFAULT_MIN_AREA_PT2,
    cluster_margin: float = DEFAULT_CLUSTER_MARGIN_PT,
    min_shapes_per_cluster: int = DEFAULT_MIN_SHAPES_PER_CLUSTER,
) -> list[pymupdf.Rect]:
    """Candidate vector-figure bboxes: drawing clusters outside tables/exclusions.

    A single isolated stroke (e.g. a decorative cover-page swoosh) is not a
    diagram: real technical drawings are built from several primitive shapes
    (lines, circles, filled arrows...) clustered together, so clusters formed
    from fewer than `min_shapes_per_cluster` original paths are discarded.
    """
    page_rect = page.rect
    shapes: list[pymupdf.Rect] = []
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        if rect.width < MIN_SHAPE_SIDE_PT or rect.height < MIN_SHAPE_SIDE_PT:
            continue  # a ruling line, not a shape
        if _is_full_page_frame(rect, page_rect):
            continue
        if _is_too_light(drawing):
            continue
        if any(rect.intersects(bbox) for bbox in table_bboxes):
            continue
        shapes.append(rect)

    regions = []
    for cluster, shape_count in _cluster_rects(shapes, cluster_margin):
        if shape_count < min_shapes_per_cluster:
            continue
        if cluster.width * cluster.height < min_area:
            continue
        if (
            cluster.height <= MAX_BAND_HEIGHT_PT
            and cluster.width / page_rect.width >= MAX_BAND_WIDTH_RATIO
        ):
            continue  # decorative divider band, not a diagram
        if any(cluster.intersects(bbox) for bbox in exclude_bboxes):
            continue
        regions.append(cluster)
    return regions


def extract_vector_figures(
    pdf_path: Path,
    pdf_sha256: str,
    media_dir: Path,
    table_bboxes_by_page: dict[int, list[pymupdf.Rect]],
    embedded_bboxes_by_page: dict[int, list[pymupdf.Rect]],
    min_area: float = DEFAULT_MIN_AREA_PT2,
    cluster_margin: float = DEFAULT_CLUSTER_MARGIN_PT,
    render_dpi: int = DEFAULT_RENDER_DPI,
    repeated_patterns: frozenset[str] = frozenset(),
) -> list[ExtractedImage]:
    figures: list[ExtractedImage] = []
    output_dir = media_dir / pdf_sha256[:12]
    page_figure_index: dict[int, int] = {}

    with pymupdf.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            table_bboxes = table_bboxes_by_page.get(page_number, [])
            embedded_bboxes = embedded_bboxes_by_page.get(page_number, [])

            regions = detect_vector_figure_regions(
                page, table_bboxes, embedded_bboxes, min_area, cluster_margin
            )
            if not regions:
                continue

            caption_blocks = page_text_blocks(page, exclude=table_bboxes)

            for region in regions:
                caption = find_caption(
                    region,
                    caption_blocks,
                    FIGURE_CAPTION_PATTERN,
                    repeated_patterns=repeated_patterns,
                )

                index_on_page = page_figure_index.get(page_number, 0)
                page_figure_index[page_number] = index_on_page + 1
                figure_id = f"{pdf_sha256[:8]}_p{page_number}_v{index_on_page}"
                figure_path = output_dir / f"{figure_id}.png"
                figure_path.parent.mkdir(parents=True, exist_ok=True)

                pixmap = page.get_pixmap(clip=region, dpi=render_dpi)
                pixmap.save(figure_path)
                sha256 = hashlib.sha256(figure_path.read_bytes()).hexdigest()

                figures.append(
                    ExtractedImage(
                        image_id=figure_id,
                        page=page_number,
                        image_path=figure_path,
                        width=pixmap.width,
                        height=pixmap.height,
                        sha256=sha256,
                        bbox=tuple(region),
                        caption=caption,
                        source="vector_render",
                        kind="figure",
                    )
                )
    return figures
