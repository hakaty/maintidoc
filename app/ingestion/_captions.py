"""Shared PyMuPDF text-block and caption-matching helpers.

Used by extract_text, extract_tables and extract_images so that "what counts
as page text outside a table" stays consistent across all three.
"""

import re
from collections.abc import Container
from dataclasses import dataclass

import pymupdf

from app.ingestion.repeated_lines import normalize_line

TABLE_CAPTION_PATTERN = re.compile(r"tableau|table", re.IGNORECASE)
FIGURE_CAPTION_PATTERN = re.compile(r"figure|fig\.", re.IGNORECASE)
MAX_CAPTION_CHARS = 300


@dataclass(frozen=True)
class TextBlock:
    bbox: pymupdf.Rect
    text: str


# PyMuPDF's find_tables() costs ~0.3-2s per page. extract_text, extract_tables
# and extract_images each need a page's table bboxes (not the Table objects
# themselves, which don't outlive the caller's `with pymupdf.open()` block),
# so this cache makes sure find_tables() runs once per page per PDF, no matter
# how many of those three call it. Keyed by (pdf path, pdf sha256, page number)
# -- sha256, not just path, so a changed file reusing the same path is never
# served another file's stale bboxes.
_table_bbox_cache: dict[tuple[str, str, int], list[pymupdf.Rect]] = {}


def set_cached_table_bboxes(
    pdf_path: str, pdf_sha256: str, page_number: int, bboxes: list[pymupdf.Rect]
) -> None:
    _table_bbox_cache[(pdf_path, pdf_sha256, page_number)] = bboxes


def get_table_bboxes(
    pdf_path: str, pdf_sha256: str, page_number: int, page: pymupdf.Page
) -> list[pymupdf.Rect]:
    key = (pdf_path, pdf_sha256, page_number)
    cached = _table_bbox_cache.get(key)
    if cached is not None:
        return cached
    bboxes = [pymupdf.Rect(table.bbox) for table in page.find_tables().tables]
    _table_bbox_cache[key] = bboxes
    return bboxes


def page_text_blocks(
    page: pymupdf.Page, exclude: list[pymupdf.Rect] | None = None
) -> list[TextBlock]:
    """Text blocks on the page, in reading order, skipping anything inside `exclude` bboxes."""
    exclude = exclude or []
    blocks: list[TextBlock] = []
    for x0, y0, x1, y1, text, _block_no, block_type in page.get_text("blocks", sort=True):
        if block_type != 0:
            continue
        rect = pymupdf.Rect(x0, y0, x1, y1)
        if any(rect.intersects(area) for area in exclude):
            continue
        stripped = text.strip()
        if stripped:
            blocks.append(TextBlock(bbox=rect, text=stripped))
    return blocks


def _vertical_gap(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    if a.y1 <= b.y0:
        return b.y0 - a.y1
    if b.y1 <= a.y0:
        return a.y0 - b.y1
    return 0.0


def _is_entirely_repeated(text: str, repeated_patterns: Container[str]) -> bool:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return False
    return all(normalize_line(line) in repeated_patterns for line in lines)


def find_caption(
    target_bbox: pymupdf.Rect,
    blocks: list[TextBlock],
    keyword_pattern: re.Pattern[str],
    max_chars: int = MAX_CAPTION_CHARS,
    repeated_patterns: Container[str] | None = None,
) -> str | None:
    """Nearest block (by vertical gap) matching `keyword_pattern`, else nearest overall.

    A block made up entirely of repeated running header/footer lines
    (`repeated_patterns`, from app.ingestion.repeated_lines.detect_repeated_lines)
    is never picked as a caption -- it carries no document-specific meaning.
    If that leaves no candidate at all, the caption is None rather than
    falling back to header/footer junk.
    """
    if repeated_patterns:
        blocks = [
            block for block in blocks if not _is_entirely_repeated(block.text, repeated_patterns)
        ]
    if not blocks:
        return None
    matching = [block for block in blocks if keyword_pattern.search(block.text)]
    candidates = matching or blocks
    nearest = min(candidates, key=lambda block: _vertical_gap(target_bbox, block.bbox))
    return nearest.text[:max_chars]
