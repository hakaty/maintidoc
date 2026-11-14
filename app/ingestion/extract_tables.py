"""Table extraction via PyMuPDF's find_tables(), wrapped as LlamaIndex TextNodes.

Table content is UNTRUSTED data: stored and displayed, never interpreted as
instructions.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pymupdf
from llama_index.core.schema import TextNode

from app.ingestion._captions import (
    TABLE_CAPTION_PATTERN,
    find_caption,
    page_text_blocks,
    set_cached_table_bboxes,
)


@dataclass
class ExtractedTable:
    table_id: str
    page: int
    markdown: str
    row_count: int
    col_count: int
    bbox: tuple[float, float, float, float]
    caption: str | None


def _clean_cell_text(text: str | None) -> str:
    """Collapse whitespace/newlines to single spaces and escape "|" for markdown."""
    if not text:
        return ""
    return " ".join(text.split()).replace("|", "\\|")


def _table_grid(page: pymupdf.Page, table: pymupdf.table.Table) -> list[list[str]]:
    """Row-major cell text, read directly from each cell's bbox.

    table.to_markdown()/table.extract() both mangle inter-word spacing on this
    PDF (e.g. "Niveau de danger" -> "Niveaudedanger", wrapped lines joined with
    a literal "<br>"); page.get_text(clip=...) on each cell's own bbox doesn't
    have that problem, since it reconstructs spacing from actual glyph gaps.
    """
    grid: list[list[str]] = []
    for row in table.rows:
        cells = []
        for bbox in row.cells:
            if bbox is None:
                cells.append("")
                continue
            text = page.get_text("text", clip=pymupdf.Rect(bbox))
            cells.append(_clean_cell_text(text))
        grid.append(cells)
    return grid


def _grid_to_markdown(grid: list[list[str]]) -> str:
    if not grid:
        return ""
    header, *rows = grid
    separator = ["---"] * len(header)
    lines = ["|" + "|".join(header) + "|", "|" + "|".join(separator) + "|"]
    lines.extend("|" + "|".join(row) + "|" for row in rows)
    return "\n".join(lines)


@lru_cache(maxsize=16)
def extract_tables(
    pdf_path: Path, pdf_sha256: str, repeated_patterns: frozenset[str] = frozenset()
) -> list[ExtractedTable]:
    tables: list[ExtractedTable] = []
    with pymupdf.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            found = page.find_tables().tables
            table_bboxes = [pymupdf.Rect(table.bbox) for table in found]
            # find_tables() is costly (~0.3-2s/page): share the result with
            # extract_text/extract_images so they don't recompute it.
            set_cached_table_bboxes(str(pdf_path), pdf_sha256, page_number, table_bboxes)
            caption_blocks = page_text_blocks(page, exclude=table_bboxes)

            for table_index, table in enumerate(found):
                bbox = pymupdf.Rect(table.bbox)
                caption = find_caption(
                    bbox, caption_blocks, TABLE_CAPTION_PATTERN, repeated_patterns=repeated_patterns
                )
                markdown = _grid_to_markdown(_table_grid(page, table))
                tables.append(
                    ExtractedTable(
                        table_id=f"{pdf_sha256[:8]}_p{page_number}_t{table_index}",
                        page=page_number,
                        markdown=markdown,
                        row_count=table.row_count,
                        col_count=table.col_count,
                        bbox=tuple(bbox),
                        caption=caption,
                    )
                )
    return tables


def table_to_node(table: ExtractedTable, pdf_sha256: str, source_file: str) -> TextNode:
    return TextNode(
        id_=table.table_id,
        text=table.markdown,
        metadata={
            "content_type": "table",
            "table_id": table.table_id,
            "page": table.page,
            "row_count": table.row_count,
            "col_count": table.col_count,
            "bbox": table.bbox,
            "caption": table.caption,
            "pdf_sha256": pdf_sha256,
            "source_file": source_file,
        },
    )
