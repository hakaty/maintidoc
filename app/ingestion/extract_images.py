"""Image extraction via PyMuPDF, with logo/tiny-image filtering and dedup.

Description and CLIP-embedding fields are left empty: filled in by Milestone 3.
"""

import hashlib
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from llama_index.core.schema import ImageNode
from PIL import Image as PILImage

from app.ingestion._captions import (
    FIGURE_CAPTION_PATTERN,
    find_caption,
    get_table_bboxes,
    page_text_blocks,
)

logger = logging.getLogger(__name__)

DEFAULT_MIN_IMAGE_PX = 80
DEFAULT_LOGO_PAGE_RATIO = 0.5
DEFAULT_ICON_MAX_PX = 150


@dataclass
class ExtractedImage:
    image_id: str
    page: int
    image_path: Path
    width: int
    height: int
    sha256: str
    bbox: tuple[float, float, float, float]
    caption: str | None
    other_pages: list[int] = field(default_factory=list)
    source: str = "embedded"  # "embedded" | "vector_render"
    kind: str = "figure"  # "figure" | "icon" (both sides < icon_max_px)


@dataclass
class SkippedImage:
    sha256: str
    reason: str
    pages: list[int]


@dataclass
class _Occurrence:
    page: int
    xref: int
    sha256: str
    width: int
    height: int
    data: bytes
    bbox: tuple[float, float, float, float]


def _collect_occurrences(doc: pymupdf.Document) -> list[_Occurrence]:
    occurrences: list[_Occurrence] = []
    for page_index, page in enumerate(doc):
        page_number = page_index + 1
        seen_xrefs_on_page: set[int] = set()
        for image in page.get_images(full=True):
            xref = image[0]
            if xref in seen_xrefs_on_page:
                continue
            seen_xrefs_on_page.add(xref)

            info = doc.extract_image(xref)
            data: bytes = info["image"]
            rects = page.get_image_rects(xref)
            bbox = tuple(rects[0]) if rects else (0.0, 0.0, 0.0, 0.0)

            occurrences.append(
                _Occurrence(
                    page=page_number,
                    xref=xref,
                    sha256=hashlib.sha256(data).hexdigest(),
                    width=info["width"],
                    height=info["height"],
                    data=data,
                    bbox=bbox,
                )
            )
    return occurrences


def _save_png(data: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = PILImage.open(io.BytesIO(data))
    if image.mode == "CMYK":
        image = image.convert("RGB")
    image.save(path, format="PNG")


def extract_images(
    pdf_path: Path,
    pdf_sha256: str,
    media_dir: Path,
    min_image_px: int = DEFAULT_MIN_IMAGE_PX,
    logo_page_ratio: float = DEFAULT_LOGO_PAGE_RATIO,
    icon_max_px: int = DEFAULT_ICON_MAX_PX,
    repeated_patterns: frozenset[str] = frozenset(),
) -> tuple[list[ExtractedImage], list[SkippedImage]]:
    kept: list[ExtractedImage] = []
    skipped: list[SkippedImage] = []

    with pymupdf.open(pdf_path) as doc:
        total_pages = len(doc)
        occurrences = _collect_occurrences(doc)

        by_sha: dict[str, list[_Occurrence]] = {}
        for occ in occurrences:
            by_sha.setdefault(occ.sha256, []).append(occ)

        output_dir = media_dir / pdf_sha256[:12]
        page_image_index: dict[int, int] = {}

        for sha256, group in by_sha.items():
            group.sort(key=lambda occ: occ.page)
            pages = sorted({occ.page for occ in group})

            if total_pages and len(pages) / total_pages >= logo_page_ratio:
                skipped.append(
                    SkippedImage(sha256=sha256, reason="repeated_on_most_pages", pages=pages)
                )
                continue

            first = group[0]
            if min(first.width, first.height) < min_image_px:
                skipped.append(SkippedImage(sha256=sha256, reason="too_small", pages=pages))
                continue

            page = doc[first.page - 1]
            table_bboxes = get_table_bboxes(str(pdf_path), pdf_sha256, first.page, page)
            caption_blocks = page_text_blocks(page, exclude=table_bboxes)
            caption = find_caption(
                pymupdf.Rect(first.bbox),
                caption_blocks,
                FIGURE_CAPTION_PATTERN,
                repeated_patterns=repeated_patterns,
            )

            index_on_page = page_image_index.get(first.page, 0)
            page_image_index[first.page] = index_on_page + 1
            image_id = f"{pdf_sha256[:8]}_p{first.page}_i{index_on_page}"
            image_path = output_dir / f"{image_id}.png"
            _save_png(first.data, image_path)

            kind = (
                "icon"
                if first.width < icon_max_px and first.height < icon_max_px
                else "figure"
            )
            kept.append(
                ExtractedImage(
                    image_id=image_id,
                    page=first.page,
                    image_path=image_path,
                    width=first.width,
                    height=first.height,
                    sha256=sha256,
                    bbox=first.bbox,
                    caption=caption,
                    other_pages=[p for p in pages if p != first.page],
                    source="embedded",
                    kind=kind,
                )
            )

    return kept, skipped


def image_to_node(image: ExtractedImage, pdf_sha256: str, source_file: str) -> ImageNode:
    return ImageNode(
        id_=image.image_id,
        image_path=str(image.image_path),
        text=image.caption or "",
        metadata={
            "content_type": "image",
            "image_id": image.image_id,
            "sha256": image.sha256,
            "page": image.page,
            "bbox": image.bbox,
            "caption": image.caption,
            "width": image.width,
            "height": image.height,
            "other_pages": image.other_pages,
            "source": image.source,
            "kind": image.kind,
            "pdf_sha256": pdf_sha256,
            "source_file": source_file,
            "description": "",
            "clip_embedding": None,
        },
    )
