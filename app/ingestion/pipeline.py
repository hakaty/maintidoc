"""End-to-end extraction pipeline: PDF -> text/table/image nodes -> data/processed/<sha[:12]>/."""

import hashlib
import importlib.metadata
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pymupdf
from llama_index.core.schema import BaseNode, ImageNode, TextNode

from app.ingestion.chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_documents
from app.ingestion.extract_images import (
    DEFAULT_ICON_MAX_PX,
    DEFAULT_LOGO_PAGE_RATIO,
    DEFAULT_MIN_IMAGE_PX,
    extract_images,
    image_to_node,
)
from app.ingestion.extract_tables import extract_tables, table_to_node
from app.ingestion.extract_text import extract_text
from app.ingestion.extract_vector_figures import (
    DEFAULT_CLUSTER_MARGIN_PT,
    DEFAULT_MIN_AREA_PT2,
    extract_vector_figures,
)
from app.ingestion.repeated_lines import detect_repeated_lines, remove_repeated_lines

logger = logging.getLogger(__name__)

_NODE_CLASSES: dict[str, type[BaseNode]] = {"TextNode": TextNode, "ImageNode": ImageNode}


class CorruptedPdfError(Exception):
    pass


@dataclass
class ExtractionParams:
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    min_image_px: int = DEFAULT_MIN_IMAGE_PX
    logo_page_ratio: float = DEFAULT_LOGO_PAGE_RATIO
    icon_max_px: int = DEFAULT_ICON_MAX_PX
    vector_min_area: float = DEFAULT_MIN_AREA_PT2
    vector_cluster_margin: float = DEFAULT_CLUSTER_MARGIN_PT


@dataclass
class ExtractionResult:
    processed_dir: Path
    manifest: dict
    unchanged: bool


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_pdf_or_raise(pdf_path: Path) -> pymupdf.Document:
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:  # pymupdf raises its own error types on malformed files
        raise CorruptedPdfError(f"{pdf_path}: cannot be opened ({exc})") from exc
    if doc.is_encrypted:
        doc.close()
        raise CorruptedPdfError(f"{pdf_path}: PDF is encrypted/password-protected")
    try:
        if doc.page_count > 0:
            doc[0].get_text()
    except Exception as exc:
        doc.close()
        raise CorruptedPdfError(f"{pdf_path}: cannot be read ({exc})") from exc
    return doc


def node_to_line(node: BaseNode) -> str:
    return json.dumps(node.to_dict(), default=str, ensure_ascii=False)


def node_from_line(line: str) -> BaseNode:
    data = json.loads(line)
    cls = _NODE_CLASSES.get(data.get("class_name", "TextNode"), TextNode)
    return cls.from_dict(data)


def _library_versions() -> dict[str, str]:
    versions = {"pymupdf": pymupdf.__version__}
    for package in ("llama-index-core", "llama-index-readers-file"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def run_extraction(
    pdf_path: Path,
    processed_root: Path,
    media_dir: Path,
    params: ExtractionParams | None = None,
    force: bool = False,
) -> ExtractionResult:
    params = params or ExtractionParams()

    if not pdf_path.is_file():
        raise CorruptedPdfError(f"{pdf_path}: file not found")

    pdf_sha256 = _sha256_file(pdf_path)
    doc = _open_pdf_or_raise(pdf_path)
    total_pages = doc.page_count
    doc.close()

    processed_dir = processed_root / pdf_sha256[:12]
    manifest_path = processed_dir / "manifest.json"
    nodes_path = processed_dir / "nodes.jsonl"

    if not force and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("pdf_sha256") == pdf_sha256 and existing.get("parameters") == asdict(
            params
        ):
            logger.info("unchanged: %s (sha256=%s)", pdf_path, pdf_sha256[:12])
            return ExtractionResult(processed_dir=processed_dir, manifest=existing, unchanged=True)

    # extract_text runs first (not extract_tables, despite find_tables() being
    # the costly step it also shares via the bbox cache -- whichever of the
    # two runs first warms that cache either way): repeated_lines is detected
    # from its output and threaded into extract_tables/extract_images/
    # extract_vector_figures so their caption picker never returns a running
    # header/footer line.
    documents = extract_text(pdf_path, pdf_sha256)
    repeated_lines = detect_repeated_lines(documents)
    repeated_patterns = frozenset(repeated_lines)
    cleaned_documents = remove_repeated_lines(documents, repeated_lines)

    tables = extract_tables(pdf_path, pdf_sha256, repeated_patterns)
    images, skipped_images = extract_images(
        pdf_path,
        pdf_sha256,
        media_dir,
        params.min_image_px,
        params.logo_page_ratio,
        params.icon_max_px,
        repeated_patterns,
    )

    table_bboxes_by_page: dict[int, list[pymupdf.Rect]] = {}
    for table in tables:
        table_bboxes_by_page.setdefault(table.page, []).append(pymupdf.Rect(table.bbox))

    embedded_bboxes_by_page: dict[int, list[pymupdf.Rect]] = {}
    for image in images:
        embedded_bboxes_by_page.setdefault(image.page, []).append(pymupdf.Rect(image.bbox))

    vector_figures = extract_vector_figures(
        pdf_path,
        pdf_sha256,
        media_dir,
        table_bboxes_by_page,
        embedded_bboxes_by_page,
        params.vector_min_area,
        params.vector_cluster_margin,
        repeated_patterns=repeated_patterns,
    )
    all_images = images + vector_figures

    image_ids_by_page: dict[int, list[str]] = {}
    for image in all_images:
        image_ids_by_page.setdefault(image.page, []).append(image.image_id)

    table_ids_by_page: dict[int, list[str]] = {}
    for table in tables:
        table_ids_by_page.setdefault(table.page, []).append(table.table_id)

    chunks = chunk_documents(
        cleaned_documents,
        pdf_sha256,
        str(pdf_path),
        image_ids_by_page,
        table_ids_by_page,
        chunk_size=params.chunk_size,
        chunk_overlap=params.chunk_overlap,
    )

    nodes: list[BaseNode] = list(chunks)
    nodes.extend(table_to_node(table, pdf_sha256, str(pdf_path)) for table in tables)
    nodes.extend(image_to_node(image, pdf_sha256, str(pdf_path)) for image in all_images)

    processed_dir.mkdir(parents=True, exist_ok=True)
    with nodes_path.open("w", encoding="utf-8") as fh:
        for node in nodes:
            fh.write(node_to_line(node) + "\n")

    scanned_pages = sorted(
        document.metadata["page"] for document in documents if document.metadata["scanned"]
    )

    images_by_source: dict[str, int] = {}
    images_by_kind: dict[str, int] = {}
    for image in all_images:
        images_by_source[image.source] = images_by_source.get(image.source, 0) + 1
        images_by_kind[image.kind] = images_by_kind.get(image.kind, 0) + 1

    skipped_by_reason: dict[str, int] = {}
    for skipped in skipped_images:
        skipped_by_reason[skipped.reason] = skipped_by_reason.get(skipped.reason, 0) + 1

    manifest = {
        "source_file": str(pdf_path),
        "pdf_sha256": pdf_sha256,
        "total_pages": total_pages,
        "extraction_date": datetime.now(UTC).isoformat(),
        "parameters": asdict(params),
        "counts": {
            "text_chunks": len(chunks),
            "tables": len(tables),
            "images_kept": len(all_images),
            "images_skipped": len(skipped_images),
            "images_by_source": images_by_source,
            "images_by_kind": images_by_kind,
            "images_skipped_by_reason": skipped_by_reason,
        },
        "scanned_pages": scanned_pages,
        "skipped_images": [asdict(skipped) for skipped in skipped_images],
        "repeated_lines_removed": sorted(repeated_lines.values()),
        "library_versions": _library_versions(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return ExtractionResult(processed_dir=processed_dir, manifest=manifest, unchanged=False)
