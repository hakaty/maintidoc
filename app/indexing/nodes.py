"""Render data/processed/<sha12>/nodes.jsonl rows into the TextNodes used by
the PGVector text index (Milestone 4): one of content_type "text", "table" or
"image_description" per source node.

All rendered text comes from document/VLM content, which is UNTRUSTED data:
stored as-is in the index, never interpreted as instructions.
"""

import re
from typing import Any

from llama_index.core.schema import TextNode

# M7.4b fix: 90 CHARACTERS still produced near-full sentence fragments in the
# table cell (e.g. "L'image montre un symbole de sécurité électrique,
# représenté par un triangle jaune avec…") -- not the short, scannable cell
# the task asks for (e.g. "triangle jaune avec « EX » noir"). Capped by WORD
# count instead, after stripping the VLM's generic lead-in clause.
MAX_PICTOGRAM_DESCRIPTION_WORDS = 12
# A flat word cap can cut a description off right before its one crucial
# identifying detail -- a real regression hit on the real manual: "Un
# triangle jaune avec un triangle noir a l'interieur, contenant les
# lettres…" lost the actual letters ("EX") that make the pictogram
# recognizable. If a quoted span or a short ALL-CAPS code (like "EX") is
# present, the cap is extended (up to this ceiling) to keep it whole rather
# than silently dropping the one detail a technician would look for.
MAX_PICTOGRAM_DESCRIPTION_WORDS_WITH_CODE = 20

_PICTOGRAM_BOILERPLATE = re.compile(
    r"^l['’]image\s+(?:montre|représente)\s+une?\s+(?:symbole|pictogramme)"
    r"(?:\s+de\s+sécurité)?,?\s*(?:signalant|représenté(?:e)?\s+par|montrant|d['’])?\s*",
    re.IGNORECASE,
)
_PICTOGRAM_CODE_PATTERN = re.compile(r'"[^"]+"|«[^»]+»|\b[A-Z]{2,4}\b')


def text_node_id(pdf_sha256: str, content_type: str, original_id: str) -> str:
    """Deterministic id: re-running the build on the same document/node never
    produces a different id for the same logical chunk/table/image, so a
    second build can detect what's already indexed instead of duplicating it.
    """
    return f"{pdf_sha256}:{content_type}:{original_id}"


def _common_metadata(raw_meta: dict[str, Any], pdf_sha256: str, source_file: str) -> dict[str, Any]:
    return {
        "pdf_sha256": pdf_sha256,
        "source_file": source_file,
        "page": raw_meta.get("page"),
        "chunk_index": raw_meta.get("chunk_index"),
        "related_image_ids": raw_meta.get("related_image_ids", []),
        "related_table_ids": raw_meta.get("related_table_ids", []),
        "kind_fr": raw_meta.get("kind_fr"),
        "clip_label": raw_meta.get("clip_label"),
    }


def _text_chunk_node(raw: dict[str, Any], pdf_sha256: str, source_file: str) -> TextNode:
    meta = raw["metadata"]
    metadata = _common_metadata(meta, pdf_sha256, source_file)
    metadata["content_type"] = "text"
    return TextNode(
        id_=text_node_id(pdf_sha256, "text", raw["id_"]),
        text=raw.get("text", ""),
        metadata=metadata,
    )


def _table_text(page: int, caption: str | None, markdown: str) -> str:
    header = f"Tableau (page {page})"
    if caption:
        header += f" — {caption}"
    return f"{header}\n{markdown}"


def _bbox_overlap_ratio(
    inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]
) -> float:
    """Fraction of `inner`'s area that falls inside `outer` -- used to decide
    whether a pictogram image "is" a given table cell, not just near it."""
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    overlap_x = max(0.0, min(ix1, ox1) - max(ix0, ox0))
    overlap_y = max(0.0, min(iy1, oy1) - max(iy0, oy0))
    overlap_area = overlap_x * overlap_y
    inner_area = max(1e-6, (ix1 - ix0) * (iy1 - iy0))
    return overlap_area / inner_area


def _short_pictogram_description(description: str | None) -> str:
    """Section 2.1 fix (M7.4), shortened further in M7.4b: a SHORT, scannable
    description for the table cell, not the full (often 1-2 sentence) VLM
    description -- the full description always remains available separately
    on the image's own "image_description" node, this is cell display text
    only. Strips the VLM's generic lead-in clause ("L'image représente un
    pictogramme de sécurité, signalant...") then clamps to
    MAX_PICTOGRAM_DESCRIPTION_WORDS words, never mid-word."""
    if not description:
        return ""
    text = description.strip().rstrip(".")
    stripped = _PICTOGRAM_BOILERPLATE.sub("", text, count=1).strip()
    if stripped:
        text = stripped
    words = text.split()
    if len(words) <= MAX_PICTOGRAM_DESCRIPTION_WORDS:
        return text

    codes = _PICTOGRAM_CODE_PATTERN.findall(text)
    cap = MAX_PICTOGRAM_DESCRIPTION_WORDS
    if codes and not any(code in " ".join(words[:cap]) for code in codes):
        # Extend word-by-word (never past the higher ceiling) until the
        # first missing code is fully included, instead of cutting right
        # before it.
        for cap in range(
            MAX_PICTOGRAM_DESCRIPTION_WORDS + 1, MAX_PICTOGRAM_DESCRIPTION_WORDS_WITH_CODE + 1
        ):
            if any(code in " ".join(words[:cap]) for code in codes):
                break
    if cap >= len(words):
        return text
    return " ".join(words[:cap]) + "…"


ImageHint = tuple[tuple[float, float, float, float], "str | None", "str | None"]


def _image_lookup_by_page(
    raw_nodes: list[dict[str, Any]],
) -> dict[int, list[ImageHint]]:
    """page -> [(bbox, description, description_status), ...] for every
    already-extracted, already-VLM-described image node. Built once and
    reused for every table on that page (section 2.1 fix)."""
    by_page: dict[int, list[ImageHint]] = {}
    for raw in raw_nodes:
        meta = raw.get("metadata", {})
        if meta.get("content_type") != "image":
            continue
        bbox = meta.get("bbox")
        page = meta.get("page")
        if not bbox or page is None:
            continue
        by_page.setdefault(page, []).append(
            (tuple(bbox), meta.get("description"), meta.get("description_status"))
        )
    return by_page


def _unescape_cell(text: str) -> str:
    return text.replace("\\|", "|")


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|")


def _fill_pictogram_cells(
    markdown: str,
    cell_bboxes: list[list[list[float] | None]] | None,
    page: int,
    images_by_page: dict[int, list[ImageHint]],
) -> str:
    """Replace an EMPTY cell whose bbox contains an already-extracted,
    already-described pictogram image with "[pictogramme : <short
    description>]" (or "[pictogramme]" if the image has no usable
    description) -- section 2.1 fix. Reuses the existing extraction +
    VLM-description pipeline output (the pictograms on a page like
    "Symboles complémentaires" are already separately extracted, embedded
    images, each already described) instead of re-cropping/re-describing:
    same result, no duplicate VLM calls. No-op if there's nothing to patch
    (older processed data without cell_bboxes, or no matching image)."""
    if not cell_bboxes:
        return markdown
    candidates = images_by_page.get(page, [])
    if not candidates:
        return markdown

    lines = markdown.splitlines()
    if len(lines) < 3:  # header + separator + at least one data row
        return markdown

    changed = False
    out_lines = lines[:2]  # header + separator, unchanged
    data_lines = lines[2:]
    # cell_bboxes includes the header row (row 0); data rows start at index 1.
    for row_index, line in enumerate(data_lines, start=1):
        if row_index >= len(cell_bboxes):
            out_lines.append(line)
            continue
        # Exactly one leading/trailing "|" (markdown table row delimiters),
        # never `.strip("|")`: a row whose FIRST cell is empty (e.g. "||Risque
        # ...|") would otherwise collapse that empty cell's own delimiter too,
        # shifting every column left by one.
        inner = line[1:-1] if line.startswith("|") and line.endswith("|") else line
        cells = inner.split("|")
        row_bboxes = cell_bboxes[row_index]
        for col_index, cell_bbox in enumerate(row_bboxes):
            if col_index >= len(cells) or cell_bbox is None:
                continue
            if _unescape_cell(cells[col_index]).strip():
                continue  # cell already has real text, leave it alone
            match = next(
                (
                    (desc, status)
                    for bbox, desc, status in candidates
                    if _bbox_overlap_ratio(bbox, tuple(cell_bbox)) > 0.5
                ),
                None,
            )
            if match is None:
                continue
            desc, status = match
            short = _short_pictogram_description(desc) if status == "ok" else ""
            placeholder = f"[pictogramme : {short}]" if short else "[pictogramme]"
            cells[col_index] = _escape_cell(placeholder)
            changed = True
        out_lines.append("|" + "|".join(cells) + "|")

    return "\n".join(out_lines) if changed else markdown


def _table_row_nodes(
    markdown: str,
    page: int,
    caption: str | None,
    table_id: str,
    pdf_sha256: str,
    source_file: str,
) -> list[TextNode]:
    """Section 2.2 fix: ALSO index each row of a table as its own chunk
    (title + header + that row + page), on top of the existing whole-table
    chunk -- a row-level match (e.g. "Risque d'atmosphère explosive") scores
    higher against a question about that one row than the whole 14-row
    table competing against it as a single chunk."""
    lines = markdown.splitlines()
    if len(lines) < 3:
        return []
    header_line, _separator, *data_lines = lines
    header = header_line.strip("|")
    title = f"Tableau (page {page})"
    if caption:
        title += f" — {caption}"

    nodes = []
    for row_index, line in enumerate(data_lines):
        row = line[1:-1] if line.startswith("|") and line.endswith("|") else line
        if not row.strip("| "):
            continue
        text = f"{title}\n{header}\n{row}"
        nodes.append(
            TextNode(
                id_=text_node_id(pdf_sha256, "table_row", f"{table_id}_r{row_index}"),
                text=text,
                metadata={
                    "content_type": "table_row",
                    "table_id": table_id,
                    "page": page,
                    "caption": caption,
                    "row_index": row_index,
                    "pdf_sha256": pdf_sha256,
                    "source_file": source_file,
                },
            )
        )
    return nodes


def _patched_table_markdown(
    raw: dict[str, Any],
    images_by_page: dict[int, list[ImageHint]] | None,
) -> str:
    meta = raw["metadata"]
    markdown = raw.get("text", "")
    if images_by_page:
        markdown = _fill_pictogram_cells(
            markdown, meta.get("cell_bboxes"), meta["page"], images_by_page
        )
    return markdown


def _table_node(
    raw: dict[str, Any],
    pdf_sha256: str,
    source_file: str,
    markdown: str,
) -> TextNode:
    meta = raw["metadata"]
    metadata = _common_metadata(meta, pdf_sha256, source_file)
    metadata["content_type"] = "table"
    metadata["table_id"] = meta.get("table_id")
    metadata["caption"] = meta.get("caption")
    metadata["row_count"] = meta.get("row_count")
    metadata["col_count"] = meta.get("col_count")
    original_id = meta.get("table_id") or raw["id_"]
    return TextNode(
        id_=text_node_id(pdf_sha256, "table", original_id),
        text=_table_text(meta["page"], meta.get("caption"), markdown),
        metadata=metadata,
    )


def _image_description_text(
    page: int,
    kind_fr: str | None,
    caption: str | None,
    description: str | None,
    visible_text: str | None,
) -> str:
    header = f"Figure (page {page})"
    if kind_fr:
        header += f" — {kind_fr}"
    segments = [f"{header}."]
    if caption:
        segments.append(f"Légende : {caption}.")
    if description:
        segments.append(f"Description : {description}.")
    if visible_text:
        segments.append(f"Texte visible : {visible_text}.")
    return " ".join(segments)


def _image_description_node(raw: dict[str, Any], pdf_sha256: str, source_file: str) -> TextNode:
    meta = raw["metadata"]
    metadata = _common_metadata(meta, pdf_sha256, source_file)
    metadata["content_type"] = "image_description"
    metadata["image_id"] = meta["image_id"]
    metadata["image_path"] = raw.get("image_path")
    metadata["description_status"] = meta.get("description_status")
    metadata["safety_relevant"] = meta.get("safety_relevant")
    text = _image_description_text(
        meta["page"],
        meta.get("kind_fr"),
        meta.get("caption"),
        meta.get("description"),
        meta.get("visible_text"),
    )
    return TextNode(
        id_=text_node_id(pdf_sha256, "image_description", meta["image_id"]),
        text=text,
        metadata=metadata,
    )


def build_text_index_nodes(
    raw_nodes: list[dict[str, Any]], pdf_sha256: str, source_file: str
) -> list[TextNode]:
    """One TextNode per source text chunk / table (+ one per table ROW,
    section 2.2) / image, for the text index."""
    images_by_page = _image_lookup_by_page(raw_nodes)
    nodes: list[TextNode] = []
    for raw in raw_nodes:
        content_type = raw.get("metadata", {}).get("content_type")
        if content_type == "text":
            nodes.append(_text_chunk_node(raw, pdf_sha256, source_file))
        elif content_type == "table":
            markdown = _patched_table_markdown(raw, images_by_page)
            nodes.append(_table_node(raw, pdf_sha256, source_file, markdown))
            meta = raw["metadata"]
            table_id = meta.get("table_id") or raw["id_"]
            nodes.extend(
                _table_row_nodes(
                    markdown, meta["page"], meta.get("caption"), table_id, pdf_sha256, source_file
                )
            )
        elif content_type == "image":
            nodes.append(_image_description_node(raw, pdf_sha256, source_file))
    return nodes


def build_image_index_nodes(
    raw_nodes: list[dict[str, Any]],
    clip_vectors: dict[str, Any],
    pdf_sha256: str,
    source_file: str,
) -> list[TextNode]:
    """One TextNode per image with a precomputed CLIP vector, for the image index."""
    nodes: list[TextNode] = []
    for raw in raw_nodes:
        meta = raw.get("metadata", {})
        if meta.get("content_type") != "image":
            continue
        image_id = meta["image_id"]
        vector = clip_vectors.get(image_id)
        if vector is None:
            continue
        node = TextNode(
            id_=text_node_id(pdf_sha256, "image", image_id),
            text=meta.get("caption") or meta.get("description") or "",
            metadata={
                "content_type": "image",
                "image_id": image_id,
                "image_path": raw.get("image_path"),
                "page": meta.get("page"),
                "caption": meta.get("caption"),
                "kind_fr": meta.get("kind_fr"),
                "clip_label": meta.get("clip_label"),
                "description": meta.get("description"),
                "pdf_sha256": pdf_sha256,
                "source_file": source_file,
            },
        )
        node.embedding = [float(x) for x in vector]
        nodes.append(node)
    return nodes
