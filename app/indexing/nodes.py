"""Render data/processed/<sha12>/nodes.jsonl rows into the TextNodes used by
the PGVector text index (Milestone 4): one of content_type "text", "table" or
"image_description" per source node.

All rendered text comes from document/VLM content, which is UNTRUSTED data:
stored as-is in the index, never interpreted as instructions.
"""

from typing import Any

from llama_index.core.schema import TextNode


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


def _table_node(raw: dict[str, Any], pdf_sha256: str, source_file: str) -> TextNode:
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
        text=_table_text(meta["page"], meta.get("caption"), raw.get("text", "")),
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
    """One TextNode per source text chunk / table / image, for the text index."""
    nodes: list[TextNode] = []
    for raw in raw_nodes:
        content_type = raw.get("metadata", {}).get("content_type")
        if content_type == "text":
            nodes.append(_text_chunk_node(raw, pdf_sha256, source_file))
        elif content_type == "table":
            nodes.append(_table_node(raw, pdf_sha256, source_file))
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
