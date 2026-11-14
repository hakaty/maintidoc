"""Per-page sentence-based chunking, linked to co-located images/tables."""

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument
from llama_index.core.schema import TextNode

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64


def chunk_documents(
    documents: list[LlamaDocument],
    pdf_sha256: str,
    source_file: str,
    image_ids_by_page: dict[int, list[str]],
    table_ids_by_page: dict[int, list[str]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextNode]:
    """SentenceSplitter applied per page Document: a chunk never spans two pages."""
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes: list[TextNode] = []

    for document in documents:
        page = document.metadata["page"]
        if not document.text.strip():
            continue

        for chunk_index, chunk_text in enumerate(splitter.split_text(document.text)):
            nodes.append(
                TextNode(
                    id_=f"{pdf_sha256[:8]}_p{page}_c{chunk_index}",
                    text=chunk_text,
                    metadata={
                        "content_type": "text",
                        "page": page,
                        "chunk_index": chunk_index,
                        "pdf_sha256": pdf_sha256,
                        "source_file": source_file,
                        "related_image_ids": image_ids_by_page.get(page, []),
                        "related_table_ids": table_ids_by_page.get(page, []),
                    },
                )
            )
    return nodes
