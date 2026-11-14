"""PGVectorStore wiring for Milestone 4: a text index and an image index.

Two Postgres tables (actual SQL names are "data_<table_name>", a LlamaIndex
convention): Settings.pgvector_text_table holds BGE-M3 text embeddings
(configurable dim, hybrid vector+French-full-text search, HNSW cosine) and
Settings.pgvector_image_table holds precomputed MobileCLIP image embeddings
(configurable dim, vector-only HNSW cosine). Node content is untrusted
document/VLM data: stored and returned as-is, never executed.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.postgres import PGVectorStore

from app.config import Settings, get_settings
from app.images.clip_encoder import ClipEncoder
from app.indexing.embedder import TextEmbedder, get_query_embedder, unload_ollama_models
from app.indexing.nodes import build_image_index_nodes, build_text_index_nodes

HNSW_KWARGS = {
    "hnsw_m": 16,
    "hnsw_ef_construction": 64,
    "hnsw_ef_search": 40,
    "hnsw_dist_method": "vector_cosine_ops",
}
SNIPPET_LENGTH = 120


def _db_params(database_url: str) -> dict[str, str | None]:
    parts = urlsplit(database_url)
    return {
        "host": parts.hostname,
        "port": str(parts.port) if parts.port else None,
        "database": parts.path.lstrip("/"),
        "user": parts.username,
        "password": parts.password,
    }


def get_text_store(settings: Settings | None = None) -> PGVectorStore:
    settings = settings or get_settings()
    return PGVectorStore.from_params(
        **_db_params(settings.database_url),
        table_name=settings.pgvector_text_table,
        embed_dim=settings.pgvector_text_dim,
        hybrid_search=settings.pgvector_hybrid_search,
        text_search_config=settings.pgvector_text_search_config,
        use_jsonb=True,
        hnsw_kwargs=HNSW_KWARGS,
    )


def get_image_store(settings: Settings | None = None) -> PGVectorStore:
    settings = settings or get_settings()
    return PGVectorStore.from_params(
        **_db_params(settings.database_url),
        table_name=settings.pgvector_image_table,
        embed_dim=settings.pgvector_image_dim,
        hybrid_search=False,
        use_jsonb=True,
        hnsw_kwargs=HNSW_KWARGS,
    )


def _pdf_filter(pdf_sha256: str) -> MetadataFilters:
    return MetadataFilters(filters=[MetadataFilter(key="pdf_sha256", value=pdf_sha256)])


def _existing_node_ids(store: PGVectorStore, pdf_sha256: str) -> set[str]:
    return {node.node_id for node in store.get_nodes(filters=_pdf_filter(pdf_sha256))}


def _count_nodes(store: PGVectorStore, pdf_sha256: str) -> int:
    return len(store.get_nodes(filters=_pdf_filter(pdf_sha256)))


def _load_clip_vectors(processed_dir: Path) -> dict[str, np.ndarray]:
    index_path = processed_dir / "clip_index.json"
    npy_path = processed_dir / "clip_embeddings.npy"
    if not index_path.is_file() or not npy_path.is_file():
        return {}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    vectors = np.load(npy_path)
    return dict(zip(index["image_ids"], vectors, strict=True))


def _load_raw_nodes(processed_dir: Path) -> list[dict]:
    lines = (processed_dir / "nodes.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@dataclass
class BuildResult:
    pdf_sha256: str
    text_nodes_total: int
    text_nodes_added: int
    image_nodes_total: int
    image_nodes_added: int
    text_count_in_db: int
    image_count_in_db: int


def build_index(
    processed_dir: Path, force: bool = False, settings: Settings | None = None
) -> BuildResult:
    """Build/update the text and image PGVector indexes for one processed document.

    Idempotent: node ids are deterministic (pdf_sha256 + content_type +
    original id), so re-running without --force only adds nodes not already
    present. --force deletes every node for this pdf_sha256 in both tables
    first, then rebuilds from the current nodes.jsonl/clip_embeddings.npy.
    """
    settings = settings or get_settings()
    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
    pdf_sha256 = manifest["pdf_sha256"]
    source_file = manifest["source_file"]

    raw_nodes = _load_raw_nodes(processed_dir)
    text_nodes = build_text_index_nodes(raw_nodes, pdf_sha256, source_file)
    clip_vectors = _load_clip_vectors(processed_dir)
    image_nodes = build_image_index_nodes(raw_nodes, clip_vectors, pdf_sha256, source_file)

    unload_ollama_models(settings)
    embedder = TextEmbedder(settings)
    try:
        if text_nodes:
            vectors = embedder.embed_texts([node.get_content() for node in text_nodes])
            for node, vector in zip(text_nodes, vectors, strict=True):
                node.embedding = vector.tolist()
    finally:
        embedder.unload()

    text_store = get_text_store(settings)
    image_store = get_image_store(settings)

    if force:
        text_store.delete_nodes(filters=_pdf_filter(pdf_sha256))
        image_store.delete_nodes(filters=_pdf_filter(pdf_sha256))
        new_text_nodes = text_nodes
        new_image_nodes = image_nodes
    else:
        existing_text_ids = _existing_node_ids(text_store, pdf_sha256)
        existing_image_ids = _existing_node_ids(image_store, pdf_sha256)
        new_text_nodes = [n for n in text_nodes if n.node_id not in existing_text_ids]
        new_image_nodes = [n for n in image_nodes if n.node_id not in existing_image_ids]

    if new_text_nodes:
        text_store.add(new_text_nodes)
    if new_image_nodes:
        image_store.add(new_image_nodes)

    return BuildResult(
        pdf_sha256=pdf_sha256,
        text_nodes_total=len(text_nodes),
        text_nodes_added=len(new_text_nodes),
        image_nodes_total=len(image_nodes),
        image_nodes_added=len(new_image_nodes),
        text_count_in_db=_count_nodes(text_store, pdf_sha256),
        image_count_in_db=_count_nodes(image_store, pdf_sha256),
    )


@dataclass
class SearchResult:
    node_id: str
    page: int | None
    content_type: str | None
    score: float
    snippet: str


def _to_search_results(result: VectorStoreQueryResult) -> list[SearchResult]:
    nodes: list[BaseNode] = result.nodes or []
    similarities = result.similarities or [0.0] * len(nodes)
    results = []
    for node, score in zip(nodes, similarities, strict=True):
        text = node.get_content()
        results.append(
            SearchResult(
                node_id=node.node_id,
                page=node.metadata.get("page"),
                content_type=node.metadata.get("content_type"),
                score=float(score),
                snippet=text[:SNIPPET_LENGTH],
            )
        )
    return results


def search_text(
    query: str,
    k: int = 5,
    content_type: str | None = None,
    mode: str = "vector",
    settings: Settings | None = None,
) -> list[SearchResult]:
    """French semantic search over the text index. query is untrusted user input,
    embedded and compared by vector similarity -- never executed as code/SQL."""
    settings = settings or get_settings()
    store = get_text_store(settings)
    embedder = get_query_embedder(settings)
    query_embedding = embedder.embed_query(query).tolist()

    filters = (
        MetadataFilters(filters=[MetadataFilter(key="content_type", value=content_type)])
        if content_type
        else None
    )
    query_mode = VectorStoreQueryMode.HYBRID if mode == "hybrid" else VectorStoreQueryMode.DEFAULT
    vs_query = VectorStoreQuery(
        query_embedding=query_embedding,
        query_str=query,
        similarity_top_k=k,
        mode=query_mode,
        filters=filters,
    )
    return _to_search_results(store.query(vs_query))


@dataclass
class ImageSearchResult:
    image_id: str
    image_path: str | None
    page: int | None
    caption: str | None
    score: float


def _to_image_search_results(result: VectorStoreQueryResult) -> list[ImageSearchResult]:
    nodes: list[BaseNode] = result.nodes or []
    similarities = result.similarities or [0.0] * len(nodes)
    results = []
    for node, score in zip(nodes, similarities, strict=True):
        results.append(
            ImageSearchResult(
                image_id=node.metadata.get("image_id", node.node_id),
                image_path=node.metadata.get("image_path"),
                page=node.metadata.get("page"),
                caption=node.metadata.get("caption"),
                score=float(score),
            )
        )
    return results


def search_image(
    image_path: Path, k: int = 5, settings: Settings | None = None
) -> list[ImageSearchResult]:
    """CLIP image-to-image search over the image index."""
    settings = settings or get_settings()
    store = get_image_store(settings)
    encoder = ClipEncoder(settings)
    try:
        vector = encoder.encode_images([image_path])[0].tolist()
    finally:
        encoder.unload()
    vs_query = VectorStoreQuery(query_embedding=vector, similarity_top_k=k)
    return _to_image_search_results(store.query(vs_query))


def search_image_text(
    text: str, k: int = 5, settings: Settings | None = None
) -> list[ImageSearchResult]:
    """CLIP text-to-image search. text must be ENGLISH (the CLIP text tower is
    English-only) -- do not pass French user queries here."""
    settings = settings or get_settings()
    store = get_image_store(settings)
    encoder = ClipEncoder(settings)
    try:
        vector = encoder.encode_texts([text])[0].tolist()
    finally:
        encoder.unload()
    vs_query = VectorStoreQuery(query_embedding=vector, similarity_top_k=k)
    return _to_image_search_results(store.query(vs_query))
