"""Node retrieval for RAG answers: vector/hybrid search over the text index,
with optional query expansion for paraphrased questions, fused via
Reciprocal Rank Fusion, then deduped/budgeted down to a context the LLM can
actually use.

question/image content is UNTRUSTED user input: embedded and compared by
vector similarity, never executed as code/SQL/instructions.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

from app.config import Settings, get_settings
from app.images.clip_encoder import ClipEncoder
from app.indexing.embedder import get_query_embedder
from app.indexing.stores import get_image_store, get_text_store
from app.llm_factory import get_llm

logger = logging.getLogger(__name__)

# Rough chars-per-token estimate: no tokenizer dependency added just for
# budgeting context, BGE-M3's own tokenizer is not representative of Qwen's.
CHARS_PER_TOKEN = 4

QUERY_REWRITE_PROMPT_PATH = Path("prompts") / "query_rewrite_v1.md"
QUERY_REWRITE_NUM_PREDICT = 150


@dataclass
class RetrievedNode:
    node_id: str
    page: int | None
    content_type: str | None
    score: float
    text: str
    image_id: str | None
    image_path: str | None
    safety_relevant: bool | None


@dataclass
class RetrievalResult:
    nodes: list[RetrievedNode]
    best_score: float
    refused: bool
    reformulations: list[str] = field(default_factory=list)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _to_retrieved_node(node: BaseNode, score: float) -> RetrievedNode:
    meta = node.metadata
    return RetrievedNode(
        node_id=node.node_id,
        page=meta.get("page"),
        content_type=meta.get("content_type"),
        score=float(score),
        text=node.get_content(),
        image_id=meta.get("image_id"),
        image_path=meta.get("image_path"),
        safety_relevant=meta.get("safety_relevant"),
    )


def _dedupe_by_best_score(
    nodes: list[BaseNode], similarities: list[float]
) -> list[RetrievedNode]:
    """Hybrid mode can return the same node via both the dense and sparse
    signal: keep only the best-scoring occurrence of each node id."""
    seen: set[str] = set()
    deduped: list[RetrievedNode] = []
    pairs = sorted(zip(nodes, similarities, strict=True), key=lambda ns: -ns[1])
    for node, score in pairs:
        if node.node_id in seen:
            continue
        seen.add(node.node_id)
        deduped.append(_to_retrieved_node(node, score))
    return deduped


def _rrf_fuse(ranked_lists: list[list[BaseNode]], k_rrf: int) -> list[BaseNode]:
    """Reciprocal Rank Fusion: combines several ranked node lists (original
    question + reformulations) into one consensus ranking, by rank only --
    not by raw similarity scores, which aren't comparable across queries."""
    rrf_scores: dict[str, float] = {}
    node_by_id: dict[str, BaseNode] = {}
    for ranked in ranked_lists:
        for rank, node in enumerate(ranked, start=1):
            rrf_scores[node.node_id] = rrf_scores.get(node.node_id, 0.0) + 1.0 / (k_rrf + rank)
            node_by_id.setdefault(node.node_id, node)
    ordered_ids = sorted(rrf_scores, key=lambda node_id: -rrf_scores[node_id])
    return [node_by_id[node_id] for node_id in ordered_ids]


def _apply_budget(
    deduped: list[RetrievedNode], max_nodes: int, budget_tokens: int
) -> list[RetrievedNode]:
    """Keep up to max_nodes nodes, best score first, without exceeding the
    token budget. Tables/figures are never truncated: a node that would blow
    the budget is skipped whole, not cut short."""
    kept: list[RetrievedNode] = []
    used_tokens = 0
    for rn in deduped:
        if len(kept) >= max_nodes:
            break
        cost = _estimate_tokens(rn.text)
        if kept and used_tokens + cost > budget_tokens:
            continue
        kept.append(rn)
        used_tokens += cost
    return kept


def expand_query(question: str, settings: Settings) -> list[str]:
    """2-3 short French reformulations using manual vocabulary (Qwen).

    Reformulations are retrieval aids ONLY: never shown to the user as
    facts, never cited. Falls back to [] (original question alone used) on
    any failure: unreachable LLM, malformed JSON, non-list response.
    """
    try:
        template = QUERY_REWRITE_PROMPT_PATH.read_text(encoding="utf-8")
        prompt = template.format(question=question)
        llm = get_llm(settings, temperature=0.0, num_predict=QUERY_REWRITE_NUM_PREDICT)
        response = llm.invoke(prompt)
        data = json.loads(str(response.content))
    except Exception:  # noqa: BLE001 -- expansion is best-effort, never blocks retrieval
        logger.warning("Query expansion failed; falling back to the original question only")
        return []
    if not isinstance(data, list):
        return []
    return [str(item).strip() for item in data if str(item).strip()][:3]


def _query_text_store(
    query_text: str,
    store,
    k: int,
    mode: str,
    settings: Settings,
    content_type: str | None = None,
    pdf_sha256: str | None = None,
) -> tuple[list[BaseNode], list[float]]:
    embedder = get_query_embedder(settings)
    query_embedding = embedder.embed_query(query_text).tolist()
    query_mode = VectorStoreQueryMode.HYBRID if mode == "hybrid" else VectorStoreQueryMode.DEFAULT
    filters = []
    if pdf_sha256:
        filters.append(MetadataFilter(key="pdf_sha256", value=pdf_sha256))
    if content_type:
        filters.append(MetadataFilter(key="content_type", value=content_type))
    metadata_filters = (
        MetadataFilters(filters=filters, condition=FilterCondition.AND) if filters else None
    )
    vs_query = VectorStoreQuery(
        query_embedding=query_embedding,
        query_str=query_text,
        similarity_top_k=k,
        mode=query_mode,
        filters=metadata_filters,
    )
    result = store.query(vs_query)
    nodes = result.nodes or []
    similarities = result.similarities or [0.0] * len(nodes)
    return nodes, similarities


def retrieve(
    question: str,
    k: int | None = None,
    mode: str = "hybrid",
    settings: Settings | None = None,
    content_type: str | None = None,
    pdf_sha256: str | None = None,
) -> RetrievalResult:
    """French semantic search over the text index (text/table/image_description).

    If the best score for the ORIGINAL question is below Settings.rag_min_score,
    returns refused=True with no nodes: the caller must not call the LLM in
    that case. Query expansion (if enabled) only affects which nodes are kept
    once past that gate, never the refusal decision itself.

    `content_type`/`pdf_sha256` scope the search (used by the agent's tools to
    bind every search server-side to the active document and, for
    search_tables/search_figures, to one content type) -- the simple RAG path
    leaves both unset and searches across the whole active document's index.
    """
    settings = settings or get_settings()
    k = k or settings.rag_k
    store = get_text_store(settings)

    nodes0, sims0 = _query_text_store(
        question, store, k, mode, settings, content_type, pdf_sha256
    )
    best_score = max(sims0, default=0.0)

    if best_score < settings.rag_min_score:
        return RetrievalResult(nodes=[], best_score=best_score, refused=True)

    reformulations: list[str] = []
    if settings.rag_query_expansion and best_score < settings.rag_expansion_skip_score:
        reformulations = expand_query(question, settings)

    if not reformulations:
        ordered = _dedupe_by_best_score(nodes0, sims0)
    else:
        ranked_lists = [nodes0]
        similarity_by_id: dict[str, float] = dict(
            zip((n.node_id for n in nodes0), sims0, strict=True)
        )
        for reform in reformulations:
            nodes_r, sims_r = _query_text_store(
                reform, store, k, mode, settings, content_type, pdf_sha256
            )
            ranked_lists.append(nodes_r)
            for node, score in zip(nodes_r, sims_r, strict=True):
                similarity_by_id[node.node_id] = max(similarity_by_id.get(node.node_id, 0.0), score)
        fused_nodes = _rrf_fuse(ranked_lists, settings.rag_rrf_k)
        ordered = [_to_retrieved_node(n, similarity_by_id[n.node_id]) for n in fused_nodes]

    kept = _apply_budget(ordered, settings.rag_max_nodes, settings.rag_context_budget_tokens)
    return RetrievalResult(
        nodes=kept, best_score=best_score, refused=False, reformulations=reformulations
    )


# Performance fix (UI section 2 report): retrieve_by_image used to build a
# fresh ClipEncoder and immediately .unload() it on every single call --
# each image search paid the full ~6-7s MobileCLIP weight-load cost again,
# even though ClipEncoder is designed to load once and be reused (see its
# own docstring). Results are unaffected: same model, same preprocessing,
# same vectors, same top-k, same scores -- only the model is now cached
# across calls instead of being reloaded from scratch every time.
_IMAGE_ENCODER_CACHE: dict[tuple[str, str], ClipEncoder] = {}


def _get_cached_image_encoder(settings: Settings) -> ClipEncoder:
    key = (settings.clip_model, settings.clip_pretrained)
    encoder = _IMAGE_ENCODER_CACHE.get(key)
    if encoder is None:
        encoder = ClipEncoder(settings)
        _IMAGE_ENCODER_CACHE[key] = encoder
    return encoder


def retrieve_by_image(
    image_path: Path, k: int = 3, settings: Settings | None = None
) -> list[RetrievedNode]:
    """CLIP image-to-image search, mapped back to the matching
    image_description nodes in the text index (for extra RAG context)."""
    settings = settings or get_settings()
    image_store = get_image_store(settings)
    encoder = _get_cached_image_encoder(settings)
    vector = encoder.encode_images([image_path])[0].tolist()

    vs_query = VectorStoreQuery(query_embedding=vector, similarity_top_k=k)
    result = image_store.query(vs_query)
    image_nodes = result.nodes or []
    image_similarities = result.similarities or [0.0] * len(image_nodes)
    score_by_image_id = {
        n.metadata["image_id"]: score
        for n, score in zip(image_nodes, image_similarities, strict=True)
        if n.metadata.get("image_id")
    }
    if not score_by_image_id:
        return []

    text_store = get_text_store(settings)
    matches: list[RetrievedNode] = []
    for image_id, score in score_by_image_id.items():
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="content_type", value="image_description"),
                MetadataFilter(key="image_id", value=image_id),
            ],
            condition=FilterCondition.AND,
        )
        found = text_store.get_nodes(filters=filters)
        if found:
            # the real CLIP cosine similarity for this image, not a placeholder
            matches.append(_to_retrieved_node(found[0], score=float(score)))
    return matches
