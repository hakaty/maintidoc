"""LangChain tools for the RAG agent.

All tools are bound server-side to ONE active document (pdf_sha256): the LLM
never chooses or receives the document identity, it only sees a query/page
number and gets results already scoped to that document.

Tool OUTPUT is rendered as <document> blocks, exactly like app.rag.service's
context -- UNTRUSTED data for the agent to cite, never instructions. Every
node any tool returns is recorded in an EvidenceLedger: the final citation
check (app/agent/agent.py) only accepts pages that came from this ledger.
"""

import json
import logging
import time
from pathlib import Path

from langchain_core.tools import tool
from llama_index.core.vector_stores.types import (
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
)

from app.config import Settings, get_settings
from app.indexing.stores import get_text_store
from app.rag.retriever import RetrievedNode, _to_retrieved_node, retrieve, retrieve_by_image

logger = logging.getLogger(__name__)

MAX_K = 10
MAX_QUERY_CHARS = 300


class EvidenceLedger:
    """Per-run record of every node any tool actually returned."""

    def __init__(self) -> None:
        self.nodes: dict[str, RetrievedNode] = {}
        self.calls: list[dict] = []

    def record(
        self, nodes: list[RetrievedNode], tool_name: str, args: dict, seconds: float
    ) -> None:
        for node in nodes:
            self.nodes[node.node_id] = node
        self.calls.append(
            {"tool": tool_name, "args": args, "n_results": len(nodes), "seconds": round(seconds, 3)}
        )

    @property
    def valid_pages(self) -> set[int]:
        return {node.page for node in self.nodes.values() if node.page is not None}


def _validate_query(query: str) -> str | None:
    """Returns the cleaned query, or None if it's empty (callers must check
    and return an informative string -- tools should never raise)."""
    query = (query or "").strip()
    if not query:
        return None
    return query[:MAX_QUERY_CHARS]


def _validate_k(k: int, max_k: int = MAX_K) -> int:
    return max(1, min(int(k), max_k))


def page_count(pdf_sha256: str) -> int | None:
    manifest_path = Path("data/processed") / pdf_sha256[:12] / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest.get("total_pages")


def _render_nodes(nodes: list[RetrievedNode]) -> str:
    """Renders nodes exactly as retrieved -- tables/figures are never
    truncated here: the per-call token budget in `_search_filtered` already
    decides which whole nodes fit, matching app.rag.retriever._apply_budget's
    "skip whole, never cut short" rule used by the simple RAG path."""
    if not nodes:
        return "(aucun résultat)"
    blocks = []
    for node in nodes:
        safety = ' safety_relevant="true"' if node.safety_relevant else ""
        header = f'<document page="{node.page}" type="{node.content_type}"{safety}>'
        blocks.append(f"{header}\n{node.text}\n</document>")
    return "\n\n".join(blocks)


def _search_filtered(
    query_text: str,
    content_type: str | None,
    pdf_sha256: str,
    k: int,
    settings: Settings,
) -> list[RetrievedNode]:
    """Reuses the FULL simple-RAG retrieval path (hybrid search, adaptive
    query expansion, RRF fusion, dedupe, token-budgeted truncation) instead
    of a one-shot vector query, scoped to this document and content type."""
    result = retrieve(
        query_text, k=k, mode="hybrid", settings=settings, content_type=content_type,
        pdf_sha256=pdf_sha256,
    )
    return result.nodes


def _query_page(page: int, pdf_sha256: str, settings: Settings) -> list[RetrievedNode]:
    store = get_text_store(settings)
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="pdf_sha256", value=pdf_sha256),
            MetadataFilter(key="page", value=page),
        ],
        condition=FilterCondition.AND,
    )
    found = store.get_nodes(filters=filters)
    return [_to_retrieved_node(n, 1.0) for n in found]


def build_tools(
    settings: Settings | None,
    pdf_sha256: str,
    ledger: EvidenceLedger,
    image_path: Path | None = None,
) -> list:
    """Tool closures bound to this document/ledger. image_path only present
    when the user attached a photo for this run -- enables similar_figures."""
    settings = settings or get_settings()
    total_pages = page_count(pdf_sha256)

    @tool
    def search_manual(query: str, k: int = 6) -> str:
        """Recherche des passages de texte pertinents dans le manuel actif."""
        safe_query = _validate_query(query)
        if safe_query is None:
            return "(la requête de recherche ne peut pas être vide)"
        start = time.monotonic()
        safe_k = _validate_k(k)
        nodes = _search_filtered(safe_query, None, pdf_sha256, safe_k, settings)
        ledger.record(
            nodes, "search_manual", {"query": safe_query, "k": safe_k}, time.monotonic() - start
        )
        return _render_nodes(nodes)

    @tool
    def search_tables(query: str, k: int = 4) -> str:
        """Recherche des tableaux pertinents dans le manuel actif (markdown intact)."""
        safe_query = _validate_query(query)
        if safe_query is None:
            return "(la requête de recherche ne peut pas être vide)"
        start = time.monotonic()
        safe_k = _validate_k(k)
        nodes = _search_filtered(safe_query, "table", pdf_sha256, safe_k, settings)
        ledger.record(
            nodes, "search_tables", {"query": safe_query, "k": safe_k}, time.monotonic() - start
        )
        return _render_nodes(nodes)

    @tool
    def search_figures(query: str, k: int = 4) -> str:
        """Recherche des figures/pictogrammes pertinents via leur description en français."""
        safe_query = _validate_query(query)
        if safe_query is None:
            return "(la requête de recherche ne peut pas être vide)"
        start = time.monotonic()
        safe_k = _validate_k(k)
        nodes = _search_filtered(safe_query, "image_description", pdf_sha256, safe_k, settings)
        ledger.record(
            nodes, "search_figures", {"query": safe_query, "k": safe_k}, time.monotonic() - start
        )
        return _render_nodes(nodes)

    @tool
    def get_page(page: int) -> str:
        """Retourne tous les passages, tableaux et figures d'une page donnée du manuel actif."""
        start = time.monotonic()
        if total_pages is not None and not (1 <= page <= total_pages):
            return f"(page invalide : ce document a {total_pages} pages, de 1 à {total_pages})"
        nodes = _query_page(page, pdf_sha256, settings)
        ledger.record(nodes, "get_page", {"page": page}, time.monotonic() - start)
        return _render_nodes(nodes)

    tools = [search_manual, search_tables, search_figures, get_page]

    if image_path is not None:

        @tool
        def similar_figures() -> str:
            """Retourne les figures du manuel actif visuellement similaires à la photo jointe."""
            start = time.monotonic()
            try:
                nodes = retrieve_by_image(image_path, k=3, settings=settings)
            except Exception as exc:  # noqa: BLE001 -- tool errors surface as text, not crashes
                logger.warning("similar_figures failed: %s", exc)
                return "(recherche par image indisponible)"
            ledger.record(nodes, "similar_figures", {}, time.monotonic() - start)
            return _render_nodes(nodes)

        tools.append(similar_figures)

    return tools
