import json
from pathlib import Path

import numpy as np
import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryResult

from app.agent import tools as tools_module
from app.agent.tools import EvidenceLedger, build_tools, page_count
from app.rag import retriever as retriever_module
from app.rag.retriever import RetrievedNode


def _node(node_id: str, page: int, content_type: str, text: str, **meta) -> TextNode:
    return TextNode(
        id_=node_id, text=text, metadata={"page": page, "content_type": content_type, **meta}
    )


class _FakeEmbedder:
    def embed_query(self, text: str) -> np.ndarray:
        return np.zeros(4, dtype=np.float32)


class _RecordingStore:
    """Records the metadata filters passed to every query/get_nodes call.

    Similarities fixed at 0.8 (above the default rag_min_score=0.50 AND
    rag_expansion_skip_score=0.65) so retrieve() never triggers the
    expand_query LLM call in these tool tests -- that path is covered by
    test_retriever.py."""

    def __init__(self, nodes_by_type: dict[str, list[TextNode]]) -> None:
        self.nodes_by_type = nodes_by_type
        self.queries: list[object] = []
        self.get_nodes_calls: list[object] = []

    def query(self, vs_query) -> VectorStoreQueryResult:
        self.queries.append(vs_query)
        content_type = None
        pdf_sha256 = None
        if vs_query.filters is not None:
            for f in vs_query.filters.filters:
                if f.key == "content_type":
                    content_type = f.value
                if f.key == "pdf_sha256":
                    pdf_sha256 = f.value
        if pdf_sha256 != "doc-under-test":
            return VectorStoreQueryResult(nodes=[], similarities=[])
        if isinstance(content_type, list):
            # search_tables/search_figures now filter on a LIST of content
            # types (M7.4: table_row chunks can also hold pictogram info) --
            # match any one of them.
            nodes = [n for ct in content_type for n in self.nodes_by_type.get(ct, [])]
        else:
            nodes = self.nodes_by_type.get(content_type or "any", [])
        return VectorStoreQueryResult(nodes=nodes, similarities=[0.8] * len(nodes))

    def get_nodes(self, filters=None):
        self.get_nodes_calls.append(filters)
        page = next((f.value for f in filters.filters if f.key == "page"), None)
        pdf_sha256 = next((f.value for f in filters.filters if f.key == "pdf_sha256"), None)
        if pdf_sha256 != "doc-under-test":
            return []
        return [n for n in self.nodes_by_type.get("any", []) if n.metadata["page"] == page]


@pytest.fixture
def ledger() -> EvidenceLedger:
    return EvidenceLedger()


@pytest.fixture(autouse=True)
def _patch_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    # search_manual/search_tables/search_figures now go through
    # app.rag.retriever.retrieve(), which resolves the embedder/store from
    # its own module namespace -- not tools_module's.
    monkeypatch.setattr(
        retriever_module, "get_query_embedder", lambda settings=None: _FakeEmbedder()
    )


def _store_with(**by_type) -> _RecordingStore:
    return _RecordingStore(by_type)


def _patch_store(monkeypatch: pytest.MonkeyPatch, store: _RecordingStore) -> None:
    """search_* tools query via app.rag.retriever; get_page queries via
    app.agent.tools directly -- patch both namespaces to the same store."""
    monkeypatch.setattr(retriever_module, "get_text_store", lambda settings=None: store)
    monkeypatch.setattr(tools_module, "get_text_store", lambda settings=None: store)


def test_search_manual_binds_to_document_and_records_evidence(
    monkeypatch: pytest.MonkeyPatch, ledger: EvidenceLedger
) -> None:
    node = _node("n1", 4, "text", "contenu pertinent")
    store = _store_with(any=[node])
    _patch_store(monkeypatch, store)

    search_manual, *_ = build_tools(None, "doc-under-test", ledger)
    result = search_manual.invoke({"query": "une question", "k": 3})

    assert "contenu pertinent" in result
    assert "n1" in ledger.nodes
    assert ledger.calls[0]["tool"] == "search_manual"


def test_search_manual_cannot_see_another_document(
    monkeypatch: pytest.MonkeyPatch, ledger: EvidenceLedger
) -> None:
    """The store only has data for "doc-under-test"; binding to a different
    pdf_sha256 must return nothing, proving the document filter is real."""
    node = _node("n1", 4, "text", "contenu")
    store = _store_with(any=[node])
    _patch_store(monkeypatch, store)

    search_manual, *_ = build_tools(None, "some-other-document", ledger)
    result = search_manual.invoke({"query": "une question", "k": 3})

    assert result == "(aucun résultat)"
    assert ledger.nodes == {}


def test_search_tables_filters_by_content_type(
    monkeypatch: pytest.MonkeyPatch, ledger: EvidenceLedger
) -> None:
    table_node = _node("t1", 35, "table", "|Cause|Remede|")
    text_node = _node("x1", 4, "text", "non pertinent")
    store = _store_with(table=[table_node], text=[text_node])
    _patch_store(monkeypatch, store)

    _, search_tables, search_figures, _ = build_tools(None, "doc-under-test", ledger)
    result = search_tables.invoke({"query": "RCD", "k": 4})

    assert "Cause" in result
    assert "t1" in ledger.nodes
    assert "x1" not in ledger.nodes


def test_search_figures_filters_by_content_type(
    monkeypatch: pytest.MonkeyPatch, ledger: EvidenceLedger
) -> None:
    figure_node = _node("f1", 5, "image_description", "Figure EX")
    store = _store_with(image_description=[figure_node])
    _patch_store(monkeypatch, store)

    _, _, search_figures, _ = build_tools(None, "doc-under-test", ledger)
    result = search_figures.invoke({"query": "explosif", "k": 4})

    assert "Figure EX" in result
    assert "f1" in ledger.nodes


def test_search_tables_never_truncates_a_table_node(
    monkeypatch: pytest.MonkeyPatch, ledger: EvidenceLedger
) -> None:
    """Tables must come back whole (no fixed-char truncation), matching the
    simple RAG path's "skip whole node, never cut short" budgeting rule."""
    big_table = "|Col1|Col2|\n" + "\n".join(f"|row{i}|val{i}|" for i in range(300))
    table_node = _node("t1", 35, "table", big_table)
    store = _store_with(table=[table_node])
    _patch_store(monkeypatch, store)

    _, search_tables, _, _ = build_tools(None, "doc-under-test", ledger)
    result = search_tables.invoke({"query": "RCD", "k": 4})

    assert big_table in result
    assert ledger.nodes["t1"].text == big_table


def test_get_page_returns_nodes_for_that_page(
    monkeypatch: pytest.MonkeyPatch, ledger: EvidenceLedger, tmp_path: Path
) -> None:
    node_p9 = _node("p9", 9, "text", "stockage")
    store = _store_with(any=[node_p9])
    monkeypatch.setattr(tools_module, "get_text_store", lambda settings=None: store)
    monkeypatch.setattr(tools_module, "page_count", lambda pdf_sha256: 48)

    *_, get_page = build_tools(None, "doc-under-test", ledger)
    result = get_page.invoke({"page": 9})

    assert "stockage" in result
    assert "p9" in ledger.nodes


def test_get_page_rejects_out_of_bounds_page(
    monkeypatch: pytest.MonkeyPatch, ledger: EvidenceLedger
) -> None:
    store = _store_with(any=[])
    monkeypatch.setattr(tools_module, "get_text_store", lambda settings=None: store)
    monkeypatch.setattr(tools_module, "page_count", lambda pdf_sha256: 48)

    *_, get_page = build_tools(None, "doc-under-test", ledger)

    assert "invalide" in get_page.invoke({"page": 0})
    assert "invalide" in get_page.invoke({"page": 999})
    assert ledger.nodes == {}


def test_page_count_reads_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    processed = tmp_path / "data" / "processed" / "abcdef123456"
    processed.mkdir(parents=True)
    (processed / "manifest.json").write_text(
        json.dumps({"total_pages": 48}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    assert page_count("abcdef123456" + "0" * 52) == 48


def test_page_count_returns_none_when_manifest_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert page_count("nonexistent" + "0" * 53) is None


def test_k_argument_is_capped(monkeypatch: pytest.MonkeyPatch, ledger: EvidenceLedger) -> None:
    node = _node("n1", 4, "text", "contenu")
    store = _store_with(any=[node])
    _patch_store(monkeypatch, store)

    search_manual, *_ = build_tools(None, "doc-under-test", ledger)
    search_manual.invoke({"query": "q", "k": 999})

    vs_query = store.queries[-1]
    assert vs_query.similarity_top_k <= 10


def test_empty_query_is_rejected(monkeypatch: pytest.MonkeyPatch, ledger: EvidenceLedger) -> None:
    store = _store_with(any=[])
    _patch_store(monkeypatch, store)

    search_manual, *_ = build_tools(None, "doc-under-test", ledger)
    result = search_manual.invoke({"query": "   ", "k": 3})

    assert "vide" in result.lower()
    assert ledger.nodes == {}


def test_similar_figures_tool_only_present_with_image_path(ledger: EvidenceLedger) -> None:
    without_image = build_tools(None, "doc-under-test", ledger)
    assert len(without_image) == 4

    with_image = build_tools(None, "doc-under-test", ledger, image_path=Path("photo.png"))
    assert len(with_image) == 5
    assert with_image[-1].name == "similar_figures"


def _retrieved_node(node_id: str, page: int, content_type: str) -> RetrievedNode:
    return RetrievedNode(
        node_id=node_id,
        page=page,
        content_type=content_type,
        score=0.9,
        text="texte",
        image_id=None,
        image_path=None,
        safety_relevant=None,
    )


def test_evidence_ledger_tracks_valid_pages() -> None:
    ledger = EvidenceLedger()
    ledger.record([_retrieved_node("n1", 4, "text")], "search_manual", {}, 0.1)
    ledger.record([_retrieved_node("n2", 9, "table")], "search_tables", {}, 0.1)

    assert ledger.valid_pages == {4, 9}
    assert len(ledger.calls) == 2
