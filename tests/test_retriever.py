from pathlib import Path

import numpy as np
import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryResult

from app.config import Settings
from app.rag import retriever as retriever_module

FAKE_DIM = 4


class _FakeEmbedder:
    def __init__(self, settings=None, for_query=False) -> None:
        self.settings = settings
        self.for_query = for_query

    def embed_query(self, text: str) -> np.ndarray:
        return np.zeros(FAKE_DIM, dtype=np.float32)

    def unload(self) -> None:
        pass


class _FakeTextStore:
    def __init__(self, nodes: list[TextNode], scores: list[float]) -> None:
        self._nodes = nodes
        self._scores = scores
        self.get_nodes_calls: list[object] = []

    def query(self, vs_query) -> VectorStoreQueryResult:
        return VectorStoreQueryResult(nodes=self._nodes, similarities=self._scores)

    def get_nodes(self, filters=None):
        self.get_nodes_calls.append(filters)
        return self._nodes


def _settings(**overrides) -> Settings:
    overrides.setdefault("rag_query_expansion", False)
    return Settings(_env_file=None, **overrides)


def _node(node_id: str, page: int, content_type: str, text: str, **meta) -> TextNode:
    return TextNode(
        id_=node_id, text=text, metadata={"page": page, "content_type": content_type, **meta}
    )


@pytest.fixture(autouse=True)
def _patch_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeEmbedder()
    monkeypatch.setattr(retriever_module, "get_query_embedder", lambda settings=None: fake)
    # rag_query_expansion defaults to False in _settings() above, so expand_query
    # (which calls the real LLM) is never reached by tests that don't opt in --
    # it is NOT patched here, so the dedicated expand_query tests below can
    # exercise the real function.


def _patch_store(monkeypatch: pytest.MonkeyPatch, store: _FakeTextStore) -> None:
    monkeypatch.setattr(retriever_module, "get_text_store", lambda settings=None: store)


def test_below_min_score_refuses_without_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    nodes = [_node("n1", 4, "text", "contenu")]
    _patch_store(monkeypatch, _FakeTextStore(nodes, [0.1]))
    settings = _settings(rag_min_score=0.45)

    result = retriever_module.retrieve("une question", settings=settings)

    assert result.refused is True
    assert result.nodes == []
    assert result.best_score == 0.1


def test_above_min_score_returns_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    nodes = [_node("n1", 4, "text", "contenu pertinent")]
    _patch_store(monkeypatch, _FakeTextStore(nodes, [0.9]))
    settings = _settings(rag_min_score=0.45)

    result = retriever_module.retrieve("une question", settings=settings)

    assert result.refused is False
    assert len(result.nodes) == 1
    assert result.nodes[0].node_id == "n1"


def test_dedupes_same_node_id_keeping_best_score(monkeypatch: pytest.MonkeyPatch) -> None:
    n1a = _node("dup", 4, "text", "contenu")
    n1b = TextNode(id_="dup", text="contenu", metadata={"page": 4, "content_type": "text"})
    _patch_store(monkeypatch, _FakeTextStore([n1a, n1b], [0.5, 0.9]))
    settings = _settings(rag_min_score=0.1)

    result = retriever_module.retrieve("une question", settings=settings)

    assert len(result.nodes) == 1
    assert result.nodes[0].score == 0.9


def test_budget_always_keeps_best_node_whole_even_if_oversized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    big_text = "x" * 20000  # ~5000 tokens at 4 chars/token, way over a small budget
    nodes = [_node("big", 4, "table", big_text)]
    _patch_store(monkeypatch, _FakeTextStore(nodes, [0.9]))
    settings = _settings(rag_min_score=0.1, rag_context_budget_tokens=100, rag_max_nodes=6)

    result = retriever_module.retrieve("une question", settings=settings)

    assert [n.node_id for n in result.nodes] == ["big"]
    assert result.nodes[0].text == big_text  # never truncated


def test_budget_skips_a_node_that_would_blow_the_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small_text = "texte court"
    big_text = "x" * 20000
    nodes = [
        _node("small", 5, "text", small_text),
        _node("big", 4, "table", big_text),
    ]
    _patch_store(monkeypatch, _FakeTextStore(nodes, [0.9, 0.8]))
    settings = _settings(rag_min_score=0.1, rag_context_budget_tokens=100, rag_max_nodes=6)

    result = retriever_module.retrieve("une question", settings=settings)

    kept_ids = [n.node_id for n in result.nodes]
    assert kept_ids == ["small"]  # "big" would blow the remaining budget: skipped whole


def test_max_nodes_caps_count_even_within_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    nodes = [_node(f"n{i}", i, "text", "petit texte") for i in range(10)]
    scores = [1.0 - i * 0.01 for i in range(10)]
    _patch_store(monkeypatch, _FakeTextStore(nodes, scores))
    settings = _settings(rag_min_score=0.1, rag_max_nodes=3, rag_context_budget_tokens=100000)

    result = retriever_module.retrieve("une question", settings=settings)

    assert len(result.nodes) == 3


def test_retrieve_by_image_maps_to_image_description_nodes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image_match = TextNode(
        id_="img1", text="", metadata={"image_id": "img1", "content_type": "image"}
    )
    description_node = _node(
        "desc1", 5, "image_description", "Figure (page 5) ...", image_id="img1"
    )

    class _FakeImageStore:
        def query(self, vs_query) -> VectorStoreQueryResult:
            return VectorStoreQueryResult(nodes=[image_match], similarities=[0.95])

    text_store = _FakeTextStore([description_node], [1.0])
    fake_image_store = _FakeImageStore()
    monkeypatch.setattr(retriever_module, "get_image_store", lambda settings=None: fake_image_store)
    monkeypatch.setattr(retriever_module, "get_text_store", lambda settings=None: text_store)

    class _FakeClipEncoder:
        def __init__(self, settings=None) -> None:
            pass

        def encode_images(self, paths):
            return np.zeros((1, FAKE_DIM), dtype=np.float32)

        def unload(self) -> None:
            pass

    monkeypatch.setattr(retriever_module, "ClipEncoder", _FakeClipEncoder)
    # The cache is module-level and persists across tests/calls (performance
    # fix, UI section 2 report) -- clear it so this test gets a fresh
    # _FakeClipEncoder instead of reusing whatever another test cached.
    monkeypatch.setattr(retriever_module, "_IMAGE_ENCODER_CACHE", {})

    image_path = tmp_path / "query.png"
    image_path.write_bytes(b"fake")

    matches = retriever_module.retrieve_by_image(image_path, settings=_settings())

    assert len(matches) == 1
    assert matches[0].node_id == "desc1"
    assert matches[0].score == pytest.approx(0.95)  # real CLIP score, not a placeholder 1.0
    assert text_store.get_nodes_calls  # a metadata filter lookup actually happened


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeRewriteLLM:
    def __init__(self, content: str) -> None:
        self.content = content

    def invoke(self, prompt: str) -> _FakeResponse:
        return _FakeResponse(self.content)


def test_expand_query_parses_json_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        retriever_module,
        "get_llm",
        lambda *a, **k: _FakeRewriteLLM('["Reformulation un ?", "Reformulation deux ?"]'),
    )

    result = retriever_module.expand_query("ma pompe fait du bruit", _settings())

    assert result == ["Reformulation un ?", "Reformulation deux ?"]


def test_expand_query_falls_back_to_empty_on_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(retriever_module, "get_llm", lambda *a, **k: _FakeRewriteLLM("not json"))

    assert retriever_module.expand_query("question", _settings()) == []


def test_expand_query_falls_back_to_empty_on_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom:
        def invoke(self, prompt: str):
            raise RuntimeError("unreachable")

    monkeypatch.setattr(retriever_module, "get_llm", lambda *a, **k: _Boom())

    assert retriever_module.expand_query("question", _settings()) == []


def test_expand_query_caps_at_three_and_ignores_non_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        retriever_module, "get_llm", lambda *a, **k: _FakeRewriteLLM('{"not": "a list"}')
    )
    assert retriever_module.expand_query("question", _settings()) == []

    monkeypatch.setattr(
        retriever_module,
        "get_llm",
        lambda *a, **k: _FakeRewriteLLM('["a", "b", "c", "d"]'),
    )
    assert retriever_module.expand_query("question", _settings()) == ["a", "b", "c"]


def test_query_expansion_fuses_reformulation_hits_via_rrf(monkeypatch: pytest.MonkeyPatch) -> None:
    """A node only found by a reformulation (not the original wording) must
    still surface in the final results -- this is the whole point of query
    expansion for paraphrased/everyday-language questions."""
    original_only = _node("orig", 4, "text", "contenu trouve par la question originale")
    reform_only = _node("reform", 9, "text", "contenu trouve seulement par la reformulation")

    class _TwoQueryStore:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def query(self, vs_query) -> VectorStoreQueryResult:
            self.queries.append(vs_query.query_str)
            if vs_query.query_str == "la question de l'utilisateur":
                return VectorStoreQueryResult(nodes=[original_only], similarities=[0.9])
            return VectorStoreQueryResult(nodes=[reform_only], similarities=[0.85])

    store = _TwoQueryStore()
    monkeypatch.setattr(retriever_module, "get_text_store", lambda settings=None: store)
    monkeypatch.setattr(
        retriever_module, "expand_query", lambda question, settings: ["une reformulation"]
    )
    settings = _settings(
        rag_min_score=0.1, rag_query_expansion=True, rag_expansion_skip_score=1.0
    )

    result = retriever_module.retrieve("la question de l'utilisateur", settings=settings)

    kept_ids = {n.node_id for n in result.nodes}
    assert kept_ids == {"orig", "reform"}
    assert result.reformulations == ["une reformulation"]
    assert len(store.queries) == 2  # original + one reformulation


def test_query_expansion_disabled_skips_reformulation_entirely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = _node("n1", 4, "text", "contenu")

    class _CountingStore:
        def __init__(self) -> None:
            self.call_count = 0

        def query(self, vs_query) -> VectorStoreQueryResult:
            self.call_count += 1
            return VectorStoreQueryResult(nodes=[node], similarities=[0.9])

    store = _CountingStore()
    monkeypatch.setattr(retriever_module, "get_text_store", lambda settings=None: store)

    def _boom(question, settings):
        raise AssertionError("expand_query must not be called when rag_query_expansion=False")

    monkeypatch.setattr(retriever_module, "expand_query", _boom)
    settings = _settings(rag_min_score=0.1, rag_query_expansion=False)

    result = retriever_module.retrieve("question", settings=settings)

    assert store.call_count == 1
    assert result.reformulations == []


def test_adaptive_expansion_skipped_when_original_score_is_confident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = _node("n1", 4, "text", "contenu")
    store = _FakeTextStore([node], [0.9])  # above rag_expansion_skip_score
    _patch_store(monkeypatch, store)

    def _boom(question, settings):
        raise AssertionError("expand_query must not be called when the score is confident")

    monkeypatch.setattr(retriever_module, "expand_query", _boom)
    settings = _settings(
        rag_min_score=0.1, rag_query_expansion=True, rag_expansion_skip_score=0.65
    )

    result = retriever_module.retrieve("question", settings=settings)

    assert result.reformulations == []


def test_adaptive_expansion_triggered_when_original_score_is_weak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = _node("n1", 4, "text", "contenu")
    store = _FakeTextStore([node], [0.55])  # below rag_expansion_skip_score
    _patch_store(monkeypatch, store)
    monkeypatch.setattr(
        retriever_module, "expand_query", lambda question, settings: ["reformulation"]
    )
    settings = _settings(
        rag_min_score=0.1, rag_query_expansion=True, rag_expansion_skip_score=0.65
    )

    result = retriever_module.retrieve("question", settings=settings)

    assert result.reformulations == ["reformulation"]


# --- M7.4b section 1: table_row crowding fix --------------------------------


def _retrieved(node_id: str, page: int, content_type: str, score: float):
    return retriever_module.RetrievedNode(
        node_id=node_id,
        page=page,
        content_type=content_type,
        score=score,
        text="x",
        image_id=None,
        image_path=None,
        safety_relevant=None,
    )


def test_cap_table_rows_per_page_keeps_only_the_best_row_per_page() -> None:
    nodes = [
        _retrieved("r1", 37, "table_row", 0.9),
        _retrieved("r2", 37, "table_row", 0.8),  # same page as r1 -- dropped
        _retrieved("t1", 27, "text", 0.7),
        _retrieved("r3", 40, "table_row", 0.6),
    ]

    kept = retriever_module._cap_table_rows_per_page(nodes)

    assert [n.node_id for n in kept] == ["r1", "t1", "r3"]


def test_cap_table_rows_per_page_never_touches_non_table_row_nodes() -> None:
    nodes = [
        _retrieved("t1", 5, "table", 0.9),
        _retrieved("t2", 5, "table", 0.8),  # same page, but not table_row -- kept
    ]

    kept = retriever_module._cap_table_rows_per_page(nodes)

    assert [n.node_id for n in kept] == ["t1", "t2"]


def test_retrieve_fetches_more_raw_candidates_than_it_keeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M7.4b regression fix (qa16): several table_row hits from the SAME
    table could previously fill the first `k` raw results, pushing the one
    correct text chunk out of the window entirely before any de-duplication
    had a chance to run. retrieve() now asks the store for more than `k`
    raw candidates (capped at 10, app.agent.tools.MAX_K's ceiling --
    test_agent_tools.py::test_k_argument_is_capped pins that ceiling)."""
    node = _node("n1", 4, "text", "contenu")
    captured: list[int] = []

    class _CapturingStore(_FakeTextStore):
        def query(self, vs_query):
            captured.append(vs_query.similarity_top_k)
            return super().query(vs_query)

    _patch_store(monkeypatch, _CapturingStore([node], [0.9]))
    settings = _settings(rag_min_score=0.1, rag_max_nodes=3)

    retriever_module.retrieve("question", k=3, settings=settings)

    assert captured == [10]  # min(max(3, 3) * 4, 10)
