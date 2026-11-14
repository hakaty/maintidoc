from app.rag.retriever import RetrievedNode
from app.rag.service import PROMPT_PATH, PROMPT_VERSION, build_prompt


def _node(**overrides) -> RetrievedNode:
    base = dict(
        node_id="n1",
        page=4,
        content_type="text",
        score=0.9,
        text="contenu du manuel",
        image_id=None,
        image_path=None,
        safety_relevant=None,
    )
    base.update(overrides)
    return RetrievedNode(**base)


def test_prompt_version_matches_file_name() -> None:
    assert PROMPT_VERSION == "rag_answer_v1"
    assert PROMPT_PATH.name == "rag_answer_v1.md"
    assert PROMPT_PATH.is_file()


def test_context_wrapped_with_page_and_type_delimiters() -> None:
    prompt = build_prompt("ma question", [_node(page=4, content_type="table")])

    assert '<document page="4" type="table">' in prompt
    assert "</document>" in prompt
    assert "contenu du manuel" in prompt
    assert "ma question" in prompt


def test_safety_relevant_node_marked_in_context() -> None:
    prompt = build_prompt("q", [_node(safety_relevant=True)])

    assert 'safety_relevant="true"' in prompt


def test_prompt_tells_model_context_is_data_not_instructions() -> None:
    template = PROMPT_PATH.read_text(encoding="utf-8")

    assert "DONNÉES" in template or "données" in template
    assert "ignore" in template.lower()


def test_prompt_requires_page_citations_and_refusal_rule() -> None:
    template = PROMPT_PATH.read_text(encoding="utf-8")

    assert "[p. N]" in template
    assert "insuffisant" in template.lower()
    assert "Sécurité" in template
