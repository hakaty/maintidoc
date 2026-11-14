import pytest
from eval import run_qa as run_qa_module
from eval.run_qa import Question, aggregate, evaluate, paraphrase_expansion_comparison

from app.config import Settings
from app.rag.service import RAGAnswer


@pytest.fixture(autouse=True)
def _no_real_groundedness_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    """evaluate() always calls _judge_groundedness, which does a real
    retrieval (DB) + real LLM call -- irrelevant to the aggregation logic
    these tests actually exercise."""
    monkeypatch.setattr(run_qa_module, "_judge_groundedness", lambda *a, **k: None)


def _settings() -> Settings:
    return Settings(_env_file=None)


def _question(**overrides) -> Question:
    base = dict(
        id="q1", group="base", question="Q?", expected_pages=[4], answerable=True, notes=""
    )
    base.update(overrides)
    return Question(**base)


def test_refused_answerable_questions_records_cause() -> None:
    questions = [
        _question(id="q1", question="Q1?", answerable=True),
        _question(id="q2", question="Q2?", answerable=True),
        _question(id="q3", question="Q3?", answerable=False),
    ]

    def _fake_answer(question, settings=None):
        if question == "Q1?":
            return RAGAnswer(answer="x", refused=True, refusal_reason="below_min_score")
        if question == "Q2?":
            return RAGAnswer(answer="ok [p. 4].", refused=False)
        return RAGAnswer(answer="refus", refused=True, refusal_reason="below_min_score")

    evaluated = evaluate(questions, _settings(), _fake_answer)
    report = aggregate(evaluated)

    refused_answerable = report["refused_answerable_questions"]
    assert len(refused_answerable) == 1
    assert refused_answerable[0]["id"] == "q1"
    assert refused_answerable[0]["cause"] == "below_min_score"


def test_refused_answerable_questions_defaults_cause_to_other_when_missing() -> None:
    questions = [_question(id="q1", answerable=True)]

    def _fake_answer(question, settings=None):
        return RAGAnswer(answer="x", refused=True, refusal_reason=None)

    evaluated = evaluate(questions, _settings(), _fake_answer)
    report = aggregate(evaluated)

    assert report["refused_answerable_questions"][0]["cause"] == "other"


def test_paraphrase_expansion_comparison_reports_both_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    questions = [
        _question(id="p1", group="paraphrase", expected_pages=[4]),
        _question(id="p2", group="paraphrase", expected_pages=[9]),
        _question(id="b1", group="base", expected_pages=[1]),  # must be excluded
    ]
    calls = []

    def _fake_answer(question, settings=None):
        calls.append(settings.rag_query_expansion)
        page = 4 if settings.rag_query_expansion else 999  # only "right" with expansion on
        from app.rag.service import Citation

        return RAGAnswer(
            answer=f"ok [p. {page}].",
            citations=[Citation(page=page, node_id="n", snippet="s")],
        )

    comparison = paraphrase_expansion_comparison(questions, _settings(), _fake_answer)

    assert comparison["with_expansion"]["n"] == 2
    assert comparison["without_expansion"]["n"] == 2
    assert comparison["with_expansion"]["expected_page_hit_rate"] == 0.5  # p1 hits, p2 doesn't
    assert comparison["without_expansion"]["expected_page_hit_rate"] == 0.0
    assert True in calls and False in calls


def test_paraphrase_expansion_comparison_empty_when_no_paraphrase_questions() -> None:
    questions = [_question(id="b1", group="base")]

    assert paraphrase_expansion_comparison(questions, _settings(), lambda *a, **k: None) == {}
