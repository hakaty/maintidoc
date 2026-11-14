from pathlib import Path

import pytest
from eval.run_retrieval import (
    Question,
    QuestionResult,
    _hit_at_k,
    _is_match,
    _reciprocal_rank,
    aggregate,
    load_questions,
)

from app.indexing.stores import SearchResult

GOLD_PATH = Path("eval/retrieval_gold_v1.yaml")


def _result(page: int, content_type: str, score: float = 1.0) -> SearchResult:
    return SearchResult(node_id="n", page=page, content_type=content_type, score=score, snippet="x")


def _question(
    expected_pages: list[int], expected_content_type: str | None, answerable: bool = True
) -> Question:
    return Question(
        id="q",
        question="?",
        expected_pages=expected_pages,
        expected_content_type=expected_content_type,
        safety=False,
        answerable=answerable,
        notes="",
    )


def test_is_match_requires_both_page_and_content_type() -> None:
    question = _question([4], "table")
    assert _is_match(_result(4, "table"), question)
    assert not _is_match(_result(5, "table"), question)
    assert not _is_match(_result(4, "text"), question)


def test_hit_at_k_true_when_any_top_k_matches() -> None:
    question = _question([4], "table")
    results = [_result(1, "text"), _result(2, "text"), _result(4, "table")]
    assert _hit_at_k(results, question, k=3)
    assert not _hit_at_k(results, question, k=2)


def test_reciprocal_rank_of_first_match() -> None:
    question = _question([4], "table")
    results = [_result(1, "text"), _result(4, "table"), _result(4, "table")]
    assert _reciprocal_rank(results, question) == pytest.approx(0.5)


def test_reciprocal_rank_zero_when_no_match() -> None:
    question = _question([4], "table")
    results = [_result(1, "text")]
    assert _reciprocal_rank(results, question) == 0.0


def test_aggregate_computes_per_content_type_and_overall() -> None:
    q_table = _question([4], "table")
    q_text = _question([6], "text")
    q_unanswerable = _question([], None, answerable=False)

    evaluated = [
        QuestionResult(
            q_table, [_result(4, "table")], hit_at_5=True, reciprocal_rank=1.0, top_score=0.9
        ),
        QuestionResult(
            q_text, [_result(1, "text")], hit_at_5=False, reciprocal_rank=0.0, top_score=0.4
        ),
        QuestionResult(
            q_unanswerable,
            [_result(1, "text", 0.2)],
            hit_at_5=False,
            reciprocal_rank=0.0,
            top_score=0.2,
        ),
    ]

    report = aggregate(evaluated)

    assert report["overall"]["n"] == 2
    assert report["overall"]["hit_at_5"] == 0.5
    assert report["by_content_type"]["table"]["hit_at_5"] == 1.0
    assert report["by_content_type"]["text"]["hit_at_5"] == 0.0
    assert len(report["unanswerable_top_scores"]) == 1
    assert report["unanswerable_top_scores"][0]["top_score"] == 0.2


def test_load_questions_from_real_gold_file_has_20_entries_with_required_distribution() -> None:
    questions = load_questions(GOLD_PATH)

    assert len(questions) == 20
    answerable = [q for q in questions if q.answerable]
    unanswerable = [q for q in questions if not q.answerable]
    assert len(unanswerable) == 2
    assert sum(q.expected_content_type == "table" for q in answerable) >= 5
    assert sum(q.expected_content_type == "image_description" for q in answerable) >= 4
    assert sum(q.safety for q in questions) >= 3
