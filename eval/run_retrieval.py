"""Evaluate text-index retrieval against eval/retrieval_gold_v1.yaml.

No LLM involved: this scores page/content_type overlap of the top-k vector
vs hybrid search results against a hand-written gold set (20 French
questions, written by reading the real PDF). Metrics are observed results
for this document and this gold set, not general guarantees.

Usage: python -m eval.run_retrieval [--gold path] [--output results.json]
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.config import Settings, get_settings
from app.indexing.stores import SearchResult, search_text

DEFAULT_GOLD_PATH = Path("eval/retrieval_gold_v1.yaml")
TOP_K = 10
HIT_AT = 5


@dataclass
class Question:
    id: str
    question: str
    expected_pages: list[int]
    expected_content_type: str | None
    safety: bool
    answerable: bool
    notes: str


def load_questions(path: Path = DEFAULT_GOLD_PATH) -> list[Question]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [Question(**item) for item in data["questions"]]


def _is_match(result: SearchResult, question: Question) -> bool:
    return (
        result.content_type == question.expected_content_type
        and result.page in question.expected_pages
    )


def _hit_at_k(results: list[SearchResult], question: Question, k: int) -> bool:
    return any(_is_match(result, question) for result in results[:k])


def _reciprocal_rank(results: list[SearchResult], question: Question) -> float:
    for rank, result in enumerate(results, start=1):
        if _is_match(result, question):
            return 1.0 / rank
    return 0.0


@dataclass
class QuestionResult:
    question: Question
    results: list[SearchResult]
    hit_at_5: bool
    reciprocal_rank: float
    top_score: float


def evaluate(
    questions: list[Question], mode: str, settings: Settings | None = None
) -> list[QuestionResult]:
    settings = settings or get_settings()
    evaluated = []
    for question in questions:
        results = search_text(question.question, k=TOP_K, mode=mode, settings=settings)
        hit = _hit_at_k(results, question, HIT_AT) if question.answerable else False
        rr = _reciprocal_rank(results, question) if question.answerable else 0.0
        top_score = results[0].score if results else 0.0
        evaluated.append(QuestionResult(question, results, hit, rr, top_score))
    return evaluated


def _metrics(subset: list[QuestionResult]) -> dict:
    if not subset:
        return {"n": 0, "hit_at_5": None, "mrr_at_10": None}
    return {
        "n": len(subset),
        "hit_at_5": round(sum(e.hit_at_5 for e in subset) / len(subset), 3),
        "mrr_at_10": round(sum(e.reciprocal_rank for e in subset) / len(subset), 3),
    }


def aggregate(evaluated: list[QuestionResult]) -> dict:
    answerable = [e for e in evaluated if e.question.answerable]
    by_type: dict[str, list[QuestionResult]] = {}
    for e in answerable:
        by_type.setdefault(e.question.expected_content_type, []).append(e)

    return {
        "overall": _metrics(answerable),
        "by_content_type": {ct: _metrics(subset) for ct, subset in sorted(by_type.items())},
        "unanswerable_top_scores": [
            {
                "id": e.question.id,
                "question": e.question.question,
                "top_score": round(e.top_score, 4),
                "top_page": e.results[0].page if e.results else None,
                "top_content_type": e.results[0].content_type if e.results else None,
            }
            for e in evaluated
            if not e.question.answerable
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.run_retrieval")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    parser.add_argument("--output", type=Path, default=None, help="Write JSON results to this path")
    args = parser.parse_args(argv)

    questions = load_questions(args.gold)
    settings = get_settings()

    report = {}
    for mode in ("vector", "hybrid"):
        evaluated = evaluate(questions, mode, settings)
        report[mode] = aggregate(evaluated)
        print(f"=== mode={mode} ===")
        print(json.dumps(report[mode], indent=2, ensure_ascii=False))

    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWritten to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
