"""Evaluate full RAG answers against eval/qa_gold_v1.yaml (+ optionally
eval/qa_user_questions.yaml if it contains questions).

Metrics (observed results for this document/gold set, not guarantees):
citation validity, expected-page hit in citations, refusal accuracy,
adversarial handled, groundedness (LLM-as-judge, same local Qwen model --
this is a self-judging setup, a known bias noted in docs/05_rapport_rag.md),
latency p50/p95, tokens/s. Reported per group, with "base" (answerable, exact
manual wording) and "paraphrase" (everyday wording) broken out separately.

Usage: python -m eval.run_qa [--gold path] [--user-gold path] [--output results.json]
"""

import argparse
import json
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.config import Settings, get_settings
from app.llm_factory import get_llm
from app.rag.retriever import retrieve
from app.rag.service import RAGAnswer, answer

DEFAULT_GOLD_PATH = Path("eval/qa_gold_v1.yaml")
DEFAULT_USER_GOLD_PATH = Path("eval/qa_user_questions.yaml")
DEFAULT_MULTITURN_PATH = Path("eval/qa_multiturn_v1.yaml")
JUDGE_PROMPT_PATH = Path("prompts") / "judge_groundedness_v1.md"
JUDGE_VERDICT_PATTERN = re.compile(r"VERDICT:\s*(OUI|PARTIEL|NON)", re.IGNORECASE)
VERDICT_SCORE = {"OUI": 1.0, "PARTIEL": 0.5, "NON": 0.0}


@dataclass
class Question:
    id: str
    group: str
    question: str
    expected_pages: list[int]
    answerable: bool
    notes: str = ""


def _load_questions(path: Path) -> list[Question]:
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [Question(**item) for item in data.get("questions") or []]


def load_all_questions(
    gold_path: Path = DEFAULT_GOLD_PATH, user_gold_path: Path = DEFAULT_USER_GOLD_PATH
) -> list[Question]:
    questions = _load_questions(gold_path)
    questions.extend(_load_questions(user_gold_path))
    return questions


def _format_context(nodes) -> str:
    blocks = []
    for node in nodes:
        safety = ' safety_relevant="true"' if node.safety_relevant else ""
        header = f'<document page="{node.page}" type="{node.content_type}"{safety}>'
        blocks.append(f"{header}\n{node.text}\n</document>")
    return "\n\n".join(blocks)


def _judge_groundedness(question: str, result: RAGAnswer, settings: Settings) -> float | None:
    if result.refused or result.error or not result.answer:
        return None
    retrieval = retrieve(question, mode="hybrid", settings=settings)
    if not retrieval.nodes:
        return None
    template = JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    prompt = template.format(
        context_blocks=_format_context(retrieval.nodes), question=question, answer=result.answer
    )
    llm = get_llm(settings, temperature=0.0, num_predict=100)
    try:
        response = llm.invoke(prompt)
    except Exception:  # noqa: BLE001 -- judging is best-effort, never blocks the eval
        return None
    match = JUDGE_VERDICT_PATTERN.search(str(response.content))
    if not match:
        return None
    return VERDICT_SCORE[match.group(1).upper()]


@dataclass
class QuestionEval:
    question: Question
    result: RAGAnswer
    expected_page_hit: bool | None
    groundedness: float | None


def evaluate(
    questions: list[Question],
    settings: Settings | None = None,
    answer_fn: Callable[..., RAGAnswer] = answer,
) -> list[QuestionEval]:
    settings = settings or get_settings()
    evaluated = []
    for question in questions:
        result = answer_fn(question.question, settings=settings)
        expected_page_hit = None
        if question.expected_pages:
            cited_pages = {c.page for c in result.citations}
            expected_page_hit = bool(cited_pages & set(question.expected_pages))
        groundedness = _judge_groundedness(question.question, result, settings)
        evaluated.append(QuestionEval(question, result, expected_page_hit, groundedness))
    return evaluated


def _latency_ms(qe: QuestionEval) -> float | None:
    timings = qe.result.timings
    retrieval = timings.get("retrieval_ms") or 0.0
    generation = timings.get("generation_ms")
    if generation is None:
        return None
    return retrieval + generation


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(pct * (len(ordered) - 1))))
    return ordered[index]


def _group_metrics(evaluated: list[QuestionEval]) -> dict:
    n = len(evaluated)
    if n == 0:
        return {"n": 0}

    non_error = [qe for qe in evaluated if qe.result.error is None]
    non_refusal = [qe for qe in non_error if not qe.result.refused]

    citation_valid = [qe for qe in non_refusal if not qe.result.citations_corrected]
    expected_hits = [qe.expected_page_hit for qe in evaluated if qe.expected_page_hit is not None]
    refusal_correct = [
        qe for qe in evaluated if qe.result.refused == (not qe.question.answerable)
    ]
    adversarial_handled = [
        qe for qe in non_error if qe.result.refused or len(qe.result.citations) > 0
    ]
    grounded_scores = [qe.groundedness for qe in evaluated if qe.groundedness is not None]
    latencies = [lat for qe in non_error if (lat := _latency_ms(qe)) is not None]
    tokens_per_s = [
        tps
        for qe in non_error
        if (tps := qe.result.timings.get("tokens_per_s")) is not None
    ]

    metrics = {
        "n": n,
        "citation_validity_rate": round(len(citation_valid) / len(non_refusal), 3)
        if non_refusal
        else None,
        "refusal_accuracy": round(len(refusal_correct) / n, 3),
        "latency_ms_p50": _percentile(latencies, 0.5),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "tokens_per_s_avg": round(statistics.mean(tokens_per_s), 1) if tokens_per_s else None,
    }
    if expected_hits:
        metrics["expected_page_hit_rate"] = round(sum(expected_hits) / len(expected_hits), 3)
    if any(qe.question.group == "adversarial" for qe in evaluated):
        metrics["adversarial_handled_rate"] = round(len(adversarial_handled) / n, 3)
    if grounded_scores:
        metrics["groundedness_avg"] = round(statistics.mean(grounded_scores), 3)
    return metrics


def _refused_answerable(evaluated: list[QuestionEval]) -> list[dict]:
    """Answerable questions that were refused anyway, with the cause --
    transparency into false refusals (expected by section 0 of M6/M7)."""
    return [
        {
            "id": qe.question.id,
            "group": qe.question.group,
            "question": qe.question.question,
            "cause": qe.result.refusal_reason or "other",
        }
        for qe in evaluated
        if qe.question.answerable and qe.result.refused
    ]


def aggregate(evaluated: list[QuestionEval]) -> dict:
    by_group: dict[str, list[QuestionEval]] = {}
    for qe in evaluated:
        by_group.setdefault(qe.question.group, []).append(qe)
    return {
        "overall": _group_metrics(evaluated),
        "by_group": {group: _group_metrics(subset) for group, subset in sorted(by_group.items())},
        "refused_answerable_questions": _refused_answerable(evaluated),
    }


def paraphrase_expansion_comparison(
    questions: list[Question],
    settings: Settings | None = None,
    answer_fn: Callable[..., RAGAnswer] = answer,
) -> dict:
    """Paraphrase-group page-hit rate with query expansion on vs off, same
    questions, same settings otherwise -- isolates the effect of 0b/M6."""
    settings = settings or get_settings()
    paraphrase_questions = [q for q in questions if q.group == "paraphrase"]
    if not paraphrase_questions:
        return {}

    results = {}
    for expansion in (True, False):
        settings.rag_query_expansion = expansion
        evaluated = evaluate(paraphrase_questions, settings, answer_fn)
        hits = [qe.expected_page_hit for qe in evaluated if qe.expected_page_hit is not None]
        results["with_expansion" if expansion else "without_expansion"] = {
            "n": len(evaluated),
            "expected_page_hit_rate": round(sum(hits) / len(hits), 3) if hits else None,
        }
    return results


def _load_conversations(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("conversations") or []


def evaluate_multiturn(
    conversations: list[dict], settings: Settings | None = None
) -> dict:
    """Agent-only (needs session memory): one fresh session_id per
    conversation, turns asked sequentially so later turns can rely on
    history for follow-ups ("et pour la version triphasée ?")."""
    import uuid

    from app.agent.agent import ask as agent_ask

    settings = settings or get_settings()
    per_turn_hits: dict[int, list[bool]] = {}
    details = []

    for conversation in conversations:
        session_id = str(uuid.uuid4())
        for turn_index, turn in enumerate(conversation["turns"]):
            result = agent_ask(turn["question"], settings=settings, session_id=session_id)
            cited_pages = {c.page for c in result.citations}
            hit = bool(cited_pages & set(turn.get("expected_pages", [])))
            per_turn_hits.setdefault(turn_index, []).append(hit)
            details.append(
                {
                    "conversation": conversation["id"],
                    "turn": turn_index + 1,
                    "question": turn["question"],
                    "hit": hit,
                    "refused": result.refused,
                }
            )

    return {
        "by_turn_hit_rate": {
            f"turn_{i + 1}": round(sum(hits) / len(hits), 3) for i, hits in per_turn_hits.items()
        },
        "details": details,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.run_qa")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    parser.add_argument("--user-gold", type=Path, default=DEFAULT_USER_GOLD_PATH)
    parser.add_argument(
        "--multiturn", type=Path, default=None, nargs="?", const=DEFAULT_MULTITURN_PATH
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=["simple", "agent"],
        default="simple",
        help="simple = app.rag.service.answer(); agent = app.agent.agent.ask()",
    )
    parser.add_argument(
        "--compare-expansion",
        action="store_true",
        help="Also report paraphrase page-hit with vs without query expansion",
    )
    args = parser.parse_args(argv)

    if args.multiturn:
        conversations = _load_conversations(args.multiturn)
        multiturn_report = evaluate_multiturn(conversations, get_settings())
        print("=== multiturn (agent) ===")
        print(json.dumps(multiturn_report, indent=2, ensure_ascii=False))
        if args.output:
            args.output.write_text(
                json.dumps(multiturn_report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return 0

    questions = load_all_questions(args.gold, args.user_gold)
    settings = get_settings()

    answer_fn: Callable[..., RAGAnswer] = answer
    if args.mode == "agent":
        from app.agent.agent import ask as agent_ask

        answer_fn = agent_ask

    evaluated = evaluate(questions, settings, answer_fn)
    report = aggregate(evaluated)
    report["mode"] = args.mode

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.compare_expansion:
        comparison = paraphrase_expansion_comparison(questions, settings, answer_fn)
        print("\n=== paraphrase: query expansion on vs off ===")
        print(json.dumps(comparison, indent=2, ensure_ascii=False))
        report["paraphrase_expansion_comparison"] = comparison

    if args.output:
        full = {
            "report": report,
            "questions": [
                {
                    "id": qe.question.id,
                    "group": qe.question.group,
                    "refused": qe.result.refused,
                    "error": qe.result.error,
                    "citations_corrected": qe.result.citations_corrected,
                    "expected_page_hit": qe.expected_page_hit,
                    "groundedness": qe.groundedness,
                    "citations": [c.page for c in qe.result.citations],
                }
                for qe in evaluated
            ],
        }
        args.output.write_text(json.dumps(full, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWritten to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
