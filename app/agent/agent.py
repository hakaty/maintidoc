"""LangChain tool-calling agent, redesigned as a PLANNER + proven GENERATOR:

- The agent loop (langchain.agents.create_agent) only decides which tools to
  call and when to stop -- its own final text is a throwaway "I'm done"
  message, never shown to the user and never cited from.
- The actual answer is written by the SAME proven generator as the simple
  RAG path (app.rag.service.build_prompt / _generate / finalize_answer),
  fed with this run's evidence ledger (deduplicated, budgeted by relevance,
  same token budget as Simple) -- so citation checking, the no-source rule,
  the safety-line dedup and citation-marker normalisation are identical in
  both modes, not two parallel implementations that can drift apart.
- Follow-ups are condensed into a standalone question (app.rag.condense)
  BEFORE the planner runs, using prior turns from this session's memory --
  the planner and generator then only ever see this run's own evidence,
  never a page cited from memory.

Falls back to the simple RAG path on any agent failure (including a
timeout) or when the ledger ends up empty.

No LangGraph custom graph: langchain.agents.create_agent is LangChain's own
built-in agent factory (it happens to compile a langgraph graph internally,
but we never define graph nodes/edges ourselves).
"""

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import HumanMessage

from app.agent.tools import EvidenceLedger, build_tools
from app.config import Settings, get_settings
from app.llm_factory import get_llm
from app.rag import service
from app.rag.condense import condense_question
from app.rag.retriever import _apply_budget
from app.rag.service import REFUSAL_TEXT, RAGAnswer, detect_injection  # noqa: F401 -- REFUSAL_TEXT re-exported for tests

logger = logging.getLogger(__name__)

PROMPT_VERSION = "agent_system_v1"
PROMPT_PATH = Path("prompts") / f"{PROMPT_VERSION}.md"
MAX_QUESTION_CHARS = 500

# Per-session memory: (question, final_answer_text) pairs, oldest first,
# trimmed to the last MAX_HISTORY_TURNS turns (~ the "last 6 messages" of
# the original spec: one human + one assistant message per turn). Used ONLY
# to condense follow-ups into a standalone question -- never replayed to the
# planner or cited from directly.
MAX_HISTORY_TURNS = 3
_session_histories: dict[str, list[tuple[str, str]]] = {}


def reset_session(session_id: str) -> None:
    """Called by the UI's "Nouvelle conversation" button."""
    _session_histories.pop(session_id, None)


def _get_history(session_id: str | None) -> list[tuple[str, str]]:
    if session_id is None:
        return []
    return list(_session_histories.get(session_id, []))


def _update_history(
    session_id: str | None,
    history: list[tuple[str, str]],
    question: str,
    answer_text: str,
) -> None:
    if session_id is None:
        return
    updated = [*history, (question, answer_text)]
    _session_histories[session_id] = updated[-MAX_HISTORY_TURNS:]


def _planner_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _default_pdf_sha256(settings: Settings) -> str:
    return hashlib.sha256(settings.pdf_path.read_bytes()).hexdigest()


def ask(
    question: str,
    image_path: Path | None = None,
    settings: Settings | None = None,
    session_id: str | None = None,
    pdf_sha256: str | None = None,
) -> RAGAnswer:
    settings = settings or get_settings()
    model = service._model_name(settings)
    question = (question or "")[:MAX_QUESTION_CHARS]
    injection_detected = detect_injection(question)

    try:
        pdf_sha256 = pdf_sha256 or _default_pdf_sha256(settings)
        history = _get_history(session_id)
        standalone_question = condense_question(question, history, settings)

        ledger = EvidenceLedger()
        tools = build_tools(settings, pdf_sha256, ledger, image_path)
        llm = get_llm(
            settings,
            temperature=settings.rag_temperature,
            num_predict=settings.rag_num_predict,
            num_ctx=settings.rag_num_ctx,
        )
        max_tool_calls = settings.agent_max_tool_calls
        timeout_seconds = settings.agent_timeout_seconds
        # *3 + 6, not *2 + 4: observed in the real agent eval that
        # ToolCallLimitMiddleware's own "end" step needs more graph steps than
        # a naive (model, tool) pair count once it forces a stop after
        # run_limit tool calls -- 3 questions hit "Recursion limit of 12
        # reached" and fell back to simple RAG even though only 4 real tool
        # calls had happened. This does not raise the tool-call cap (still
        # max_tool_calls, enforced by the middleware) or the timeout.
        recursion_limit = max_tool_calls * 3 + 6
        middleware = [ToolCallLimitMiddleware(run_limit=max_tool_calls, exit_behavior="end")]
        agent = create_agent(
            llm, tools, system_prompt=_planner_system_prompt(), middleware=middleware
        )

        run_config = {"recursion_limit": recursion_limit}
        t0 = time.monotonic()
        # Not a `with` block: on timeout we must return immediately without
        # blocking for the background call to actually finish (Python can't
        # forcibly kill a thread) -- shutdown(wait=False) just stops tracking it.
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            agent.invoke,
            {"messages": [HumanMessage(content=standalone_question)]},
            config=run_config,
        )
        try:
            future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            executor.shutdown(wait=False)
            raise TimeoutError(f"agent exceeded {timeout_seconds}s timeout") from None
        else:
            executor.shutdown(wait=False)
        planning_ms = round((time.monotonic() - t0) * 1000, 1)

        # Evidence ledger -> the proven generator, exactly like Simple RAG:
        # best score first, token-budgeted (tables/figures never truncated,
        # a node that would blow the budget is skipped whole -- see
        # app.rag.retriever._apply_budget), restricted to THIS run's nodes.
        nodes_sorted = sorted(ledger.nodes.values(), key=lambda n: -n.score)
        budgeted = _apply_budget(
            nodes_sorted, settings.rag_max_nodes, settings.rag_context_budget_tokens
        )

        prompt = service.build_prompt(standalone_question, budgeted)
        if injection_detected:
            prompt += service.INJECTION_REMINDER

        gen_start = time.monotonic()
        outcome = service._generate(prompt, settings, llm=llm)
        generation_ms = round((time.monotonic() - gen_start) * 1000, 1)

        prepared = service.PreparedAnswer(
            nodes=budgeted,
            prompt=prompt,
            model=model,
            provider=settings.llm_provider,
            retrieval_ms=round(sum(c["seconds"] for c in ledger.calls) * 1000, 1),
            injection_detected=injection_detected,
            standalone_question=standalone_question,
        )
        answer_obj = service.finalize_answer(
            outcome.text, prepared, settings, generation_ms, outcome.tokens_per_s, llm=llm
        )
        answer_obj.mode_used = "agent"
        answer_obj.timings["planning_ms"] = planning_ms

        trace = list(ledger.calls)
        if history:
            trace = [
                {
                    "tool": "condense_question",
                    "args": {"query": standalone_question},
                    "n_results": 0,
                    "seconds": 0.0,
                },
                *trace,
            ]
        answer_obj.tool_calls = trace

        _update_history(session_id, history, question, answer_obj.answer)
        return answer_obj

    except Exception as exc:  # noqa: BLE001 -- any agent failure falls back to simple RAG
        logger.warning("Agent failed (%s); falling back to simple RAG", exc)
        fallback = service.answer(question, image_path=image_path, settings=settings)
        fallback.mode_used = "fallback_rag"
        return fallback
