"""Follow-up question condensation: rewrites a follow-up ("et pour la
version triphasée ?") into a standalone French question using prior turns,
so retrieval never depends on conversation memory -- only this run's
evidence is ever cited, per the no-citation-from-memory rule.

Shared by both the Simple RAG path (app/rag/service.py) and the agent
(app/agent/agent.py) so follow-ups work the same way in both modes.
"""

import logging
from pathlib import Path

from app.config import Settings, get_settings
from app.llm_factory import get_llm

logger = logging.getLogger(__name__)

PROMPT_PATH = Path("prompts") / "condense_question_v1.md"
NUM_PREDICT = 80
MAX_HISTORY_TURNS = 3


def condense_question(
    question: str, history: list[tuple[str, str]], settings: Settings | None = None
) -> str:
    """`history` is a list of (question, answer_text) tuples, oldest first.

    Best-effort: on any failure (unreachable LLM, malformed response) falls
    back to the original question unchanged -- condensation never blocks the
    answer pipeline, it only helps retrieval find the right passages.
    """
    if not history:
        return question
    settings = settings or get_settings()
    try:
        template = PROMPT_PATH.read_text(encoding="utf-8")
        recent = history[-MAX_HISTORY_TURNS:]
        history_text = "\n".join(f"Q: {q}\nR: {a}" for q, a in recent)
        prompt = template.format(history=history_text, question=question)
        llm = get_llm(settings, temperature=0.0, num_predict=NUM_PREDICT)
        response = llm.invoke(prompt)
        text = str(response.content).strip().strip('"')
    except Exception as exc:  # noqa: BLE001 -- best-effort, never blocks retrieval
        logger.warning("Question condensation failed; using the original question: %s", exc)
        return question
    return text or question
