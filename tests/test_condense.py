from dataclasses import dataclass

import pytest

from app.config import Settings
from app.rag import condense as condense_module


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


@dataclass
class _FakeResponse:
    content: str


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> _FakeResponse:
        self.calls.append(prompt)
        return _FakeResponse(content=self.reply)


def test_no_history_returns_question_unchanged_without_calling_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*a, **k):
        raise AssertionError("LLM must not be called when there is no history")

    monkeypatch.setattr(condense_module, "get_llm", _boom)

    result = condense_module.condense_question("question autonome", [], _settings())

    assert result == "question autonome"


def test_history_present_calls_llm_and_returns_its_text(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _FakeLLM("Quelle est l'étape suivante si la réinitialisation du RCD ne suffit pas ?")
    monkeypatch.setattr(condense_module, "get_llm", lambda *a, **k: llm)

    history = [("Que faire si le RCD s'est déclenché ?", "Réinitialiser le RCD [p. 35].")]
    result = condense_module.condense_question(
        "et si ça ne suffit pas ?", history, _settings()
    )

    assert result == "Quelle est l'étape suivante si la réinitialisation du RCD ne suffit pas ?"
    assert "RCD" in llm.calls[0]  # the history was included in the prompt


def test_llm_failure_falls_back_to_original_question(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*a, **k):
        raise RuntimeError("Ollama unreachable")

    monkeypatch.setattr(condense_module, "get_llm", _boom)

    result = condense_module.condense_question(
        "et si ça ne suffit pas ?", [("q", "a")], _settings()
    )

    assert result == "et si ça ne suffit pas ?"


def test_history_truncated_to_max_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _FakeLLM("question reformulée")
    monkeypatch.setattr(condense_module, "get_llm", lambda *a, **k: llm)
    history = [(f"q{i}", f"a{i}") for i in range(10)]

    condense_module.condense_question("suivi", history, _settings())

    prompt = llm.calls[0]
    assert "q0" not in prompt  # oldest turns dropped
    assert "q9" in prompt  # most recent turn kept
