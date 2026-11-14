from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.llm_factory import get_llm, get_vlm


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_get_llm_ollama_returns_chat_ollama() -> None:
    settings = _settings(llm_provider="ollama", ollama_llm_model="qwen2.5:7b-instruct")

    llm = get_llm(settings)

    assert isinstance(llm, ChatOllama)
    assert llm.model == "qwen2.5:7b-instruct"


def test_get_vlm_ollama_returns_chat_ollama() -> None:
    settings = _settings(llm_provider="ollama", ollama_vlm_model="qwen2.5vl:3b")

    vlm = get_vlm(settings)

    assert isinstance(vlm, ChatOllama)
    assert vlm.model == "qwen2.5vl:3b"


def test_get_llm_openrouter_returns_chat_openai() -> None:
    settings = _settings(
        llm_provider="openrouter",
        openrouter_api_key="sk-test",
        openrouter_llm_model="qwen/qwen-2.5-7b-instruct",
    )

    llm = get_llm(settings)

    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "qwen/qwen-2.5-7b-instruct"


def test_get_llm_ollama_passes_generation_overrides() -> None:
    settings = _settings(llm_provider="ollama")

    llm = get_llm(settings, temperature=0.1, num_predict=600, num_ctx=8192)

    assert llm.temperature == 0.1
    assert llm.num_predict == 600
    assert llm.num_ctx == 8192


def test_get_llm_openrouter_passes_generation_overrides() -> None:
    settings = _settings(llm_provider="openrouter", openrouter_api_key="sk-test")

    llm = get_llm(settings, temperature=0.1, num_predict=600)

    assert llm.temperature == 0.1
    assert llm.max_tokens == 600


def test_get_vlm_openrouter_returns_chat_openai() -> None:
    settings = _settings(
        llm_provider="openrouter",
        openrouter_api_key="sk-test",
        openrouter_vlm_model="qwen/qwen-2.5-vl-3b-instruct",
    )

    vlm = get_vlm(settings)

    assert isinstance(vlm, ChatOpenAI)
    assert vlm.model_name == "qwen/qwen-2.5-vl-3b-instruct"
