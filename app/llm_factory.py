from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings

# Chat model clients are cheap but not free to build, and re-creating one per
# question discards nothing on the Ollama side -- but on the demo/UI path we
# want ONE client per (provider, model, generation params) for the lifetime
# of the process, both to avoid rebuilding httpx connection pools per call and
# so every call shares the same `keep_alive` setting consistently.
_client_cache: dict[tuple, BaseChatModel] = {}


def _build_ollama(model: str, settings: Settings, **kwargs: object) -> ChatOllama:
    kwargs.setdefault("keep_alive", settings.ollama_keep_alive)
    return ChatOllama(base_url=settings.ollama_base_url, model=model, **kwargs)


def _build_openrouter(model: str, settings: Settings, **kwargs: object) -> ChatOpenAI:
    # settings.openrouter_api_key is guaranteed non-empty here: Settings refuses
    # to construct with llm_provider="openrouter" and no key (see app/config.py).
    return ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        model=model,
        **kwargs,
    )


def get_llm(
    settings: Settings | None = None,
    *,
    temperature: float | None = None,
    num_predict: int | None = None,
    num_ctx: int | None = None,
) -> BaseChatModel:
    settings = settings or get_settings()
    if settings.llm_provider == "ollama":
        key = (
            "ollama",
            settings.ollama_llm_model,
            settings.ollama_base_url,
            settings.ollama_keep_alive,
            temperature,
            num_predict,
            num_ctx,
        )
        if key not in _client_cache:
            kwargs: dict[str, object] = {}
            if temperature is not None:
                kwargs["temperature"] = temperature
            if num_predict is not None:
                kwargs["num_predict"] = num_predict
            if num_ctx is not None:
                kwargs["num_ctx"] = num_ctx
            _client_cache[key] = _build_ollama(settings.ollama_llm_model, settings, **kwargs)
        return _client_cache[key]

    key = (
        "openrouter",
        settings.openrouter_llm_model,
        settings.openrouter_base_url,
        temperature,
        num_predict,
    )
    if key not in _client_cache:
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if num_predict is not None:
            kwargs["max_tokens"] = num_predict
        _client_cache[key] = _build_openrouter(settings.openrouter_llm_model, settings, **kwargs)
    return _client_cache[key]


def get_vlm(settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    if settings.llm_provider == "ollama":
        return _build_ollama(settings.ollama_vlm_model, settings)
    return _build_openrouter(settings.openrouter_vlm_model, settings)
