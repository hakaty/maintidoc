from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "LLM_PROVIDER",
        "OPENROUTER_API_KEY",
        "DATABASE_URL",
        "OLLAMA_LLM_MODEL",
        "OLLAMA_VLM_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=None)

    assert settings.llm_provider == "ollama"
    assert settings.database_url == "postgresql://maintidoc:maintidoc@127.0.0.1:5432/maintidoc"
    assert settings.ollama_llm_model == "qwen2.5:7b-instruct"
    assert settings.ollama_vlm_model == "qwen2.5vl:3b"
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_batch_size == 16
    assert settings.embedding_max_length == 1024
    assert settings.pgvector_text_table == "maintidoc_text"
    assert settings.pgvector_image_table == "maintidoc_images"
    assert settings.pgvector_text_dim == 1024
    assert settings.pgvector_image_dim == 512
    assert settings.pgvector_hybrid_search is True
    assert settings.pgvector_text_search_config == "french"
    assert settings.media_dir == Path("data/media")
    assert settings.pdf_path == Path("data/pdf/document.pdf")
    assert settings.models_dir == Path("models")


def test_provider_switch_to_openrouter() -> None:
    settings = Settings(_env_file=None, llm_provider="openrouter", openrouter_api_key="sk-test")

    assert settings.llm_provider == "openrouter"
    assert settings.openrouter_api_key == "sk-test"


def test_missing_openrouter_key_raises() -> None:
    with pytest.raises(ValidationError, match="OPENROUTER_API_KEY"):
        Settings(_env_file=None, llm_provider="openrouter", openrouter_api_key="")
