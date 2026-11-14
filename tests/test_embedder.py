from pathlib import Path

import numpy as np
import pytest
import torch

from app.config import Settings
from app.indexing import embedder as embedder_module
from app.indexing.embedder import (
    EmbedderDownloadError,
    TextEmbedder,
    get_query_embedder,
    unload_ollama_models,
    unload_query_embedder,
)

FAKE_DIM = 8


class _FakeSentenceTransformerModel:
    def get_embedding_dimension(self) -> int:
        return FAKE_DIM


class _FakeHuggingFaceEmbedding:
    """Deterministic stand-in: one call per text, no network/model download."""

    instances: list["_FakeHuggingFaceEmbedding"] = []

    def __init__(self, model_name: str, device: str, cache_folder: str, **kwargs) -> None:
        self.model_name = model_name
        self.device = device
        self.cache_folder = cache_folder
        self.kwargs = kwargs
        self._model = _FakeSentenceTransformerModel()
        _FakeHuggingFaceEmbedding.instances.append(self)

    def get_text_embedding_batch(self, texts: list[str], show_progress: bool = False):
        return [[float(len(text) % 7)] * FAKE_DIM for text in texts]

    def get_query_embedding(self, query: str):
        return [float(len(query) % 7)] * FAKE_DIM


@pytest.fixture
def fake_hf_embedding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _FakeHuggingFaceEmbedding.instances.clear()
    monkeypatch.setattr(embedder_module, "HuggingFaceEmbedding", _FakeHuggingFaceEmbedding)
    monkeypatch.setattr(embedder_module, "_model_cache_size_mb", lambda cache_dir, name: 123.4)


def _settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, models_dir=tmp_path / "models")


def test_embedding_dim_read_from_model(fake_hf_embedding: None, tmp_path: Path) -> None:
    text_embedder = TextEmbedder(_settings(tmp_path))
    assert text_embedder.embedding_dim == FAKE_DIM


def test_embed_texts_shape_and_dtype(fake_hf_embedding: None, tmp_path: Path) -> None:
    text_embedder = TextEmbedder(_settings(tmp_path))
    vectors = text_embedder.embed_texts(["une phrase", "une autre phrase plus longue"])

    assert vectors.shape == (2, FAKE_DIM)
    assert vectors.dtype == np.float32


def test_embed_texts_empty_list(fake_hf_embedding: None, tmp_path: Path) -> None:
    text_embedder = TextEmbedder(_settings(tmp_path))
    vectors = text_embedder.embed_texts([])

    assert vectors.shape == (0, FAKE_DIM)


def test_embed_query_shape(fake_hf_embedding: None, tmp_path: Path) -> None:
    text_embedder = TextEmbedder(_settings(tmp_path))
    vector = text_embedder.embed_query("une question")

    assert vector.shape == (FAKE_DIM,)


def test_device_never_silently_cpu_when_cuda_expected(
    fake_hf_embedding: None, tmp_path: Path
) -> None:
    text_embedder = TextEmbedder(_settings(tmp_path))
    text_embedder.embed_query("x")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert text_embedder.device == expected


def test_passes_configured_batch_size_and_max_length(
    fake_hf_embedding: None, tmp_path: Path
) -> None:
    settings = Settings(
        _env_file=None,
        models_dir=tmp_path / "models",
        embedding_batch_size=4,
        embedding_max_length=256,
    )
    text_embedder = TextEmbedder(settings)
    text_embedder.embed_query("x")

    instance = _FakeHuggingFaceEmbedding.instances[-1]
    assert instance.kwargs["embed_batch_size"] == 4
    assert instance.kwargs["max_length"] == 256
    assert instance.kwargs["normalize"] is True


def test_cache_folder_points_at_models_dir(fake_hf_embedding: None, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    text_embedder = TextEmbedder(settings)
    text_embedder.embed_query("x")

    instance = _FakeHuggingFaceEmbedding.instances[-1]
    assert instance.cache_folder == str(settings.models_dir)
    assert settings.models_dir.is_dir()


def test_load_seconds_and_cache_size_recorded(fake_hf_embedding: None, tmp_path: Path) -> None:
    text_embedder = TextEmbedder(_settings(tmp_path))
    text_embedder.embed_query("x")

    assert text_embedder.load_seconds is not None
    assert text_embedder.load_seconds >= 0.0
    assert text_embedder.cache_size_mb == 123.4


def test_unload_resets_and_can_reload(fake_hf_embedding: None, tmp_path: Path) -> None:
    text_embedder = TextEmbedder(_settings(tmp_path))
    text_embedder.embed_query("x")
    text_embedder.unload()

    vectors = text_embedder.embed_texts(["hello"])
    assert vectors.shape == (1, FAKE_DIM)


def test_load_failure_raises_clear_download_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _boom(*args, **kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(embedder_module, "HuggingFaceEmbedding", _boom)

    text_embedder = TextEmbedder(_settings(tmp_path))
    with pytest.raises(EmbedderDownloadError, match="offline"):
        text_embedder.embed_query("x")


def test_unload_ollama_models_posts_keep_alive_zero_for_both_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def _fake_post(url: str, json: dict, timeout: float):
        calls.append(json)

    monkeypatch.setattr(embedder_module.httpx, "post", _fake_post)

    settings = Settings(_env_file=None, llm_provider="ollama")
    unload_ollama_models(settings)

    models_told_to_unload = {call["model"] for call in calls}
    assert models_told_to_unload == {settings.ollama_vlm_model, settings.ollama_llm_model}
    assert all(call["keep_alive"] == 0 for call in calls)


def test_unload_ollama_models_noop_for_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        embedder_module.httpx, "post", lambda *a, **k: calls.append(k) or calls.append(a)
    )

    settings = Settings(_env_file=None, llm_provider="openrouter", openrouter_api_key="sk-test")
    unload_ollama_models(settings)

    assert calls == []


@pytest.fixture(autouse=True)
def _reset_query_embedder_singleton():
    embedder_module._query_embedder = None
    yield
    embedder_module._query_embedder = None


def test_get_query_embedder_returns_same_instance_across_calls(
    fake_hf_embedding: None, tmp_path: Path
) -> None:
    settings = _settings(tmp_path)

    first = get_query_embedder(settings)
    second = get_query_embedder(settings)

    assert first is second


def test_get_query_embedder_loads_model_only_once_across_many_queries(
    fake_hf_embedding: None, tmp_path: Path
) -> None:
    settings = _settings(tmp_path)

    for _ in range(5):
        embedder = get_query_embedder(settings)
        embedder.embed_query("une question")

    assert len(_FakeHuggingFaceEmbedding.instances) == 1


def test_unload_query_embedder_resets_singleton(fake_hf_embedding: None, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    get_query_embedder(settings).embed_query("x")

    unload_query_embedder()

    get_query_embedder(settings).embed_query("y")
    assert len(_FakeHuggingFaceEmbedding.instances) == 2


@pytest.mark.integration
def test_real_bge_m3_loads_and_embeds() -> None:
    """Loads the real configured BGE-M3 model (network/GPU required).

    Uses the project's real models_dir (not tmp_path): the weights are ~2.2GB,
    so re-downloading them into a fresh tmp dir on every test run would be
    wasteful -- models/ is git-ignored and exactly meant for this cache.
    """
    try:
        text_embedder = TextEmbedder(Settings(_env_file=None))
        vectors = text_embedder.embed_texts(["Ceci est une phrase de test en francais."])
    except EmbedderDownloadError as exc:
        pytest.skip(f"Real BGE-M3 model unavailable: {exc}")

    assert vectors.shape[0] == 1
    assert vectors.shape[1] == text_embedder.embedding_dim
    assert text_embedder.embedding_dim == 1024
    np.testing.assert_allclose(np.linalg.norm(vectors[0]), 1.0, atol=1e-2)
    text_embedder.unload()
