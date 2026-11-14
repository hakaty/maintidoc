"""BGE-M3 text embeddings via LlamaIndex's HuggingFaceEmbedding.

Weights are cached under Settings.models_dir (git-ignored, HF_HOME points
there too): downloaded once, resumed automatically on interruption by
huggingface_hub. Device cuda + fp16 when available, embeddings L2-normalised.
Before loading, Ollama's VLM/LLM models are told to unload (keep_alive 0) so
BGE-M3 does not have to share VRAM with a model Ollama is still holding onto
from a previous run.
"""

import logging
import os
import time
from pathlib import Path

import httpx
import numpy as np
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.config import Settings, get_settings
from app.device import resolve_device, resolve_query_embedding_device

logger = logging.getLogger(__name__)

OLLAMA_TIMEOUT_SECONDS = 120.0


class EmbedderDownloadError(RuntimeError):
    """The embedding model could not be loaded (likely offline and not cached)."""


def _ensure_hf_home(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))


def unload_ollama_models(settings: Settings | None = None) -> None:
    """Free Ollama's VRAM (both the VLM and the text LLM) before BGE-M3 loads."""
    settings = settings or get_settings()
    if settings.llm_provider != "ollama":
        return
    for model in (settings.ollama_vlm_model, settings.ollama_llm_model):
        try:
            httpx.post(
                f"{settings.ollama_base_url}/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError:
            logger.warning("Could not unload Ollama model %s (server unreachable)", model)


def _model_cache_size_mb(cache_dir: Path, model_name: str) -> float | None:
    from huggingface_hub import scan_cache_dir

    try:
        info = scan_cache_dir(str(cache_dir))
    except Exception:
        return None
    for repo in info.repos:
        if repo.repo_id == model_name:
            return repo.size_on_disk / (1024**2)
    return None


class TextEmbedder:
    """Lazily loads Settings.embedding_model (BAAI/bge-m3 by default) once per process."""

    def __init__(self, settings: Settings | None = None, for_query: bool = False) -> None:
        self._settings = settings or get_settings()
        self._for_query = for_query
        self._embedding: HuggingFaceEmbedding | None = None
        self._device: str | None = None
        self.load_seconds: float | None = None
        self.cache_size_mb: float | None = None

    @property
    def embedding_dim(self) -> int:
        self._ensure_loaded()
        assert self._embedding is not None
        return self._embedding._model.get_embedding_dimension()

    @property
    def device(self) -> str:
        self._ensure_loaded()
        assert self._device is not None
        return self._device

    @property
    def peak_vram_mb(self) -> float | None:
        """Peak CUDA memory allocated since the model was (re)loaded, in MB. None on CPU."""
        if self._device != "cuda":
            return None
        return torch.cuda.max_memory_allocated() / (1024**2)

    def _ensure_loaded(self) -> None:
        if self._embedding is not None:
            return

        cache_dir = self._settings.models_dir
        _ensure_hf_home(cache_dir)

        model_name = self._settings.embedding_model
        device = (
            resolve_query_embedding_device(self._settings)
            if self._for_query
            else resolve_device(self._settings, "BGE-M3")
        )
        self._device = device
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            logger.info("Loading %s on GPU: %s", model_name, torch.cuda.get_device_name(0))
        else:
            logger.info("Loading %s on CPU (fp32)", model_name)

        model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
        start = time.monotonic()
        try:
            self._embedding = HuggingFaceEmbedding(
                model_name=model_name,
                device=device,
                cache_folder=str(cache_dir),
                embed_batch_size=self._settings.embedding_batch_size,
                max_length=self._settings.embedding_max_length,
                normalize=True,
                model_kwargs=model_kwargs,
            )
        except Exception as exc:
            raise EmbedderDownloadError(
                f"Could not load embedding model {model_name!r} with cache_folder="
                f"{cache_dir} (offline and not already cached?): {exc}"
            ) from exc
        self.load_seconds = round(time.monotonic() - start, 2)
        self.cache_size_mb = _model_cache_size_mb(cache_dir, model_name)
        logger.info(
            "Loaded %s in %.2fs (cache on disk: %s MB)",
            model_name,
            self.load_seconds,
            self.cache_size_mb,
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """L2-normalised embeddings, shape (len(texts), embedding_dim). Content is untrusted."""
        self._ensure_loaded()
        assert self._embedding is not None
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        vectors = self._embedding.get_text_embedding_batch(texts, show_progress=False)
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """L2-normalised embedding for a search query (BGE-M3 uses the same tower for both)."""
        self._ensure_loaded()
        assert self._embedding is not None
        return np.asarray(self._embedding.get_query_embedding(text), dtype=np.float32)

    def unload(self) -> None:
        """Free VRAM: drop the model and clear the CUDA cache."""
        if self._embedding is None:
            return
        device = self._device
        peak_mb = self.peak_vram_mb
        self._embedding = None
        self._device = None
        if device == "cuda":
            torch.cuda.empty_cache()
        if peak_mb is not None:
            logger.info("Unloaded text embedder (peak VRAM: %.1f MB)", peak_mb)
        else:
            logger.info("Unloaded text embedder")


# Process-wide singleton for query-time embeddings: a RAG service answers many
# questions per process, and reloading BGE-M3 from disk on every single query
# (observed ~5-8s each) dominated end-to-end latency (M5 follow-up). Loaded
# once, kept resident until unload_query_embedder() is called explicitly
# (e.g. the UI only needs to call it when truly shutting down).
_query_embedder: TextEmbedder | None = None


def get_query_embedder(settings: Settings | None = None) -> TextEmbedder:
    global _query_embedder
    if _query_embedder is None:
        _query_embedder = TextEmbedder(settings or get_settings(), for_query=True)
    return _query_embedder


def unload_query_embedder() -> None:
    global _query_embedder
    if _query_embedder is not None:
        _query_embedder.unload()
        _query_embedder = None
