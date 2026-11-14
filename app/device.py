"""Shared CUDA/CPU device resolution for BGE-M3 and MobileCLIP.

DEVICE=auto (default) picks cuda if available, else falls back to cpu --
and that fallback is always logged, never silent. DEVICE=cuda forces GPU use
and raises immediately if none is available (instead of silently running on
CPU). DEVICE=cpu forces CPU even if a GPU is present.
"""

import logging

import torch

from app.config import Settings

logger = logging.getLogger(__name__)


def resolve_device(settings: Settings, component: str) -> str:
    choice = settings.device
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"DEVICE=cuda requested for {component} but no CUDA device is available"
            )
        return "cuda"
    if choice == "cpu":
        logger.warning("%s: CPU mode (DEVICE=cpu)", component)
        return "cpu"
    # auto
    if torch.cuda.is_available():
        return "cuda"
    logger.warning("%s: CPU mode (no CUDA device detected)", component)
    return "cpu"


def resolve_query_embedding_device(settings: Settings) -> str:
    """EMBED_QUERY_DEVICE overrides DEVICE for single-query embeddings only,
    so a query can run on CPU while the GPU is busy holding the LLM (Qwen)."""
    choice = settings.embed_query_device or settings.device
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "EMBED_QUERY_DEVICE=cuda requested but no CUDA device is available"
            )
        return "cuda"
    if choice == "cpu":
        logger.warning("Query embedding: CPU mode (EMBED_QUERY_DEVICE=cpu)")
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    logger.warning("Query embedding: CPU mode (no CUDA device detected)")
    return "cpu"
