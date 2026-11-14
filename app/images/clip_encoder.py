"""MobileCLIP image/text embeddings via open_clip.

The text encoder is ENGLISH-ONLY (the underlying CLIP text tower was trained
on English captions): only use encode_texts() with English queries (M4/M5
zero-shot labels, image<->text search) or for image-to-image search via
encode_images() on both sides. French user queries must not be passed to
encode_texts() directly.
"""

import logging
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image as PILImage

from app.config import Settings, get_settings
from app.device import resolve_device

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 32


def get_embedding_dim(clip_model: str) -> int | None:
    """Embedding dim from open_clip's model config, without loading any weights."""
    config = open_clip.get_model_config(clip_model)
    return config["embed_dim"] if config else None


class ClipEncoder:
    """Lazily loads the configured MobileCLIP model once per process."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: open_clip.model.CLIP | None = None
        self._preprocess = None
        self._tokenizer = None
        self._device: str | None = None
        self._dim: int | None = None

    @property
    def device(self) -> str:
        self._ensure_loaded()
        assert self._device is not None
        return self._device

    @property
    def embedding_dim(self) -> int:
        self._ensure_loaded()
        assert self._dim is not None
        return self._dim

    @property
    def logit_scale(self) -> float:
        """The model's learned temperature, for calibrated zero-shot softmax."""
        self._ensure_loaded()
        if hasattr(self._model, "logit_scale"):
            return float(self._model.logit_scale.exp().item())
        return 100.0  # standard CLIP default when a model has no learned scale

    @property
    def peak_vram_mb(self) -> float | None:
        """Peak CUDA memory allocated since the model was (re)loaded, in MB.

        None on CPU. Tracked via torch.cuda.reset_peak_memory_stats() at load
        time + torch.cuda.max_memory_allocated(), not a point-in-time reading.
        """
        if self._device != "cuda":
            return None
        return torch.cuda.max_memory_allocated() / (1024**2)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        model_name = self._settings.clip_model
        pretrained = self._settings.clip_pretrained
        device = resolve_device(self._settings, "MobileCLIP")
        self._device = device
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            logger.info(
                "Loading %s/%s on GPU: %s", model_name, pretrained, torch.cuda.get_device_name(0)
            )
        else:
            logger.info("Loading %s/%s on CPU (fp32)", model_name, pretrained)

        self._dim = get_embedding_dim(model_name)

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model = model.to(device).eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(model_name)

        if self._dim is None:
            # Fallback for a model_name not in open_clip's config registry:
            # a single forward pass reveals the true output dimension.
            dummy = preprocess(PILImage.new("RGB", (224, 224))).unsqueeze(0).to(device)
            with torch.no_grad():
                self._dim = int(model.encode_image(dummy).shape[-1])

    def _autocast(self):
        if self._device == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return torch.autocast(device_type="cpu", enabled=False)

    @staticmethod
    def _l2_normalize(array: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(array, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        return array / norms

    def encode_images(
        self, image_paths: list[Path], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> np.ndarray:
        """L2-normalised image embeddings, shape (len(image_paths), embedding_dim)."""
        self._ensure_loaded()
        if not image_paths:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        all_features: list[np.ndarray] = []
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            tensors = torch.stack(
                [self._preprocess(PILImage.open(path).convert("RGB")) for path in batch_paths]
            ).to(self._device)
            with torch.no_grad(), self._autocast():
                features = self._model.encode_image(tensors)
            all_features.append(features.float().cpu().numpy())

        embeddings = np.concatenate(all_features, axis=0)
        return self._l2_normalize(embeddings).astype(np.float32)

    def encode_texts(self, texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
        """L2-normalised English-only text embeddings, shape (len(texts), embedding_dim)."""
        self._ensure_loaded()
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        all_features: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokens = self._tokenizer(batch).to(self._device)
            with torch.no_grad(), self._autocast():
                features = self._model.encode_text(tokens)
            all_features.append(features.float().cpu().numpy())

        embeddings = np.concatenate(all_features, axis=0)
        return self._l2_normalize(embeddings).astype(np.float32)

    def unload(self) -> None:
        """Free VRAM: drop the model and clear the CUDA cache."""
        if self._model is None:
            return
        device = self._device
        peak_mb = self.peak_vram_mb
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._dim = None
        self._device = None
        if device == "cuda":
            torch.cuda.empty_cache()
        if peak_mb is not None:
            logger.info("Unloaded CLIP model (peak VRAM: %.1f MB)", peak_mb)
        else:
            logger.info("Unloaded CLIP model")
            return
        logger.info("Unloaded CLIP model")
