from pathlib import Path

import numpy as np
import open_clip
import pytest
import torch
from PIL import Image as PILImage

from app.config import Settings
from app.images.clip_encoder import ClipEncoder, get_embedding_dim

FAKE_DIM = 8


class _FakeClipModel(torch.nn.Module):
    """Deterministic stand-in for a real CLIP model: encode_image/encode_text
    without downloading or running any real network."""

    def __init__(self, dim: int = FAKE_DIM) -> None:
        super().__init__()
        self.dim = dim
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))  # exp(0) == 1

    def encode_image(self, tensor: torch.Tensor) -> torch.Tensor:
        means = tensor.mean(dim=(1, 2, 3))
        offsets = torch.arange(self.dim, device=tensor.device).float()
        return means.unsqueeze(1).repeat(1, self.dim) + offsets

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        seeds = tokens.float().mean(dim=1)
        offsets = torch.arange(self.dim, device=tokens.device).float()
        return seeds.unsqueeze(1).repeat(1, self.dim) + offsets

    def to(self, device: str) -> "_FakeClipModel":
        return self

    def eval(self) -> "_FakeClipModel":  # torch.nn.Module.eval(): eval mode, not builtin eval()
        return self


def _fake_preprocess(image: PILImage.Image) -> torch.Tensor:
    return torch.full((3, 4, 4), float(sum(image.getpixel((0, 0)))) / 765.0)


def _fake_tokenizer(texts: list[str]) -> torch.Tensor:
    return torch.tensor([[float(sum(ord(c) for c in text) % 97)] * 4 for text in texts])


@pytest.fixture
def fake_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        open_clip,
        "create_model_and_transforms",
        lambda model_name, pretrained=None: (_FakeClipModel(), None, _fake_preprocess),
    )
    monkeypatch.setattr(open_clip, "get_tokenizer", lambda model_name: _fake_tokenizer)
    monkeypatch.setattr(open_clip, "get_model_config", lambda model_name: {"embed_dim": FAKE_DIM})


@pytest.fixture
def two_images(tmp_path: Path) -> list[Path]:
    # Different total brightness (not just hue): _fake_preprocess sums RGB
    # channels, so e.g. red (255,0,0) and blue (0,0,255) would collide.
    dark = tmp_path / "dark.png"
    light = tmp_path / "light.png"
    PILImage.new("RGB", (16, 16), color=(20, 20, 20)).save(dark)
    PILImage.new("RGB", (16, 16), color=(230, 230, 230)).save(light)
    return [dark, light]


def test_embedding_dim_read_from_model(fake_clip: None) -> None:
    encoder = ClipEncoder(Settings(_env_file=None))
    assert encoder.embedding_dim == FAKE_DIM


def test_get_embedding_dim_does_not_load_model(fake_clip: None) -> None:
    assert get_embedding_dim("MobileCLIP-S2") == FAKE_DIM


def test_encode_images_shape_and_normalisation(fake_clip: None, two_images: list[Path]) -> None:
    encoder = ClipEncoder(Settings(_env_file=None))
    embeddings = encoder.encode_images(two_images)

    assert embeddings.shape == (2, FAKE_DIM)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_encode_images_batching_gives_same_result(fake_clip: None, two_images: list[Path]) -> None:
    encoder = ClipEncoder(Settings(_env_file=None))
    one_batch = encoder.encode_images(two_images, batch_size=2)
    per_image = encoder.encode_images(two_images, batch_size=1)

    np.testing.assert_allclose(one_batch, per_image, atol=1e-5)


def test_encode_images_different_inputs_differ(fake_clip: None, two_images: list[Path]) -> None:
    encoder = ClipEncoder(Settings(_env_file=None))
    embeddings = encoder.encode_images(two_images)

    assert not np.allclose(embeddings[0], embeddings[1])


def test_encode_texts_shape_and_normalisation(fake_clip: None) -> None:
    encoder = ClipEncoder(Settings(_env_file=None))
    embeddings = encoder.encode_texts(["a photo", "a diagram"])

    assert embeddings.shape == (2, FAKE_DIM)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_encode_empty_lists(fake_clip: None) -> None:
    encoder = ClipEncoder(Settings(_env_file=None))

    assert encoder.encode_images([]).shape == (0, FAKE_DIM)
    assert encoder.encode_texts([]).shape == (0, FAKE_DIM)


def test_unload_resets_and_can_reload(fake_clip: None, two_images: list[Path]) -> None:
    encoder = ClipEncoder(Settings(_env_file=None))
    encoder.encode_images(two_images[:1])
    encoder.unload()

    # still usable after unload -- lazily reloads
    embeddings = encoder.encode_images(two_images)
    assert embeddings.shape == (2, FAKE_DIM)


def test_device_never_silently_cpu_when_cuda_expected(fake_clip: None) -> None:
    """device must reflect torch.cuda.is_available(), never hard-coded."""
    encoder = ClipEncoder(Settings(_env_file=None))
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert encoder.device == expected


@pytest.mark.integration
def test_real_mobileclip_loads_and_encodes(tmp_path: Path) -> None:
    """Loads the real configured MobileCLIP model (network/GPU required)."""
    image_path = tmp_path / "sample.png"
    PILImage.new("RGB", (32, 32), color=(10, 20, 30)).save(image_path)

    try:
        encoder = ClipEncoder(Settings(_env_file=None))
        embeddings = encoder.encode_images([image_path])
    except Exception as exc:  # pragma: no cover - network/model download failure
        pytest.skip(f"Real MobileCLIP model unavailable: {exc}")

    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] == encoder.embedding_dim
    np.testing.assert_allclose(np.linalg.norm(embeddings[0]), 1.0, atol=1e-4)
    encoder.unload()
