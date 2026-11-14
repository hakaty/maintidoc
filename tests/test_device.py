import logging

import pytest

from app.config import Settings
from app.device import resolve_device, resolve_query_embedding_device


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_auto_picks_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.device.torch.cuda.is_available", lambda: True)
    assert resolve_device(_settings(device="auto"), "X") == "cuda"


def test_auto_falls_back_to_cpu_and_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("app.device.torch.cuda.is_available", lambda: False)
    with caplog.at_level(logging.WARNING):
        device = resolve_device(_settings(device="auto"), "X")
    assert device == "cpu"
    assert "CPU mode" in caplog.text


def test_explicit_cpu_forced_even_with_cuda_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.device.torch.cuda.is_available", lambda: True)
    assert resolve_device(_settings(device="cpu"), "X") == "cpu"


def test_explicit_cuda_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.device.torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="DEVICE=cuda"):
        resolve_device(_settings(device="cuda"), "X")


def test_explicit_cuda_succeeds_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.device.torch.cuda.is_available", lambda: True)
    assert resolve_device(_settings(device="cuda"), "X") == "cuda"


def test_query_device_defaults_to_device_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.device.torch.cuda.is_available", lambda: True)
    settings = _settings(device="cpu", embed_query_device=None)
    assert resolve_query_embedding_device(settings) == "cpu"


def test_query_device_overrides_device_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.device.torch.cuda.is_available", lambda: True)
    settings = _settings(device="cuda", embed_query_device="cpu")
    assert resolve_query_embedding_device(settings) == "cpu"


def test_query_device_cuda_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.device.torch.cuda.is_available", lambda: False)
    settings = _settings(device="cpu", embed_query_device="cuda")
    with pytest.raises(RuntimeError, match="EMBED_QUERY_DEVICE=cuda"):
        resolve_query_embedding_device(settings)
