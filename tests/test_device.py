import torch

from utils.device import get_best_device


def test_get_best_device_returns_supported_backend():
    assert get_best_device().type in {"cuda", "mps", "cpu"}


def test_get_best_device_prefers_mps_when_available_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert get_best_device().type == "mps"
