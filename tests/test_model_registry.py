import pytest

from models import model as model_registry


def test_convnext_is_available_without_mambavision():
    assert "convnext_tiny_fpn1234concat" in model_registry.MODEL


def test_missing_mambavision_has_actionable_error(monkeypatch):
    monkeypatch.setattr(model_registry, "_MAMBAVISION_IMPORT_ERROR", ImportError("missing"))

    with pytest.raises(ImportError, match="optional CUDA dependencies"):
        model_registry.initialize_model("mambavision_small", num_points=4)
