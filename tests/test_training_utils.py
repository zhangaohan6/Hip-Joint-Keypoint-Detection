import pytest
from torch.utils.data import TensorDataset
import torch

from train_hip_crop_keypoints import limit_dataset, set_global_seed, write_training_log


def test_limit_dataset_is_deterministic():
    dataset = TensorDataset(torch.arange(20))

    first = limit_dataset(dataset, 5, seed=7)
    second = limit_dataset(dataset, 5, seed=7)

    assert first.indices == second.indices
    assert len(first) == 5


def test_limit_dataset_preserves_small_dataset():
    dataset = TensorDataset(torch.arange(3))

    assert limit_dataset(dataset, 5) is dataset


def test_limit_dataset_rejects_non_positive_limit():
    with pytest.raises(ValueError, match="positive"):
        limit_dataset(TensorDataset(torch.arange(3)), 0)


def test_global_seed_repeats_torch_values():
    set_global_seed(9)
    first = torch.rand(3)
    set_global_seed(9)
    second = torch.rand(3)

    assert torch.equal(first, second)


def test_training_log_is_written_per_epoch(tmp_path):
    path = tmp_path / "metrics.txt"
    write_training_log(path, [1.2], [0.1], [8.0], [1.0], [0.08], [6.0])

    text = path.read_text(encoding="utf-8")
    assert "Epoch 1" in text
    assert "Val Pixel Error = 6.0000" in text
