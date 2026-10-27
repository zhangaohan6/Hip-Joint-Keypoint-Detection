import numpy as np
import pytest

from utils.measurement_uncertainty import estimate_angle_uncertainty


POINTS_8 = np.array(
    [
        [10, 20], [20, 20], [15, 30], [18, 35],
        [90, 20], [80, 20], [85, 30], [82, 35],
    ],
    dtype=np.float64,
)


def test_zero_noise_collapses_interval():
    result = estimate_angle_uncertainty(POINTS_8, keypoint_std_px=0, samples=10)

    for median, low, high in result.values():
        assert median == pytest.approx(low)
        assert median == pytest.approx(high)


def test_uncertainty_is_deterministic_for_seed():
    first = estimate_angle_uncertainty(POINTS_8, 2.0, samples=100, seed=7)
    second = estimate_angle_uncertainty(POINTS_8, 2.0, samples=100, seed=7)

    assert first == second


def test_negative_noise_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        estimate_angle_uncertainty(POINTS_8, -1.0)
