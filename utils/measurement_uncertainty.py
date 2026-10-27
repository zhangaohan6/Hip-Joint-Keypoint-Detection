from __future__ import annotations

import numpy as np

from utils.hip_geometry import (
    calculate_acetabular_index_angles,
    unify_keypoints_format,
)


def estimate_angle_uncertainty(
    points: np.ndarray,
    keypoint_std_px: float,
    samples: int = 1_000,
    seed: int = 42,
) -> dict[str, tuple[float, float, float]]:
    """Estimate AI-angle sensitivity to isotropic keypoint localization noise.

    The returned tuples contain median, 2.5th percentile, and 97.5th
    percentile. This is an engineering sensitivity estimate, not calibrated
    clinical or model uncertainty.
    """
    points_12 = unify_keypoints_format(points).astype(np.float64)
    if keypoint_std_px < 0:
        raise ValueError("keypoint_std_px must be non-negative")
    if samples < 1:
        raise ValueError("samples must be positive")

    rng = np.random.default_rng(seed)
    noisy_points = points_12 + rng.normal(
        loc=0.0,
        scale=keypoint_std_px,
        size=(samples, *points_12.shape),
    )
    angles = np.asarray(
        [calculate_acetabular_index_angles(sample) for sample in noisy_points],
        dtype=np.float64,
    )

    def summarize(values: np.ndarray) -> tuple[float, float, float]:
        median, low, high = np.percentile(values, [50.0, 2.5, 97.5])
        return float(median), float(low), float(high)

    return {
        "left": summarize(angles[:, 0]),
        "right": summarize(angles[:, 1]),
    }
