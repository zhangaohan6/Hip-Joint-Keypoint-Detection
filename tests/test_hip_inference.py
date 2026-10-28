from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from utils.hip_inference import (
    CropBox,
    detect_bilateral_rois,
    prepare_crop_box,
    restore_crop_keypoints,
)


def test_prepare_crop_box_matches_square_expansion_and_clamping():
    box = prepare_crop_box((100, 80), (10.0, 20.0, 50.0, 40.0))

    assert box == CropBox(x1=6.0, y1=6.0, x2=54.0, y2=54.0)


def test_restore_crop_keypoints_maps_resize_space_to_full_image():
    box = CropBox(10.0, 20.0, 110.0, 220.0)
    points = np.array([[0.0, 0.0], [112.0, 112.0], [224.0, 224.0]])

    restored = restore_crop_keypoints(points, box, input_size=224)

    np.testing.assert_allclose(
        restored,
        np.array([[10.0, 20.0], [60.0, 120.0], [110.0, 220.0]]),
    )


def test_restore_crop_keypoints_reverses_mirror_and_reordering():
    box = CropBox(0.0, 0.0, 224.0, 224.0)
    mirrored_points = np.array([[213.0, 20.0], [193.0, 40.0]])

    restored = restore_crop_keypoints(
        mirrored_points,
        box,
        input_size=224,
        mirrored=True,
        reorder_indices=[1, 0],
    )

    np.testing.assert_allclose(restored, np.array([[30.0, 40.0], [10.0, 20.0]]))


class FakeDetector:
    def __init__(self, classes, scores, boxes):
        self.boxes = SimpleNamespace(
            cls=torch.tensor(classes),
            conf=torch.tensor(scores),
            xyxy=torch.tensor(boxes),
        )

    def predict(self, **kwargs):
        return [SimpleNamespace(boxes=self.boxes)]


def test_detect_bilateral_rois_selects_best_box_per_side():
    detector = FakeDetector(
        classes=[0, 1, 0],
        scores=[0.7, 0.8, 0.9],
        boxes=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
    )

    detections, confidences = detect_bilateral_rois(
        image=Image.new("L", (20, 20)),
        detector=detector,
    )

    assert detections == {
        "LeftHip": (9.0, 10.0, 11.0, 12.0),
        "RightHip": (5.0, 6.0, 7.0, 8.0),
    }
    assert confidences == pytest.approx({"left": 0.9, "right": 0.8})


def test_detect_bilateral_rois_rejects_missing_side():
    detector = FakeDetector(classes=[0], scores=[0.9], boxes=[[1, 2, 3, 4]])

    with pytest.raises(ValueError, match="right hip ROI"):
        detect_bilateral_rois(
            image=Image.new("L", (20, 20)),
            detector=detector,
        )
