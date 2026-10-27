import numpy as np

from utils.hip_inference import CropBox, prepare_crop_box, restore_crop_keypoints


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
