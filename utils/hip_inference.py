from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from config import Dataset as DataConfig
from datasets.hip_crop_keypoints import read_detection_for_image
from datasets.transforms import get_hip_base_transform
from models.model import initialize_model
from utils.device import get_best_device
from utils.evaluation import extract_info_from_model_path
from utils.keypoints import get_pred_coords


@dataclass(frozen=True)
class CropBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


def prepare_crop_box(
    image_size: tuple[int, int],
    detection: tuple[float, float, float, float],
    crop_expand: float = 0.10,
    keep_square: bool = True,
) -> CropBox:
    """Reproduce the deterministic crop geometry used by the training dataset."""
    width, height = image_size
    x1, y1, x2, y2 = detection

    if y2 - y1 < 2:
        box_width = max(2.0, x2 - x1)
        y2 = min(float(height), y1 + box_width)
        if y2 - y1 < box_width:
            y1 = max(0.0, y2 - box_width)

    if keep_square:
        side_length = max(max(2.0, x2 - x1), max(2.0, y2 - y1))
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        x1, x2 = center_x - side_length / 2.0, center_x + side_length / 2.0
        y1, y2 = center_y - side_length / 2.0, center_y + side_length / 2.0

    box_width = max(2.0, x2 - x1)
    box_height = max(2.0, y2 - y1)
    x1 -= box_width * crop_expand
    x2 += box_width * crop_expand
    y1 -= box_height * crop_expand
    y2 += box_height * crop_expand

    return CropBox(
        x1=max(0.0, x1),
        y1=max(0.0, y1),
        x2=min(float(width), x2),
        y2=min(float(height), y2),
    )


def restore_crop_keypoints(
    points: np.ndarray,
    crop_box: CropBox,
    input_size: int,
    mirrored: bool = False,
    reorder_indices: list[int] | None = None,
) -> np.ndarray:
    """Map model coordinates from a resized crop back to the full image."""
    restored = np.asarray(points, dtype=np.float32).copy()
    if mirrored:
        restored[:, 0] = (input_size - 1) - restored[:, 0]
        if reorder_indices is not None:
            inverse_order = np.argsort(reorder_indices)
            restored = restored[inverse_order]

    restored[:, 0] = crop_box.x1 + restored[:, 0] * crop_box.width / input_size
    restored[:, 1] = crop_box.y1 + restored[:, 1] * crop_box.height / input_size
    return restored


def load_keypoint_model(model_path: Path, model_name: str = "convnext_tiny_fpn1234concat"):
    head_type, input_size, _, _, _, split_ratio, _ = extract_info_from_model_path(model_path)
    nx = int(input_size * split_ratio) if split_ratio is not None else None
    ny = int(input_size * split_ratio) if split_ratio is not None else None
    model = initialize_model(
        model_name,
        num_points=4,
        head_type=head_type,
        input_size=(input_size, input_size),
        Nx=nx,
        Ny=ny,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    device = get_best_device()
    model.to(device).eval()
    return model, device, head_type, input_size, nx, ny


def predict_bilateral_keypoints(
    image: Image.Image,
    image_name: str,
    detections_dir: Path,
    model_bundle,
) -> tuple[np.ndarray, dict[str, CropBox]]:
    """Predict four keypoints per hip and return them in full-image coordinates."""
    model, device, head_type, input_size, nx, ny = model_bundle
    detections = read_detection_for_image(str(detections_dir), image_name)
    transform = get_hip_base_transform(input_size)
    grayscale = image.convert("L")
    predicted_sides = []
    crop_boxes = {}

    for side in ("left", "right"):
        crop_box = prepare_crop_box(
            grayscale.size,
            detections[DataConfig.SIDE_LABELS[side]],
        )
        crop = grayscale.crop(
            tuple(int(round(value)) for value in (crop_box.x1, crop_box.y1, crop_box.x2, crop_box.y2))
        )
        tensor = transform(crop)
        mirrored = side == "right"
        if mirrored:
            tensor = TF.hflip(tensor)

        with torch.inference_mode():
            outputs = model(tensor.unsqueeze(0).to(device))
            points = get_pred_coords(
                outputs,
                head_type=head_type,
                Nx=nx,
                Ny=ny,
                input_size=input_size,
            )[0].cpu().numpy()

        predicted_sides.append(
            restore_crop_keypoints(
                points,
                crop_box,
                input_size,
                mirrored=mirrored,
                reorder_indices=[0, 1, 2, 3] if mirrored else None,
            )
        )
        crop_boxes[side] = crop_box

    return np.concatenate(predicted_sides), crop_boxes
