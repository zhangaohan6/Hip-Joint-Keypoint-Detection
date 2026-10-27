from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.hip_crop_keypoints import HipCropKeypointDataset, MirroredToSideDataset
from datasets.transforms import get_hip_base_transform
from models.model import initialize_model
from utils.device import get_best_device
from utils.evaluation import extract_info_from_model_path
from utils.keypoints import get_pred_coords


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize baseline keypoint predictions")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("data/val"))
    parser.add_argument("--model_name", default="convnext_tiny_fpn1234concat")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--eval_samples", type=int, default=0, help="0 evaluates the full split")
    parser.add_argument("--mirror_to_side", choices=["left", "right"], default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("baseline_outputs"))
    args = parser.parse_args()

    head_type, input_size, _, _, _, split_ratio, _ = extract_info_from_model_path(
        args.model_path
    )
    nx = int(input_size * split_ratio) if split_ratio is not None else None
    ny = int(input_size * split_ratio) if split_ratio is not None else None

    base_dataset = HipCropKeypointDataset(
        img_dir=args.data_dir / "images",
        annotation_dir=args.data_dir / "annotations",
        detections_dir=args.data_dir / "detections",
        side=args.side,
        transform=get_hip_base_transform(input_size),
        input_size=input_size,
        bbox_jitter=False,
    )
    dataset = (
        MirroredToSideDataset(base_dataset, target_side=args.mirror_to_side)
        if args.mirror_to_side
        else base_dataset
    )
    device = get_best_device()
    model = initialize_model(
        args.model_name,
        num_points=base_dataset.num_keypoints,
        head_type=head_type,
        input_size=(input_size, input_size),
        Nx=nx,
        Ny=ny,
    )
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.to(device).eval()

    plot_count = min(args.num_samples, len(dataset))
    eval_count = len(dataset) if args.eval_samples == 0 else min(args.eval_samples, len(dataset))
    if plot_count < 1 or eval_count < 1:
        raise ValueError("num_samples and evaluated sample count must be positive")
    cols = min(4, plot_count)
    rows = int(np.ceil(plot_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    results = []

    for index in range(eval_count):
        image, keypoints, _, image_name = dataset[index]
        with torch.inference_mode():
            outputs = model(image.unsqueeze(0).to(device))
            prediction = get_pred_coords(
                outputs,
                head_type=head_type,
                Nx=nx,
                Ny=ny,
                input_size=input_size,
            )[0].cpu().numpy()

        target = keypoints.numpy().reshape(-1, 2)
        distances = np.linalg.norm(prediction - target, axis=1)
        results.append(
            {
                "image": image_name,
                "mean_error_px": float(distances.mean()),
                "max_error_px": float(distances.max()),
            }
        )

        if index < plot_count:
            axis = axes[index // cols][index % cols]
            axis.imshow(image.permute(1, 2, 0).numpy(), cmap="gray")
            axis.scatter(target[:, 0], target[:, 1], c="lime", s=45, label="GT")
            axis.scatter(prediction[:, 0], prediction[:, 1], c="red", s=55, marker="x", label="Pred")
            for gt, pred in zip(target, prediction):
                axis.plot([gt[0], pred[0]], [gt[1], pred[1]], color="yellow", linewidth=1)
            axis.set_title(f"{image_name}\nmean error: {distances.mean():.1f}px")
            axis.axis("off")

    for index in range(plot_count, rows * cols):
        axes[index // cols][index % cols].axis("off")
    axes[0][0].legend(loc="lower right")
    fig.suptitle(f"{args.side.title()} hip baseline: GT vs prediction")
    fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / f"{args.side}_gt_vs_prediction.png"
    metrics_path = args.output_dir / f"{args.side}_sample_metrics.json"
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"device={device}")
    print(f"evaluated_samples={eval_count}")
    print(f"visualized_samples={plot_count}")
    print(f"mean_error_px={np.mean([item['mean_error_px'] for item in results]):.4f}")
    print(f"figure={figure_path}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
