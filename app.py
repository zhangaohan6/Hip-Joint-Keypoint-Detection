from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from utils.csv_parser import parse_csv_get_points
from utils.hip_geometry import (
    calculate_acetabular_index_angles,
    unify_keypoints_format,
)
from utils.hip_inference import load_keypoint_model, predict_bilateral_keypoints
from utils.measurement_uncertainty import estimate_angle_uncertainty


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset" / "mtddh_xray_2d"
DETECTIONS_ROOT = DATASET_ROOT / "detections"
MODEL_PATH = (
    PROJECT_ROOT
    / "weights"
    / "convnext_tiny_fpn1234concat_simcc_2d_sr3.0_sigma4.0_"
      "cropleft_mirror_224_30_0.0001_64_best.pth"
)


def find_samples(dataset_root: Path) -> dict[str, tuple[Path, Path]]:
    images_dir = dataset_root / "images"
    annotations_dir = dataset_root / "annotations"
    if not images_dir.exists() or not annotations_dir.exists():
        return {}

    samples: dict[str, tuple[Path, Path]] = {}
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        annotation_path = annotations_dir / f"{image_path.stem}.csv"
        if annotation_path.exists():
            samples[image_path.stem] = (image_path, annotation_path)
    return samples


@st.cache_resource(show_spinner="Loading ConvNeXt-SimCC checkpoint...")
def load_cached_model(model_path: str):
    return load_keypoint_model(Path(model_path))


def draw_measurement_overlay(
    image: Image.Image,
    prediction: np.ndarray,
    ground_truth: np.ndarray | None = None,
) -> Image.Image:
    points_12 = unify_keypoints_format(prediction)
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    p1, p3, p7, p9 = points_12[0], points_12[2], points_12[6], points_12[8]
    draw.line([tuple(p3), tuple(p7)], fill=(0, 255, 255), width=3)
    draw.line([tuple(p1), tuple(p3)], fill=(255, 128, 0), width=3)
    draw.line([tuple(p9), tuple(p7)], fill=(255, 128, 0), width=3)

    if ground_truth is not None:
        for prediction_point, target_point in zip(prediction, ground_truth):
            draw.line(
                [tuple(prediction_point), tuple(target_point)],
                fill=(255, 220, 0),
                width=2,
            )
            x, y = map(float, target_point)
            radius = 5
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=(0, 255, 80),
                width=3,
            )

    for index, point in enumerate(prediction):
        x, y = map(float, point)
        radius = 6
        draw.line((x - radius, y - radius, x + radius, y + radius), fill=(255, 40, 40), width=3)
        draw.line((x - radius, y + radius, x + radius, y - radius), fill=(255, 40, 40), width=3)
        draw.text((x + 8, y - 8), str(index + 1), fill=(255, 255, 255))
    return canvas


def main() -> None:
    st.set_page_config(page_title="Trustworthy Hip Imaging AI", layout="wide")
    st.title("Trustworthy Hip Imaging AI")
    st.caption("MTDDH automatic keypoint and acetabular-index measurement explorer")
    st.warning(
        "Research prototype only. The measurements and sensitivity intervals "
        "must not be used for diagnosis or treatment decisions."
    )

    samples = find_samples(DATASET_ROOT)
    if not samples:
        st.error(
            "Prepared MTDDH samples were not found. Run `bash prepare_MTDDH_dataset` first."
        )
        return

    selected_stem = st.sidebar.selectbox("Public MTDDH sample", list(samples))
    keypoint_std_px = st.sidebar.slider(
        "Assumed keypoint error (pixels)", 0.0, 10.0, 2.0, 0.5
    )
    monte_carlo_samples = st.sidebar.select_slider(
        "Sensitivity samples", options=[100, 250, 500, 1_000, 2_000], value=1_000
    )
    qa_threshold = st.sidebar.slider(
        "QA flag: max 95% interval width (degrees)", 1.0, 20.0, 5.0, 0.5
    )

    if not MODEL_PATH.exists():
        st.error(f"Trained checkpoint was not found: `{MODEL_PATH}`")
        return

    image_path, annotation_path = samples[selected_stem]
    image = Image.open(image_path)
    ground_truth = parse_csv_get_points(annotation_path)
    model_bundle = load_cached_model(str(MODEL_PATH))
    with st.spinner("Running bilateral keypoint inference..."):
        prediction, _ = predict_bilateral_keypoints(
            image,
            image_path.name,
            DETECTIONS_ROOT,
            model_bundle,
        )

    points_12 = unify_keypoints_format(prediction)
    left_angle, right_angle = calculate_acetabular_index_angles(points_12)
    gt_left_angle, gt_right_angle = calculate_acetabular_index_angles(
        unify_keypoints_format(ground_truth)
    )
    uncertainty = estimate_angle_uncertainty(
        prediction,
        keypoint_std_px=keypoint_std_px,
        samples=monte_carlo_samples,
    )

    overlay = draw_measurement_overlay(image, prediction, ground_truth)
    point_errors = np.linalg.norm(prediction - ground_truth, axis=1)
    image_col, metrics_col = st.columns([2, 1])
    with image_col:
        st.image(
            overlay,
            caption=f"{selected_stem} — red × prediction, green ○ ground truth",
            width="stretch",
        )
    with metrics_col:
        st.subheader("Model prediction")
        st.metric(
            "Left AI angle",
            f"{left_angle:.2f}°",
            delta=f"{left_angle - gt_left_angle:+.2f}° vs GT",
            delta_color="off",
        )
        st.metric(
            "Right AI angle",
            f"{right_angle:.2f}°",
            delta=f"{right_angle - gt_right_angle:+.2f}° vs GT",
            delta_color="off",
        )
        st.metric("Absolute asymmetry", f"{abs(left_angle - right_angle):.2f}°")
        st.metric("Mean keypoint error", f"{point_errors.mean():.2f} px")

        st.subheader("Localization sensitivity")
        widest_interval = 0.0
        for side in ("left", "right"):
            median, low, high = uncertainty[side]
            widest_interval = max(widest_interval, high - low)
            st.write(f"**{side.title()}**: {median:.2f}° (95%: {low:.2f}–{high:.2f}°)")

        if widest_interval > qa_threshold:
            st.error(
                f"Engineering QA flag: interval width {widest_interval:.2f}° "
                f"exceeds {qa_threshold:.2f}°."
            )
        else:
            st.success(
                f"Engineering QA pass: widest interval is {widest_interval:.2f}°."
            )

    with st.expander("What this uncertainty means"):
        st.write(
            "The interval comes from repeatedly perturbing every annotated keypoint "
            "with Gaussian pixel noise and recalculating the angle. In prediction "
            "mode it is centered on model outputs. It measures "
            "geometric sensitivity only; it is not a calibrated model confidence "
            "or a clinical reference range."
        )

    with st.expander("Pipeline scope and data use"):
        st.write(
            "The keypoints are produced by the trained ConvNeXt-SimCC model. This "
            "demo uses the public MTDDH detection boxes only to define the left and "
            "right hip crops; ground-truth keypoints are used solely for the displayed "
            "error comparison. Arbitrary full-image upload requires the separate ROI "
            "detector planned for the next stage."
        )


if __name__ == "__main__":
    main()
