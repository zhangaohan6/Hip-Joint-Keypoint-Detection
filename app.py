from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, UnidentifiedImageError

from utils.csv_parser import parse_csv_get_points
from utils.hip_geometry import (
    calculate_acetabular_index_angles,
    unify_keypoints_format,
)
from utils.hip_inference import (
    detect_bilateral_rois,
    load_keypoint_model,
    load_roi_detector,
    predict_keypoints_from_detections,
)
from utils.measurement_uncertainty import estimate_angle_uncertainty


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset" / "mtddh_xray_2d"
MODEL_PATH = (
    PROJECT_ROOT
    / "weights"
    / "convnext_tiny_fpn1234concat_simcc_2d_sr3.0_sigma4.0_"
      "cropleft_mirror_224_30_0.0001_64_best.pth"
)
YOLO_MODEL_PATH = Path(
    os.environ.get(
        "HIP_YOLO_WEIGHTS",
        PROJECT_ROOT
        / "runs"
        / "train"
        / "yolo11n_mtddh_640"
        / "weights"
        / "best.pt",
    )
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


@st.cache_resource(show_spinner="Loading YOLO11n bilateral ROI detector...")
def load_cached_detector(model_path: str):
    return load_roi_detector(Path(model_path))


def draw_measurement_overlay(
    image: Image.Image,
    prediction: np.ndarray,
    ground_truth: np.ndarray | None = None,
    detections: dict[str, tuple[float, float, float, float]] | None = None,
    detection_confidences: dict[str, float] | None = None,
) -> Image.Image:
    points_12 = unify_keypoints_format(prediction)
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    if detections is not None:
        for side, label, color in (
            ("left", "LeftHip", (40, 180, 255)),
            ("right", "RightHip", (190, 80, 255)),
        ):
            if label not in detections:
                continue
            x1, y1, x2, y2 = detections[label]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            confidence = (detection_confidences or {}).get(side)
            text = label if confidence is None else f"{label} {confidence:.2f}"
            draw.text((x1 + 4, y1 + 4), text, fill=color)

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
    input_options = ["Upload full pelvis X-ray"]
    if samples:
        input_options.insert(0, "Public MTDDH sample")
    input_mode = st.sidebar.radio("Input source", input_options)

    selected_stem = None
    uploaded_file = None
    if input_mode == "Public MTDDH sample":
        selected_stem = st.sidebar.selectbox("Public MTDDH sample", list(samples))
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Pelvis X-ray", type=["jpg", "jpeg", "png"]
        )
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
    if not YOLO_MODEL_PATH.exists():
        st.error(
            f"YOLO checkpoint was not found: `{YOLO_MODEL_PATH}`. "
            "Set `HIP_YOLO_WEIGHTS` to an alternate checkpoint path."
        )
        return

    if input_mode == "Public MTDDH sample":
        image_path, annotation_path = samples[selected_stem]
        image = Image.open(image_path).convert("RGB")
        ground_truth = parse_csv_get_points(annotation_path)
        image_label = selected_stem
    else:
        if uploaded_file is None:
            st.info("Upload a full pelvis X-ray to run automatic bilateral inference.")
            return
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except (UnidentifiedImageError, OSError):
            st.error("The uploaded file could not be decoded as an image.")
            return
        ground_truth = None
        image_label = Path(uploaded_file.name).stem

    detector = load_cached_detector(str(YOLO_MODEL_PATH))
    model_bundle = load_cached_model(str(MODEL_PATH))
    try:
        with st.spinner("Detecting bilateral hip ROIs..."):
            detections, detection_confidences = detect_bilateral_rois(image, detector)
        with st.spinner("Running bilateral keypoint inference..."):
            prediction, _ = predict_keypoints_from_detections(
                image, detections, model_bundle
            )
    except ValueError as error:
        st.error(f"Automatic inference stopped: {error}")
        return

    points_12 = unify_keypoints_format(prediction)
    left_angle, right_angle = calculate_acetabular_index_angles(points_12)
    if ground_truth is not None:
        gt_left_angle, gt_right_angle = calculate_acetabular_index_angles(
            unify_keypoints_format(ground_truth)
        )
    uncertainty = estimate_angle_uncertainty(
        prediction,
        keypoint_std_px=keypoint_std_px,
        samples=monte_carlo_samples,
    )

    overlay = draw_measurement_overlay(
        image,
        prediction,
        ground_truth,
        detections,
        detection_confidences,
    )
    image_col, metrics_col = st.columns([2, 1])
    with image_col:
        st.image(
            overlay,
            caption=(
                f"{image_label} — red × prediction, blue/purple YOLO ROI"
                + (", green ○ ground truth" if ground_truth is not None else "")
            ),
            width="stretch",
        )
    with metrics_col:
        st.subheader("Model prediction")
        if ground_truth is not None:
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
        else:
            st.metric("Left AI angle", f"{left_angle:.2f}°")
            st.metric("Right AI angle", f"{right_angle:.2f}°")
        st.metric("Absolute asymmetry", f"{abs(left_angle - right_angle):.2f}°")
        if ground_truth is not None:
            point_errors = np.linalg.norm(prediction - ground_truth, axis=1)
            st.metric("Mean keypoint error", f"{point_errors.mean():.2f} px")
        st.caption(
            "YOLO confidence: "
            f"left {detection_confidences['left']:.3f}, "
            f"right {detection_confidences['right']:.3f}"
        )

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
            "The YOLO11n detector finds the left and right hip ROIs, then the trained "
            "ConvNeXt-SimCC model predicts four keypoints per side. For public MTDDH "
            "samples, ground-truth keypoints are used only for the displayed error "
            "comparison and never as model input. Uploaded images have no ground-truth "
            "comparison."
        )


if __name__ == "__main__":
    main()
