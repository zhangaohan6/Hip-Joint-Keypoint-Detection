# 🦴 Deep Learning-Based Keypoint Detection for Predicting Developmental Dysplasia of the Hip from Pediatric Hip Radiographs

<div align="center">
  <div>
    <a href="https://github.com/tana0101/Hip-Joint-Keypoint-Detection/blob/main/README_zh_TW.md">🇹🇼 Traditional Chinese</a> |
    <a href="https://github.com/tana0101/Hip-Joint-Keypoint-Detection/blob/main/README.md">🌏 English</a> |
    <a href="https://deepwiki.com/tana0101/Hip-Joint-Keypoint-Detection">📚 DeepWiki</a> |
    <a href="https://github.com/tana0101/Hip-Joint-Keypoint-Detection/issues">❓ Issues</a> |
    📝 Paper (restricted by confidentiality agreement)
  </div>

<br>

  <img src="src/img/project_banner.png" style="width: 70%;"/>

<br>

  <a href="https://app.codacy.com/gh/tana0101/Hip-Joint-Keypoint-Detection/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade"><img src="https://app.codacy.com/project/badge/Grade/800c026fb9d1418e9cb735d1455c3383"/></a>
  <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/tana0101/Hip-Joint-Keypoint-Detection">
  <img alt="Using Python version" src="https://img.shields.io/badge/python-3.10-blue.svg">
  <a href="https://deepwiki.com/tana0101/Hip-Joint-Keypoint-Detection"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white"/>
  <img alt="Ultralytics YOLO" src="https://img.shields.io/badge/Ultralytics%20YOLO-%23000000.svg?style=flat&logo=ultralytics&logoColor=white"/>
</div>

## 📋 Overview

Developmental Dysplasia of the Hip (**DDH**) may lead to long-term impairment of a child's walking ability if it is not diagnosed early. In clinical practice, DDH assessment relies heavily on manual interpretation of X-ray images by physicians, which may result in subjective measurement variability.

This project proposes a **top-down, two-stage deep learning detection system** to assist with DDH measurement and classification:

1. **Automatic Detection and Localization**: YOLO is used to accurately localize the hip region of interest (ROI).
2. **Keypoint Prediction**: Anatomical keypoints are localized within each ROI, with either 12 or 8 hip keypoints annotated depending on the dataset.
3. **Clinical Measurement**: The system automatically calculates the **Acetabular Index (AI) angle** and outputs the **IHDI classification** result.

> **Research-only Notice**: This project is a prototype system developed for computer science research purposes. Its outputs must not be used directly as a basis for clinical diagnosis.

## ✨ Key Features

- 🧩 **Highly Modular Architecture (Modular Design)**  
  Object detection and keypoint prediction are decoupled, with a unified model interface designed to support rapid replacement of experimental configurations:
  - **Backbones**: Built-in support for ConvNeXt (V1/V2) and HRNet, with extensible interfaces for additional network architectures.
  - **Prediction Heads**: Supports the official SimCC implementation, modified SimCC variants, conventional Direct Regression, and Heatmap-based methods.

- ⚙️ **Microservice and Backend Integration (Backend-Ready)**  
  The core inference pipeline has been encapsulated as a backend Inference API and is currently being tested at National Cheng Kung University Hospital.

- 📊 **Comprehensive Evaluation Framework**  
  Includes complete K-Fold cross-validation and data visualization workflows, with automatic generation of confusion matrices and error-distribution plots.

## 💾 Dataset

- Datasets are stored under the `dataset/` directory.
- The annotation tool is located in `Keypoint-Annotation-Tool/`.

### 🏥 NCKUH_IHDI (National Cheng Kung University Hospital Dataset)

<img src="src/img/sample_IHDI.jpg" style="width: 30%;"/>

This study uses retrospectively collected hip X-ray images from National Cheng Kung University Hospital acquired between June 25, 2019 and January 7, 2025, with patient ages ranging from 1 to 59 months at the time of imaging. A total of 622 images were initially included, and 557 images remained after outlier exclusion. Each image was manually annotated by a clinical physician with **12 keypoints**, together with LeftHip / RightHip object labels for training the detection stage.

- Image annotation: each image corresponds to one `.csv` file in the following format:

```text
"(x1,y1)","(x2,y2)",...,"(x12,y12)"
```

- Note: Due to medical privacy and data protection requirements, this dataset cannot be publicly released.

<hr>

### 🌍 MTDDH (Public Dataset)

<img src="src/img/sample_MTDDH.jpg" style="width: 30%;"/>

- Source: [open-hip-dysplasia](https://github.com/radoss-org/open-hip-dysplasia.git)
- Dataset size: 1,666 hip X-ray images after outlier exclusion
- Annotations:
  - **8 keypoints**
  - LeftHip / RightHip object labels

<hr>

### 📊 Dataset Statistics

**Acetabular Index (AI) Distribution**

<img src="dataset/xray_IHDI_AI_Distribution.png" />

<img src="dataset/mtddh_xray_2d_AI_Distribution.png" />

**IHDI Classification Distribution**

<div style="display: flex; justify-content: space-between; gap: 10px;">
  <img src="dataset/xray_IHDI_IHDI_Distribution.png" style="width: 49%;" />
  <img src="dataset/mtddh_xray_2d_IHDI_Distribution.png" style="width: 49%;" />
</div>

## 🏆 Model Performance & Results

The `ConvNeXtTinyMS` backbone with the `SimCC 2D` head was evaluated using 5-fold cross-validation on the public MTDDH dataset. The aggregated OOF (Out-of-Fold) results are shown below:

<p align="center">
  <img src="src/img/avg_dists.png" width="75%">
  <br>
  <b>(a) Keypoint Distance Error</b>
</p>

<p align="center">
  <img src="src/img/AI_angle_errors.png" width="75%">
  <br>
  <b>(b) AI Angle Error</b>
</p>

<p align="center">
  <img src="src/img/CM_4Class_all.png" width="75%">
  <br>
  <b>(c) IHDI 4-Class Confusion Matrix</b>
</p>

<p align="center">
  <img src="src/img/bland_altman_overall_ai_angle.png" width="53%">
  <img src="src/img/scatter_overall_ai_angle.png" width="36%">
  <br>
  <b>(d) Bland-Altman & Scatter Plot of AI Angle</b>
</p>

## 🛠️ Methodology

This project adopts a **top-down two-stage keypoint detection pipeline**:

1. **🔍 Object Detection and Single-Side Cropping**: YOLO detects the LeftHip / RightHip regions and crops the corresponding ROIs to reduce background interference.
2. **🧠 Single-Side Keypoint Detection**: Keypoint detection is performed independently on each cropped hip ROI, supporting comparisons across multiple Backbone / Head configurations.

### Head Architecture

<img src="src/img/head_design.png" style="width: 99%;" />

This project supports multiple keypoint head designs to accommodate different model characteristics and experimental requirements:

- **SimCC 2D / SimCC 2D Deconv**:  
  The official SimCC-based approaches convert coordinate regression into independent one-dimensional classification distributions along the x- and y-axes, with coordinates decoded using soft-argmax.

- **SimCC 1D (Custom Variant)**:  
  Global Average Pooling is used to compress the feature map, followed by fully connected layers that predict the x- and y-axis distributions with reduced computational complexity.

- **Direct Regression**:  
  Fully connected layers directly regress the `(x, y)` coordinates.

- **Heatmap**:  
  Follows the original HRNet implementation by producing two-dimensional heatmaps trained with MSE loss. During decoding, Argmax is used to identify the peak response, followed by a 0.25-pixel sub-pixel shift toward the second-highest response to compensate for quantization error.

> 💡 **For detailed implementations of the Head models and decoding logic, please refer to [`models/head.py`](models/head.py).**

### Backbone Architecture

The currently supported Backbone architectures are:

- ConvNeXtV1
  - `ConvNeXtTinyCustom`
- ConvNeXtV1 + Feature Pyramid Network (multi-scale features)
  - `ConvNeXtTinyMS` (`convnext_tiny_fpn1234concat`)
- HRNet
  - `HRNetW32Custom`
  - `HRNetW48Custom`

Models with the `Custom` suffix are modified and optimized based on the corresponding official implementations to better suit the hip keypoint detection task.

> 💡 **For all Backbone implementations, please refer to [`models/model.py`](models/model.py).**

🚧 **Other Backbones, such as EfficientNet and InceptionNeXt, are still under development and testing to ensure compatibility with different Head architectures.** 🚧

### Other Techniques

- **🔄 Data Augmentation**: Random Rotation / Random Translation
- **📉 Loss Functions**
  - **Direct Regression (MSE Loss)**
  - **Heatmap (MSE Loss)**
  - **SimCC Series (KL Divergence Loss)**
- **⚙️ Optimizers**: AdamW
- **📈 LR Schedulers**: Cosine Annealing + Warmup
- **Decoder**: Expectation, Heuristic

> 💡 **For detailed Decoder implementations, please refer to [`utils/simcc.py`](utils/simcc.py).**

## 📂 Project Structure

```text
Hip-Joint-Keypoint-Detection/
├── dataset/                          # 💾 Dataset directory
│   ├── xray_IHDI_2_clean/           # NCKUH dataset (Private)
│   └── mtddh_xray_2d/               # Public dataset
├── datasets/                         # Dataset loading and preprocessing modules
├── models/                           # 🧠 Model and Head definitions / implementations
├── src/                              # Core project resources and images
│   └── img/                          # Images used in README
├── utils/                            # 🛠️ Common utility functions
├── weights/                          # 📥 Trained model weights (.pth)
├── logs/                             # 📝 Training logs and loss curves
├── results/                          # 📊 Statistical and evaluation outputs
├── experiment/                       # Records of previous experiments
├── Keypoint-Annotation-Tool/         # 🖊️ Keypoint annotation tool
├── train_yolo.py                     # [Training] YOLO object detector
├── train_full_image_keypoints.py     # [Training] Bilateral full-image keypoint model
├── train_hip_crop_keypoints.py       # [Training] Single-side hip keypoint model
├── predict_full_image_keypoints.py   # [Inference] Full-image prediction and evaluation
├── predict_hip_crop_keypoints.py     # [Inference] Single-side prediction and evaluation
├── split.py                          # [Utility] Dataset split (Train/Val/Test)
├── split_from_json.py                # [Utility] Split dataset using the public split JSON
├── kfold_split.py                    # [Utility] K-Fold dataset splitting
├── kfold_train_yolo.py               # [K-Fold] YOLO cross-validation training
├── kfold_train_hip_crop_keypoints.py # [K-Fold] Keypoint cross-validation training
├── kfold_predict_hip_crop_keypoints.py # [K-Fold] Cross-validation evaluation
├── requirements.txt                  # 📦 Project dependencies
├── README.md                         # 🇬🇧 English documentation
└── README_zh_TW.md                   # 🇹🇼 Traditional Chinese documentation
```

## 🚀 Workflow Pipeline

The standard experimental and inference pipeline is as follows:

1. 📂 **Data Preparation**: Prepare the dataset first. For the public dataset, the `prepare_MTDDH_dataset` script can be used. Then run `split.py` or `kfold_split.py` to split the dataset.
2. 🔍 **Stage 1 (Detection)**: Run `train_yolo.py` or `kfold_train_yolo.py` to train the YOLO model for hip ROI detection and cropping, reducing background interference.
3. 🧠 **Stage 2 (Keypoint)**: Run `train_hip_crop_keypoints.py` or `kfold_train_hip_crop_keypoints.py` and specify the desired Backbone and Head to train the single-side keypoint model.
4. 📈 **Evaluation**: Run `predict_hip_crop_keypoints.py` or `kfold_predict_hip_crop_keypoints.py` to perform dataset inference and calculate AI angle errors and IHDI classification accuracy.

### Apple Silicon MVP quickstart

The ConvNeXt/HRNet workflow runs on Apple Silicon without the optional
CUDA-only MambaVision dependencies.

```bash
uv python install 3.10
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt

PATH="$PWD/.venv/bin:$PATH" bash prepare_MTDDH_dataset
.venv/bin/python split.py \
  --dataset dataset/mtddh_xray_2d --out data \
  --train 0.8 --val 0.1 --test 0.1 --seed 42

.venv/bin/pytest -q
.venv/bin/streamlit run app.py
```

When the trained checkpoints are present, the Streamlit app accepts either a
public MTDDH sample or an uploaded full pelvis X-ray. YOLO11n detects both hip
ROIs, then ConvNeXt-SimCC predicts four keypoints per side and maps them back to
the full image before calculating left/right acetabular-index angles. Public
sample ground-truth keypoints are used only for the displayed comparison; they
are never model input. The Monte Carlo interval is an engineering sensitivity
estimate, not calibrated model confidence or clinical uncertainty.

By default, the app loads the detector from
`runs/train/yolo11n_mtddh_640/weights/best.pt`. Set `HIP_YOLO_WEIGHTS` to use a
checkpoint elsewhere. Generated model weights remain intentionally gitignored.

### Reproducible 5-epoch smoke baseline

```bash
.venv/bin/python train_hip_crop_keypoints.py \
  --data_dir data --model_name convnext_tiny_fpn1234concat \
  --input_size 224 --epochs 5 --learning_rate 0.0001 --batch_size 8 \
  --side left --mirror --head_type simcc_2d --split_ratio 3.0 --sigma 4.0 \
  --max_train_samples 128 --max_val_samples 32 \
  --num_workers 0 --no_show --seed 42
```

On an Apple M4 using MPS, this smoke run completed in about 40 seconds. The
best epoch reached validation loss `2.2078`, NME `0.0371`, and crop-space mean
pixel error `12.8050`. Evaluation over the full 166-image validation split
gave mean crop-space errors of `10.6564 px` for left hips and `11.3691 px` for
mirrored right hips. These are pipeline smoke-test results from only 128
training samples, not final model performance.

### Full 30-epoch baseline

```bash
.venv/bin/python train_hip_crop_keypoints.py \
  --data_dir data --model_name convnext_tiny_fpn1234concat \
  --input_size 224 --epochs 30 --learning_rate 0.0001 --batch_size 64 \
  --side left --mirror --head_type simcc_2d --split_ratio 3.0 --sigma 4.0 \
  --num_workers 0 --no_show --seed 42
```

On an Apple M4 using MPS, the full run completed in about 67 minutes. The
best validation crop-space error was `4.0597 px`. Full-split evaluation of the
saved EMA checkpoint produced mean errors of `3.3838 px` (left) and `3.5996 px`
(mirrored right) on validation, and `3.5427 px` (left) and `3.6588 px`
(mirrored right) on test. Predictions remain a research baseline and must not
be interpreted as clinical measurements.

### YOLO11n bilateral ROI detector

```bash
.venv/bin/python train_yolo.py \
  --data data/data.yaml --model yolo11n.pt \
  --epochs 50 --patience 10 --imgsz 640 --batch 16 \
  --device mps --workers 0 \
  --project runs/train --name yolo11n_mtddh_640 \
  --pretrained --seed 42 --fliplr 0 --flipud 0 \
  --degrees 0 --shear 0 --perspective 0 --mosaic 0 --mixup 0
```

Training stopped at epoch 42 after ten epochs without improvement; the best
checkpoint was from epoch 32. On the held-out 168-image test split (336 hip
ROIs), it reached precision `0.9995`, recall `1.0000`, mAP50 `0.9950`, and
mAP50-95 `0.8683`. Left/right mAP50-95 was `0.8754`/`0.8612`. Generated model
weights and run directories are intentionally gitignored.

> 💡 **For detailed environment setup, the K-Fold cross-validation workflow, and complete command-line configurations, please refer to the [📖 Advanced Usage Guide](docs/USAGE_GUIDE.md).**

## 🔧 Analysis & Visualization Tools

In addition to the main training and evaluation pipeline, this project provides several utilities for dataset analysis and model result visualization:

- `visualize_single_simcc.py`  
  Visualizes keypoint predictions and SimCC probability distributions for a single image. This tool is useful for inspecting model predictions and analyzing abnormal or failure cases.  
  Results are saved to `simcc_vis_results/`.

- `analyze_hip_dataset.py`  
  Analyzes dataset statistics and distributions, including sample counts, AI, and IHDI distributions, and generates corresponding statistical plots.  
  The generated results can be found in the corresponding dataset directory under `dataset/`.

- `compare_dataset_annotations.py`  
  Compares keypoints, AI measurements, and related measurement differences between different annotation versions of the same dataset. This tool can be used to evaluate annotation consistency.  
  Results are saved to `annotation_comparison_results/`.

## ⚖️ License

This project is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

For details, please refer to the [LICENSE](LICENSE) file.

### Core Dependencies & Open-Source Acknowledgements

- **[Ultralytics YOLO](https://docs.ultralytics.com/)** (AGPL-3.0)
  - **Impact**: Because this project relies on YOLO for core training and inference functionality, the overall project follows the AGPL-3.0 requirements. If this project, or a modified version of it, is provided as a network service or publicly distributed, the corresponding source code must be made available.

- **[ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2)** (Meta Research)
  - **Code**: MIT License.
  - **Pretrained Weights**: **CC-BY-NC 4.0 (non-commercial use only)**.
  - **⚠️ Note**: If the official ConvNeXt V2 pretrained weights are used, the project is restricted to academic research or other non-commercial use under the applicable weight license.

- **[MambaVision](https://github.com/NVlabs/MambaVision)** (NVIDIA)
  - **License**: NVIDIA Source Code License-NC, which generally includes non-commercial-use restrictions. Please refer to the original repository for details.

- **[SimCC](https://github.com/leeyegy/SimCC)** (MIT)
- **[ConvNeXt V1](https://github.com/facebookresearch/ConvNeXt)** (MIT)
- **[EfficientNet](https://docs.pytorch.org/vision/main/models/efficientnet.html)** (BSD-3-Clause via TorchVision)
- **[InceptionNeXt](https://github.com/sail-sg/inceptionnext)** (Apache-2.0)
- **[HRNet (Bottom-Up)](https://github.com/HRNet/HRNet-Bottom-Up-Pose-Estimation)** (MIT)

We sincerely thank the authors and contributors of open-source projects such as SimCC and HRNet for their contributions to the deep learning community. For academic discussion or backend integration collaboration, please feel free to contact us through GitHub Issues.
