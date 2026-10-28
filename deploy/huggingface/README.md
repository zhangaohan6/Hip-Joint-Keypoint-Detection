---
title: Trustworthy Hip Imaging AI
emoji: 🦴
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
license: agpl-3.0
short_description: Research demo for bilateral hip ROI and keypoint inference.
tags:
  - medical-imaging
  - computer-vision
  - streamlit
  - research
---

# Trustworthy Hip Imaging AI

Upload a full pelvis X-ray to run automatic bilateral hip ROI detection,
eight-keypoint prediction, acetabular-index measurement, and engineering
sensitivity analysis.

> **Research prototype only.** Outputs must not be used for diagnosis or
> treatment decisions. The sensitivity interval is not calibrated clinical
> uncertainty.

The Docker build downloads the published MTDDH MVP checkpoints and verifies
their SHA256 digests before installation. Application code does not write
uploaded radiographs to persistent storage.

Source, methodology, tests, and model details:
[zhangaohan6/Hip-Joint-Keypoint-Detection](https://github.com/zhangaohan6/Hip-Joint-Keypoint-Detection/tree/feat/mtddh-mvp)
