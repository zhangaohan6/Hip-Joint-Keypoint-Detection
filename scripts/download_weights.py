#!/usr/bin/env python3
"""Download the reproducible MTDDH MVP checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen


RELEASE_BASE_URL = (
    "https://github.com/zhangaohan6/Hip-Joint-Keypoint-Detection/"
    "releases/download/mtddh-mvp-v0.1.0"
)


@dataclass(frozen=True)
class Asset:
    name: str
    destination: Path
    sha256: str


ASSETS = (
    Asset(
        name="mtddh-yolo11n-roi.pt",
        destination=Path("runs/train/yolo11n_mtddh_640/weights/best.pt"),
        sha256="a6d39033d5638fd0c0642f83920c96b24e1aa4442535f04544982d1e34de6376",
    ),
    Asset(
        name="mtddh-convnext-simcc-keypoints.pth",
        destination=Path(
            "weights/convnext_tiny_fpn1234concat_simcc_2d_sr3.0_sigma4.0_"
            "cropleft_mirror_224_30_0.0001_64_best.pth"
        ),
        sha256="d8180903be685495f0f0694cbe558f098334e89e92235ac3189079a2f20b6e25",
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_asset(asset: Asset, project_root: Path, base_url: str, force: bool) -> Path:
    destination = project_root / asset.destination
    if destination.exists() and not force:
        if file_sha256(destination) == asset.sha256:
            print(f"Verified existing {asset.destination}")
            return destination
        raise RuntimeError(
            f"Existing file failed SHA256 verification: {asset.destination}. "
            "Re-run with --force to replace it."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{base_url.rstrip('/')}/{asset.name}"
    if urlparse(url).scheme not in {"http", "https"}:
        raise ValueError("Checkpoint URLs must use HTTP or HTTPS.")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".download",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            print(f"Downloading {url}")
            with urlopen(url) as response:  # nosec B310: scheme restricted above
                shutil.copyfileobj(response, temporary_file, length=1024 * 1024)

        actual_sha256 = file_sha256(temporary_path)
        if actual_sha256 != asset.sha256:
            raise RuntimeError(
                f"SHA256 mismatch for {asset.name}: expected {asset.sha256}, "
                f"got {actual_sha256}"
            )
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(f"Installed {asset.destination}")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--base-url",
        default=RELEASE_BASE_URL,
        help="Release asset base URL (mainly useful for mirrors and tests)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing checkpoints even when present",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for asset in ASSETS:
        download_asset(
            asset,
            project_root=args.project_root.resolve(),
            base_url=args.base_url,
            force=args.force,
        )


if __name__ == "__main__":
    main()
