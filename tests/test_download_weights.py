import hashlib
from pathlib import Path

import pytest

from scripts.download_weights import Asset, download_asset


def make_asset(source_dir: Path, content: bytes) -> Asset:
    name = "checkpoint.pt"
    (source_dir / name).write_bytes(content)
    return Asset(
        name=name,
        destination=Path("weights/checkpoint.pt"),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def test_download_asset_installs_and_verifies_checkpoint(tmp_path):
    source_dir = tmp_path / "source"
    project_root = tmp_path / "project"
    source_dir.mkdir()
    asset = make_asset(source_dir, b"checkpoint-content")

    destination = download_asset(
        asset,
        project_root=project_root,
        base_url=source_dir.as_uri(),
        force=False,
    )

    assert destination.read_bytes() == b"checkpoint-content"
    assert download_asset(asset, project_root, source_dir.as_uri(), False) == destination


def test_download_asset_rejects_bad_checksum_without_installing(tmp_path):
    source_dir = tmp_path / "source"
    project_root = tmp_path / "project"
    source_dir.mkdir()
    asset = make_asset(source_dir, b"corrupt-checkpoint")
    asset = Asset(asset.name, asset.destination, "0" * 64)

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        download_asset(asset, project_root, source_dir.as_uri(), force=False)

    assert not (project_root / asset.destination).exists()
    assert not list((project_root / "weights").glob("*.download"))


def test_download_asset_requires_force_for_invalid_existing_file(tmp_path):
    source_dir = tmp_path / "source"
    project_root = tmp_path / "project"
    source_dir.mkdir()
    asset = make_asset(source_dir, b"valid-checkpoint")
    destination = project_root / asset.destination
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"stale")

    with pytest.raises(RuntimeError, match="Re-run with --force"):
        download_asset(asset, project_root, source_dir.as_uri(), force=False)

    download_asset(asset, project_root, source_dir.as_uri(), force=True)
    assert destination.read_bytes() == b"valid-checkpoint"
