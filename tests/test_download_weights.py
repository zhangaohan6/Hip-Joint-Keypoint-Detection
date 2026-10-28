import hashlib
import io
from pathlib import Path

import pytest

from scripts.download_weights import Asset, download_asset


def make_asset(content: bytes) -> Asset:
    name = "checkpoint.pt"
    return Asset(
        name=name,
        destination=Path("weights/checkpoint.pt"),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def test_download_asset_installs_and_verifies_checkpoint(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    content = b"checkpoint-content"
    asset = make_asset(content)
    monkeypatch.setattr(
        "scripts.download_weights.urlopen", lambda _: io.BytesIO(content)
    )

    destination = download_asset(
        asset,
        project_root=project_root,
        base_url="https://example.test/release",
        force=False,
    )

    assert destination.read_bytes() == b"checkpoint-content"
    assert (
        download_asset(asset, project_root, "https://example.test/release", False)
        == destination
    )


def test_download_asset_rejects_bad_checksum_without_installing(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    content = b"corrupt-checkpoint"
    asset = make_asset(content)
    asset = Asset(asset.name, asset.destination, "0" * 64)
    monkeypatch.setattr(
        "scripts.download_weights.urlopen", lambda _: io.BytesIO(content)
    )

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        download_asset(
            asset, project_root, "https://example.test/release", force=False
        )

    assert not (project_root / asset.destination).exists()
    assert not list((project_root / "weights").glob("*.download"))


def test_download_asset_requires_force_for_invalid_existing_file(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    content = b"valid-checkpoint"
    asset = make_asset(content)
    monkeypatch.setattr(
        "scripts.download_weights.urlopen", lambda _: io.BytesIO(content)
    )
    destination = project_root / asset.destination
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"stale")

    with pytest.raises(RuntimeError, match="Re-run with --force"):
        download_asset(
            asset, project_root, "https://example.test/release", force=False
        )

    download_asset(asset, project_root, "https://example.test/release", force=True)
    assert destination.read_bytes() == b"valid-checkpoint"


def test_download_asset_rejects_non_http_url(tmp_path):
    asset = make_asset(b"checkpoint")

    with pytest.raises(ValueError, match="HTTP or HTTPS"):
        download_asset(asset, tmp_path, "file:///tmp/release", force=False)
