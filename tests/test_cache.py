"""Tests for the decoded-image cache, including the auto-build-on-missing behaviour
added after a Colab session reset wiped a previously-built cache mid-project (the repo
survives via git; /content and everything in it does not)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fedswarm.data.cache import build_and_save_cache, ensure_cache
from fedswarm.data.datasets import load_cache


def _tiny_dataset(root: Path) -> pd.DataFrame:
    """A miniature dataset plus the manifest describing it, mirroring the real
    manifest's schema closely enough for build_and_save_cache to run. Testing/ must
    exist (even empty) for find_split_parent to recognize this as a valid dataset root."""
    (root / "Training" / "glioma").mkdir(parents=True)
    (root / "Testing" / "glioma").mkdir(parents=True)
    paths = []
    for i in range(3):
        p = root / "Training" / "glioma" / f"img_{i}.jpg"
        Image.new("L", (64, 64), color=50 + i * 30).save(p)
        paths.append(f"Training/glioma/img_{i}.jpg")

    return pd.DataFrame(
        {
            "path": paths,
            "label": "glioma",
            "pseudo_patient_id": range(3),
            "split": "train",
            "is_representative": True,
            "is_mixed_label": False,
        }
    )


def test_ensure_cache_builds_when_missing(tmp_path: Path) -> None:
    manifest = _tiny_dataset(tmp_path)
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    cache_dir = tmp_path / "cache"

    array_path = ensure_cache(size=32, manifest_path=manifest_path, cache_dir=cache_dir, root=tmp_path)

    assert array_path.exists()
    array = np.load(array_path)
    assert array.shape == (3, 32, 32)
    assert (cache_dir / "images_32_meta.json").exists()


def test_ensure_cache_does_not_rebuild_when_present(tmp_path: Path) -> None:
    manifest = _tiny_dataset(tmp_path)
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    cache_dir = tmp_path / "cache"

    first = ensure_cache(size=32, manifest_path=manifest_path, cache_dir=cache_dir, root=tmp_path)
    mtime_before = first.stat().st_mtime_ns

    # Second call: even though the manifest could in principle differ, ensure_cache's
    # contract is "build only if missing" -- so this must be a no-op, not a full rebuild.
    second = ensure_cache(size=32, manifest_path=manifest_path, cache_dir=cache_dir, root=tmp_path)

    assert second == first
    assert second.stat().st_mtime_ns == mtime_before


def test_load_cache_auto_builds_by_default(tmp_path: Path) -> None:
    manifest = _tiny_dataset(tmp_path)
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    cache_dir = tmp_path / "cache"

    assert not (cache_dir / "images_32.npy").exists()

    array = load_cache(size=32, cache_dir=cache_dir, manifest_path=manifest_path, root=tmp_path)

    assert array.shape == (3, 32, 32)


def test_load_cache_raises_when_auto_build_disabled(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"

    with pytest.raises(FileNotFoundError, match="No cache at"):
        load_cache(size=32, cache_dir=cache_dir, auto_build=False)


def test_build_and_save_cache_writes_array_and_metadata(tmp_path: Path) -> None:
    manifest = _tiny_dataset(tmp_path)
    out_dir = tmp_path / "out"

    array_path = build_and_save_cache(manifest, tmp_path, size=16, out_dir=out_dir)

    assert array_path == out_dir / "images_16.npy"
    meta_path = out_dir / "images_16_meta.json"
    assert meta_path.exists()

    array = np.load(array_path)
    assert array.dtype == np.uint8
    assert array.shape == (3, 16, 16)
