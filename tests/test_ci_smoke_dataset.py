"""Tests for scripts/ci_smoke_dataset.py -- the synthetic dataset the CI smoke job runs
against, since a CI runner has no Kaggle credentials.

This file exists because the generator silently broke the smoke job. It emitted a
two-way train/test manifest while `fl/app.py` builds its server-held validation loader
from `split == "val"` unconditionally, so the loader came back empty and the run died
inside the simulation with `need at least one array to concatenate` -- surfaced by
`flwr run` as a bare "Exit Code: 700". The same two-way/three-way mismatch had already
broken the unit-test fixtures once before.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _generate(out_dir: Path) -> pd.DataFrame:
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "ci_smoke_dataset.py"), "--out-dir", str(out_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return pd.read_csv(out_dir / "manifest.csv")


def test_the_manifest_is_three_way_like_the_real_one(tmp_path: Path) -> None:
    """`val` is the one that matters and the one that was missing. A CI dataset shaped
    differently from the real Phase 1.3 manifest tests a pipeline the project does not
    run."""
    manifest = _generate(tmp_path / "ds")

    assert set(manifest["split"]) == {"train", "val", "test"}
    for split in ("train", "val", "test"):
        assert (manifest["split"] == split).sum() > 0, f"{split} split is empty"


def test_every_split_carries_more_than_one_class(tmp_path: Path) -> None:
    """A split with one class makes macro-F1 and ROC-AUC degenerate, which would make the
    smoke job's own assertions meaningless even when it passes."""
    manifest = _generate(tmp_path / "ds")

    for split in ("train", "val", "test"):
        labels = set(manifest.loc[manifest["split"] == split, "label"])
        assert len(labels) > 1, f"{split} has only {labels}"


def test_the_cache_and_manifest_agree(tmp_path: Path) -> None:
    """The run reads images from the cache and labels from the manifest; a length
    mismatch between them is an indexing bug that surfaces as wrong labels, not a crash."""
    out_dir = tmp_path / "ds"
    manifest = _generate(out_dir)

    import numpy as np

    caches = list((out_dir / "cache").glob("images_*.npy"))
    assert caches, "no image cache was written"
    images = np.load(caches[0], mmap_mode="r")
    assert len(images) == len(manifest)
