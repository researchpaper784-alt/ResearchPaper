"""Phase 1.3 — dataset objects over the manifest and the decoded cache.

Nothing here re-derives the split or touches the filesystem per item: the manifest decides
membership, the cache holds pixels, and this module only joins them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from fedswarm.data.cache import CACHE_DIR, ensure_cache
from fedswarm.data.download import CLASSES
from fedswarm.data.splits import MANIFEST_CSV

LABEL_TO_INDEX = {label: i for i, label in enumerate(CLASSES)}


def load_manifest(path: Path | str = MANIFEST_CSV) -> pd.DataFrame:
    manifest = Path(path)
    if not manifest.exists():
        raise FileNotFoundError(
            f"No manifest at {manifest}. Run `python -m fedswarm.data.splits` first."
        )
    return pd.read_csv(manifest)


def load_cache(
    size: int = 112,
    cache_dir: Path | str = CACHE_DIR,
    manifest_path: Path | str = MANIFEST_CSV,
    root: Path | str | None = None,
    auto_build: bool = True,
) -> np.ndarray:
    """Loads the decoded-image cache, building it first if missing (the common case
    after a Colab/Kaggle session reset wipes /content but the git-cloned repo survives).
    Pass auto_build=False to get the old fail-fast behaviour instead."""
    array_path = Path(cache_dir) / f"images_{size}.npy"
    if not array_path.exists():
        if not auto_build:
            raise FileNotFoundError(
                f"No cache at {array_path}. Run `python -m fedswarm.data.cache --size {size}`."
            )
        ensure_cache(size, manifest_path, cache_dir, root)
    return np.load(array_path, mmap_mode="r")


def select(
    manifest: pd.DataFrame,
    split: str | None = None,
    representatives_only: bool = True,
    drop_mixed_label: bool = False,
) -> np.ndarray:
    """Row indices into the manifest (and therefore into the cache).

    `representatives_only` collapses each pseudo-patient to a single image. It defaults to
    True because this dataset variant is 34% redundant overall and 68% redundant in the
    `notumor` class; keeping every copy would reweight both training and evaluation toward
    a handful of distinct patients.

    `drop_mixed_label` excludes components whose images disagree on their class, for the
    sensitivity analysis the plan asks for around the known SARTAJ mislabeling.
    """
    mask = pd.Series(True, index=manifest.index)
    if split is not None:
        mask &= manifest["split"] == split
    if representatives_only:
        mask &= manifest["is_representative"]
    if drop_mixed_label:
        mask &= ~manifest["is_mixed_label"]
    return np.flatnonzero(mask.to_numpy())


class ManifestDataset(Dataset):
    """Indexes into the cached uint8 array via manifest row positions."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        images: np.ndarray,
        indices: np.ndarray,
        transform=None,
    ) -> None:
        if len(manifest) != len(images):
            raise ValueError(
                f"manifest has {len(manifest)} rows but cache has {len(images)} images; "
                "the cache was built from a different manifest -- rebuild it."
            )
        self.manifest = manifest
        self.images = images
        self.indices = np.asarray(indices)
        self.transform = transform
        self.labels = (
            manifest["label"].map(LABEL_TO_INDEX).to_numpy()[self.indices].astype(np.int64)
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        row = self.indices[i]
        image = torch.from_numpy(np.array(self.images[row])).unsqueeze(0)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[i])

    def class_counts(self) -> dict[str, int]:
        counts = np.bincount(self.labels, minlength=len(CLASSES))
        return {CLASSES[i]: int(n) for i, n in enumerate(counts)}
