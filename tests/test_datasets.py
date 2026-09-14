"""Phase 1.3 tests — cache, transforms, and dataset layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from fedswarm.data.cache import letterbox
from fedswarm.data.datasets import ManifestDataset, select
from fedswarm.data.transforms import ToFloatAndReplicate


def _tiny_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "path": [f"img_{i}.jpg" for i in range(6)],
            "label": ["glioma", "glioma", "meningioma", "notumor", "pituitary", "pituitary"],
            "pseudo_patient_id": [0, 0, 1, 2, 3, 3],
            "split": ["train", "train", "train", "val", "test", "test"],
            "is_representative": [True, False, True, True, True, False],
            "is_mixed_label": [False, False, True, False, False, False],
        }
    )


def test_letterbox_preserves_aspect_ratio_and_pads_to_square() -> None:
    wide = Image.new("L", (200, 50), color=255)
    out = letterbox(wide, 100)

    assert out.size == (100, 100)
    array = np.asarray(out)
    # Content occupies a 100x25 band; the rest is zero padding.
    assert array.max() == 255
    assert array[0, :].sum() == 0, "top rows should be padding"
    assert array[50, :].sum() > 0, "the middle band should carry the image"


def test_letterbox_does_not_distort_a_square() -> None:
    square = Image.new("L", (64, 64), color=128)
    out = letterbox(square, 32)

    assert out.size == (32, 32)
    assert np.asarray(out).min() > 0, "a square image needs no padding"


def test_to_float_and_replicate_makes_three_channels_in_unit_range() -> None:
    image = torch.randint(0, 256, (1, 8, 8), dtype=torch.uint8)
    out = ToFloatAndReplicate()(image)

    assert out.shape == (3, 8, 8)
    assert out.dtype == torch.float32
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert torch.equal(out[0], out[1]) and torch.equal(out[1], out[2])


def test_select_defaults_to_one_row_per_pseudo_patient() -> None:
    manifest = _tiny_manifest()
    indices = select(manifest, representatives_only=True)

    assert len(indices) == manifest["pseudo_patient_id"].nunique()


def test_select_filters_by_split_and_mixed_label() -> None:
    manifest = _tiny_manifest()

    train = select(manifest, split="train")
    assert set(manifest.loc[train, "split"]) == {"train"}

    clean = select(manifest, split="train", drop_mixed_label=True)
    assert not manifest.loc[clean, "is_mixed_label"].any()
    assert len(clean) < len(train)


def test_dataset_rejects_a_cache_built_from_a_different_manifest() -> None:
    manifest = _tiny_manifest()
    wrong_size_cache = np.zeros((3, 8, 8), dtype=np.uint8)

    with pytest.raises(ValueError, match="rebuild it"):
        ManifestDataset(manifest, wrong_size_cache, np.array([0]))


def test_dataset_returns_correct_label_for_each_index() -> None:
    manifest = _tiny_manifest()
    images = np.zeros((6, 8, 8), dtype=np.uint8)
    indices = select(manifest, representatives_only=True)
    dataset = ManifestDataset(manifest, images, indices)

    assert len(dataset) == 4
    assert dataset.class_counts() == {
        "glioma": 1,
        "meningioma": 1,
        "notumor": 1,
        "pituitary": 1,
    }
    _, label = dataset[0]
    assert label == 0  # glioma
