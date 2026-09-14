"""Phase 1.3 tests — split integrity.

Two layers: synthetic tests of the splitting logic that run anywhere (CI included), and
tests against the real committed manifest that skip when the dataset is not present.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fedswarm.data.splits import (
    MANIFEST_CSV,
    assign_splits,
    component_labels,
    summarize,
)


def _synthetic_components(n_per_class: int = 200, seed: int = 0) -> pd.DataFrame:
    rows = []
    component = 0
    for label in ("glioma", "meningioma", "notumor", "pituitary"):
        for _ in range(n_per_class):
            rows.append(
                {
                    "pseudo_patient_id": component,
                    "component_label": label,
                    "is_mixed_label": False,
                    "n_images": 1,
                }
            )
            component += 1
    return pd.DataFrame(rows)


def test_component_labels_uses_majority_and_flags_mixed() -> None:
    df = pd.DataFrame(
        {
            "pseudo_patient_id": [0, 0, 0, 1, 1],
            "label": ["glioma", "glioma", "meningioma", "notumor", "notumor"],
            "path": ["a", "b", "c", "d", "e"],
        }
    )
    out = component_labels(df).set_index("pseudo_patient_id")

    assert out.loc[0, "component_label"] == "glioma"
    assert bool(out.loc[0, "is_mixed_label"]) is True
    assert out.loc[1, "component_label"] == "notumor"
    assert bool(out.loc[1, "is_mixed_label"]) is False


def test_assign_splits_covers_every_component_exactly_once() -> None:
    components = _synthetic_components()
    out = assign_splits(components, seed=0)

    assert len(out) == len(components)
    assert set(out["split"]) == {"train", "val", "test"}
    assert out["pseudo_patient_id"].nunique() == len(components)


def test_assign_splits_hits_target_proportions() -> None:
    components = _synthetic_components()
    out = assign_splits(components, seed=0)
    fractions = out["split"].value_counts(normalize=True)

    assert fractions["train"] == pytest.approx(0.70, abs=0.02)
    assert fractions["val"] == pytest.approx(0.10, abs=0.02)
    assert fractions["test"] == pytest.approx(0.20, abs=0.02)


def test_assign_splits_is_deterministic_under_a_fixed_seed() -> None:
    components = _synthetic_components()
    first = assign_splits(components, seed=7)["split"].tolist()
    second = assign_splits(components, seed=7)["split"].tolist()
    third = assign_splits(components, seed=8)["split"].tolist()

    assert first == second
    assert first != third


def test_assign_splits_is_stratified() -> None:
    components = _synthetic_components()
    out = assign_splits(components, seed=0)

    global_fractions = out["component_label"].value_counts(normalize=True)
    for _, group in out.groupby("split"):
        split_fractions = group["component_label"].value_counts(normalize=True)
        for label, expected in global_fractions.items():
            assert abs(split_fractions.get(label, 0.0) - expected) < 0.02


# --- tests against the real manifest ------------------------------------------------


def _load_manifest() -> pd.DataFrame:
    if not Path(MANIFEST_CSV).exists():
        pytest.skip(f"{MANIFEST_CSV} not built; run `python -m fedswarm.data.splits`")
    return pd.read_csv(MANIFEST_CSV)


def test_no_cross_split_leakage() -> None:
    """The acceptance criterion: pseudo-patient IDs must not intersect across splits."""
    manifest = _load_manifest()

    ids = {
        split: set(group["pseudo_patient_id"])
        for split, group in manifest.groupby("split")
    }
    splits = sorted(ids)
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            overlap = ids[a] & ids[b]
            assert not overlap, f"{len(overlap)} pseudo-patients shared between {a} and {b}"


def test_manifest_stratification_within_two_points() -> None:
    manifest = _load_manifest()
    stats = summarize(manifest, seed=0)

    assert stats["max_class_deviation_pp"] < 2.0
    assert stats["n_components_spanning_splits"] == 0


def test_manifest_has_one_representative_per_pseudo_patient() -> None:
    manifest = _load_manifest()
    per_component = manifest.groupby("pseudo_patient_id")["is_representative"].sum()

    assert (per_component == 1).all(), "every pseudo-patient needs exactly one representative"


def test_manifest_covers_every_image_once() -> None:
    manifest = _load_manifest()

    assert manifest["path"].is_unique
    assert manifest["split"].isin({"train", "val", "test"}).all()
