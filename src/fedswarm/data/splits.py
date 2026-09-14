"""Phase 1.3 — canonical train/val/test splits and the manifest.

Splitting happens at the **pseudo-patient** level (the connected components found in Phase
1.2), never at the image level, so cross-split leakage is zero by construction rather than
by hope. Within that constraint the split is stratified by class.

The manifest decouples *which files* from *how they are loaded*: partitioning, caching,
augmentation and model code all read it and never re-derive the split.

Run: python -m fedswarm.data.splits
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from fedswarm.data.download import find_split_parent, resolve_root

PSEUDO_PATIENTS_CSV = Path("data/processed/pseudo_patients.csv")
MANIFEST_CSV = Path("data/processed/manifest.csv")
STATS_JSON = Path("data/processed/split_stats.json")

# 10 folds of ~10% each, allocated 7 / 1 / 2.
N_FOLDS = 10
VAL_FOLDS = (7,)
TEST_FOLDS = (8, 9)


def choose_representatives(df: pd.DataFrame, split_parent: Path) -> pd.Series:
    """One representative image per pseudo-patient: the highest-resolution member, with
    the path as a deterministic tie-break.

    Why collapse at all: the `notumor` class is 67.9% redundant in this dataset variant, so
    keeping every copy would silently reweight training toward a handful of distinct
    patients and make test metrics a measure of those few patients repeated.
    """
    pixels = []
    for rel_path in df["path"]:
        with Image.open(split_parent / rel_path) as img:
            pixels.append(img.size[0] * img.size[1])
    work = df.assign(_pixels=pixels)

    # Sort so the preferred representative is first within each group.
    work = work.sort_values(["pseudo_patient_id", "_pixels", "path"], ascending=[True, False, True])
    winners = work.groupby("pseudo_patient_id", sort=False).head(1).index
    return df.index.isin(winners)


def component_labels(df: pd.DataFrame) -> pd.DataFrame:
    """One row per pseudo-patient, carrying the label used for stratification.

    A component spanning several labels (25 of them here) gets its majority label for
    stratification purposes only; individual images keep their own labels. Such components
    are flagged so a sensitivity analysis can exclude them later.
    """
    rows = []
    for component, group in df.groupby("pseudo_patient_id"):
        counts = Counter(group["label"])
        majority, _ = counts.most_common(1)[0]
        rows.append(
            {
                "pseudo_patient_id": component,
                "component_label": majority,
                "is_mixed_label": len(counts) > 1,
                "n_images": len(group),
            }
        )
    return pd.DataFrame(rows)


def assign_splits(components: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Stratified, group-respecting 70/10/20 assignment.

    StratifiedGroupKFold guarantees each group lands wholly in one fold while keeping class
    proportions near-constant across folds; folds are then allocated to train/val/test.
    Each component is its own group here, which means the group constraint is trivially
    satisfied and the value comes from the stratification.
    """
    y = components["component_label"].to_numpy()
    groups = components["pseudo_patient_id"].to_numpy()
    x = np.zeros((len(components), 1))

    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_of = np.empty(len(components), dtype=int)
    for fold_index, (_, holdout) in enumerate(splitter.split(x, y, groups)):
        fold_of[holdout] = fold_index

    split_name = np.where(
        np.isin(fold_of, TEST_FOLDS), "test", np.where(np.isin(fold_of, VAL_FOLDS), "val", "train")
    )
    return components.assign(fold=fold_of, split=split_name)


def build_manifest(
    pseudo_patients_csv: Path = PSEUDO_PATIENTS_CSV,
    root: Path | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(pseudo_patients_csv)
    split_parent = find_split_parent(resolve_root(root))

    df["is_representative"] = choose_representatives(df, split_parent)

    components = assign_splits(component_labels(df), seed=seed)
    df = df.merge(
        components[["pseudo_patient_id", "split", "is_mixed_label", "component_label"]],
        on="pseudo_patient_id",
        how="left",
    )

    df = df.rename(columns={"split_x": "original_split", "split_y": "split"})
    if "original_split" not in df.columns:  # merge did not collide
        df = df.rename(columns={"split": "split"})

    # `source_component` is in the plan's manifest schema but is not recoverable from this
    # dataset -- see docs/OPEN_QUESTIONS.md. Column kept so the schema stays stable.
    df["source_component"] = "unknown"

    manifest = df[
        [
            "path",
            "label",
            "pseudo_patient_id",
            "split",
            "phash",
            "sha256",
            "source_component",
            "original_split",
            "is_representative",
            "is_mixed_label",
        ]
    ].sort_values(["split", "label", "path"])

    stats = summarize(manifest, seed)
    return manifest, stats


def summarize(manifest: pd.DataFrame, seed: int) -> dict:
    representative = manifest[manifest["is_representative"]]

    global_dist = representative["label"].value_counts(normalize=True).to_dict()
    per_split = {}
    max_deviation = 0.0
    for split, group in representative.groupby("split"):
        dist = group["label"].value_counts(normalize=True).to_dict()
        deviations = {c: abs(dist.get(c, 0.0) - global_dist[c]) * 100 for c in global_dist}
        max_deviation = max(max_deviation, max(deviations.values()))
        per_split[split] = {
            "n_pseudo_patients": int(len(group)),
            "n_images_all": int((manifest["split"] == split).sum()),
            "class_counts": group["label"].value_counts().to_dict(),
            "class_fractions": {k: round(v, 4) for k, v in dist.items()},
            "max_deviation_pp": round(max(deviations.values()), 3),
        }

    # Leakage must be zero by construction: verify rather than assert it in a comment.
    per_component_splits = manifest.groupby("pseudo_patient_id")["split"].nunique()
    n_leaked = int((per_component_splits > 1).sum())

    return {
        "seed": seed,
        "n_images_total": int(len(manifest)),
        "n_pseudo_patients": int(manifest["pseudo_patient_id"].nunique()),
        "n_mixed_label_components": int(
            manifest[manifest["is_mixed_label"]]["pseudo_patient_id"].nunique()
        ),
        "global_class_fractions": {k: round(v, 4) for k, v in global_dist.items()},
        "per_split": per_split,
        "max_class_deviation_pp": round(max_deviation, 3),
        "n_components_spanning_splits": n_leaked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pseudo-patients", default=str(PSEUDO_PATIENTS_CSV))
    parser.add_argument("--out", default=str(MANIFEST_CSV))
    parser.add_argument("--stats-out", default=str(STATS_JSON))
    args = parser.parse_args()

    manifest, stats = build_manifest(Path(args.pseudo_patients), args.root, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    Path(args.stats_out).write_text(json.dumps(stats, indent=2))

    print(f"images {stats['n_images_total']}, pseudo-patients {stats['n_pseudo_patients']}")
    print(f"mixed-label components: {stats['n_mixed_label_components']}")
    print()
    print(f"{'split':<8}{'units':>8}{'images':>9}   class fractions")
    for split in ("train", "val", "test"):
        s = stats["per_split"][split]
        fractions = "  ".join(f"{k}={v:.3f}" for k, v in sorted(s["class_fractions"].items()))
        print(f"{split:<8}{s['n_pseudo_patients']:>8}{s['n_images_all']:>9}   {fractions}")
    print()
    print(f"max class deviation from global: {stats['max_class_deviation_pp']:.2f} pp (limit 2.0)")
    print(f"components spanning >1 split:    {stats['n_components_spanning_splits']} (must be 0)")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
