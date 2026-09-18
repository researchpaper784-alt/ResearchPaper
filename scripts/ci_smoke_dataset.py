"""Phase 10 -- builds a tiny, fully synthetic dataset + manifest + image cache so CI
can run a real `flwr run .` smoke test without the real MRI dataset (no Kaggle
credentials available to a CI runner). Mirrors the exact fixture pattern
`tests/test_fl_app.py::task_manifest_and_cache` already uses and relies on being
correct (same `_tiny_dataset_rows`-style rows, same `build_and_save_cache` call) --
this script just writes it to fixed, predictable paths instead of `tmp_path`, so a
CI workflow step can point `flwr run . --run-config "..."` at them.

Run:
  .venv/bin/python scripts/ci_smoke_dataset.py --out-dir /tmp/fedswarm_ci_smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedswarm.data.cache import build_and_save_cache  # noqa: E402


def _tiny_dataset_rows(root: Path, n: int = 40) -> list[dict]:
    """A three-way split, matching the real Phase 1.3 manifest.

    The `val` rows are the point. This generator originally emitted train/test only,
    while `fl/app.py` builds its server-held `global_val_loader` from `split == "val"`
    unconditionally -- so the loader came back empty and the run died mid-simulation in
    `eval/evaluator.py` with `need at least one array to concatenate`, reported by
    `flwr run` as a bare "Exit Code: 700". That is the second of the two bugs that kept
    this smoke job red; the first was `flower-superlink` not being on PATH, which
    stopped the job before it ever got far enough to hit this one.

    The same two-way/three-way mismatch already bit the unit-test fixtures once
    (docs/PROGRESS_REVIEW.md, Phase 3 fix 3). Keeping CI's manifest three-way is what
    stops it recurring a third time.
    """
    (root / "Training" / "glioma").mkdir(parents=True)
    (root / "Testing" / "glioma").mkdir(parents=True)
    rows = []
    for i in range(n):
        if i < n - 16:
            split_dir, split = "Training", "train"
        elif i < n - 8:
            # Val lives under Training/ on disk: the real pipeline carves it out of the
            # training images rather than from the held-out test set.
            split_dir, split = "Training", "val"
        else:
            split_dir, split = "Testing", "test"
        rel = f"{split_dir}/glioma/img_{i}.jpg"
        Image.new("L", (32, 32), color=(i * 7) % 256).save(root / rel)
        rows.append(
            {
                "path": rel,
                "label": "glioma" if i % 2 == 0 else "meningioma",
                "pseudo_patient_id": i,
                "split": split,
                "is_representative": True,
                "is_mixed_label": False,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--image-size", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_root = out_dir / "raw"

    manifest = pd.DataFrame(_tiny_dataset_rows(raw_root))
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    cache_dir = out_dir / "cache"
    build_and_save_cache(manifest, raw_root, size=args.image_size, out_dir=cache_dir)

    print(f"Wrote synthetic manifest to {manifest_path}")
    print(f"Wrote synthetic image cache to {cache_dir}")


if __name__ == "__main__":
    main()
