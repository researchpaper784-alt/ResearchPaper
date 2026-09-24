"""A synthetic dataset heterogeneous enough to reproduce the project's fitness problem locally.

`ci_smoke_dataset.py` exists to make CI fast, and it succeeds by being degenerate: 40 images,
labels alternating by parity, and pixel content that has **nothing to do with the label**
(`color=(i*7)%256`). Nothing there can produce client heterogeneity, because every client is
learning the same unlearnable mapping -- so a FedACO run on it has `fedavg_fitness` around
+0.86, `corner_margin` already negative, and tells you nothing about the regime the paper runs
in. This generator is the opposite trade: slower, and structured so label skew produces real
disagreement between clients.

**Why that matters.** The question blocking this project is whether the data-free fitness has a
usable optimum on heterogeneous data. Its signature on real MRI, measured over three Kaggle
sessions:

    F at the FedAvg point   mean -0.0005 to -0.0212, negative in 7 of 15 rounds
    corner_margin           +0.31 to +0.62, positive in 15 of 15 rounds
    FedACO vs FedAvg        0.20 macro-F1 BEHIND

Each of those cost a GPU session to obtain. If a local dataset reproduces the signature, the
same questions get answered in minutes -- and every fitness change can be tested before anyone
spends compute on it.

**How the heterogeneity is built.** Each class gets a distinct spatial pattern (a
low-frequency sinusoid at its own orientation and phase) plus per-image noise. A client holding
only classes {0, 1} therefore has a genuinely different local optimum from one holding {2, 3},
and its update points somewhere else -- which is what makes the Gram matrix's off-diagonals
small and the dispersion term large. Label-parity images cannot do that at any size.

⚠️ **This is not the real dataset and a result here is not a paper result.** It reproduces a
*failure signature*, which is a much weaker claim than reproducing the data. Anything measured
here has to be confirmed on the real gate before it goes in the paper -- the log-spaced-level
episode is what happens when synthetic evidence is trusted to transfer.

Run:
  python scripts/synthetic_heterogeneous_dataset.py --out-dir /tmp/het --num-images 600 --image-size 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedswarm.data.cache import build_and_save_cache  # noqa: E402
from fedswarm.data.download import CLASSES  # noqa: E402


def _class_pattern(class_index: int, size: int, amplitude: float) -> np.ndarray:
    """A distinct low-frequency pattern per class.

    Low frequency on purpose: a small CNN can fit it in a handful of rounds, so a 15-round run
    reaches the regime where clients have actually diverged rather than all still being near
    the initialization. Orientation and phase both vary with the class, so no two classes are
    related by a simple intensity shift -- which is what `ci_smoke_dataset`'s per-image colour
    amounts to, and why it produces no heterogeneity.
    """
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / max(size - 1, 1)
    angle = class_index * np.pi / len(CLASSES)
    projected = x * np.cos(angle) + y * np.sin(angle)
    wave = np.sin(2.0 * np.pi * (1.5 + class_index) * projected + class_index)
    return 0.5 + amplitude * wave


def _rows(
    root: Path, num_images: int, size: int, seed: int, amplitude: float, noise: float
) -> list[dict]:
    """A three-way split, matching the real Phase 1.3 manifest.

    The `val` rows matter for the same reason they do in `ci_smoke_dataset.py`: `fl/app.py`
    builds its server-held `global_val_loader` from `split == "val"` unconditionally, and a
    two-way manifest leaves that loader empty, which kills the run mid-simulation with
    `need at least one array to concatenate` reported as a bare "Exit Code: 700".
    """
    rng = np.random.default_rng(seed)
    n_test = max(len(CLASSES) * 4, num_images // 5)
    n_val = max(len(CLASSES) * 4, num_images // 5)
    rows: list[dict] = []
    for index in range(num_images):
        class_index = index % len(CLASSES)
        label = CLASSES[class_index]
        if index < num_images - n_test - n_val:
            split_dir, split = "Training", "train"
        elif index < num_images - n_test:
            # Val is carved out of the training images, as the real pipeline does -- not taken
            # from the held-out test set.
            split_dir, split = "Training", "val"
        else:
            split_dir, split = "Testing", "test"

        pattern = _class_pattern(class_index, size, amplitude)
        noisy = np.clip(pattern + rng.normal(0.0, noise, pattern.shape), 0.0, 1.0)
        directory = root / split_dir / label
        directory.mkdir(parents=True, exist_ok=True)
        relative = f"{split_dir}/{label}/img_{index:05d}.jpg"
        Image.fromarray((noisy * 255).astype(np.uint8), mode="L").save(root / relative)

        rows.append(
            {
                "path": relative,
                "label": label,
                # One image per pseudo-patient: the real manifest groups by patient to stop
                # leakage across splits, and giving every image its own id keeps the
                # partitioner's grouping logic exercised without inventing a patient structure
                # this generator has no basis for.
                "pseudo_patient_id": index,
                "split": split,
                "is_representative": True,
                "is_mixed_label": False,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-images", type=int, default=600)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    # Difficulty, and it has to be tunable rather than fixed. At amplitude 0.35 / noise 0.18
    # the classes are trivially separable: a local run reached final macro-F1 = 1.0000, the
    # model saturated, every client's delta pointed the same way, and F at the FedAvg point
    # stayed comfortably positive (+0.17) -- so the run reproduced the corner problem but not
    # the zero-centred-fitness problem. The real runs sit at 0.13-0.33 macro-F1. A fixture that
    # is easier than the task it stands in for reproduces only the failures that do not depend
    # on difficulty, and there is no way to know which those are in advance.
    parser.add_argument("--amplitude", type=float, default=0.15, help="class signal strength")
    parser.add_argument("--noise", type=float, default=0.45, help="per-image pixel noise")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_root = out_dir / "raw"

    manifest = pd.DataFrame(
        _rows(raw_root, args.num_images, args.image_size, args.seed, args.amplitude, args.noise)
    )
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    cache_dir = out_dir / "cache"
    build_and_save_cache(manifest, raw_root, size=args.image_size, out_dir=cache_dir)

    counts = manifest.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(f"\nWrote {len(manifest)} images to {manifest_path}")
    print(f"Cache: {cache_dir}\n")
    print(counts.to_string())
    print(
        "\nClass-correlated patterns, so dirichlet label skew produces genuinely different "
        "local optima per client -- which is what `ci_smoke_dataset.py` cannot do."
    )
    print(
        f"signal/noise = {args.amplitude}/{args.noise}. Check the resulting run's final "
        "macro-F1: near 1.0 means the task saturated and the fixture is easier than the one "
        "it stands in for."
    )


if __name__ == "__main__":
    main()
