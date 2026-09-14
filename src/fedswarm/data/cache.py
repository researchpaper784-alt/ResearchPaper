"""Phase 1.3 — decoded-image cache.

Federated simulation re-reads every client's images every round. Decoding ~11,400 JPEGs
per round through PIL is an estimated ~45 s/round single-threaded and would dominate the
round time (see docs/EXPERIMENT_LOG.md). Decoding once into a uint8 array removes that
cost entirely: at 112 the whole dataset is ~90 MB and stays resident in RAM.

Images are stored single-channel; the 3-channel replication ImageNet backbones expect
happens at load time, so the cache is 3x smaller.

Run: python -m fedswarm.data.cache --size 112
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from fedswarm.data.download import find_split_parent, resolve_root
from fedswarm.data.splits import MANIFEST_CSV

CACHE_DIR = Path("data/processed/cache")


def letterbox(img: Image.Image, size: int) -> Image.Image:
    """Resize preserving aspect ratio, then zero-pad to a square.

    Zero-padding is the right fill here because MRI background is already black; stretching
    instead would distort anatomy, and anatomical proportions carry diagnostic signal.
    """
    width, height = img.size
    scale = size / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = img.resize(new_size, Image.BICUBIC)

    canvas = Image.new("L", (size, size), color=0)
    canvas.paste(resized, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    return canvas


def build_cache(manifest: pd.DataFrame, split_parent: Path, size: int) -> np.ndarray:
    array = np.zeros((len(manifest), size, size), dtype=np.uint8)
    for i, rel_path in enumerate(tqdm(manifest["path"], desc=f"caching@{size}")):
        with Image.open(split_parent / rel_path) as img:
            array[i] = np.asarray(letterbox(img.convert("L"), size), dtype=np.uint8)
    return array


def dataset_statistics(array: np.ndarray, manifest: pd.DataFrame) -> dict:
    """Mean/std computed on the **training** representatives only.

    Using val or test images here would leak distributional information about the held-out
    data into preprocessing -- a small leak, but free to avoid.
    """
    mask = (manifest["split"] == "train") & manifest["is_representative"]
    train_pixels = array[mask.to_numpy()].astype(np.float64) / 255.0

    return {
        "n_images_used": int(mask.sum()),
        "mean": float(train_pixels.mean()),
        "std": float(train_pixels.std()),
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
        "note": (
            "`mean`/`std` are single-channel, computed over training representatives, for "
            "from-scratch models. ImageNet statistics are recorded for pretrained "
            "backbones. Both are reported so the paper can state which was used where."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--size", type=int, default=112)
    parser.add_argument("--manifest", default=str(MANIFEST_CSV))
    parser.add_argument("--out-dir", default=str(CACHE_DIR))
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    split_parent = find_split_parent(resolve_root(args.root))

    array = build_cache(manifest, split_parent, args.size)
    stats = dataset_statistics(array, manifest)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    array_path = out_dir / f"images_{args.size}.npy"
    np.save(array_path, array)

    meta = {
        "size": args.size,
        "n_images": int(len(manifest)),
        "dtype": "uint8",
        "layout": "single-channel (N, size, size); replicate to 3 channels at load time",
        "manifest_order": "row i of the array corresponds to row i of the manifest",
        "manifest_sha_of_paths": pd.util.hash_pandas_object(manifest["path"]).sum().item(),
        "statistics": stats,
        "bytes": int(array.nbytes),
    }
    (out_dir / f"images_{args.size}_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\ncached {len(manifest)} images at {args.size}x{args.size} "
          f"({array.nbytes / 1e6:.1f} MB)")
    print(f"train-set mean {stats['mean']:.4f}  std {stats['std']:.4f} "
          f"(over {stats['n_images_used']} training representatives)")
    print(f"Wrote {array_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
