"""Recovers `source_component` (figshare / sartaj / br35h) per manifest image.

Why this exists: the plan's `source_shift` partitioning regime (§1.4) needs to know
which of the three datasets merged into the Kaggle "Brain Tumor MRI Dataset" each image
came from, but the merge itself discards that label. This module downloads/reads each
upstream source, hashes it the same way Phase 1.2 hashed the merged dataset, and matches
by nearest perceptual-hash distance.

Figshare (Cheng et al.) ships as MATLAB v7.3 (.mat/HDF5) files, one per image, each a
`cjdata` struct with `image` (raw int16 512x512), `label` (1/2/3 = meningioma/glioma/
pituitary), and `PID` (a real patient ID string) -- unlike our merged JPEGs, this source
carries genuine ground-truth patient identity, which is a free cross-check on our own
phash-based pseudo-patient clustering from Phase 1.2.

Run: python -m fedswarm.data.source_provenance --figshare data/raw/figshare/extracted
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

FIGSHARE_LABEL_TO_CLASS = {1: "meningioma", 2: "glioma", 3: "pituitary"}
HASH_SIZE = 8  # matches Phase 1.2's dedup.py, so distances are directly comparable


def _phash_bitstring(image: Image.Image) -> str:
    """64-char '0'/'1' string, matching the exact format dedup.py persists into
    manifest.csv (`"".join(str(int(b)) for b in phash.hash.flatten())`) -- NOT
    imagehash's own str(), which is a 16-char hex encoding of the same bits and is not
    directly comparable to what's stored in the manifest without conversion."""
    bits = imagehash.phash(image, hash_size=HASH_SIZE).hash.flatten()
    return "".join(str(int(b)) for b in bits)


def _decode_pid(raw: np.ndarray) -> str:
    """cjdata.PID is stored as an (N, 1) array of uint16 char codes."""
    return "".join(chr(int(c[0])) for c in raw)


def _to_uint8_image(raw: np.ndarray) -> Image.Image:
    """Min-max normalize the raw int16 MRI slice to an 8-bit grayscale image, the same
    representation Phase 1.2 hashed the merged JPEGs in."""
    arr = raw.astype(np.float64)
    lo, hi = arr.min(), arr.max()
    scaled = np.zeros_like(arr, dtype=np.uint8) if hi <= lo else ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="L")


def load_figshare(directory: Path, show_progress: bool = True) -> list[dict[str, Any]]:
    """One record per .mat file: source, patient_id, label, phash (hex string)."""
    paths = sorted(Path(directory).glob("*.mat"))
    if not paths:
        raise FileNotFoundError(f"No .mat files found under {directory}")

    records = []
    for path in tqdm(paths, desc="figshare", disable=not show_progress):
        with h5py.File(path, "r") as f:
            g = f["cjdata"]
            patient_id = _decode_pid(g["PID"][()])
            label_code = int(g["label"][()][0, 0])
            # h5py reads MATLAB's column-major array storage in row-major order, so the
            # raw array is transposed relative to the displayed image -- confirmed
            # empirically: without .T essentially nothing matches the Kaggle JPEGs
            # (a handful of coincidental hash collisions out of ~22M comparisons); with
            # .T, hundreds of exact (distance-0) matches appear immediately.
            image = _to_uint8_image(g["image"][()].T)
            phash = _phash_bitstring(image)

        records.append(
            {
                "source": "figshare",
                "source_file": path.name,
                "patient_id": patient_id,
                "label": FIGSHARE_LABEL_TO_CLASS.get(label_code, f"unknown_{label_code}"),
                "phash": phash,
            }
        )
    return records


def load_flat_image_dir(
    directory: Path, source_name: str, show_progress: bool = True
) -> list[dict[str, Any]]:
    """One record per image under `directory` (any depth), for sources that ship as plain
    JPEG/PNG folders (SARTAJ, Br35H) rather than Figshare's per-file .mat structs."""
    suffixes = {".jpg", ".jpeg", ".png"}
    paths = sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in suffixes)
    if not paths:
        raise FileNotFoundError(f"No images found under {directory}")

    records = []
    for path in tqdm(paths, desc=source_name, disable=not show_progress):
        with Image.open(path) as img:
            phash = _phash_bitstring(img.convert("L"))
        records.append(
            {
                "source": source_name,
                "source_file": str(path.relative_to(directory)),
                "patient_id": None,
                "label": path.parent.name.lower(),
                "phash": phash,
            }
        )
    return records


def _phash_to_bits(bit_strings: list[str]) -> np.ndarray:
    """'0'/'1' bit strings (the manifest's persisted phash format) -> (N, hash_size**2)
    float32 bit matrix, for the same matmul-based Hamming-distance trick used in Phase
    1.2's dedup.py."""
    n_bits = HASH_SIZE * HASH_SIZE
    bits = np.zeros((len(bit_strings), n_bits), dtype=np.float32)
    for i, bit_str in enumerate(bit_strings):
        bits[i] = [int(b) for b in bit_str]
    return bits


def match_manifest_to_sources(
    manifest: pd.DataFrame,
    upstream_records: list[dict[str, Any]],
    max_distance: int = 5,
    chunk_size: int = 512,
) -> pd.DataFrame:
    """For each manifest row, find the nearest upstream record by phash Hamming distance.

    Returns columns: source_component, source_match_distance, source_file, matched (bool).
    A manifest row with no upstream match within `max_distance` gets source_component
    'unmatched' -- expected for near-duplicate slices that only barely survived Phase
    1.2's own threshold-5 dedup against the *merged* set, and worth reporting honestly
    rather than forcing a low-confidence assignment.
    """
    if "phash" not in manifest.columns:
        raise ValueError(
            "manifest has no 'phash' column -- rebuild it with the current "
            "fedswarm.data.splits (which now carries phash through from dedup.py)"
        )

    manifest_bits = _phash_to_bits(manifest["phash"].tolist())
    upstream_bits = _phash_to_bits([r["phash"] for r in upstream_records])
    upstream_complement = 1.0 - upstream_bits

    best_distance = np.full(len(manifest), 999, dtype=np.int32)
    best_index = np.full(len(manifest), -1, dtype=np.int64)

    for start in range(0, len(manifest), chunk_size):
        stop = min(start + chunk_size, len(manifest))
        block = manifest_bits[start:stop]
        block_complement = 1.0 - block
        distances = block @ upstream_complement.T + block_complement @ upstream_bits.T

        nearest = distances.argmin(axis=1)
        nearest_distance = distances[np.arange(len(nearest)), nearest]
        best_distance[start:stop] = nearest_distance.astype(np.int32)
        best_index[start:stop] = nearest

    matched = best_distance <= max_distance
    source_component = np.where(
        matched,
        [upstream_records[i]["source"] if i >= 0 else "unmatched" for i in best_index],
        "unmatched",
    )
    source_file = np.where(
        matched,
        [upstream_records[i]["source_file"] if i >= 0 else "" for i in best_index],
        "",
    )

    return pd.DataFrame(
        {
            "source_component": source_component,
            "source_match_distance": best_distance,
            "source_file": source_file,
            "matched": matched,
        },
        index=manifest.index,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figshare", default="data/raw/figshare/extracted")
    parser.add_argument("--sartaj", default=None, help="Path to extracted SARTAJ dataset")
    parser.add_argument("--br35h", default=None, help="Path to extracted Br35H dataset")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--max-distance", type=int, default=5)
    parser.add_argument("--out", default="data/processed/source_provenance.csv")
    parser.add_argument("--report", default="data/processed/source_provenance_report.json")
    args = parser.parse_args()

    upstream: list[dict[str, Any]] = []
    upstream += load_figshare(Path(args.figshare))
    if args.sartaj:
        upstream += load_flat_image_dir(Path(args.sartaj), "sartaj")
    if args.br35h:
        upstream += load_flat_image_dir(Path(args.br35h), "br35h")

    print(f"\nLoaded {len(upstream)} upstream records: "
          f"{ {s: sum(1 for r in upstream if r['source'] == s) for s in set(r['source'] for r in upstream)} }")

    manifest = pd.read_csv(args.manifest)
    matches = match_manifest_to_sources(manifest, upstream, max_distance=args.max_distance)

    # manifest.csv carries a placeholder source_component=="unknown" column (Phase 1.3,
    # for schema stability before this module existed) -- replace it with the real one.
    out = manifest.drop(columns=["source_component"], errors="ignore").join(matches)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    counts = matches["source_component"].value_counts().to_dict()
    report = {
        "n_manifest_images": int(len(manifest)),
        "n_upstream_records": len(upstream),
        "upstream_sources": sorted(set(r["source"] for r in upstream)),
        "max_distance": args.max_distance,
        "source_component_counts": counts,
        "match_rate": float(matches["matched"].mean()),
        "distance_distribution": {
            str(k): int(v) for k, v in matches["source_match_distance"].value_counts().sort_index().items()
        },
    }
    Path(args.report).write_text(json.dumps(report, indent=2))

    print(f"\nsource_component counts: {counts}")
    print(f"match rate (distance <= {args.max_distance}): {report['match_rate']:.1%}")
    print(f"\nWrote {args.out}")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
