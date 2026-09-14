"""Calibrate the near-duplicate threshold (Phase 1.2).

The plan requires the phash threshold to be chosen by inspection rather than assumed, and
requires its false-positive behaviour to be recorded. This script hashes the dataset once,
then sweeps the threshold, and writes contact sheets of sampled flagged pairs bucketed by
Hamming distance so they can actually be looked at.

Run: .venv/bin/python scripts/audit_threshold.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from fedswarm.data.dedup import (
    compute_hashes,
    components_from_edges,
    exact_duplicate_edges,
    list_images,
    near_duplicate_edges,
)
from fedswarm.data.download import find_split_parent, resolve_root

OUT_DIR = Path("data/processed/audit")
THUMB = 200


def summarize(records, sha_list, bits, threshold) -> dict:
    edges = near_duplicate_edges(bits, threshold)
    all_edges = list(set(exact_duplicate_edges(sha_list)) | set(edges))
    labels = components_from_edges(len(records), all_edges)

    splits_per_component = defaultdict(set)
    labels_per_component = defaultdict(set)
    size_per_component: dict[int, int] = defaultdict(int)
    for rec, label in zip(records, labels):
        key = int(label)
        splits_per_component[key].add(rec["split"])
        labels_per_component[key].add(rec["label"])
        size_per_component[key] += 1

    leaked = [c for c, s in splits_per_component.items() if len(s) > 1]
    mixed = [c for c, ls in labels_per_component.items() if len(ls) > 1]

    return {
        "threshold": threshold,
        "n_phash_edges": len(edges),
        "n_components": int(len(set(labels.tolist()))),
        "n_leaked_components": len(leaked),
        "n_leaked_images": sum(size_per_component[c] for c in leaked),
        "n_mixed_label_components": len(mixed),
        "n_mixed_label_images": sum(size_per_component[c] for c in mixed),
    }


def pair_distances(bits: np.ndarray, pairs: list[tuple[int, int]]) -> list[int]:
    return [int(np.sum(bits[i] != bits[j])) for i, j in pairs]


def contact_sheet(records, pairs, distances, split_parent: Path, out: Path, title: str) -> None:
    """Side-by-side pairs, labelled with Hamming distance, split and class."""
    if not pairs:
        return
    rows = len(pairs)
    width = THUMB * 2 + 30
    height = rows * (THUMB + 34)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    for row, ((i, j), dist) in enumerate(zip(pairs, distances)):
        y = row * (THUMB + 34)
        for col, idx in enumerate((i, j)):
            rec = records[idx]
            with Image.open(split_parent / rec["path"]) as img:
                thumb = img.convert("L").resize((THUMB, THUMB), Image.BICUBIC)
            sheet.paste(thumb, (col * (THUMB + 30), y + 26))
        a, b = records[i], records[j]
        draw.text(
            (2, y + 8),
            f"d={dist}  |  {a['split']}/{a['label']}/{Path(a['path']).name}"
            f"   VS   {b['split']}/{b['label']}/{Path(b['path']).name}",
            fill="black",
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"Wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--max-threshold", type=int, default=12)
    parser.add_argument("--pairs-per-bucket", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    split_parent = find_split_parent(resolve_root(args.root))
    records = list_images(split_parent)
    print(f"{len(records)} images; hashing once ...")
    sha_list, phash_bits, _ = compute_hashes(records, show_progress=False)

    sweep = []
    for threshold in range(0, args.max_threshold + 1):
        row = summarize(records, sha_list, phash_bits, threshold)
        sweep.append(row)
        print(
            f"t={threshold:>2}  edges={row['n_phash_edges']:>6}  "
            f"components={row['n_components']:>5}  "
            f"leaked_imgs={row['n_leaked_images']:>5}  "
            f"mixed_label_components={row['n_mixed_label_components']:>4}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "threshold_sweep.json").write_text(json.dumps(sweep, indent=2))

    # Contact sheets bucketed by distance, so the boundary can be judged visually.
    wide_edges = near_duplicate_edges(phash_bits, args.max_threshold)
    distances = pair_distances(phash_bits, wide_edges)
    exact_pairs = set(exact_duplicate_edges(sha_list))

    buckets: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for pair, dist in zip(wide_edges, distances):
        if pair in exact_pairs:
            continue  # byte-identical, nothing to judge
        buckets[f"d{dist:02d}"].append(pair)

    rng = np.random.default_rng(args.seed)
    for bucket, pairs in sorted(buckets.items()):
        picks = rng.choice(len(pairs), size=min(args.pairs_per_bucket, len(pairs)), replace=False)
        chosen = [pairs[int(p)] for p in picks]
        contact_sheet(
            records,
            chosen,
            pair_distances(phash_bits, chosen),
            split_parent,
            OUT_DIR / f"pairs_{bucket}.png",
            bucket,
        )

    print(f"\nBucket sizes: { {k: len(v) for k, v in sorted(buckets.items())} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
