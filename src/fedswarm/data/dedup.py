"""Phase 1.2 — de-duplication and cross-split leakage audit.

This is the credibility step for the whole project. The merged Brain Tumor MRI dataset
contains near-duplicate slices (consecutive slices of one patient volume look nearly
identical). If near-duplicates straddle the train/test boundary, the model is tested on
what it trained on and every number in the paper is inflated.

What this module does:

  1. Exact duplicates    -- SHA256 over raw file bytes.
  2. Near duplicates     -- perceptual hash (phash), with dhash computed as an
                            independent cross-check on the agreement rate.
  3. Connected components over the union of those edges. **Each component is treated as
     one pseudo-patient**, and is the atomic unit for splitting and for assignment to a
     federated client from here on.
  4. Cross-split leakage -- any component whose images appear in more than one of the
     dataset's original Training/Testing splits.

Run:  python -m fedswarm.data.dedup --phash-threshold 5
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

from fedswarm.data.download import (
    CLASSES,
    IMAGE_SUFFIXES,
    SPLITS,
    find_split_parent,
    resolve_root,
)

REPORT_PATH = Path("data/processed/leakage_report.json")


def list_images(split_parent: Path) -> list[dict[str, str]]:
    """Every image across both original splits, with its split and class label."""
    records = []
    for split in SPLITS:
        for cls in CLASSES:
            class_dir = split_parent / split / cls
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    records.append(
                        {
                            "path": str(path.relative_to(split_parent)),
                            "abs_path": str(path),
                            "split": split,
                            "label": cls,
                        }
                    )
    return records


def compute_hashes(
    records: list[dict[str, str]], hash_size: int = 8, show_progress: bool = True
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (sha256 per image, phash bit matrix, dhash bit matrix).

    Bit matrices are float32 (N, hash_size**2) so Hamming distances can be computed with
    BLAS matmuls rather than a Python loop over pairs.
    """
    sha_list: list[str] = []
    phash_bits: list[np.ndarray] = []
    dhash_bits: list[np.ndarray] = []

    iterator = tqdm(records, desc="hashing", disable=not show_progress)
    for rec in iterator:
        path = Path(rec["abs_path"])
        sha_list.append(hashlib.sha256(path.read_bytes()).hexdigest())
        with Image.open(path) as img:
            img = img.convert("L")
            phash_bits.append(imagehash.phash(img, hash_size=hash_size).hash.flatten())
            dhash_bits.append(imagehash.dhash(img, hash_size=hash_size).hash.flatten())

    return (
        sha_list,
        np.asarray(phash_bits, dtype=np.float32),
        np.asarray(dhash_bits, dtype=np.float32),
    )


def exact_duplicate_edges(sha_list: list[str]) -> list[tuple[int, int]]:
    """Edges between images with byte-identical content."""
    by_digest: dict[str, list[int]] = defaultdict(list)
    for idx, digest in enumerate(sha_list):
        by_digest[digest].append(idx)

    edges = []
    for indices in by_digest.values():
        anchor = indices[0]
        for other in indices[1:]:
            edges.append((anchor, other))
    return edges


def near_duplicate_edges(
    bits: np.ndarray, threshold: int, chunk_size: int = 512
) -> list[tuple[int, int]]:
    """Pairs whose Hamming distance is <= threshold.

    For binary matrices, Hamming(i, j) = A[i]·(1-A[j]) + (1-A[i])·A[j], so the full
    pairwise distance matrix is two matmuls. Chunked over rows to bound memory.
    """
    n = bits.shape[0]
    complement = 1.0 - bits
    edges = []

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        block = bits[start:stop]
        block_complement = complement[start:stop]
        distances = block @ complement.T + block_complement @ bits.T

        rows, cols = np.nonzero(distances <= threshold)
        for row, col in zip(rows, cols):
            i = start + int(row)
            j = int(col)
            if i < j:  # upper triangle only, skips self-pairs
                edges.append((i, j))

    return edges


def components_from_edges(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Connected-component label per image. Isolated images get their own component."""
    if edges:
        rows = np.array([e[0] for e in edges])
        cols = np.array([e[1] for e in edges])
        data = np.ones(len(edges))
        adjacency = coo_matrix((data, (rows, cols)), shape=(n, n))
    else:
        adjacency = coo_matrix((n, n))

    _, labels = connected_components(adjacency, directed=False)
    return labels


def audit(
    split_parent: Path,
    phash_threshold: int = 5,
    hash_size: int = 8,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run the full audit and return the report as a dict."""
    records = list_images(split_parent)
    if not records:
        raise FileNotFoundError(f"No images found under {split_parent}")

    sha_list, phash_bits, dhash_bits = compute_hashes(
        records, hash_size=hash_size, show_progress=show_progress
    )

    exact_edges = exact_duplicate_edges(sha_list)
    phash_edges = near_duplicate_edges(phash_bits, phash_threshold)
    dhash_edges = near_duplicate_edges(dhash_bits, phash_threshold)

    # dhash is an independent cross-check: how often does it agree that a phash-flagged
    # pair is a duplicate? Low agreement means the threshold is probably too loose.
    phash_set, dhash_set = set(phash_edges), set(dhash_edges)
    agreement = len(phash_set & dhash_set) / len(phash_set) if phash_set else None

    all_edges = list(set(exact_edges) | phash_set)
    labels = components_from_edges(len(records), all_edges)

    for rec, label, digest, phash_row in zip(records, labels, sha_list, phash_bits):
        rec["pseudo_patient_id"] = int(label)
        rec["sha256"] = digest
        rec["phash"] = "".join(str(int(b)) for b in phash_row.astype(np.uint8))

    # Cross-split leakage: a component whose images live in more than one original split.
    splits_per_component: dict[int, set[str]] = defaultdict(set)
    labels_per_component: dict[int, set[str]] = defaultdict(set)
    for rec in records:
        splits_per_component[rec["pseudo_patient_id"]].add(rec["split"])
        labels_per_component[rec["pseudo_patient_id"]].add(rec["label"])

    leaked = sorted(c for c, s in splits_per_component.items() if len(s) > 1)
    leaked_images = [r for r in records if r["pseudo_patient_id"] in set(leaked)]

    # A component spanning two class labels means the duplicate detection crossed a class
    # boundary -- either a threshold that is too loose, or genuine dataset mislabeling.
    mixed_label = sorted(c for c, ls in labels_per_component.items() if len(ls) > 1)

    n_exact_dupe_images = len(records) - len(set(sha_list))
    component_sizes = Counter(int(label) for label in labels)
    multi_image_components = {c: n for c, n in component_sizes.items() if n > 1}

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scanned_path": str(split_parent),
        "params": {
            "phash_threshold": phash_threshold,
            "hash_size": hash_size,
            "hash_bits": int(phash_bits.shape[1]),
        },
        "totals": {
            "n_images": len(records),
            "n_unique_sha256": len(set(sha_list)),
            "n_exact_duplicate_images": int(n_exact_dupe_images),
            "n_pseudo_patients": int(len(component_sizes)),
            "n_multi_image_components": len(multi_image_components),
            "largest_component_size": max(component_sizes.values()),
        },
        "edges": {
            "n_exact_edges": len(exact_edges),
            "n_phash_edges": len(phash_edges),
            "n_dhash_edges": len(dhash_edges),
            "phash_dhash_agreement": agreement,
        },
        "cross_split_leakage": {
            "n_leaked_components": len(leaked),
            "n_leaked_images": len(leaked_images),
            "leaked_component_ids": leaked[:200],
            "note": (
                "Counted against the dataset's ORIGINAL Training/Testing split. This is "
                "the number to report in the paper as the leakage that existed before "
                "our re-split. Phase 1.3 rebuilds splits at the pseudo-patient level so "
                "this becomes zero by construction."
            ),
        },
        "mixed_label_components": {
            "n": len(mixed_label),
            "component_ids": mixed_label[:200],
            "note": (
                "Components spanning more than one class label. Either the near-duplicate "
                "threshold is too loose, or these are genuine mislabelings (the SARTAJ "
                "component of this dataset has documented glioma mislabeling). Inspect "
                "before trusting the threshold."
            ),
        },
    }
    return {"report": report, "records": records}


def sample_flagged_pairs(
    records: list[dict[str, str]], edges: list[tuple[int, int]], n: int, seed: int = 0
) -> list[dict[str, str]]:
    """A reproducible sample of flagged pairs, for manual eyeballing of the threshold."""
    rng = np.random.default_rng(seed)
    if not edges:
        return []
    picks = rng.choice(len(edges), size=min(n, len(edges)), replace=False)
    return [
        {"a": records[edges[int(p)][0]]["path"], "b": records[edges[int(p)][1]]["path"]}
        for p in picks
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="Dataset root (default: $FEDSWARM_DATA_ROOT)")
    parser.add_argument("--phash-threshold", type=int, default=5, help="Max Hamming distance")
    parser.add_argument("--hash-size", type=int, default=8, help="phash/dhash size (bits=size^2)")
    parser.add_argument("--report", default=str(REPORT_PATH), help="Output JSON path")
    parser.add_argument(
        "--records-out",
        default="data/processed/pseudo_patients.csv",
        help="Per-image pseudo-patient assignment, consumed by Phase 1.3",
    )
    args = parser.parse_args()

    try:
        split_parent = find_split_parent(resolve_root(args.root))
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        return 1

    result = audit(
        split_parent, phash_threshold=args.phash_threshold, hash_size=args.hash_size
    )
    report, records = result["report"], result["records"]

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    records_path = Path(args.records_out)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "split", "label", "pseudo_patient_id", "sha256", "phash"]
        )
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec[k] for k in writer.fieldnames})

    totals = report["totals"]
    leakage = report["cross_split_leakage"]
    print()
    print(f"images                     {totals['n_images']}")
    print(f"exact duplicate images     {totals['n_exact_duplicate_images']}")
    print(f"pseudo-patients            {totals['n_pseudo_patients']}")
    print(f"multi-image components     {totals['n_multi_image_components']}")
    print(f"largest component          {totals['largest_component_size']}")
    print(f"phash/dhash agreement      {report['edges']['phash_dhash_agreement']}")
    print()
    print(f"⚠️  CROSS-SPLIT LEAKS (original split): {leakage['n_leaked_components']} components, "
          f"{leakage['n_leaked_images']} images")
    print(f"⚠️  mixed-label components: {report['mixed_label_components']['n']}")
    print()
    print(f"Wrote {report_path}")
    print(f"Wrote {records_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
