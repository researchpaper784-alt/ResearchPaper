"""What does component-level pseudo-patient splitting catch that image-level de-duplication misses?

Three published pipelines de-duplicate this dataset at the IMAGE level (MTA-Swin 2026,
WICA-Net-M 2026, Saifullah et al. ICTAI 2025) and two of them state patient-level leakage as
unsolved. This project instead builds a near-duplicate graph, takes connected components, and
keeps each component on one side of every split. The claim that this is better needs a number,
and the number is the TRANSITIVE part of the graph: images that are not near-duplicates of each
other directly, but are joined through a chain.

A greedy pairwise image-level de-duplicator -- keep an image unless it is within the threshold of
one already kept -- guarantees no two survivors are near-duplicates. It does not guarantee no
two survivors are *related*: A and C can both survive while each is a near-duplicate of a
dropped B. Those survivor pairs are split independently, so an image-level pipeline can put A in
train and C in test. This script counts them.

Reconstructed exactly from `data/processed/manifest.csv` (its `phash` and `sha256` columns are
the inputs `fedswarm.data.dedup.audit` built the graph from), so it needs no images.

    python scripts/measure_component_vs_image_dedup.py [--threshold 5] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fedswarm.data.dedup import (  # noqa: E402
    components_from_edges,
    exact_duplicate_edges,
    near_duplicate_edges,
)


def bits_from_manifest(manifest: pd.DataFrame) -> np.ndarray:
    return np.array([[c == "1" for c in h] for h in manifest["phash"]], dtype=np.float32)


def greedy_image_level_survivors(bits: np.ndarray, sha: list[str], threshold: int,
                                 order: np.ndarray) -> list[int]:
    """Keep an image unless it is an exact or near duplicate of one already kept.

    This is the image-level scheme: pairwise, order-dependent, and it guarantees no two
    survivors are within the threshold -- which is all it guarantees.
    """
    kept: list[int] = []
    kept_bits = np.empty((0, bits.shape[1]), dtype=np.float32)
    seen_sha: set[str] = set()
    for i in order:
        if sha[i] in seen_sha:
            continue
        if len(kept):
            row = bits[i]
            dist = kept_bits @ (1.0 - row) + (1.0 - kept_bits) @ row
            if (dist <= threshold).any():
                continue
        kept.append(int(i))
        kept_bits = np.vstack([kept_bits, bits[i]])
        seen_sha.add(sha[i])
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data/processed/manifest.csv"))
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5, help="orderings for the greedy scheme")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # dtype=str: the phash column is a 64-character 0/1 string, and pandas otherwise parses it
    # as an integer -- dropping every leading zero and silently shortening the hash.
    manifest = pd.read_csv(args.manifest, dtype={"phash": str, "sha256": str})
    bits = bits_from_manifest(manifest)
    sha = manifest["sha256"].tolist()
    n = len(manifest)

    edges = sorted(set(exact_duplicate_edges(sha))
                   | set(near_duplicate_edges(bits, args.threshold)))
    labels = components_from_edges(n, edges)

    # Sanity: the reconstruction must reproduce the committed pseudo-patient partition, or
    # every number below describes a different graph from the one the splits were built on.
    committed = manifest["pseudo_patient_id"].to_numpy()
    same_partition = (
        len(set(zip(labels.tolist(), committed.tolist())))
        == len(set(labels.tolist()))
        == len(set(committed.tolist()))
    )

    direct = set(edges)
    members: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(labels):
        members[int(c)].append(i)
    multi = {c: m for c, m in members.items() if len(m) > 1}

    same_component_pairs = sum(len(m) * (len(m) - 1) // 2 for m in multi.values())
    indirect_pairs = same_component_pairs - len(direct)
    non_clique = sum(
        1 for m in multi.values()
        if len(m) * (len(m) - 1) // 2 > sum(1 for a in m for b in m if a < b and (a, b) in direct)
    )

    # The image-level scheme, over several orderings because greedy de-dup is order-dependent.
    rng = np.random.default_rng(0)
    residual = []
    for _ in range(args.seeds):
        survivors = greedy_image_level_survivors(bits, sha, args.threshold, rng.permutation(n))
        by_comp: dict[int, int] = defaultdict(int)
        for i in survivors:
            by_comp[int(labels[i])] += 1
        related_pairs = sum(k * (k - 1) // 2 for k in by_comp.values() if k > 1)
        comps_with_multiple = sum(1 for k in by_comp.values() if k > 1)
        images_in_those = sum(k for k in by_comp.values() if k > 1)
        residual.append({
            "survivors": len(survivors),
            "components_keeping_2plus_survivors": comps_with_multiple,
            "survivors_in_those_components": images_in_those,
            "related_survivor_pairs": related_pairs,
        })

    report = {
        "threshold": args.threshold,
        "n_images": n,
        "n_components": len(members),
        "reconstruction_matches_committed_partition": bool(same_partition),
        "n_multi_image_components": len(multi),
        "n_direct_edges": len(direct),
        "n_same_component_pairs": same_component_pairs,
        "n_indirect_pairs": indirect_pairs,
        "n_non_clique_components": non_clique,
        "image_level_greedy": {
            "orderings": args.seeds,
            "per_ordering": residual,
            "mean_related_survivor_pairs": float(np.mean([r["related_survivor_pairs"] for r in residual])),
            "mean_survivors_sharing_a_component": float(
                np.mean([r["survivors_in_those_components"] for r in residual])),
            "mean_survivors": float(np.mean([r["survivors"] for r in residual])),
        },
    }

    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    return 0 if same_partition else 1


if __name__ == "__main__":
    raise SystemExit(main())
