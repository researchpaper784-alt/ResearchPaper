"""Phase 1.4 — federated partitioning.

Partitions the **training pseudo-patients** across clients. Never partitions images: two
near-identical slices of one patient landing on two different clients would leak across
the federation the same way they leaked across train/test in Phase 1.2.

`val` and `test` stay global and server-side; they are never partitioned.

Deviation from the plan, deliberately: the plan suggests reusing
`flwr_datasets.partitioner.{Iid,Dirichlet,Pathological}Partitioner`. Those operate on rows
of a Hugging Face `Dataset`, whereas our unit is a pseudo-patient row in a manifest and our
required output is a JSON-cacheable index assignment. Converting manifest -> HF Dataset ->
partition -> back to indices is more code and more fragile than drawing the Dirichlet
directly, and mixing two partitioner styles (since `quantity_skew` must be custom anyway)
is worse than one consistent one. The algorithms below are the standard formulations.

Run: python -m fedswarm.data.partition --regime dirichlet --alpha 0.3 --num-clients 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from fedswarm.data.datasets import LABEL_TO_INDEX, load_manifest, select
from fedswarm.data.download import CLASSES

PARTITION_DIR = Path("data/processed/partitions")
REGIMES = ("iid", "dirichlet", "pathological", "quantity_skew", "source_shift")


@dataclass(frozen=True)
class PartitionSpec:
    regime: str
    num_clients: int = 20
    seed: int = 0
    alpha: float | None = None
    classes_per_client: int | None = None
    skew_sigma: float | None = None
    min_client_size: int = 10
    local_val_fraction: float = 0.1
    max_attempts: int = 100

    def key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


@dataclass
class Partition:
    spec: PartitionSpec
    client_train: dict[int, list[int]] = field(default_factory=dict)
    client_val: dict[int, list[int]] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "spec": asdict(self.spec),
            "key": self.spec.key(),
            "client_train": {str(k): v for k, v in self.client_train.items()},
            "client_val": {str(k): v for k, v in self.client_val.items()},
            "diagnostics": self.diagnostics,
        }


# --- regimes -------------------------------------------------------------------------
# Each returns a list of arrays of positions into the `labels` array.


def _iid(labels: np.ndarray, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = rng.permutation(len(labels))
    return [np.sort(chunk) for chunk in np.array_split(order, k)]


def _dirichlet(
    labels: np.ndarray, k: int, alpha: float, rng: np.random.Generator
) -> list[np.ndarray]:
    """Label skew: for each class, draw client proportions from Dir(alpha)."""
    buckets: list[list[np.ndarray]] = [[] for _ in range(k)]
    for class_index in np.unique(labels):
        class_positions = np.flatnonzero(labels == class_index)
        rng.shuffle(class_positions)
        proportions = rng.dirichlet(np.repeat(alpha, k))
        cut_points = (np.cumsum(proportions) * len(class_positions)).astype(int)[:-1]
        for client, piece in enumerate(np.split(class_positions, cut_points)):
            buckets[client].append(piece)
    return [np.sort(np.concatenate(b)) if b else np.array([], dtype=int) for b in buckets]


def _pathological(
    labels: np.ndarray, k: int, classes_per_client: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Shard assignment, as in the original FedAvg paper: sort by label, cut into
    k * classes_per_client contiguous shards, give each client that many shards."""
    order = np.argsort(labels, kind="stable")
    shards = np.array_split(order, k * classes_per_client)
    shard_order = rng.permutation(len(shards))

    buckets = []
    for client in range(k):
        picked = shard_order[client * classes_per_client : (client + 1) * classes_per_client]
        buckets.append(np.sort(np.concatenate([shards[s] for s in picked])))
    return buckets


def _quantity_skew(
    labels: np.ndarray, k: int, sigma: float, rng: np.random.Generator
) -> list[np.ndarray]:
    """Label-balanced but size-skewed: client sizes ~ lognormal.

    This regime attacks FedAvg's n_k weighting directly, which is exactly what FedSwarm
    replaces, so it is where the largest gains should appear if the method works at all.
    """
    weights = rng.lognormal(mean=0.0, sigma=sigma, size=k)
    weights /= weights.sum()

    order = rng.permutation(len(labels))
    cut_points = (np.cumsum(weights) * len(labels)).astype(int)[:-1]
    return [np.sort(piece) for piece in np.split(order, cut_points)]


def _build_buckets(
    labels: np.ndarray, spec: PartitionSpec, rng: np.random.Generator
) -> list[np.ndarray]:
    if spec.regime == "iid":
        return _iid(labels, spec.num_clients, rng)
    if spec.regime == "dirichlet":
        if spec.alpha is None:
            raise ValueError("dirichlet regime requires --alpha")
        return _dirichlet(labels, spec.num_clients, spec.alpha, rng)
    if spec.regime == "pathological":
        if spec.classes_per_client is None:
            raise ValueError("pathological regime requires --classes-per-client")
        return _pathological(labels, spec.num_clients, spec.classes_per_client, rng)
    if spec.regime == "quantity_skew":
        if spec.skew_sigma is None:
            raise ValueError("quantity_skew regime requires --skew-sigma")
        return _quantity_skew(labels, spec.num_clients, spec.skew_sigma, rng)
    if spec.regime == "source_shift":
        raise NotImplementedError(
            "source_shift needs a per-image source_component label, which this dataset "
            "does not carry. See the source_shift section of docs/OPEN_QUESTIONS.md -- it "
            "requires hash-matching against the upstream Figshare/SARTAJ/Br35H datasets "
            "and is an open author decision."
        )
    raise ValueError(f"Unknown regime {spec.regime!r}; expected one of {REGIMES}")


# --- diagnostics ---------------------------------------------------------------------


def class_matrix(buckets: list[np.ndarray], labels: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(buckets), len(CLASSES)), dtype=int)
    for client, positions in enumerate(buckets):
        for class_index, count in zip(*np.unique(labels[positions], return_counts=True)):
            matrix[client, class_index] = count
    return matrix


def heterogeneity_index(matrix: np.ndarray) -> float:
    """Mean pairwise Jensen-Shannon **divergence** (base 2) between client label
    distributions: 0 = identical, 1 = maximally disjoint.

    scipy returns the JS *distance*, so it is squared here to get divergence. Reported into
    every result file so gains can be plotted against measured heterogeneity rather than
    against a nominal Dirichlet alpha.
    """
    totals = matrix.sum(axis=1, keepdims=True)
    if np.any(totals == 0):
        return float("nan")
    distributions = matrix / totals

    divergences = [
        jensenshannon(distributions[i], distributions[j], base=2) ** 2
        for i in range(len(matrix))
        for j in range(i + 1, len(matrix))
    ]
    return float(np.nanmean(divergences)) if divergences else 0.0


# --- driver --------------------------------------------------------------------------


def build_partition(manifest: pd.DataFrame, spec: PartitionSpec) -> Partition:
    """Pure function of (manifest, spec). Same spec and seed always yields the same split."""
    train_rows = select(manifest, split="train", representatives_only=True)
    labels = manifest.loc[train_rows, "label"].map(LABEL_TO_INDEX).to_numpy()

    rng = np.random.default_rng(spec.seed)
    for attempt in range(spec.max_attempts):
        buckets = _build_buckets(labels, spec, rng)
        sizes = np.array([len(b) for b in buckets])
        if sizes.min() >= spec.min_client_size:
            break
    else:
        raise RuntimeError(
            f"Could not produce {spec.num_clients} clients with >= {spec.min_client_size} "
            f"samples each after {spec.max_attempts} attempts (regime={spec.regime}, "
            f"alpha={spec.alpha}). Reduce num_clients, lower min_client_size, or raise alpha."
        )

    partition = Partition(spec=spec)
    for client, positions in enumerate(buckets):
        shuffled = rng.permutation(positions)
        n_val = max(1, int(round(len(shuffled) * spec.local_val_fraction)))
        n_val = min(n_val, len(shuffled) - 1)  # never leave local-train empty
        partition.client_val[client] = sorted(int(train_rows[i]) for i in shuffled[:n_val])
        partition.client_train[client] = sorted(int(train_rows[i]) for i in shuffled[n_val:])

    matrix = class_matrix(buckets, labels)
    partition.diagnostics = {
        "js_divergence": heterogeneity_index(matrix),
        "client_sizes": [int(s) for s in sizes],
        "class_matrix": matrix.tolist(),
        "classes": list(CLASSES),
        "n_units_total": int(len(train_rows)),
        "attempts_needed": attempt + 1,
        "size_gini": float(_gini(sizes)),
    }
    return partition


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of client sizes; 0 = perfectly even, ->1 = concentrated."""
    sorted_values = np.sort(values.astype(float))
    n = len(sorted_values)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n)


def load_or_build(
    manifest: pd.DataFrame, spec: PartitionSpec, cache_dir: Path = PARTITION_DIR
) -> Partition:
    cache_path = Path(cache_dir) / f"{spec.key()}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text())
        return Partition(
            spec=spec,
            client_train={int(k): v for k, v in payload["client_train"].items()},
            client_val={int(k): v for k, v in payload["client_val"].items()},
            diagnostics=payload["diagnostics"],
        )

    partition = build_partition(manifest, spec)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(partition.to_dict(), indent=2))
    return partition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=REGIMES, required=True)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--classes-per-client", type=int, default=None)
    parser.add_argument("--skew-sigma", type=float, default=None)
    parser.add_argument("--min-client-size", type=int, default=10)
    args = parser.parse_args()

    spec = PartitionSpec(
        regime=args.regime,
        num_clients=args.num_clients,
        seed=args.seed,
        alpha=args.alpha,
        classes_per_client=args.classes_per_client,
        skew_sigma=args.skew_sigma,
        min_client_size=args.min_client_size,
    )
    partition = load_or_build(load_manifest(), spec)
    diagnostics = partition.diagnostics

    print(f"regime={spec.regime} K={spec.num_clients} seed={spec.seed} key={spec.key()}")
    print(f"JS divergence  {diagnostics['js_divergence']:.4f}")
    print(f"size gini      {diagnostics['size_gini']:.4f}")
    print(f"client sizes   min={min(diagnostics['client_sizes'])} "
          f"max={max(diagnostics['client_sizes'])} total={sum(diagnostics['client_sizes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
