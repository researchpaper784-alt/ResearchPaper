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
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from fedswarm.data.datasets import LABEL_TO_INDEX, load_manifest, select
from fedswarm.data.download import CLASSES

PARTITION_DIR = Path("data/processed/partitions")
SOURCE_PROVENANCE_CSV = Path("data/processed/source_provenance.csv")
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


@lru_cache(maxsize=4)
def load_pseudo_patient_source_labels(
    path: Path | str = SOURCE_PROVENANCE_CSV,
) -> dict[int, str]:
    """One source label per pseudo-patient: majority vote among that pseudo-patient's
    matched member images (source_provenance.py matches per-image, since near-duplicate
    siblings can differ in whether any single one crosses the match threshold).
    Pseudo-patients with zero matched members are omitted from the returned dict --
    callers treat "not in this dict" as "provenance unknown", not as a fourth source.

    Cached: build_partition's retry loop can call this many times per source_shift
    build (once per attempt when the random split doesn't hit min_client_size), and
    re-reading + re-grouping a multi-thousand-row CSV each time is wasted work for data
    that never changes within a process.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No source provenance at {path}. Run "
            "`python -m fedswarm.data.source_provenance --figshare ... --sartaj ... "
            "--br35h ...` first (see docs/OPEN_QUESTIONS.md)."
        )

    df = pd.read_csv(path)
    labels: dict[int, str] = {}
    for pseudo_patient_id, group in df.groupby("pseudo_patient_id"):
        matched = group[group["matched"]]
        if len(matched):
            labels[int(pseudo_patient_id)] = matched["source_component"].mode().iloc[0]
    return labels


def _source_shift(
    pseudo_patient_ids: np.ndarray,
    k: int,
    rng: np.random.Generator,
    source_labels: dict[int, str] | None = None,
) -> tuple[list[np.ndarray], int]:
    """Partition by which upstream dataset (Figshare/SARTAJ/Br35H) each pseudo-patient's
    images came from -- the realistic cross-site setting: real heterogeneity between
    hospitals is feature/acquisition shift (different scanners, protocols), not just
    label imbalance (plan §1.4).

    Clients are split into source-aligned groups sized proportionally to each source's
    available pseudo-patient count; within a group, that source's pseudo-patients are
    split evenly across its clients. Pseudo-patients with no recovered provenance (see
    load_pseudo_patient_source_labels) are EXCLUDED from this regime rather than dumped
    into an arbitrary bucket, which would blur the cross-site signal this regime exists
    to isolate. Returns (buckets, n_excluded) so the caller can report the exclusion
    honestly rather than silently shrinking the dataset.

    `source_labels` defaults to loading the real recovered-provenance CSV; tests inject
    a small synthetic dict directly instead.
    """
    if source_labels is None:
        source_labels = load_pseudo_patient_source_labels()

    positions_by_source: dict[str, list[int]] = {}
    n_excluded = 0
    for position, pseudo_patient_id in enumerate(pseudo_patient_ids):
        source = source_labels.get(int(pseudo_patient_id))
        if source is None:
            n_excluded += 1
        else:
            positions_by_source.setdefault(source, []).append(position)

    if not positions_by_source:
        raise RuntimeError(
            "No pseudo-patient in this training split has recovered provenance -- "
            "cannot build source_shift partitions."
        )

    sources = sorted(positions_by_source)
    sizes = np.array([len(positions_by_source[s]) for s in sources])
    total = sizes.sum()

    # Proportional client allocation, each source with any data gets >=1 client, and the
    # largest source absorbs the rounding remainder so allocations sum to exactly k.
    raw_allocation = sizes / total * k
    allocation = np.maximum(1, np.floor(raw_allocation)).astype(int)
    while allocation.sum() > k:
        allocation[np.argmax(allocation)] -= 1
    while allocation.sum() < k:
        allocation[np.argmax(sizes)] += 1

    buckets: list[np.ndarray] = []
    for source, n_clients_for_source in zip(sources, allocation):
        positions = rng.permutation(positions_by_source[source])
        buckets.extend(np.sort(chunk) for chunk in np.array_split(positions, n_clients_for_source))

    return buckets, n_excluded


def _build_buckets(
    labels: np.ndarray,
    spec: PartitionSpec,
    rng: np.random.Generator,
    pseudo_patient_ids: np.ndarray | None = None,
    source_labels: dict[int, str] | None = None,
) -> tuple[list[np.ndarray], int]:
    """Returns (buckets, n_excluded) -- n_excluded is always 0 except for source_shift,
    which legitimately cannot place pseudo-patients with no recovered provenance."""
    if spec.regime == "iid":
        return _iid(labels, spec.num_clients, rng), 0
    if spec.regime == "dirichlet":
        if spec.alpha is None:
            raise ValueError("dirichlet regime requires --alpha")
        return _dirichlet(labels, spec.num_clients, spec.alpha, rng), 0
    if spec.regime == "pathological":
        if spec.classes_per_client is None:
            raise ValueError("pathological regime requires --classes-per-client")
        return _pathological(labels, spec.num_clients, spec.classes_per_client, rng), 0
    if spec.regime == "quantity_skew":
        if spec.skew_sigma is None:
            raise ValueError("quantity_skew regime requires --skew-sigma")
        return _quantity_skew(labels, spec.num_clients, spec.skew_sigma, rng), 0
    if spec.regime == "source_shift":
        if pseudo_patient_ids is None:
            raise ValueError("source_shift regime requires pseudo_patient_ids")
        return _source_shift(pseudo_patient_ids, spec.num_clients, rng, source_labels)
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


def build_partition(
    manifest: pd.DataFrame,
    spec: PartitionSpec,
    source_labels: dict[int, str] | None = None,
) -> Partition:
    """Pure function of (manifest, spec). Same spec and seed always yields the same split.

    For every regime except source_shift, the returned buckets cover all of train_rows.
    source_shift legitimately cannot place pseudo-patients with no recovered upstream
    provenance (see load_pseudo_patient_source_labels) -- those are excluded, and the
    count is recorded in diagnostics rather than silently dropped.

    `source_labels` is only consulted for the source_shift regime; passing it explicitly
    (rather than relying on the default disk-backed lookup) is how tests inject a small
    synthetic mapping instead of depending on the real multi-thousand-row CSV.
    """
    train_rows = select(manifest, split="train", representatives_only=True)
    labels = manifest.loc[train_rows, "label"].map(LABEL_TO_INDEX).to_numpy()
    pseudo_patient_ids = manifest.loc[train_rows, "pseudo_patient_id"].to_numpy()

    rng = np.random.default_rng(spec.seed)
    for attempt in range(spec.max_attempts):
        buckets, n_excluded = _build_buckets(
            labels, spec, rng, pseudo_patient_ids, source_labels
        )
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
        "n_units_placed": int(sizes.sum()),
        "n_units_excluded_no_provenance": n_excluded,
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
