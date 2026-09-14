"""Tests for the source_shift partitioning regime (plan §1.4's fifth regime).

Uses synthetic pseudo-patient -> source dicts throughout, injected via build_partition's
source_labels parameter, so these tests are independent of the real multi-thousand-row
source_provenance.csv and of whether it's been (re)generated recently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fedswarm.data.partition import PartitionSpec, build_partition


def _synthetic_manifest(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    labels = rng.choice(["glioma", "meningioma", "notumor", "pituitary"], size=n)
    return pd.DataFrame(
        {
            "path": [f"img_{i}.jpg" for i in range(n)],
            "label": labels,
            "pseudo_patient_id": np.arange(n),
            "split": "train",
            "is_representative": True,
            "is_mixed_label": False,
        }
    )


def _synthetic_source_labels(n_a: int, n_b: int, n_c: int, offset: int = 0) -> dict[int, str]:
    """pseudo-patients [offset, offset+n_a) -> 'a', next n_b -> 'b', next n_c -> 'c'.
    Anything beyond n_a+n_b+n_c (up to the manifest size) is left unmatched."""
    labels: dict[int, str] = {}
    i = offset
    for _ in range(n_a):
        labels[i] = "a"
        i += 1
    for _ in range(n_b):
        labels[i] = "b"
        i += 1
    for _ in range(n_c):
        labels[i] = "c"
        i += 1
    return labels


def test_excludes_pseudo_patients_with_no_provenance() -> None:
    manifest = _synthetic_manifest(300)
    source_labels = _synthetic_source_labels(n_a=100, n_b=60, n_c=40)  # 200 known, 100 not

    spec = PartitionSpec(regime="source_shift", num_clients=10, seed=0, min_client_size=1)
    partition = build_partition(manifest, spec, source_labels=source_labels)

    assert partition.diagnostics["n_units_excluded_no_provenance"] == 100
    assert partition.diagnostics["n_units_placed"] == 200
    assert partition.diagnostics["n_units_total"] == 300


def test_excluded_units_are_never_assigned_to_any_client() -> None:
    manifest = _synthetic_manifest(300)
    source_labels = _synthetic_source_labels(n_a=100, n_b=60, n_c=40)
    excluded_positions = set(range(200, 300))  # the un-labeled tail

    spec = PartitionSpec(regime="source_shift", num_clients=10, seed=0, min_client_size=1)
    partition = build_partition(manifest, spec, source_labels=source_labels)

    assigned = set()
    for client in partition.client_train:
        assigned.update(partition.client_train[client])
        assigned.update(partition.client_val[client])

    assert assigned.isdisjoint(excluded_positions)
    assert len(assigned) == 200


def test_client_count_scales_with_source_size() -> None:
    """Source 'a' has 10x source 'b's or 'c's data; proportional allocation should give
    it roughly 10x the clients (500/600*12=10.0 vs 50/600*12=1.0 each) -- which in turn
    means per-client size is roughly EQUAL across sources, not lopsided. Verified by
    tracing which original position range each client's units fall in, since the source
    label itself isn't threaded into the returned Partition object."""
    manifest = _synthetic_manifest(600)
    source_labels = _synthetic_source_labels(n_a=500, n_b=50, n_c=50)  # a=[0,500) b=[500,550) c=[550,600)

    spec = PartitionSpec(regime="source_shift", num_clients=12, seed=0, min_client_size=1)
    partition = build_partition(manifest, spec, source_labels=source_labels)

    clients_touching_a = clients_touching_b = clients_touching_c = 0
    for client in partition.client_train:
        units = partition.client_train[client] + partition.client_val[client]
        if any(u < 500 for u in units):
            clients_touching_a += 1
        if any(500 <= u < 550 for u in units):
            clients_touching_b += 1
        if any(550 <= u < 600 for u in units):
            clients_touching_c += 1

    assert clients_touching_a == 10
    assert clients_touching_b == 1
    assert clients_touching_c == 1

    # per-client size should be roughly EQUAL across sources despite 'a' having 10x the
    # data -- that's what proportional client allocation buys you.
    sizes = partition.diagnostics["client_sizes"]
    assert max(sizes) / min(sizes) < 1.5


def test_each_source_gets_at_least_one_client() -> None:
    """Even a tiny source shouldn't be squeezed out to zero clients by rounding."""
    manifest = _synthetic_manifest(300)
    source_labels = _synthetic_source_labels(n_a=180, n_b=15, n_c=5)  # c is tiny

    spec = PartitionSpec(regime="source_shift", num_clients=10, seed=0, min_client_size=1)
    partition = build_partition(manifest, spec, source_labels=source_labels)

    # every placed unit must belong to source a, b, or c -- none should be silently
    # dropped for a source that received zero clients.
    assert partition.diagnostics["n_units_placed"] == 200


def test_buckets_are_disjoint_and_no_client_is_empty() -> None:
    manifest = _synthetic_manifest(300)
    source_labels = _synthetic_source_labels(n_a=120, n_b=80, n_c=0)

    spec = PartitionSpec(regime="source_shift", num_clients=8, seed=0, min_client_size=1)
    partition = build_partition(manifest, spec, source_labels=source_labels)

    assert len(partition.client_train) == 8
    seen = set()
    for client in range(8):
        assert partition.client_train[client], f"client {client} has no train data"
        for idx in partition.client_train[client] + partition.client_val[client]:
            assert idx not in seen, f"unit {idx} assigned to more than one client"
            seen.add(idx)


def test_reproducible_under_fixed_seed() -> None:
    manifest = _synthetic_manifest(300)
    source_labels = _synthetic_source_labels(n_a=100, n_b=60, n_c=40)
    spec = PartitionSpec(regime="source_shift", num_clients=10, seed=3, min_client_size=1)

    first = build_partition(manifest, spec, source_labels=source_labels)
    second = build_partition(manifest, spec, source_labels=source_labels)

    assert first.client_train == second.client_train
    assert first.client_val == second.client_val


def test_different_seeds_reshuffle_within_source_assignment() -> None:
    manifest = _synthetic_manifest(300)
    source_labels = _synthetic_source_labels(n_a=100, n_b=60, n_c=40)

    a = build_partition(
        manifest,
        PartitionSpec(regime="source_shift", num_clients=10, seed=0, min_client_size=1),
        source_labels=source_labels,
    )
    b = build_partition(
        manifest,
        PartitionSpec(regime="source_shift", num_clients=10, seed=1, min_client_size=1),
        source_labels=source_labels,
    )

    assert a.client_train != b.client_train


def test_raises_when_nothing_has_provenance() -> None:
    manifest = _synthetic_manifest(50)
    spec = PartitionSpec(regime="source_shift", num_clients=5, seed=0, min_client_size=1)

    with pytest.raises(RuntimeError, match="[Nn]o pseudo-patient"):
        build_partition(manifest, spec, source_labels={})


def test_diagnostics_are_computable_over_placed_units_only() -> None:
    """js_divergence/size_gini must not choke on a partition that doesn't cover every
    training unit -- they're computed from the returned buckets, which is correct."""
    manifest = _synthetic_manifest(300)
    source_labels = _synthetic_source_labels(n_a=100, n_b=60, n_c=40)
    spec = PartitionSpec(regime="source_shift", num_clients=10, seed=0, min_client_size=1)

    partition = build_partition(manifest, spec, source_labels=source_labels)

    assert np.isfinite(partition.diagnostics["js_divergence"])
    assert np.isfinite(partition.diagnostics["size_gini"])
