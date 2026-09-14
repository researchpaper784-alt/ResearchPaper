"""Phase 1.4 tests — federated partitioning.

The acceptance criteria the plan states for every regime: partitions are disjoint, cover
every training pseudo-patient, are reproducible under a fixed seed, and leave no client
empty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fedswarm.data.partition import (
    PartitionSpec,
    build_partition,
    class_matrix,
    heterogeneity_index,
)

SPECS = {
    "iid": PartitionSpec(regime="iid", num_clients=8, seed=0, min_client_size=5),
    "dirichlet": PartitionSpec(
        regime="dirichlet", num_clients=8, seed=0, alpha=0.3, min_client_size=5
    ),
    "pathological": PartitionSpec(
        regime="pathological", num_clients=8, seed=0, classes_per_client=2, min_client_size=5
    ),
    "quantity_skew": PartitionSpec(
        regime="quantity_skew", num_clients=8, seed=0, skew_sigma=1.0, min_client_size=5
    ),
}


@pytest.fixture
def synthetic_manifest() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
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


def _all_assigned(partition) -> list[int]:
    assigned = []
    for client in partition.client_train:
        assigned += partition.client_train[client] + partition.client_val[client]
    return assigned


@pytest.mark.parametrize("regime", sorted(SPECS))
def test_partitions_are_disjoint(synthetic_manifest, regime) -> None:
    partition = build_partition(synthetic_manifest, SPECS[regime])
    assigned = _all_assigned(partition)

    assert len(assigned) == len(set(assigned)), f"{regime}: a unit was assigned twice"


@pytest.mark.parametrize("regime", sorted(SPECS))
def test_partitions_cover_every_training_unit(synthetic_manifest, regime) -> None:
    partition = build_partition(synthetic_manifest, SPECS[regime])

    assert set(_all_assigned(partition)) == set(range(len(synthetic_manifest)))


@pytest.mark.parametrize("regime", sorted(SPECS))
def test_no_client_is_empty(synthetic_manifest, regime) -> None:
    spec = SPECS[regime]
    partition = build_partition(synthetic_manifest, spec)

    assert len(partition.client_train) == spec.num_clients
    for client in range(spec.num_clients):
        assert partition.client_train[client], f"{regime}: client {client} has no train data"
        assert partition.client_val[client], f"{regime}: client {client} has no val data"


@pytest.mark.parametrize("regime", sorted(SPECS))
def test_partition_is_reproducible(synthetic_manifest, regime) -> None:
    first = build_partition(synthetic_manifest, SPECS[regime])
    second = build_partition(synthetic_manifest, SPECS[regime])

    assert first.client_train == second.client_train
    assert first.client_val == second.client_val


def test_different_seeds_give_different_partitions(synthetic_manifest) -> None:
    a = build_partition(synthetic_manifest, PartitionSpec(regime="iid", num_clients=8, seed=0))
    b = build_partition(synthetic_manifest, PartitionSpec(regime="iid", num_clients=8, seed=1))

    assert a.client_train != b.client_train


def test_local_val_fraction_is_respected(synthetic_manifest) -> None:
    spec = PartitionSpec(regime="iid", num_clients=8, seed=0, local_val_fraction=0.1)
    partition = build_partition(synthetic_manifest, spec)

    for client in partition.client_train:
        total = len(partition.client_train[client]) + len(partition.client_val[client])
        assert len(partition.client_val[client]) == pytest.approx(0.1 * total, abs=1)


# --- regime-specific behaviour -------------------------------------------------------


def test_iid_is_nearly_homogeneous(synthetic_manifest) -> None:
    partition = build_partition(synthetic_manifest, SPECS["iid"])

    assert partition.diagnostics["js_divergence"] < 0.1
    assert partition.diagnostics["size_gini"] < 0.05


def test_dirichlet_heterogeneity_decreases_as_alpha_increases(synthetic_manifest) -> None:
    divergences = [
        build_partition(
            synthetic_manifest,
            PartitionSpec(
                regime="dirichlet", num_clients=8, seed=0, alpha=alpha, min_client_size=1
            ),
        ).diagnostics["js_divergence"]
        for alpha in (0.1, 1.0, 10.0)
    ]

    assert divergences[0] > divergences[1] > divergences[2]


def test_pathological_limits_classes_per_client(synthetic_manifest) -> None:
    spec = SPECS["pathological"]
    partition = build_partition(synthetic_manifest, spec)
    matrix = np.array(partition.diagnostics["class_matrix"])

    classes_held = (matrix > 0).sum(axis=1)
    # Shards are contiguous in label-sorted order, so one shard may straddle a label
    # boundary; that allows at most one extra class per client.
    assert classes_held.max() <= spec.classes_per_client + 1
    assert classes_held.mean() <= spec.classes_per_client + 0.5


def test_quantity_skew_skews_size_but_not_labels(synthetic_manifest) -> None:
    partition = build_partition(synthetic_manifest, SPECS["quantity_skew"])

    assert partition.diagnostics["size_gini"] > 0.15, "sizes should be uneven"
    assert partition.diagnostics["js_divergence"] < 0.1, "labels should stay balanced"


def test_source_shift_raises_with_a_pointer_to_the_open_question(synthetic_manifest) -> None:
    spec = PartitionSpec(regime="source_shift", num_clients=8, seed=0)

    with pytest.raises(NotImplementedError, match="OPEN_QUESTIONS"):
        build_partition(synthetic_manifest, spec)


def test_impossible_min_client_size_raises_rather_than_looping_forever(
    synthetic_manifest,
) -> None:
    spec = PartitionSpec(
        regime="iid", num_clients=8, seed=0, min_client_size=10_000, max_attempts=3
    )

    with pytest.raises(RuntimeError, match="Could not produce"):
        build_partition(synthetic_manifest, spec)


# --- diagnostics ---------------------------------------------------------------------


def test_heterogeneity_index_is_zero_for_identical_clients() -> None:
    matrix = np.tile(np.array([10, 20, 30, 40]), (5, 1))

    assert heterogeneity_index(matrix) == pytest.approx(0.0, abs=1e-9)


def test_heterogeneity_index_is_one_for_disjoint_clients() -> None:
    matrix = np.eye(4, dtype=int) * 10

    assert heterogeneity_index(matrix) == pytest.approx(1.0, abs=1e-9)


def test_class_matrix_counts_match_the_labels() -> None:
    labels = np.array([0, 0, 1, 2, 3, 3, 3])
    buckets = [np.array([0, 1, 2]), np.array([3, 4, 5, 6])]

    matrix = class_matrix(buckets, labels)

    assert matrix[0].tolist() == [2, 1, 0, 0]
    assert matrix[1].tolist() == [0, 0, 1, 3]
