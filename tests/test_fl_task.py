"""Phase 3.1 tests -- fl/task.py's strategy-agnostic model/data/train/eval logic.

`local_train`/`local_evaluate`/`per_class_confusion_counts` take plain PyTorch objects
(no Flower types), so they're tested directly with synthetic tensors -- no manifest,
cache, or Flower runtime needed. `load_client_data`/`partition_spec_from_run_config`
need a real manifest+cache+partition on disk, built via the same tiny-dataset pattern
`test_cache.py` already uses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from fedswarm.data.cache import build_and_save_cache
from fedswarm.fl.task import (
    build_model_from_run_config,
    load_client_data,
    local_evaluate,
    local_train,
    partition_spec_from_run_config,
    per_class_confusion_counts,
)
from fedswarm.models.simple_cnn import SimpleCNN


def _synthetic_loader(n: int = 24, num_classes: int = 4, size: int = 16, batch_size: int = 8) -> DataLoader:
    rng = torch.Generator().manual_seed(0)
    images = torch.rand(n, 3, size, size, generator=rng)
    labels = torch.randint(0, num_classes, (n,), generator=rng)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=True)


def test_local_train_returns_every_field_the_plan_lists() -> None:
    model = SimpleCNN(num_classes=4, norm="groupnorm")
    loader = _synthetic_loader()

    metrics = local_train(model, loader, torch.device("cpu"), epochs=2, lr=0.01)

    for key in ("num-examples", "num_batches", "train_loss_before", "train_loss_after", "update_norm"):
        assert key in metrics, f"missing {key}"
    assert metrics["num-examples"] == len(loader.dataset)
    assert metrics["num_batches"] == 2 * len(loader)  # 2 epochs


def test_local_train_update_norm_is_nonzero_without_fedprox() -> None:
    """Regression guard: update_norm must reflect real parameter movement whether or
    not mu (FedProx) is active -- it is a general-purpose field (plan §4.2's r_k
    heuristic, loss-based baselines), not a FedProx-only diagnostic."""
    model = SimpleCNN(num_classes=4, norm="groupnorm")
    loader = _synthetic_loader()

    metrics = local_train(model, loader, torch.device("cpu"), epochs=1, lr=0.1, mu=0.0)

    assert metrics["update_norm"] > 0.0


def test_fedprox_term_pulls_update_norm_down() -> None:
    """A realistic proximal penalty should shrink the update relative to plain local
    SGD on the same data/init -- not a precise bound, just the qualitative direction
    FedProx is supposed to have. mu is deliberately modest: gradient descent on the
    quadratic penalty (mu/2)*||w - w_global||^2 is only stable for lr*mu < 2, and a
    too-large mu (tried 50 here first) overshoots past w_global and diverges instead
    of constraining -- the opposite of what this test checks, and not a FedProx
    property, just basic step-size instability."""
    loader = _synthetic_loader()

    torch.manual_seed(0)
    plain = SimpleCNN(num_classes=4, norm="groupnorm")
    metrics_plain = local_train(plain, loader, torch.device("cpu"), epochs=3, lr=0.1, mu=0.0)

    torch.manual_seed(0)
    prox = SimpleCNN(num_classes=4, norm="groupnorm")
    metrics_prox = local_train(prox, loader, torch.device("cpu"), epochs=3, lr=0.1, mu=1.0)

    assert metrics_prox["update_norm"] < metrics_plain["update_norm"]


def test_per_class_confusion_counts_sum_to_correct_pooled_metric() -> None:
    """The whole point of reporting counts instead of a per-client macro-F1 (plan Step
    3.1): summing two clients' counts must reconstruct the same confusion matrix as
    pooling their raw labels/predictions directly."""
    labels_a = np.array([0, 0, 1, 1])
    preds_a = np.array([0, 1, 1, 1])
    labels_b = np.array([2, 2, 3, 0])
    preds_b = np.array([2, 3, 3, 0])

    counts_a = per_class_confusion_counts(labels_a, preds_a)
    counts_b = per_class_confusion_counts(labels_b, preds_b)

    pooled_labels = np.concatenate([labels_a, labels_b])
    pooled_preds = np.concatenate([preds_a, preds_b])
    counts_pooled = per_class_confusion_counts(pooled_labels, pooled_preds)

    for cls in counts_pooled:
        for stat in ("tp", "fp", "fn", "support"):
            assert counts_a[cls][stat] + counts_b[cls][stat] == counts_pooled[cls][stat]


def test_per_class_confusion_counts_handle_a_class_absent_locally() -> None:
    """A client holding only 2 of 4 classes (plan's explicit example of why per-client
    macro-F1 is undefined) must still produce well-defined per-class counts."""
    labels = np.array([0, 0, 1, 1])
    preds = np.array([0, 1, 0, 1])

    counts = per_class_confusion_counts(labels, preds)

    assert set(counts) == {"glioma", "meningioma", "notumor", "pituitary"}
    assert counts["notumor"] == {"tp": 0, "fp": 0, "fn": 0, "support": 0}
    assert counts["pituitary"] == {"tp": 0, "fp": 0, "fn": 0, "support": 0}


def test_local_evaluate_shape() -> None:
    model = SimpleCNN(num_classes=4, norm="groupnorm")
    loader = _synthetic_loader()

    result = local_evaluate(model, loader, torch.device("cpu"))

    assert result["num-examples"] == len(loader.dataset)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert set(result["per_class_counts"]) == {"glioma", "meningioma", "notumor", "pituitary"}


def test_build_model_from_run_config_reads_flat_keys() -> None:
    model = build_model_from_run_config(
        {"model-name": "simple_cnn", "num-classes": 4, "model-pretrained": False, "model-norm": "batchnorm"}
    )
    assert isinstance(model, SimpleCNN)


def test_partition_spec_from_run_config_only_reads_the_relevant_regime_key() -> None:
    """Context.run_config is flat dict[str, primitive] (verified against the installed
    API) -- PartitionSpec's regime-specific fields (alpha, classes_per_client, ...)
    must come from individually-named keys, and only the one matching `regime` should
    actually be consulted (the others may be absent without error)."""
    spec = partition_spec_from_run_config({"regime": "dirichlet", "alpha": 0.3}, num_clients=10)
    assert spec.regime == "dirichlet"
    assert spec.alpha == 0.3
    assert spec.classes_per_client is None


@pytest.fixture
def fl_manifest_and_cache(tmp_path: Path) -> tuple[Path, Path]:
    """A tiny on-disk dataset (manifest + decoded cache) big enough to partition across
    a few clients and hold a global test split -- mirrors test_cache.py's fixture
    pattern, extended with train/test rows and enough pseudo-patients per class to
    clear PartitionSpec's default min_client_size."""
    root = tmp_path / "raw"
    (root / "Training" / "glioma").mkdir(parents=True)
    (root / "Testing" / "glioma").mkdir(parents=True)

    rows = []
    for i in range(40):
        split_dir, split = ("Training", "train") if i < 32 else ("Testing", "test")
        rel = f"{split_dir}/glioma/img_{i}.jpg"
        Image.new("L", (32, 32), color=(i * 7) % 256).save(root / rel)
        rows.append(
            {
                "path": rel,
                "label": "glioma" if i % 2 == 0 else "meningioma",
                "pseudo_patient_id": i,
                "split": split,
                "is_representative": True,
                "is_mixed_label": False,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    cache_dir = tmp_path / "cache"
    build_and_save_cache(manifest, root, size=16, out_dir=cache_dir)
    return manifest_path, cache_dir


def test_load_client_data_partitions_are_disjoint_and_nonempty(fl_manifest_and_cache) -> None:
    manifest_path, cache_dir = fl_manifest_and_cache
    run_config = {
        "regime": "iid",
        "num-clients": 4,
        "min-client-size": 2,
        "image-size": 16,
        "cache-dir": str(cache_dir),
        "manifest-path": str(manifest_path),
        "partition-cache-dir": str(cache_dir.parent / "partitions"),
        "local-batch-size": 4,
    }

    seen_train_indices: set[int] = set()
    for client_id in range(4):
        train_loader, val_loader = load_client_data(client_id, run_config)
        assert len(train_loader.dataset) > 0
        assert len(val_loader.dataset) > 0
        client_indices = set(train_loader.dataset.indices.tolist()) | set(val_loader.dataset.indices.tolist())
        assert seen_train_indices.isdisjoint(client_indices), f"client {client_id} overlaps another client"
        seen_train_indices |= client_indices


def test_load_client_data_unknown_partition_id_raises(fl_manifest_and_cache) -> None:
    manifest_path, cache_dir = fl_manifest_and_cache
    run_config = {
        "regime": "iid",
        "num-clients": 4,
        "min-client-size": 2,
        "image-size": 16,
        "cache-dir": str(cache_dir),
        "manifest-path": str(manifest_path),
        "partition-cache-dir": str(cache_dir.parent / "partitions"),
    }
    with pytest.raises(KeyError):
        load_client_data(99, run_config)
