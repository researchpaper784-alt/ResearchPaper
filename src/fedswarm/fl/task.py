"""Phase 3.1 -- shared, strategy-agnostic model/data/train/eval logic for the Flower
ClientApp (and for the server's centralized test-set evaluation, which trains no model
of its own but needs the same `build_model_from_run_config`).

Deliberately holds nothing Flower-specific (no ArrayRecord/Message/Context) so it is
plain, directly unit-testable PyTorch. `client_app.py` is the thin adapter that
translates Flower's Message/Context objects into calls here and back -- the plan's Step
3.1 design goal ("the client reads its instructions from config, not by branching on
which strategy is active") means the *strategy-specific* term (currently just FedProx's
`mu`) is the only thing that varies per strategy, and it is a single `if mu > 0` here,
not a different client per strategy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from fedswarm.data.datasets import ManifestDataset, load_cache, load_manifest
from fedswarm.data.download import CLASSES
from fedswarm.data.partition import PartitionSpec, load_or_build
from fedswarm.data.splits import MANIFEST_CSV
from fedswarm.data.transforms import build_transforms
from fedswarm.eval.evaluator import evaluate
from fedswarm.models.factory import build_model

RunConfig = dict  # Context.run_config's real type: dict[str, bool | float | int | str]


def partition_spec_from_run_config(run_config: RunConfig, num_clients: int) -> PartitionSpec:
    """Rebuilds the `PartitionSpec` every client and the server need from `Context.
    run_config`'s flat primitives.

    Flower restricts `run_config` to `dict[str, bool | float | int | str]` (verified,
    `docs/FLOWER_API_NOTES.md`) -- no nested structures -- so `PartitionSpec`'s fields
    travel as individually-named flat keys rather than one nested dict, and only the
    key relevant to the configured `regime` is actually read.
    """
    regime = run_config["regime"]
    return PartitionSpec(
        regime=regime,
        num_clients=num_clients,
        seed=int(run_config.get("partition-seed", 0)),
        alpha=float(run_config["alpha"]) if regime == "dirichlet" else None,
        classes_per_client=(
            int(run_config["classes-per-client"]) if regime == "pathological" else None
        ),
        skew_sigma=float(run_config["skew-sigma"]) if regime == "quantity_skew" else None,
        min_client_size=int(run_config.get("min-client-size", 10)),
        local_val_fraction=float(run_config.get("local-val-fraction", 0.1)),
    )


def build_model_from_run_config(run_config: RunConfig) -> nn.Module:
    return build_model(
        name=str(run_config.get("model-name", "simple_cnn")),
        num_classes=int(run_config.get("num-classes", 4)),
        pretrained=bool(run_config.get("model-pretrained", False)),
        norm=str(run_config.get("model-norm", "groupnorm")),
    )


def load_client_data(partition_id: int, run_config: RunConfig) -> tuple[DataLoader, DataLoader]:
    """This client's local-train / local-val loaders, per the plan's Step 1.4 design:
    each client's data is a fixed subset of the cached, de-duplicated manifest, split
    90/10 into local-train and local-val (the fraction is itself part of the partition
    spec, cached alongside the assignment)."""
    size = int(run_config.get("image-size", 112))
    cache_dir = Path(str(run_config.get("cache-dir", "data/processed/cache")))
    manifest_path = str(run_config.get("manifest-path", str(MANIFEST_CSV)))
    num_clients = int(run_config.get("num-clients", 20))
    batch_size = int(run_config.get("local-batch-size", 32))
    normalization = str(run_config.get("normalization", "dataset"))

    manifest = load_manifest(manifest_path)
    images = load_cache(size, cache_dir)
    spec = partition_spec_from_run_config(run_config, num_clients)
    partition_cache_dir = Path(str(run_config.get("partition-cache-dir", "data/processed/partitions")))
    partition = load_or_build(manifest, spec, partition_cache_dir)

    if partition_id not in partition.client_train:
        raise KeyError(
            f"partition-id {partition_id} has no assignment in partition {spec.key()} "
            f"(num_clients={num_clients}) -- node_config/run_config disagree on K."
        )

    train_transform = build_transforms(train=True, size=size, scheme=normalization, cache_dir=cache_dir)
    eval_transform = build_transforms(train=False, size=size, scheme=normalization, cache_dir=cache_dir)

    train_ds = ManifestDataset(manifest, images, np.array(partition.client_train[partition_id]), train_transform)
    val_ds = ManifestDataset(manifest, images, np.array(partition.client_val[partition_id]), eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=min(128, max(1, len(val_ds))), shuffle=False, num_workers=0)
    return train_loader, val_loader


def local_train(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    mu: float = 0.0,
) -> dict:
    """Runs `epochs` of local SGD. Returns every scalar the plan's Step 3.1 says the
    server might need to weight this client (`num-examples`, `train_loss_before/after`,
    `update_norm`, `num_batches`) -- FedAvg only uses `num-examples`, but FedACO's
    heuristic (plan §4.2) and the loss-based/FedNolowe baseline (Phase 5) both need the
    others, and building them into every client now means no strategy added later has
    to touch this function again.

    `train_loss_before`/`train_loss_after` are the mean loss over the first and last
    local epoch respectively (cheap: reuses losses already computed during training,
    rather than paying for a dedicated extra forward pass purely to measure "before").

    `mu > 0` adds a FedProx proximal term `(mu/2) * ||w - w_global||^2` against the
    weights the client received (snapshotted before any local step) -- present now,
    unused until Phase 5 wires a FedProx strategy through the `mu` config key; see
    docs/OPEN_QUESTIONS.md for what Phase 5 still needs to add on the server side.
    """
    model.to(device)
    # Always snapshotted, not just when mu > 0: update_norm is a standard field every
    # strategy may weight by (FedACO's r_k heuristic, plan §4.2; loss-based baselines),
    # not a FedProx-only concern -- gating this behind mu would silently report 0.0 for
    # every non-FedProx round, which is wrong, not just unused.
    global_params = [p.detach().clone() for p in model.parameters()]

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()

    epoch_losses: list[float] = []
    num_batches = 0
    for _ in range(epochs):
        batch_losses = []
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), labels)
            if mu > 0:
                prox = sum(
                    (p - g).pow(2).sum() for p, g in zip(model.parameters(), global_params)
                )
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            num_batches += 1
        epoch_losses.append(sum(batch_losses) / max(1, len(batch_losses)))

    update_norm = float(
        torch.sqrt(sum((p.detach() - g).pow(2).sum() for p, g in zip(model.parameters(), global_params)))
    )

    return {
        "num-examples": len(loader.dataset),
        "num_batches": num_batches,
        "train_loss_before": epoch_losses[0] if epoch_losses else float("nan"),
        "train_loss_after": epoch_losses[-1] if epoch_losses else float("nan"),
        "update_norm": update_norm,
    }


def per_class_confusion_counts(labels: np.ndarray, predictions: np.ndarray) -> dict[str, dict[str, int]]:
    """Per-class TP/FP/FN/support, additive across clients (unlike a per-client macro-F1
    or recall rate). Exists because the plan explicitly flags naively averaging
    per-client macro-F1 as wrong under label skew: a client holding only 2 of 4 classes
    has an undefined 4-class macro-F1, but its per-class confusion counts are always
    well-defined and summing them across clients reconstructs the exact pooled
    confusion matrix, from which a correct global macro-F1 can be recomputed."""
    cm = confusion_matrix(labels, predictions, labels=range(len(CLASSES)))
    counts = {}
    for i, cls in enumerate(CLASSES):
        tp = int(cm[i, i])
        support = int(cm[i, :].sum())
        counts[cls] = {
            "tp": tp,
            "fn": support - tp,
            "fp": int(cm[:, i].sum()) - tp,
            "support": support,
        }
    return counts


def local_evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """The current (received, not locally-trained) global model evaluated against this
    client's local-val split -- used for Flower's round-level federated evaluation
    (`@app.evaluate()`), which is separate from the server's own centralized test-set
    evaluation (`server_app.py`'s `evaluate_fn`, which needs none of this client-side
    per-class bookkeeping since it evaluates the pooled test set directly)."""
    metrics = evaluate(model, loader, device)

    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            all_labels.append(labels.numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
    labels_arr = np.concatenate(all_labels) if all_labels else np.array([], dtype=int)
    preds_arr = np.concatenate(all_preds) if all_preds else np.array([], dtype=int)

    return {
        "num-examples": len(loader.dataset),
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "per_class_counts": per_class_confusion_counts(labels_arr, preds_arr),
    }


__all__ = [
    "build_model_from_run_config",
    "load_client_data",
    "local_evaluate",
    "local_train",
    "partition_spec_from_run_config",
    "per_class_confusion_counts",
]
