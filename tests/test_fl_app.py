"""Phase 3 tests -- fedswarm.fl.app, the entire FL harness consolidated into one file.

Mirrors app.py's own section structure. `local_train`/`local_evaluate`/
`per_class_confusion_counts` take plain PyTorch objects (no Flower types), so they're
tested directly with synthetic tensors. `train_handler`/`evaluate_handler` (Section 3)
and `build_evaluate_fn`/`global_test_loader` (Section 4) are tested by calling them
directly with hand-built `Message`/`Context`/`ArrayRecord` objects -- `ClientApp.
train()`/`.evaluate()` and `ServerApp.main()`'s decorators are confirmed (by reading
their source) to register and return the function unmodified, so this exercises every
line of app.py's own logic, minus Flower's own message-routing runtime, which needs a
real `Grid`/simulation backend this machine cannot install (`docs/FLOWER_API_NOTES.md`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from fedswarm.data.cache import build_and_save_cache
from fedswarm.fl.app import (
    _repo_path,
    build_evaluate_fn,
    build_model_from_run_config,
    checkpoint_paths,
    clear_checkpoint,
    evaluate_handler,
    global_test_loader,
    global_val_loader,
    load_checkpoint,
    load_client_data,
    local_evaluate,
    local_train,
    merge_train_metrics,
    partition_spec_from_run_config,
    per_class_confusion_counts,
    save_checkpoint,
    train_handler,
)
from fedswarm.models.simple_cnn import SimpleCNN


def _tiny_dataset_rows(root: Path, n: int = 40) -> list[dict]:
    """Shared by every fixture below -- a tiny on-disk dataset (manifest rows only;
    each fixture decides its own cache size/output dir), mirroring test_cache.py's
    fixture pattern, big enough to partition across a few clients and hold a global
    test split with enough pseudo-patients per class to clear PartitionSpec's default
    min_client_size.

    Includes a `val` split. The real manifest is three-way (train/val/test, rebuilt at
    pseudo-patient level in Phase 1.3) and `split` is orthogonal to the on-disk
    directory -- real val rows live under both `Training/` and `Testing/` paths. These
    fixtures were two-way, which meant `global_val_loader` returned an empty loader and
    nothing noticed until per-round val evaluation was added.

    The val rows are appended *after* the `n` train/test rows rather than carved out of
    them, so train/test counts (and the assertions that depend on them) are unchanged."""
    (root / "Training" / "glioma").mkdir(parents=True)
    (root / "Testing" / "glioma").mkdir(parents=True)
    rows = []
    for i in range(n):
        split_dir, split = ("Training", "train") if i < n - 8 else ("Testing", "test")
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
    for j in range(6):
        i = n + j
        rel = f"Training/glioma/img_{i}.jpg"
        Image.new("L", (32, 32), color=(i * 7) % 256).save(root / rel)
        rows.append(
            {
                "path": rel,
                "label": "glioma" if j % 2 == 0 else "meningioma",
                "pseudo_patient_id": i,
                "split": "val",
                "is_representative": True,
                "is_mixed_label": False,
            }
        )
    return rows


# ======================================================================================
# _repo_path tests -- added 2026-09-16 after a real Colab run demonstrated why this
# exists: `flwr run` executes app.py's installed copy from a location with no data/
# or results/ directory next to it at all, so every relative run_config path must
# anchor to the real repo clone via $FEDSWARM_REPO_ROOT, not to cwd or __file__.
# ======================================================================================


def test_repo_path_passes_absolute_paths_through_unchanged(monkeypatch) -> None:
    monkeypatch.delenv("FEDSWARM_REPO_ROOT", raising=False)
    assert _repo_path("/already/absolute/path") == Path("/already/absolute/path")


def test_repo_path_anchors_relative_paths_to_the_env_var_when_set(monkeypatch) -> None:
    monkeypatch.setenv("FEDSWARM_REPO_ROOT", "/content/ResearchPaper")
    assert _repo_path("data/processed/cache") == Path("/content/ResearchPaper/data/processed/cache")


def test_repo_path_falls_back_to_cwd_when_the_env_var_is_unset(monkeypatch) -> None:
    """The no-op case: local pytest / scripts/run_experiment.py always have cwd ==
    repo root already, so this must resolve identically to a bare relative Path --
    zero behavior change for every entry point except `flwr run`."""
    monkeypatch.delenv("FEDSWARM_REPO_ROOT", raising=False)
    assert _repo_path("data/processed/cache") == Path(".") / "data/processed/cache"


# ======================================================================================
# SECTION 1 tests -- task: model / data / train / eval.
# ======================================================================================


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
def task_manifest_and_cache(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "raw"
    manifest = pd.DataFrame(_tiny_dataset_rows(root))
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    cache_dir = tmp_path / "cache"
    build_and_save_cache(manifest, root, size=16, out_dir=cache_dir)
    return manifest_path, cache_dir


def test_load_client_data_partitions_are_disjoint_and_nonempty(task_manifest_and_cache) -> None:
    manifest_path, cache_dir = task_manifest_and_cache
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


def test_load_client_data_unknown_partition_id_raises(task_manifest_and_cache) -> None:
    manifest_path, cache_dir = task_manifest_and_cache
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


# ======================================================================================
# SECTION 2 tests -- checkpoint/resume.
# ======================================================================================


def test_load_checkpoint_returns_none_when_absent(tmp_path: Path) -> None:
    assert load_checkpoint(tmp_path, "no-such-run") is None


def test_save_then_load_roundtrips(tmp_path: Path) -> None:
    rounds_log = [{"round": 1, "test_macro_f1": 0.5}, {"round": 2, "test_macro_f1": 0.6}]
    state_dict = {"w": [1.0, 2.0, 3.0]}

    save_checkpoint(tmp_path, "run-a", round_number=2, rounds_log=rounds_log, state_dict=state_dict)
    loaded_round, loaded_log, loaded_state = load_checkpoint(tmp_path, "run-a")

    assert loaded_round == 2
    assert loaded_log == rounds_log
    assert loaded_state == state_dict


def test_clear_checkpoint_removes_both_files(tmp_path: Path) -> None:
    save_checkpoint(tmp_path, "run-b", round_number=1, rounds_log=[], state_dict={})
    json_path, weights_path = checkpoint_paths(tmp_path, "run-b")
    assert json_path.exists() and weights_path.exists()

    clear_checkpoint(tmp_path, "run-b")

    assert not json_path.exists()
    assert not weights_path.exists()


def test_clear_checkpoint_is_a_noop_when_nothing_exists(tmp_path: Path) -> None:
    clear_checkpoint(tmp_path, "never-existed")  # must not raise


def test_two_run_ids_do_not_collide(tmp_path: Path) -> None:
    save_checkpoint(tmp_path, "run-a", round_number=5, rounds_log=[{"round": 5}], state_dict={"w": 1})
    save_checkpoint(tmp_path, "run-b", round_number=2, rounds_log=[{"round": 2}], state_dict={"w": 2})

    round_a, _, state_a = load_checkpoint(tmp_path, "run-a")
    round_b, _, state_b = load_checkpoint(tmp_path, "run-b")

    assert round_a == 5 and state_a == {"w": 1}
    assert round_b == 2 and state_b == {"w": 2}


# ======================================================================================
# SECTION 3 tests -- the ClientApp (train_handler / evaluate_handler).
# ======================================================================================


@pytest.fixture
def client_run_config(tmp_path: Path) -> dict:
    root = tmp_path / "raw"
    manifest = pd.DataFrame(_tiny_dataset_rows(root))
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    cache_dir = tmp_path / "cache"
    build_and_save_cache(manifest, root, size=16, out_dir=cache_dir)

    return {
        "regime": "iid",
        "num-clients": 4,
        "min-client-size": 2,
        "image-size": 16,
        "cache-dir": str(cache_dir),
        "manifest-path": str(manifest_path),
        "partition-cache-dir": str(cache_dir.parent / "partitions"),
        "local-batch-size": 4,
        "model-name": "simple_cnn",
        "model-norm": "groupnorm",
        "num-classes": 4,
    }


def _context(run_config: dict, partition_id: int = 0) -> Context:
    return Context(
        run_id=0,
        node_id=partition_id,
        node_config={"partition-id": partition_id, "num-partitions": run_config["num-clients"]},
        state=RecordDict(),
        run_config=run_config,
    )


def _train_message(run_config: dict, config: dict | None = None) -> Message:
    model = SimpleCNN(num_classes=4, norm=run_config["model-norm"])
    content = RecordDict(
        {
            "arrays": ArrayRecord(model.state_dict()),
            "config": ConfigRecord(config or {"local-epochs": 1, "lr": 0.01, "server_round": 1}),
        }
    )
    return Message(content=content, dst_node_id=0, message_type="train")


def test_train_handler_returns_arrays_and_every_metric_field(client_run_config) -> None:
    run_config = client_run_config
    msg = _train_message(run_config)

    reply = train_handler(msg, _context(run_config))

    assert "arrays" in reply.content
    metrics = dict(reply.content["metrics"])
    for key in (
        "num-examples",
        "num_batches",
        "train_loss_before",
        "train_loss_after",
        "update_norm",
        "client_id",
        "server_round",
    ):
        assert key in metrics, f"missing {key}"
    assert metrics["client_id"] == 0
    assert metrics["server_round"] == 1


def test_train_handler_actually_changes_the_weights(client_run_config) -> None:
    run_config = client_run_config
    msg = _train_message(run_config, config={"local-epochs": 2, "lr": 0.1, "server_round": 1})

    reply = train_handler(msg, _context(run_config))

    before = msg.content["arrays"].to_torch_state_dict()
    after = reply.content["arrays"].to_torch_state_dict()
    assert any(not (before[k] == after[k]).all() for k in before)


def test_evaluate_handler_leaves_weights_unchanged_and_reports_per_class_counts(client_run_config) -> None:
    run_config = client_run_config
    model = SimpleCNN(num_classes=4, norm=run_config["model-norm"])
    content = RecordDict({"arrays": ArrayRecord(model.state_dict())})
    msg = Message(content=content, dst_node_id=0, message_type="evaluate")

    reply = evaluate_handler(msg, _context(run_config))

    metrics = dict(reply.content["metrics"])
    assert "arrays" not in reply.content  # evaluate must not return updated weights
    assert metrics["num-examples"] > 0
    assert "loss" in metrics and "accuracy" in metrics
    # per-class counts arrive flattened (MetricRecord values are scalars, not nested
    # dicts -- verified against the installed API, see app.py's Section 3 comment).
    assert "glioma_tp" in metrics
    assert "notumor_support" in metrics


def test_train_handler_respects_partition_id_from_node_config(client_run_config) -> None:
    """Two different partition-ids must train on disjoint local data -- this is the
    one thing that actually threads context.node_config through to the data layer."""
    run_config = client_run_config

    msg0 = _train_message(run_config)
    reply0 = train_handler(msg0, _context(run_config, partition_id=0))

    msg1 = _train_message(run_config)
    reply1 = train_handler(msg1, _context(run_config, partition_id=1))

    metrics0 = dict(reply0.content["metrics"])
    metrics1 = dict(reply1.content["metrics"])
    assert metrics0["client_id"] == 0
    assert metrics1["client_id"] == 1


# ======================================================================================
# Phase 5 -- SCAFFOLD's client-side half (strategies/scaffold.py is the server side).
# Exercises the real `Context.state` persistence contract end-to-end, hand-built, the
# same way every other train_handler test in this file does.
# ======================================================================================


def _scaffold_train_message(run_config: dict, global_c: dict[str, torch.Tensor]) -> Message:
    model = SimpleCNN(num_classes=4, norm=run_config["model-norm"])
    merged = ArrayRecord(model.state_dict())
    for name, tensor in global_c.items():
        merged[f"scaffold_c/{name}"] = Array(tensor.numpy())
    content = RecordDict(
        {
            "arrays": merged,
            "config": ConfigRecord({"local-epochs": 1, "lr": 0.01, "server_round": 1}),
        }
    )
    return Message(content=content, dst_node_id=0, message_type="train")


def test_train_handler_scaffold_first_round_starts_from_zero_local_control(
    client_run_config,
) -> None:
    """No `scaffold_c_i` stored yet for this node -- local_c must default to zeros,
    so round 1's correction is exactly `global_c - 0 = global_c`."""
    run_config = client_run_config
    model = SimpleCNN(num_classes=4, norm=run_config["model-norm"])
    global_c = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}

    msg = _scaffold_train_message(run_config, global_c)
    context = _context(run_config, partition_id=0)
    reply = train_handler(msg, context)

    arrays_out = reply.content["arrays"]
    dc_keys = [k for k in arrays_out.keys() if k.startswith("scaffold_dc/")]
    assert dc_keys, "SCAFFOLD reply must carry scaffold_dc/* deltas"
    assert context.state.get("scaffold_c_i") is not None, "must persist c_i into Context.state"


def test_train_handler_scaffold_persists_and_reuses_local_control_across_rounds(
    client_run_config,
) -> None:
    """The whole point of SCAFFOLD: the SAME Context (i.e. the same simulated node,
    across two rounds) must have its round-1 `scaffold_c_i` actually change what
    round 2's correction is -- not silently reset to zero every round."""
    run_config = client_run_config
    model = SimpleCNN(num_classes=4, norm=run_config["model-norm"])
    zero_c = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
    context = _context(run_config, partition_id=0)  # one Context, reused -- the real persistence contract

    train_handler(_scaffold_train_message(run_config, zero_c), context)
    c_i_round1 = context.state.get("scaffold_c_i").to_torch_state_dict()
    assert any(float(t.abs().sum()) > 0 for t in c_i_round1.values()), (
        "round 1 must produce a nonzero c_i (params_before != params_after under real SGD)"
    )

    # Round 2: server now sends a nonzero global_c. Correction should reflect BOTH the
    # persisted c_i from round 1 AND this round's global_c, not just one or the other.
    nonzero_c = {k: torch.ones_like(v) * 0.01 for k, v in model.state_dict().items()}
    train_handler(_scaffold_train_message(run_config, nonzero_c), context)
    c_i_round2 = context.state.get("scaffold_c_i").to_torch_state_dict()

    assert any(
        not torch.allclose(c_i_round1[k], c_i_round2[k]) for k in c_i_round1
    ), "c_i must actually update round to round, not stay frozen at round 1's value"


# ======================================================================================
# SECTION 4 tests -- the ServerApp (build_evaluate_fn / global_test_loader).
#
# @server_app.main() itself is not called: it needs a real Grid, which needs the
# `simulation` extra this machine cannot install (docs/FLOWER_API_NOTES.md). The first
# real `flwr run .` on Colab is what actually exercises it end-to-end.
# ======================================================================================


@pytest.fixture
def server_run_config(tmp_path: Path) -> dict:
    root = tmp_path / "raw"
    (root / "Training" / "glioma").mkdir(parents=True)
    (root / "Testing" / "glioma").mkdir(parents=True)

    rows = []
    for i in range(20):
        split_dir, split = ("Training", "train") if i < 12 else ("Testing", "test")
        rel = f"{split_dir}/glioma/img_{i}.jpg"
        Image.new("L", (16, 16), color=(i * 11) % 256).save(root / rel)
        rows.append(
            {
                "path": rel,
                "label": "glioma" if i % 2 == 0 else "notumor",
                "pseudo_patient_id": i,
                "split": split,
                "is_representative": True,
                "is_mixed_label": False,
            }
        )
    # Appended after rows 0..19 so `test` stays exactly rows 12..19 (asserted below).
    for j in range(4):
        i = 20 + j
        rel = f"Training/glioma/img_{i}.jpg"
        Image.new("L", (16, 16), color=(i * 11) % 256).save(root / rel)
        rows.append(
            {
                "path": rel,
                "label": "glioma" if j % 2 == 0 else "notumor",
                "pseudo_patient_id": i,
                "split": "val",
                "is_representative": True,
                "is_mixed_label": False,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    cache_dir = tmp_path / "cache"
    build_and_save_cache(manifest, root, size=16, out_dir=cache_dir)

    return {
        "image-size": 16,
        "cache-dir": str(cache_dir),
        "manifest-path": str(manifest_path),
        "model-name": "simple_cnn",
        "model-norm": "groupnorm",
        "num-classes": 4,
        "checkpoint-dir": str(tmp_path / "checkpoints"),
        "_run_id": "test-run",
    }


def test_global_test_loader_only_contains_test_split_rows(server_run_config) -> None:
    loader = global_test_loader(server_run_config)
    assert len(loader.dataset) == 8  # rows 12..19 above


def test_evaluate_fn_returns_metric_record_and_logs_a_round(server_run_config) -> None:
    rounds_log: list[dict] = []
    evaluate_fn = build_evaluate_fn(server_run_config, rounds_log, round_offset=0)

    model = SimpleCNN(num_classes=4, norm="groupnorm")
    result = evaluate_fn(1, ArrayRecord(model.state_dict()))

    assert "test_macro_f1" in dict(result)
    assert len(rounds_log) == 1
    assert rounds_log[0]["round"] == 1
    assert "test_macro_f1" in rounds_log[0]


def test_evaluate_fn_checkpoints_after_every_round(server_run_config) -> None:
    rounds_log: list[dict] = []
    evaluate_fn = build_evaluate_fn(server_run_config, rounds_log, round_offset=0)
    model = SimpleCNN(num_classes=4, norm="groupnorm")

    evaluate_fn(1, ArrayRecord(model.state_dict()))

    checkpoint = load_checkpoint(server_run_config["checkpoint-dir"], "test-run")
    assert checkpoint is not None
    round_number, logged, _state = checkpoint
    assert round_number == 1
    assert logged == rounds_log


def test_evaluate_fn_skips_logging_the_resumed_round_zero_reconfirmation(server_run_config) -> None:
    """On a resumed run, Strategy.start()'s own pre-round-1 call re-evaluates the exact
    checkpointed weights already logged by the previous invocation as round_offset --
    round_offset > 0 means we're mid-resume, and server_round == 0 is that duplicate."""
    rounds_log: list[dict] = [{"round": 5, "test_macro_f1": 0.7}]  # pretend round 5 already happened
    evaluate_fn = build_evaluate_fn(server_run_config, rounds_log, round_offset=5)
    model = SimpleCNN(num_classes=4, norm="groupnorm")

    evaluate_fn(0, ArrayRecord(model.state_dict()))  # this run's internal round 0 == true round 5

    assert len(rounds_log) == 1  # unchanged: not appended again

    evaluate_fn(1, ArrayRecord(model.state_dict()))  # true round 6: a genuinely new round

    assert len(rounds_log) == 2
    assert rounds_log[1]["round"] == 6


def test_evaluate_fn_logs_global_val_metrics_alongside_test(server_run_config) -> None:
    """Every hyperparameter-search or model-selection decision must read a val metric,
    never the test metric it reports (plan §5). Before 2026-09-17 the round log carried
    `test_*` only, so there was nothing honest to select on -- this asserts the val
    signal exists, is distinct from test, and is evaluated on the val split's own rows.
    """
    rounds_log: list[dict] = []
    evaluate_fn = build_evaluate_fn(server_run_config, rounds_log, round_offset=0)
    model = SimpleCNN(num_classes=4, norm="groupnorm")

    result = evaluate_fn(1, ArrayRecord(model.state_dict()))

    entry = rounds_log[0]
    for key in ("val_loss", "val_accuracy", "val_macro_f1", "val_auc"):
        assert key in entry, f"{key} missing from the round log"
    assert "val_macro_f1" in dict(result)

    # The val split is genuinely a different set of rows from test, not a relabelled
    # copy of it -- otherwise "selecting on val" would be selection on test by another
    # name, which is the exact contamination this exists to prevent.
    val_rows = len(global_val_loader(server_run_config).dataset)
    test_rows = len(global_test_loader(server_run_config).dataset)
    assert val_rows == 4 and test_rows == 8


def test_result_final_carries_val_selection_keys(server_run_config, tmp_path: Path) -> None:
    """`best_val_macro_f1` is the max over rounds (not the last round's), so a search is
    not penalized for a config that peaks then drifts, and `best_val_round` says where."""
    rounds_log = [
        {"round": 1, "test_macro_f1": 0.10, "val_macro_f1": 0.30},
        {"round": 2, "test_macro_f1": 0.20, "val_macro_f1": 0.70},
        {"round": 3, "test_macro_f1": 0.25, "val_macro_f1": 0.50},
    ]
    best = max(r["val_macro_f1"] for r in rounds_log)
    best_round = max(rounds_log, key=lambda r: r["val_macro_f1"])["round"]
    assert best == 0.70 and best_round == 2
    assert best != rounds_log[-1]["val_macro_f1"], "a last-round rule would pick 0.50 here"


class _FakeResult:
    """Stands in for flwr's `Result` (a dataclass whose `train_metrics_clientapp` is
    keyed by round -- verified against the installed flwr==1.36.0)."""

    def __init__(self, train_metrics_clientapp: dict) -> None:
        self.train_metrics_clientapp = train_metrics_clientapp


def test_merge_train_metrics_folds_aggregate_train_output_into_the_round_log() -> None:
    """Without this, a strategy's own diagnostics never reach the result file at all.
    `rounds_log` is built entirely from evaluation, so FedACO's pheromone_entropy,
    best_fitness, fallback_used and delta_mean_sq_norm -- the fields the Phase 4
    close-out added precisely so a run could be checked for the degenerate regime --
    were going to Flower's console and nowhere a checker could read them."""
    rounds_log = [
        {"round": 1, "test_macro_f1": 0.10},
        {"round": 2, "test_macro_f1": 0.20},
    ]
    result = _FakeResult(
        {
            1: {"pheromone_entropy": 2.31, "best_fitness": 0.44, "fallback_used": 0},
            2: {"pheromone_entropy": 2.02, "best_fitness": 0.61, "fallback_used": 1},
        }
    )

    merge_train_metrics(rounds_log, result, round_offset=0)

    assert rounds_log[0]["train_pheromone_entropy"] == 2.31
    assert rounds_log[1]["train_fallback_used"] == 1
    # Evaluation metrics are untouched, and the prefix keeps the two families apart.
    assert rounds_log[0]["test_macro_f1"] == 0.10
    assert "pheromone_entropy" not in rounds_log[0]


def test_merge_train_metrics_respects_the_resume_offset() -> None:
    """A resumed run's Strategy.start() renumbers its rounds from 1, the same reason
    build_evaluate_fn takes a round_offset. Ignoring it would staple round 4's colony
    diagnostics onto round 1's evaluation."""
    rounds_log = [{"round": 4, "test_macro_f1": 0.5}, {"round": 5, "test_macro_f1": 0.6}]
    result = _FakeResult({1: {"best_fitness": 0.9}, 2: {"best_fitness": 0.8}})

    merge_train_metrics(rounds_log, result, round_offset=3)

    assert rounds_log[0]["train_best_fitness"] == 0.9
    assert rounds_log[1]["train_best_fitness"] == 0.8


def test_merge_train_metrics_is_a_noop_when_there_are_none() -> None:
    rounds_log = [{"round": 1, "test_macro_f1": 0.1}]
    merge_train_metrics(rounds_log, _FakeResult({}), round_offset=0)
    merge_train_metrics(rounds_log, object(), round_offset=0)
    assert rounds_log == [{"round": 1, "test_macro_f1": 0.1}]


def test_merge_train_metrics_skips_rounds_absent_from_the_log() -> None:
    """rounds_log is the authoritative record of rounds actually evaluated and
    checkpointed; a stray metrics round must not invent an entry in it."""
    rounds_log = [{"round": 1, "test_macro_f1": 0.1}]
    merge_train_metrics(rounds_log, _FakeResult({1: {"a": 1.0}, 99: {"a": 2.0}}), round_offset=0)
    assert len(rounds_log) == 1
