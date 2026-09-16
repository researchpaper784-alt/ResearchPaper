"""Phase 3.1 tests -- the Flower ClientApp's train/evaluate handlers, called directly.

`ClientApp.train()`/`.evaluate()` decorators register and return the function
unmodified (confirmed by reading their source, see client_app.py's module docstring),
so `train(msg, context)` and `evaluate(msg, context)` below exercise the exact code
Flower's runtime would call -- everything except the runtime's own message routing,
which needs a real `Grid`/simulation backend this machine cannot install
(`docs/FLOWER_API_NOTES.md`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from PIL import Image

from fedswarm.data.cache import build_and_save_cache
from fedswarm.fl.client_app import evaluate, train
from fedswarm.models.simple_cnn import SimpleCNN


@pytest.fixture
def fl_manifest_and_cache(tmp_path: Path) -> dict:
    """Same tiny on-disk fixture as test_fl_task.py, packaged as the run_config dict
    the handlers actually consume (context.run_config)."""
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


def test_train_handler_returns_arrays_and_every_metric_field(fl_manifest_and_cache) -> None:
    run_config = fl_manifest_and_cache
    msg = _train_message(run_config)

    reply = train(msg, _context(run_config))

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


def test_train_handler_actually_changes_the_weights(fl_manifest_and_cache) -> None:
    run_config = fl_manifest_and_cache
    msg = _train_message(run_config, config={"local-epochs": 2, "lr": 0.1, "server_round": 1})

    reply = train(msg, _context(run_config))

    before = msg.content["arrays"].to_torch_state_dict()
    after = reply.content["arrays"].to_torch_state_dict()
    assert any(not (before[k] == after[k]).all() for k in before)


def test_evaluate_handler_leaves_weights_unchanged_and_reports_per_class_counts(fl_manifest_and_cache) -> None:
    run_config = fl_manifest_and_cache
    model = SimpleCNN(num_classes=4, norm=run_config["model-norm"])
    content = RecordDict({"arrays": ArrayRecord(model.state_dict())})
    msg = Message(content=content, dst_node_id=0, message_type="evaluate")

    reply = evaluate(msg, _context(run_config))

    metrics = dict(reply.content["metrics"])
    assert "arrays" not in reply.content  # evaluate must not return updated weights
    assert metrics["num-examples"] > 0
    assert "loss" in metrics and "accuracy" in metrics
    # per-class counts arrive flattened (MetricRecord values are scalars, not nested
    # dicts -- verified against the installed API, see client_app.py's comment).
    assert "glioma_tp" in metrics
    assert "notumor_support" in metrics


def test_train_handler_respects_partition_id_from_node_config(fl_manifest_and_cache) -> None:
    """Two different partition-ids must train on disjoint local data -- this is the
    one thing that actually threads context.node_config through to the data layer."""
    run_config = fl_manifest_and_cache

    msg0 = _train_message(run_config)
    reply0 = train(msg0, _context(run_config, partition_id=0))

    msg1 = _train_message(run_config)
    reply1 = train(msg1, _context(run_config, partition_id=1))

    metrics0 = dict(reply0.content["metrics"])
    metrics1 = dict(reply1.content["metrics"])
    assert metrics0["client_id"] == 0
    assert metrics1["client_id"] == 1
