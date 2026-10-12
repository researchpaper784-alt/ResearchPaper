"""Verifies the fix for the SCAFFOLD control-variate blow-up found and fixed 2026-10-11
(docs/OPEN_QUESTIONS.md, "ROOT CAUSE FOUND AND REPRODUCED" / "FIXED, confirmed over the full
100-round budget") entirely in-process -- no Ray, no `flwr run` subprocess, so it works even in
environments where the per-run isolated simulation env is broken.

Drives the real `Scaffold` strategy and the real `train_handler` through 100 rounds by hand,
the same way `tests/test_fl_app.py`'s and `tests/test_baselines.py`'s existing scaffold unit
tests build a single round's Message/Context, just looped with real non-trivial training data.

Prints the global model's parameter norm and the global control variate's norm every 10 rounds
(and every round near a change in behavior). Before the fix, this diverged from O(10) to
O(10^7) within 10 rounds and reached NaN by round ~16, under dirichlet(0.3), K=10,
min-client-size=5, local-batch-size=32 -- settings matching `configs/experiment/main_reduced.yaml`.
With the fix (the correction clip in `fl/app.py::train_handler`, plus the global_c decay and
clip in `strategies/scaffold.py::Scaffold`), `||w||` instead grows smoothly from ~68 to ~74
over the full 100 rounds -- ordinary training, not divergence.

Use this to confirm any further change to SCAFFOLD (a different clip/decay value, a different
mechanism entirely) still keeps `||w||` bounded before spending any Kaggle GPU-hours re-running
SCAFFOLD's arm of a real sweep -- a regression here would be silent otherwise, the same way the
original bug was.

Run: .venv/bin/python scripts/debug_scaffold_divergence.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, RecordDict
from PIL import Image

from fedswarm.data.cache import build_and_save_cache
from fedswarm.fl.app import train_handler
from fedswarm.models.simple_cnn import SimpleCNN
from fedswarm.strategies.scaffold import Scaffold

NUM_ROUNDS = 100
NUM_CLIENTS = 10


def _synthetic_rows(root: Path, n: int = 2000) -> list[dict]:
    """A fixture big enough that `build_partition` reliably clears `min-client-size=5` at
    `dirichlet(alpha=0.3)` across 10 clients -- smaller fixtures (hundreds of images) hit
    `RuntimeError: Could not produce N clients with >= min_client_size samples`, a real but
    unrelated partition-sizing failure that also makes every client's reply error (and so
    also freezes `arrays` via the exact same `aggregate_train` returns `None` path this script
    exists to study) -- worth knowing about if this script is adapted with a smaller `n`.
    """
    (root / "Training" / "glioma").mkdir(parents=True)
    (root / "Testing" / "glioma").mkdir(parents=True)
    rows = []
    for i in range(n):
        split_dir, split = ("Training", "train") if i < n - 16 else ("Testing", "test")
        rel = f"{split_dir}/glioma/img_{i}.jpg"
        Image.new("L", (32, 32), color=(i * 7) % 256).save(root / rel)
        rows.append({
            "path": rel, "label": "glioma" if i % 2 == 0 else "meningioma",
            "pseudo_patient_id": i, "split": split,
            "is_representative": True, "is_mixed_label": False,
        })
    for j in range(8):
        i = n + j
        rel = f"Training/glioma/img_{i}.jpg"
        Image.new("L", (32, 32), color=(i * 7) % 256).save(root / rel)
        rows.append({
            "path": rel, "label": "glioma" if j % 2 == 0 else "meningioma",
            "pseudo_patient_id": i, "split": "val",
            "is_representative": True, "is_mixed_label": False,
        })
    return rows


def main() -> None:
    tmp = Path("/tmp/scaffold_divergence_repro")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    root = tmp / "raw"
    manifest = pd.DataFrame(_synthetic_rows(root))
    manifest_path = tmp / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    cache_dir = tmp / "cache"
    build_and_save_cache(manifest, root, size=16, out_dir=cache_dir)

    # Matches configs/experiment/main_reduced.yaml's real settings except image-size (16, not
    # 112 -- keeps this fast; the real model is larger and plausibly diverges even faster).
    run_config = {
        "regime": "dirichlet",
        "alpha": 0.3,
        "num-clients": NUM_CLIENTS,
        "min-client-size": 5,
        "image-size": 16,
        "cache-dir": str(cache_dir),
        "manifest-path": str(manifest_path),
        "partition-cache-dir": str(cache_dir.parent / "partitions"),
        "local-batch-size": 32,
        "model-name": "simple_cnn",
        "model-norm": "groupnorm",
        "num-classes": 4,
        "local-epochs": 1,
        "local-lr": 0.01,
        "deterministic": True,
        "seed": 0,
    }

    strategy = Scaffold(
        fraction_train=1.0, min_train_nodes=NUM_CLIENTS,
        min_evaluate_nodes=NUM_CLIENTS, min_available_nodes=NUM_CLIENTS,
    )
    model = SimpleCNN(num_classes=4, norm=run_config["model-norm"])
    arrays = ArrayRecord(model.state_dict())

    class FakeGrid:
        def get_node_ids(self) -> list[int]:
            return list(range(NUM_CLIENTS))

    contexts = {
        i: Context(
            run_id=0, node_id=i,
            node_config={"partition-id": i, "num-partitions": NUM_CLIENTS},
            state=RecordDict(), run_config=run_config,
        )
        for i in range(NUM_CLIENTS)
    }

    grid = FakeGrid()
    for round_num in range(1, NUM_ROUNDS + 1):
        out_messages = list(strategy.configure_train(round_num, arrays, ConfigRecord(), grid))

        replies = []
        for msg in out_messages:
            node_id = msg.metadata.dst_node_id
            try:
                replies.append(train_handler(msg, contexts[node_id]))
            except Exception as exc:
                print(f"round {round_num}: !! train_handler RAISED for node {node_id}: "
                      f"{type(exc).__name__}: {exc}")

        agg_arrays, _ = strategy.aggregate_train(round_num, replies)
        if agg_arrays is None:
            print(f"round {round_num}: !! aggregate_train returned None "
                  f"({len(replies)}/{len(out_messages)} replies ok) -- arrays stay frozen")
            continue

        old_w = arrays.to_torch_state_dict()
        new_w = agg_arrays.to_torch_state_dict()
        changed = any(not torch.allclose(old_w[k], new_w[k]) for k in old_w)
        has_nan = any(torch.isnan(v).any() for v in new_w.values())
        w_norm = sum(float(v.float().norm()) for v in new_w.values())
        c_norm = sum(float(v.float().norm()) for v in strategy._global_c.values())
        if round_num % 10 == 0 or not changed or has_nan or round_num <= 3:
            print(f"round {round_num}: ok changed={changed} nan={has_nan} "
                  f"||w||={w_norm:.3f} ||global_c||={c_norm:.3f} "
                  f"replies={len(replies)}/{len(out_messages)}")
        arrays = agg_arrays


if __name__ == "__main__":
    main()
