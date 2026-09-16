"""Phase 3.2 -- the Flower ServerApp: drives `FedAvg` (verified `flwr.serverapp.
strategy.FedAvg`, not the dead `fl.server.strategy` API), evaluates the global model on
the held-out **global** test set every round (never partitioned across clients -- plan
Step 1.4), and writes one result JSON per run in the Phase 6 result-contract schema
(`utils/results.write_result`), the same schema Phase 2's centralized runs already use.

⚠️ Not run end-to-end on this machine, same caveat as `client_app.py`: no local `Grid`/
simulation backend (`docs/FLOWER_API_NOTES.md`). `build_evaluate_fn` and the checkpoint
round-remapping below are unit-tested directly (`tests/test_server_app.py`) since they
are plain functions Flower's runtime calls into, not Flower internals themselves --
`@app.main()`'s decorator (confirmed by reading its source) returns `main` unmodified,
but calling it needs a real `Grid`, which only `flwr run` on Colab/Kaggle can supply.
The first real run there is what confirms `strategy.start()`'s actual round-by-round
behavior, not anything this file can self-certify.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fedswarm.data.datasets import ManifestDataset, load_cache, load_manifest, select
from fedswarm.data.splits import MANIFEST_CSV
from fedswarm.data.transforms import build_transforms
from fedswarm.eval.evaluator import evaluate as evaluate_model
from fedswarm.fl.checkpoint import clear_checkpoint, load_checkpoint, save_checkpoint
from fedswarm.fl.task import RunConfig, build_model_from_run_config
from fedswarm.utils.results import make_run_id, write_result
from fedswarm.utils.seed import seed_everything

app = ServerApp()


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def global_test_loader(run_config: RunConfig) -> torch.utils.data.DataLoader:
    """The server-held, never-partitioned global test split (plan Step 1.4) -- the same
    manifest/cache/transform pipeline `scripts/run_experiment.py` uses for centralized
    training, so the FL ceiling and the centralized ceiling are measured identically."""
    size = int(run_config.get("image-size", 112))
    cache_dir = Path(str(run_config.get("cache-dir", "data/processed/cache")))
    manifest_path = str(run_config.get("manifest-path", str(MANIFEST_CSV)))
    normalization = str(run_config.get("normalization", "dataset"))

    manifest = load_manifest(manifest_path)
    images = load_cache(size, cache_dir)
    indices = select(manifest, split="test", representatives_only=True)
    transform = build_transforms(train=False, size=size, scheme=normalization, cache_dir=cache_dir)
    dataset = ManifestDataset(manifest, images, indices, transform)
    return torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)


def build_evaluate_fn(run_config: RunConfig, rounds_log: list[dict], round_offset: int):
    """Closes over `rounds_log` so every call appends to the same list the caller holds
    a reference to -- `Strategy.start()` calls `evaluate_fn` itself and only returns a
    `Result` at the very end, so this is how per-round data escapes to the code that
    writes the result file (and checkpoints after every round, not just at the end).

    `round_offset` maps this invocation's `server_round` (always renumbered from 0 by
    a fresh `strategy.start()` call) back to the true round count across a resumed run.
    On a resumed run, `strategy.start()`'s own pre-round-1 call with `server_round=0` is
    a re-evaluation of the exact checkpointed weights already logged as the last entry
    of the *previous* invocation -- skip logging (and re-checkpointing) that duplicate.
    """
    device = _device()
    test_loader = global_test_loader(run_config)
    checkpoint_dir = str(run_config.get("checkpoint-dir", "results/fl/_checkpoints"))
    run_id = str(run_config.get("_run_id", "unknown"))

    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        true_round = server_round + round_offset
        model = build_model_from_run_config(run_config).to(device)
        model.load_state_dict(arrays.to_torch_state_dict())
        metrics = evaluate_model(model, test_loader, device)

        if server_round == 0 and round_offset > 0:
            print(f"[round {true_round}] re-confirmed resumed checkpoint (not re-logged)")
            return MetricRecord({"test_macro_f1": metrics["macro_f1"]})

        rounds_log.append(
            {
                "round": true_round,
                "test_loss": metrics["loss"],
                "test_accuracy": metrics["accuracy"],
                "test_macro_f1": metrics["macro_f1"],
                "test_auc": metrics.get("auc_ovr_macro"),
                "per_class_recall": metrics["per_class_recall"],
            }
        )
        save_checkpoint(checkpoint_dir, run_id, true_round, rounds_log, model.state_dict())
        print(
            f"[round {true_round}] test_macro_f1={metrics['macro_f1']:.4f} "
            f"test_acc={metrics['accuracy']:.4f}"
        )
        return MetricRecord(
            {"test_macro_f1": metrics["macro_f1"], "test_accuracy": metrics["accuracy"]}
        )

    return evaluate_fn


@app.main()
def main(grid: Grid, context: Context) -> None:
    run_config = context.run_config
    seed = int(run_config.get("seed", 0))
    deterministic = bool(run_config.get("deterministic", True))
    seed_everything(seed, deterministic=deterministic)

    resolved_config = {"run_config": dict(run_config), "strategy": "fedavg", "seed": seed}
    run_id = make_run_id(resolved_config, seed)
    run_config = {**run_config, "_run_id": run_id}  # threaded through to build_evaluate_fn

    num_rounds = int(run_config.get("num-rounds", 2))
    checkpoint_dir = str(run_config.get("checkpoint-dir", "results/fl/_checkpoints"))

    model = build_model_from_run_config(run_config)
    initial_arrays = ArrayRecord(model.state_dict())
    rounds_log: list[dict] = []
    round_offset = 0

    resumed = load_checkpoint(checkpoint_dir, run_id)
    if resumed is not None:
        round_offset, rounds_log, state_dict = resumed
        initial_arrays = ArrayRecord(state_dict)
        print(f"Resuming {run_id} from round {round_offset} ({len(rounds_log)} rounds already logged)")

    remaining_rounds = max(0, num_rounds - round_offset)
    output_dir = Path(str(run_config.get("output-dir", "results/fl")))

    if remaining_rounds == 0:
        print(f"{run_id} already completed {num_rounds} rounds -- nothing to do.")
        return

    strategy = FedAvg(
        fraction_train=float(run_config.get("fraction-train", 1.0)),
        min_train_nodes=int(run_config.get("min-train-nodes", 2)),
        min_evaluate_nodes=int(run_config.get("min-evaluate-nodes", 2)),
        min_available_nodes=int(run_config.get("min-available-nodes", 2)),
    )
    train_config = ConfigRecord(
        {
            "local-epochs": int(run_config.get("local-epochs", 2)),
            "lr": float(run_config.get("local-lr", 0.01)),
            "mu": float(run_config.get("mu", 0.0)),
        }
    )
    evaluate_fn = build_evaluate_fn(run_config, rounds_log, round_offset)

    t_start = time.time()
    strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=remaining_rounds,
        train_config=train_config,
        evaluate_fn=evaluate_fn,
    )
    wall_clock_s = time.time() - t_start

    write_result(
        output_dir / f"{run_id}.json",
        config=resolved_config,
        rounds=rounds_log,
        final={
            "wall_clock_s": wall_clock_s,
            "num_rounds_completed": num_rounds,
            "final_test_macro_f1": rounds_log[-1]["test_macro_f1"] if rounds_log else None,
        },
        seed=seed,
    )
    clear_checkpoint(checkpoint_dir, run_id)
    print(f"Wrote {output_dir / f'{run_id}.json'}")


__all__ = ["app", "main", "build_evaluate_fn", "global_test_loader"]
