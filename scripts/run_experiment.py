"""Phase 2.1 — train and evaluate a single centralized model configuration (the
performance ceiling every FL method is measured against).

Correction to this file's own earlier claim: FL runs do **not** dispatch through this
script. `Strategy.start()` (Phase 3, `fl/server_app.py`) needs a real `Grid`, which only
Flower's own runtime supplies -- an FL experiment is launched with `flwr run .`, reading
`[tool.flwr.app.config]` in `pyproject.toml` (override via `flwr run . --run-config
"..."`), not via `--config` YAML layers. This script stays the single entry point for
centralized training only.

Run:
  .venv/bin/python scripts/run_experiment.py \
      --config configs/base.yaml configs/model/simple_cnn.yaml configs/experiment/centralized.yaml \
      --seed 0
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedswarm.data.datasets import ManifestDataset, load_cache, load_manifest, select  # noqa: E402
from fedswarm.data.transforms import build_transforms  # noqa: E402
from fedswarm.eval.evaluator import evaluate  # noqa: E402
from fedswarm.models.factory import build_model, count_parameters  # noqa: E402
from fedswarm.utils.config import config_hash, load_config  # noqa: E402
from fedswarm.utils.results import write_result  # noqa: E402
from fedswarm.utils.seed import seed_everything  # noqa: E402


def pick_device(deterministic: bool = True) -> torch.device:
    """CUDA when available; otherwise CPU, deliberately skipping MPS.

    Measured on this machine: torch.use_deterministic_algorithms(True) makes MPS fall
    back to CPU per-op for several kernels used here (GroupNorm, antialiased resize),
    which round-trips every such op across the MPS/CPU boundary. Result: 152s/epoch on
    MPS+deterministic vs 26s/epoch on MPS alone -- and CPU+deterministic (36s/epoch) beats
    MPS+deterministic outright, so there is no reason to pay the MPS tax while determinism
    is on. Not spending time root-causing further since Kaggle GPU (CUDA) is the actual
    training environment (see docs/OPEN_QUESTIONS.md); MPS stays available via
    `deterministic=False` for quick non-reproducible local exploration.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        not deterministic
        and getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def _worker_init_fn(worker_id: int) -> None:
    """Reseed the worker's Python/NumPy RNGs from the seed PyTorch already assigned it.

    DataLoader automatically gives each worker process its own torch RNG stream
    (base_seed + worker_id), which is all the augmentation pipeline actually needs --
    every randomized torchvision v1 transform used here (RandomResizedCrop,
    RandomHorizontalFlip, RandomRotation, ColorJitter) draws from torch's global
    generator, not Python's `random` or NumPy's. This reseeds those two anyway, at
    near-zero cost, as insurance against the well-documented gotcha where forked
    worker processes otherwise inherit and share the *parent's* random/NumPy state --
    identical across workers -- if anything (now or in a future edit) ever calls them.
    """
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def _num_workers(cfg) -> int:
    """Parallel data loading workers for the augmentation pipeline (CPU-bound: PIL/
    torchvision transforms per image, no GPU involved). With num_workers=0 this runs
    serially in the main process and the GPU sits idle between batches -- for a small
    model this CPU-bound augmentation cost dominates wall-clock, not the GPU step, which
    matters because Colab/Kaggle meter *wall-clock* GPU-session time against your quota
    regardless of GPU utilization.

    Explicit `training.num_workers` in config always wins. Otherwise: 0 unchanged on
    CPU (this machine's local runs and existing timings stay exactly as measured), a
    small nonzero default when CUDA is available, capped by actual core count.
    """
    configured = cfg.training.get("num_workers", None)
    if configured is not None:
        return int(configured)
    if not torch.cuda.is_available():
        return 0
    return max(0, min(4, os.cpu_count() or 0))


def build_loaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    manifest = load_manifest(cfg.data.manifest)
    images = load_cache(cfg.data.image_size, cfg.data.cache_dir)

    drop_mixed = cfg.training.get("drop_mixed_label", False)
    train_transform = build_transforms(
        train=True, size=cfg.data.image_size, scheme=cfg.data.normalization, cache_dir=cfg.data.cache_dir
    )
    eval_transform = build_transforms(
        train=False, size=cfg.data.image_size, scheme=cfg.data.normalization, cache_dir=cfg.data.cache_dir
    )

    splits = {}
    for split, transform in (("train", train_transform), ("val", eval_transform), ("test", eval_transform)):
        indices = select(manifest, split=split, representatives_only=True, drop_mixed_label=drop_mixed)
        splits[split] = ManifestDataset(manifest, images, indices, transform)

    num_workers = _num_workers(cfg)
    loader_kwargs = dict(
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    loaders = {
        "train": DataLoader(splits["train"], batch_size=cfg.training.batch_size, shuffle=True, **loader_kwargs),
        "val": DataLoader(splits["val"], batch_size=128, shuffle=False, **loader_kwargs),
        "test": DataLoader(splits["test"], batch_size=128, shuffle=False, **loader_kwargs),
    }
    class_counts = {split: splits[split].class_counts() for split in splits}
    return loaders["train"], loaders["val"], loaders["test"], class_counts


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        n += len(labels)
    return total_loss / n


def run(cfg, seed: int) -> dict:
    # `training.deterministic: false` is an explicit escape hatch, not the default --
    # torch.use_deterministic_algorithms(True) is known (see seed.py, and this repo's
    # own recorded MPS timings: 152s/epoch deterministic vs 26s/epoch not) to disable
    # fast cuDNN/MPS kernel paths for some ops, which can dominate epoch wall-clock far
    # more than anything data-loading-side. Override it to test that on a slow run
    # instead of guessing; leave it True for any run whose numbers go in the paper.
    want_deterministic = cfg.training.get("deterministic", True)
    deterministic = seed_everything(seed, deterministic=want_deterministic)
    device = pick_device(deterministic=deterministic)

    train_loader, val_loader, test_loader, class_counts = build_loaders(cfg)
    model = build_model(
        cfg.model.name, cfg.model.num_classes, cfg.model.pretrained, cfg.model.norm
    ).to(device)
    n_params = count_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.max_epochs)

    best_metric, best_state, best_epoch, patience_counter = -1.0, None, 0, 0
    rounds = []
    t_start = time.time()

    for epoch in range(1, cfg.training.max_epochs + 1):
        t_epoch = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        epoch_time_s = time.time() - t_epoch

        rounds.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
                "lr": scheduler.get_last_lr()[0],
                "epoch_time_s": epoch_time_s,
            }
        )

        current = val_metrics[cfg.training.early_stop_metric]
        if current > best_metric:
            best_metric, best_epoch, patience_counter = current, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        print(
            f"  epoch {epoch:>3}  train_loss={train_loss:.4f}  "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"({epoch_time_s:.1f}s)"
            f"{'  *' if epoch == best_epoch else ''}"
        )

        if patience_counter >= cfg.training.early_stop_patience:
            print(f"  early stop at epoch {epoch} (best was epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    wall_clock_s = time.time() - t_start

    final = {
        "best_epoch": best_epoch,
        "n_epochs_run": len(rounds),
        "n_params": n_params,
        "wall_clock_s": wall_clock_s,
        "device": str(device),
        "class_counts": class_counts,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }

    print(
        f"\n  best epoch {best_epoch}: test_macro_f1={test_metrics['macro_f1']:.4f} "
        f"test_accuracy={test_metrics['accuracy']:.4f}  ({wall_clock_s:.1f}s, {n_params:,} params)"
    )
    return {"rounds": rounds, "final": final}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", required=True, help="One or more YAML layers, merged in order")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--override", nargs="*", default=[], help="Dotlist overrides, e.g. model.norm=batchnorm")
    args = parser.parse_args()

    cfg = load_config(*args.config, overrides=[f"seed={args.seed}", *args.override])
    resolved = {
        "model": dict(cfg.model),
        "training": dict(cfg.training),
        "data": dict(cfg.data),
        "seed": args.seed,
    }

    print(f"config: {' + '.join(args.config)}  seed={args.seed}  hash={config_hash(cfg)}")
    print(f"model: {cfg.model.name}  pretrained={cfg.model.pretrained}  norm={cfg.model.norm}")

    outcome = run(cfg, args.seed)

    out_path = Path(cfg.output_dir) / f"{cfg.model.name}_{cfg.data.image_size}_{cfg.model.norm}_{args.seed}.json"
    write_result(
        out_path,
        config=resolved,
        rounds=outcome["rounds"],
        final=outcome["final"],
        seed=args.seed,
    )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
