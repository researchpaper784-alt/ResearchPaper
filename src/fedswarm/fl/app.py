"""fedswarm.fl.app -- the entire Phase 3 Flower FL harness in one file.

Consolidates what were four separate modules (task.py, checkpoint.py, client_app.py,
server_app.py) into one, for a single file that is easy to read start-to-finish and
easy to hand to Colab. Launched exactly the same way either way -- via `flwr run .`,
reading `[tool.flwr.app.components]` in `pyproject.toml`, which now points both
component strings at this one file (`fedswarm.fl.app:client_app` /
`fedswarm.fl.app:server_app`).

**Runs on CPU with zero code changes.** Every device lookup below is
`torch.device("cuda" if torch.cuda.is_available() else "cpu")` -- on a Colab runtime
with no GPU attached (Runtime > Change runtime type > CPU), `torch.cuda.is_available()`
is simply `False` everywhere and training/evaluation runs on CPU. `ray` (the simulation
backend `flwr run` uses) needs no GPU either; it only requests one from client_resources
if you ask it to. Nothing here does. The only cost of CPU-only is wall-clock, and this
harness's default smoke config (2 clients, 2 rounds, SimpleCNN@112) is intentionally
small for exactly this reason.

Verified against the installed `flwr==1.36.0` Message API (not the dead
`fl.client.NumPyClient` / `fl.server.strategy` API -- see `docs/FLOWER_API_NOTES.md`).
Deliberately NOT using `flwr.simulation.run_simulation()` even though it would let this
run without a repo clone or `flwr run` at all: Flower's own source marks it deprecated
("please use `flwr run` in the CLI instead... will be removed in a future version"),
and this project's own rule (`CLAUDE.md`) is to never knowingly write against an API
that is on its way out.

⚠️ Not run end-to-end on this machine: `flwr[simulation]` has no wheel for Intel macOS
(`docs/FLOWER_API_NOTES.md`), so there is no local `Grid`/simulation backend to route a
real `Message` through the Flower runtime here. What *is* verified locally: every
handler is a plain, undecorated-in-effect function (`ClientApp.train()`/`.evaluate()`
and `ServerApp.main()`'s decorators register and return the function unmodified,
confirmed by reading their source) -- `tests/test_fl_app.py` calls them directly with
hand-built `Message`/`Context`/`ArrayRecord` objects, exercising every line of this
file's own logic, minus the runtime's own message routing. The first real `flwr run .`
(on Colab, CPU or GPU) is what confirms that, not anything this file can self-certify.

⚠️ **2026-09-16, from that first real run**: `flwr run` does not execute this file from
the cloned repo at all. Its own log makes this explicit -- `Successfully installed
fedswarm to /root/.flwr/apps/fedswarm.fedswarm.0.0.1.<hash>` -- the Simulation Runtime
packages the app and re-installs it into an isolated location, then runs client/server
code from *there*, in a separate `uv sync`-built environment. That installed copy has
no `data/` or `results/` directory at all (neither is part of the Python package --
both are gitignored, outside `[tool.hatch.build.targets.wheel]`'s scope). Every
relative `run_config` path (`cache-dir`, `manifest-path`, `output-dir`, ...) was
silently resolving against whatever that isolated process's cwd happened to be, not
the repo -- consistent with what the first run actually did: no visible errors, ~12
minutes elapsed (plausibly rebuilding the whole cache from scratch in the wrong place),
and no `results/fl/*.json` ever appearing under the actual clone. Fixed below with the
same pattern this file already used for the *raw dataset* root (`FEDSWARM_DATA_ROOT`,
`fedswarm.data.download.resolve_root`) -- an env var, not a relative path, anchoring
every data/results path to the real clone regardless of where the installed copy's
code actually executes from. Set once, in the Colab cell that calls `flwr run`:

    import os
    os.environ["FEDSWARM_REPO_ROOT"] = "/content/ResearchPaper"
    !flwr run . --stream

Unset (the common case: local pytest, `scripts/run_experiment.py`), every path below
resolves exactly as it always did -- relative to the current working directory, which
*is* the repo root for every other entry point this project has. Not yet re-verified
against a real Colab run; the next one is what actually confirms this fix.
"""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from fedswarm.data.datasets import ManifestDataset, load_cache, load_manifest, select
from fedswarm.data.download import CLASSES
from fedswarm.data.partition import PartitionSpec, load_or_build
from fedswarm.data.splits import MANIFEST_CSV
from fedswarm.data.transforms import build_transforms
from fedswarm.eval.evaluator import evaluate as evaluate_model
from fedswarm.models.factory import build_model
from fedswarm.strategies.factory import strategy_from_run_config
from fedswarm.utils.results import make_run_id, write_result
from fedswarm.utils.seed import seed_everything

RunConfig = dict  # Context.run_config's real type: dict[str, bool | float | int | str]


def _device() -> torch.device:
    """Shared by both the client and server handlers below -- CPU whenever no CUDA
    device is visible, which is the whole point of this file on an exhausted-GPU-quota
    Colab session."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _repo_path(path_str: str) -> Path:
    """Anchors a `run_config` path string to the real repo clone, not to whatever
    directory `flwr run` actually executes this file's installed copy from (see the
    module docstring's 2026-09-16 note) -- and not to an assumption about `__file__`'s
    own location either, since the installed copy lives somewhere with no `data/` or
    `results/` directory next to it at all.

    An already-absolute path passes through unchanged. A relative one resolves
    against `$FEDSWARM_REPO_ROOT` if set, else the current working directory --
    which *is* the repo root for every other entry point this project has
    (`scripts/run_experiment.py`, `pytest`), so this is a no-op change for all of
    those; only `flwr run`'s isolated execution needs the env var set at all.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    root = Path(os.environ.get("FEDSWARM_REPO_ROOT", "."))
    return root / path


# ======================================================================================
# SECTION 1 -- task: model / data / train / eval.
#
# Strategy-agnostic by construction (plan Step 3.1): holds no Flower types (no
# ArrayRecord/Message/Context), so every function here is plain, directly
# unit-testable PyTorch. Section 3 (ClientApp) is the thin adapter that translates
# Flower's Message/Context objects into calls here and back.
# ======================================================================================


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
    cache_dir = _repo_path(str(run_config.get("cache-dir", "data/processed/cache")))
    manifest_path = str(_repo_path(str(run_config.get("manifest-path", str(MANIFEST_CSV)))))
    num_clients = int(run_config.get("num-clients", 20))
    batch_size = int(run_config.get("local-batch-size", 32))
    normalization = str(run_config.get("normalization", "dataset"))

    manifest = load_manifest(manifest_path)
    images = load_cache(size, cache_dir)
    spec = partition_spec_from_run_config(run_config, num_clients)
    partition_cache_dir = _repo_path(str(run_config.get("partition-cache-dir", "data/processed/partitions")))
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
    correction: "dict[str, torch.Tensor] | None" = None,
    label_flip: bool = False,
    num_classes: int = 4,
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
    weights the client received (snapshotted before any local step) -- wired to the
    real installed `FedProx` strategy's `"proximal-mu"` config key in `train_handler`
    (Phase 5, `strategies/factory.py`).

    `correction`, keyed by `model.named_parameters()` name, is SCAFFOLD's `c - c_i`
    control-variate correction (Phase 5, `strategies/scaffold.py`): added as a linear
    term `sum_p <p, correction[name]>` to the loss, whose gradient w.r.t. each
    parameter is exactly `correction[name]` itself -- the standard way to inject a
    constant additive term into a gradient via autograd rather than hand-editing
    `param.grad` after `loss.backward()`.

    `label_flip` (Phase 8, R1) trains against `(label + 1) % num_classes` for every
    batch -- a designated-attacker client's own local labels, permuted, not a random
    per-round corruption, so the same clients are corrupted for the whole run
    (`train_handler` decides which clients via `attacker-fraction`).
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
            if label_flip:
                labels = (labels + 1) % num_classes
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), labels)
            if mu > 0:
                prox = sum(
                    (p - g).pow(2).sum() for p, g in zip(model.parameters(), global_params)
                )
                loss = loss + (mu / 2.0) * prox
            if correction is not None:
                loss = loss + sum(
                    (p * correction[name].to(device)).sum()
                    for name, p in model.named_parameters()
                )
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


def is_attacker_client(partition_id: int, num_partitions: int, attacker_fraction: float) -> bool:
    """Phase 8, R1/R2: a FIXED subset of clients is corrupted for the whole run, not
    re-chosen each round -- the first `round(attacker_fraction * num_partitions)`
    partition-ids, a deterministic function of identity alone, so this project's
    attack-fraction sweeps (10%/20%/30%) are reproducible without any extra state."""
    return partition_id < round(attacker_fraction * num_partitions)


def corrupt_update(
    trained_state: "OrderedDict[str, torch.Tensor]",
    global_state: "OrderedDict[str, torch.Tensor]",
    mode: str,
    sigma: float,
) -> "OrderedDict[str, torch.Tensor]":
    """Phase 8, R1/R2 -- corrupts an attacker client's reported update *after*
    training, simulating a Byzantine client rather than a training-time behavior.
    `mode="gaussian_noise"`: adds N(0, sigma^2) to every trained parameter.
    `mode="sign_flip"`: reports `2*global - trained`, i.e. the negated update
    `global - (trained - global)` -- the classic sign-flipping attack."""
    if mode == "gaussian_noise":
        return OrderedDict(
            (k, v + sigma * torch.randn_like(v)) for k, v in trained_state.items()
        )
    if mode == "sign_flip":
        return OrderedDict((k, 2 * global_state[k] - v) for k, v in trained_state.items())
    raise ValueError(f"Unknown attack mode: {mode!r}, expected 'gaussian_noise' or 'sign_flip'")


def apply_dp_noise(
    state: "OrderedDict[str, torch.Tensor]", sigma: float
) -> "OrderedDict[str, torch.Tensor]":
    """Phase 8, R4 -- Gaussian mechanism noise on EVERY client's update (an honest
    privacy behavior every client applies, unlike `corrupt_update`'s attacker-only
    corruption). `sigma=0.0` is a no-op returning `state` unchanged, not a copy --
    callers relying on this being a fresh object when sigma>0 should not assume the
    same when sigma==0."""
    if sigma <= 0.0:
        return state
    return OrderedDict((k, v + sigma * torch.randn_like(v)) for k, v in state.items())


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
    (Section 3's `evaluate_handler`), which is separate from the server's own
    centralized test-set evaluation (Section 4's `build_evaluate_fn`, which evaluates
    the pooled test set directly and needs none of this client-side bookkeeping)."""
    metrics = evaluate_model(model, loader, device)

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


def _global_split_loader(run_config: RunConfig, split: str) -> DataLoader:
    size = int(run_config.get("image-size", 112))
    cache_dir = _repo_path(str(run_config.get("cache-dir", "data/processed/cache")))
    manifest_path = str(_repo_path(str(run_config.get("manifest-path", str(MANIFEST_CSV)))))
    normalization = str(run_config.get("normalization", "dataset"))

    manifest = load_manifest(manifest_path)
    images = load_cache(size, cache_dir)
    indices = select(manifest, split=split, representatives_only=True)
    transform = build_transforms(train=False, size=size, scheme=normalization, cache_dir=cache_dir)
    dataset = ManifestDataset(manifest, images, indices, transform)
    return DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)


def global_test_loader(run_config: RunConfig) -> DataLoader:
    """The server-held, never-partitioned global test split (plan Step 1.4) -- the same
    manifest/cache/transform pipeline `scripts/run_experiment.py` uses for centralized
    training, so the FL ceiling and the centralized ceiling are measured identically."""
    return _global_split_loader(run_config, "test")


def global_val_loader(run_config: RunConfig) -> DataLoader:
    """The server-held global **val** split (`data/splits.py`'s three-way train/val/
    test, distinct from each client's own `local-val-fraction` carved out of its own
    partition) -- for `server_val`-style fitness (plan §4.4) and FedLAW (Phase 5,
    `strategies/fedlaw.py`), both of which need weights *chosen* against a set the
    final reported metric (always the **test** split) was never touched by."""
    return _global_split_loader(run_config, "val")


# ======================================================================================
# SECTION 2 -- checkpoint/resume for one FL run.
#
# `Strategy.start()` has no `start_round` parameter (verified, `docs/FLOWER_API_NOTES.
# md`) -- resuming means calling it again with the checkpointed weights as
# `initial_arrays` and `num_rounds` reduced by however many rounds already completed.
# Section 4 does that remapping; this section only persists and restores the
# (round, rounds_log, state_dict) triple. Saves after every round, not just at the
# end, so a lost Colab runtime mid-sweep costs at most the round in flight.
# ======================================================================================


def checkpoint_paths(checkpoint_dir: Path | str, run_id: str) -> tuple[Path, Path]:
    directory = Path(checkpoint_dir)
    return directory / f"{run_id}.json", directory / f"{run_id}.pt"


def save_checkpoint(
    checkpoint_dir: Path | str,
    run_id: str,
    round_number: int,
    rounds_log: list[dict],
    state_dict: dict,
) -> None:
    json_path, weights_path = checkpoint_paths(checkpoint_dir, run_id)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"round": round_number, "rounds_log": rounds_log}, default=str))
    torch.save(state_dict, weights_path)


def load_checkpoint(checkpoint_dir: Path | str, run_id: str) -> tuple[int, list[dict], dict] | None:
    """Returns (last_completed_round, rounds_log_so_far, state_dict), or None if no
    checkpoint exists for this run_id yet (the common case: first attempt at a run)."""
    json_path, weights_path = checkpoint_paths(checkpoint_dir, run_id)
    if not json_path.exists() or not weights_path.exists():
        return None
    payload = json.loads(json_path.read_text())
    state_dict = torch.load(weights_path, map_location="cpu")
    return payload["round"], payload["rounds_log"], state_dict


def clear_checkpoint(checkpoint_dir: Path | str, run_id: str) -> None:
    """Called once a run's final result is written -- a lingering checkpoint file
    always means "this run_id did not finish," so a completed run must not leave one
    behind for a later invocation to mistakenly resume from."""
    json_path, weights_path = checkpoint_paths(checkpoint_dir, run_id)
    json_path.unlink(missing_ok=True)
    weights_path.unlink(missing_ok=True)


# ======================================================================================
# SECTION 3 -- the ClientApp.
#
# `@client_app.train()`/`@client_app.evaluate()` register and return the function
# unmodified (confirmed by reading the decorator source) -- named `train_handler`/
# `evaluate_handler` here, not `train`/`evaluate`, only because this file also imports
# an `evaluate` function (Section 1's `evaluate_model`, aliased on import for exactly
# this reason) and merging four originally-separate modules into one namespace means
# that collision has to be resolved somewhere; Flower does not care what the Python
# name is, only the decorator matters for message routing.
# ======================================================================================

client_app = ClientApp()


def _partition_id(context: Context) -> int:
    return int(context.node_config["partition-id"])


# Prefix convention shared with strategies/scaffold.py: the global control variate
# `c` and a client's own control-variate delta `dc_i` are model-shaped, so both
# travel inside the same ArrayRecord as the model weights rather than needing a third
# top-level RecordDict key.
_SCAFFOLD_CONTROL_PREFIX = "scaffold_c/"
_SCAFFOLD_DELTA_PREFIX = "scaffold_dc/"


@client_app.train()
def train_handler(msg: Message, context: Context) -> Message:
    model = build_model_from_run_config(context.run_config)
    full_state = msg.content["arrays"].to_torch_state_dict()
    model_keys = list(model.state_dict().keys())
    model.load_state_dict(OrderedDict((k, full_state[k]) for k in model_keys))
    global_state_before_training = OrderedDict((k, model.state_dict()[k].clone()) for k in model_keys)

    train_loader, _ = load_client_data(_partition_id(context), context.run_config)
    config = msg.content["config"]

    # Phase 8, R1/R2/R4 -- robustness/stress tests. A FIXED subset of clients (by
    # partition-id, see is_attacker_client) simulates a Byzantine participant for
    # the whole run; dp-noise-sigma, separately, applies to every client regardless
    # of attacker status (an honest privacy mechanism, not an attack).
    num_partitions = int(context.node_config.get("num-partitions", 1))
    attacker_fraction = float(context.run_config.get("attacker-fraction", 0.0))
    attack_mode = str(context.run_config.get("attack-mode", "none"))
    attack_noise_sigma = float(context.run_config.get("attack-noise-sigma", 0.1))
    dp_noise_sigma = float(context.run_config.get("dp-noise-sigma", 0.0))
    is_attacker = attacker_fraction > 0.0 and is_attacker_client(
        _partition_id(context), num_partitions, attacker_fraction
    )

    # SCAFFOLD (strategies/scaffold.py): present only when that strategy is active.
    scaffold_keys = [k for k in full_state if k.startswith(_SCAFFOLD_CONTROL_PREFIX)]
    correction = None
    global_c: "OrderedDict[str, torch.Tensor] | None" = None
    local_c: "OrderedDict[str, torch.Tensor] | None" = None
    params_before: "OrderedDict[str, torch.Tensor] | None" = None
    if scaffold_keys:
        global_c = OrderedDict(
            (k[len(_SCAFFOLD_CONTROL_PREFIX) :], full_state[k]) for k in scaffold_keys
        )
        stored = context.state.get("scaffold_c_i")
        local_c = (
            stored.to_torch_state_dict()
            if stored is not None
            else OrderedDict((name, torch.zeros_like(t)) for name, t in global_c.items())
        )
        correction = OrderedDict((name, global_c[name] - local_c[name]) for name in global_c)
        params_before = OrderedDict((k, model.state_dict()[k].clone()) for k in global_c)

    metrics = local_train(
        model,
        train_loader,
        _device(),
        epochs=int(config.get("local-epochs", context.run_config.get("local-epochs", 2))),
        lr=float(config.get("lr", context.run_config.get("local-lr", 0.01))),
        # "proximal-mu" is the installed flwr.serverapp.strategy.FedProx's real,
        # verified config key (its configure_train injects `config["proximal-mu"] =
        # self.proximal_mu` -- read directly from
        # flwr/serverapp/strategy/fedprox.py). "mu" is kept as a second fallback only
        # for this project's own pre-Phase-5 tests/configs that predate wiring the
        # real built-in FedProx; a bare static run_config default of 0.0 if neither
        # is set (plain FedAvg).
        mu=float(config.get("proximal-mu", config.get("mu", context.run_config.get("mu", 0.0)))),
        correction=correction,
        label_flip=is_attacker and attack_mode == "label_flip",
        num_classes=int(context.run_config.get("num-classes", 4)),
    )
    metrics["client_id"] = _partition_id(context)
    metrics["server_round"] = int(config.get("server_round", -1))
    metrics["is_attacker"] = int(is_attacker)

    trained_state = OrderedDict((k, model.state_dict()[k]) for k in model_keys)
    if is_attacker and attack_mode in ("gaussian_noise", "sign_flip"):
        trained_state = corrupt_update(
            trained_state, global_state_before_training, attack_mode, attack_noise_sigma
        )
    trained_state = apply_dp_noise(trained_state, dp_noise_sigma)

    arrays_out = ArrayRecord(trained_state)
    if global_c is not None and local_c is not None and params_before is not None:
        lr = float(config.get("lr", context.run_config.get("local-lr", 0.01)))
        tau = max(metrics["num_batches"], 1)
        new_local_c = OrderedDict(
            (
                name,
                local_c[name]
                - global_c[name]
                + (params_before[name] - model.state_dict()[name]) / (tau * lr),
            )
            for name in global_c
        )
        context.state["scaffold_c_i"] = ArrayRecord(new_local_c)
        for name in global_c:
            delta = (new_local_c[name] - local_c[name]).numpy()
            arrays_out[f"{_SCAFFOLD_DELTA_PREFIX}{name}"] = Array(delta)

    content = RecordDict(
        {
            "arrays": arrays_out,
            "metrics": MetricRecord(metrics),
        }
    )
    return Message(content, reply_to=msg)


@client_app.evaluate()
def evaluate_handler(msg: Message, context: Context) -> Message:
    model = build_model_from_run_config(context.run_config)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    _, val_loader = load_client_data(_partition_id(context), context.run_config)
    result = local_evaluate(model, val_loader, _device())

    # MetricRecord values are scalars (bool | int | float | str, or lists of those) --
    # verified via ConfigRecord/MetricRecord's constructor signature, which is why the
    # per-class confusion counts are flattened into individually-keyed scalars here
    # rather than nested as `{"per_class_counts": {...}}`.
    flat_metrics: dict = {
        "num-examples": result["num-examples"],
        "loss": result["loss"],
        "accuracy": result["accuracy"],
        "client_id": _partition_id(context),
    }
    for cls, counts in result["per_class_counts"].items():
        for key, value in counts.items():
            flat_metrics[f"{cls}_{key}"] = value

    content = RecordDict({"metrics": MetricRecord(flat_metrics)})
    return Message(content, reply_to=msg)


# ======================================================================================
# SECTION 4 -- the ServerApp.
#
# `@server_app.main()` drives whichever `Strategy` `strategy-name` selects (Phase 5,
# `strategies/factory.py::strategy_from_run_config` -- FedAvg is only the default),
# evaluates the global model on the held-out **global** test set every round (never
# partitioned across clients -- plan Step 1.4), and writes one result JSON per run in
# the Phase 6 result-contract
# schema (`utils/results.write_result`), the same schema Phase 2's centralized runs
# already use.
# ======================================================================================

server_app = ServerApp()


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
    checkpoint_dir = _repo_path(str(run_config.get("checkpoint-dir", "results/fl/_checkpoints")))
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


def merge_train_metrics(
    rounds_log: list[dict], train_metrics_clientapp: dict[int, MetricRecord], round_offset: int
) -> None:
    """`Strategy.start()` only returns its `Result` at the very end (verified,
    `docs/FLOWER_API_NOTES.md`), so every round's own `aggregate_train` diagnostics --
    FedACO's `alpha`/`alpha_entropy`/`fallback_used`/`aco_time_ms`/`pheromone_entropy`
    (plan §6.1's schema), or whatever a simpler baseline's `train_metrics_aggr_fn`
    returns -- can only be merged into `rounds_log` after the fact, not from inside
    `build_evaluate_fn`'s closure (which never sees `Result` at all). Mutates
    `rounds_log` in place. `result.train_metrics_clientapp` is keyed by `strategy.
    start()`'s own internal round numbering (always renumbered from 1 on a fresh
    call), remapped the same way `build_evaluate_fn` remaps `server_round` ->
    `true_round`. A `server_round` with no matching `rounds_log` entry (shouldn't
    happen; defensive only) is silently skipped rather than raising."""
    rounds_by_number = {entry["round"]: entry for entry in rounds_log}
    for server_round, metrics in train_metrics_clientapp.items():
        entry = rounds_by_number.get(server_round + round_offset)
        if entry is not None:
            entry["train_metrics"] = dict(metrics)


@server_app.main()
def main(grid: Grid, context: Context) -> None:
    run_config = context.run_config
    seed = int(run_config.get("seed", 0))
    deterministic = bool(run_config.get("deterministic", True))
    seed_everything(seed, deterministic=deterministic)

    strategy_name = str(run_config.get("strategy-name", "fedavg")).lower()
    resolved_config = {"run_config": dict(run_config), "strategy": strategy_name, "seed": seed}
    run_id = make_run_id(resolved_config, seed)
    run_config = {**run_config, "_run_id": run_id}  # threaded through to build_evaluate_fn

    num_rounds = int(run_config.get("num-rounds", 2))
    checkpoint_dir = _repo_path(str(run_config.get("checkpoint-dir", "results/fl/_checkpoints")))

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
    output_dir = _repo_path(str(run_config.get("output-dir", "results/fl")))

    if remaining_rounds == 0:
        print(f"{run_id} already completed {num_rounds} rounds -- nothing to do.")
        return

    device = _device()
    needs_val_loader = strategy_name == "fedlaw" or (
        strategy_name == "fedaco" and str(run_config.get("fedaco-fitness-mode", "data_free")) == "server_val"
    )
    val_loader = global_val_loader(run_config) if needs_val_loader else None
    strategy = strategy_from_run_config(run_config, model=model, val_loader=val_loader, device=device)
    train_config = ConfigRecord(
        {
            "local-epochs": int(run_config.get("local-epochs", 2)),
            "lr": float(run_config.get("local-lr", 0.01)),
            # Plain FedAvg/most baselines never read this; FedProx overwrites it with
            # its own "proximal-mu" key in its own configure_train (see train_handler's
            # comment) -- kept here only as the pre-Phase-5 static fallback.
            "mu": float(run_config.get("mu", 0.0)),
        }
    )
    evaluate_fn = build_evaluate_fn(run_config, rounds_log, round_offset)

    t_start = time.time()
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=remaining_rounds,
        train_config=train_config,
        evaluate_fn=evaluate_fn,
    )
    wall_clock_s = time.time() - t_start

    merge_train_metrics(rounds_log, result.train_metrics_clientapp, round_offset)

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


__all__ = [
    "client_app",
    "server_app",
    "main",
    "train_handler",
    "evaluate_handler",
    "build_evaluate_fn",
    "merge_train_metrics",
    "global_test_loader",
    "global_val_loader",
    "build_model_from_run_config",
    "load_client_data",
    "local_train",
    "local_evaluate",
    "per_class_confusion_counts",
    "is_attacker_client",
    "corrupt_update",
    "apply_dp_noise",
    "partition_spec_from_run_config",
    "checkpoint_paths",
    "save_checkpoint",
    "load_checkpoint",
    "clear_checkpoint",
]
