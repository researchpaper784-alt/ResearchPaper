"""Phase 6, Step 6.2 -- sweep runner with resume (plan §6.2). Every FL run this repo
produces goes through `flwr run . --run-config "..."` (Flower's own CLI, needing the
`simulation` extra this dev machine cannot install -- `docs/FLOWER_API_NOTES.md`), so
this module is split deliberately: everything here is plain Python with zero Flower
runtime dependency and is fully unit-tested on this machine (config-grid expansion,
run_id prediction, resume-skip, lock handling, ordering, command-string construction);
only `execute_run`'s actual subprocess call needs a real Colab/Kaggle environment, and
even that takes an injectable `runner` so the orchestration loop (`run_sweep`) is
testable without ever invoking `flwr`. `scripts/run_sweep.py` is the thin CLI wrapper.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from fedswarm.utils.results import make_run_id

DEFAULT_LOCK_TIMEOUT_S = 3600.0


@dataclass(frozen=True)
class RunSpec:
    """One planned `flwr run`. `label` is a human-readable name (e.g.
    "fedaco/dirichlet_0.3/seed=2") used in logs and the manifest, not in the run_id
    itself (the run_id is derived purely from `overrides`, matching `fl/app.py`'s own
    hashing, so two RunSpecs with identical overrides but different labels correctly
    collide -- they *are* the same run)."""

    label: str
    group: str
    overrides: dict[str, Any] = field(default_factory=dict)


# ======================================================================================
# Config-grid expansion
# ======================================================================================


def resolve_entry_overrides(entry: dict[str, Any], base_dir: str | Path | None = None) -> dict[str, Any]:
    """An entry is either `{"name": ..., "overrides": {...}}` (inline) or
    `{"name": ..., "file": "configs/strategy/fedaco.yaml"}` (a reference to one of
    the standalone per-strategy/per-partition YAML files, resolved relative to
    `base_dir` -- typically the experiment YAML's own directory). Reusing those
    files here is what lets `configs/experiment/*.yaml` list strategies/partitions
    by name without re-typing every override inline in every experiment config."""
    if "overrides" in entry:
        return dict(entry["overrides"])
    path = Path(entry["file"])
    if base_dir is not None and not path.is_absolute():
        path = Path(base_dir) / path
    with open(path) as f:
        loaded = yaml.safe_load(f)
    return loaded or {}


def expand_grid(
    strategies: list[dict[str, Any]],
    partitions: list[dict[str, Any]],
    seeds: list[int],
    base_overrides: dict[str, Any] | None = None,
    group: str = "main",
    base_dir: str | Path | None = None,
) -> list[RunSpec]:
    """Full factorial: strategies x partitions x seeds (plan §6.3's primary grid).
    Each of `strategies`/`partitions` is `{"name": str, "overrides": dict}` or
    `{"name": str, "file": "configs/.../x.yaml"}` (see `resolve_entry_overrides`).
    Seeds vary slowest in the returned list's natural order (all strategies x
    partitions for seed[0], then seed[1], ...) -- `order_seed_first` relies on this,
    but callers that don't need seed-first ordering can just use this list directly.
    """
    runs = []
    for seed in seeds:
        for partition in partitions:
            for strategy in strategies:
                overrides = {
                    **(base_overrides or {}),
                    **resolve_entry_overrides(partition, base_dir),
                    **resolve_entry_overrides(strategy, base_dir),
                    "seed": seed,
                }
                label = f"{strategy['name']}/{partition['name']}/seed={seed}"
                runs.append(RunSpec(label=label, group=group, overrides=overrides))
    return runs


def granular_runs(config_path: str | Path, base_dir: str | Path) -> list[RunSpec]:
    """Expand a granular experiment config (strategies x partitions x seeds) into runs.

    The one expansion `scripts/run_sweep_granular.py` and the Kaggle notebooks share, so a
    notebook asking "which cells belong to this sweep?" gets the same answer the runner acts
    on. Before this existed each notebook re-derived it by hand, and the first attempt called
    a helper that lived only inside the script and omitted a required argument.
    """
    path = Path(config_path)
    spec = yaml.safe_load(path.read_text())
    return order_seed_first(
        expand_grid(
            strategies=spec["strategies"],
            partitions=spec["partitions"],
            seeds=spec["seeds"],
            base_overrides=spec.get("base_overrides", {}),
            group=path.stem,
            base_dir=Path(base_dir),
        )
    )


def planned_run_ids(runs: list[RunSpec], pyproject_path: str | Path) -> dict[str, str]:
    """{run_id: label} for every run, predicted exactly as `run_sweep` would name them.

    Side-effect free: `run_sweep(dry_run=True)` writes no manifest entry and takes no lock.

    The ids hash the resolved config INCLUDING pyproject's defaults, so they change when a
    default does -- which is the point when filtering results, and the trap when the gate's
    fitness fix has been applied in one session and not the next. Call this after
    `scripts/apply_gate_fix.py`, never before.
    """
    entries = run_sweep(runs, pyproject_path=str(pyproject_path), dry_run=True)
    return {e["run_id"]: e["label"] for e in entries}


def order_seed_first(runs: list[RunSpec]) -> list[RunSpec]:
    """Plan §6.2: "Order runs so that one complete seed of every configuration
    finishes first -- that way a partial sweep still yields a full (if noisy)
    table." Groups by each run's `overrides["seed"]`, preserving both the original
    within-seed order and the order seed values were first seen (not necessarily
    numeric ascending -- whatever order the config listed them in)."""
    seen_seed_order: list[Any] = []
    by_seed: dict[Any, list[RunSpec]] = {}
    for run in runs:
        seed = run.overrides.get("seed")
        if seed not in by_seed:
            by_seed[seed] = []
            seen_seed_order.append(seed)
        by_seed[seed].append(run)
    return [run for seed in seen_seed_order for run in by_seed[seed]]


# ======================================================================================
# run_id prediction -- must exactly mirror fl/app.py::main()'s resolved_config
# ======================================================================================


def pyproject_flat_defaults(pyproject_path: str | Path) -> dict[str, Any]:
    """`[tool.flwr.app.config]`'s flat key/value defaults -- the same table `flwr
    run` merges CLI `--run-config` overrides into before `Context.run_config` ever
    reaches `fl/app.py`."""
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    return data["tool"]["flwr"]["app"]["config"]


def predict_run_id(defaults: dict[str, Any], overrides: dict[str, Any]) -> str:
    """Reproduces `fl/app.py::main()`'s exact `resolved_config`/`make_run_id` call --
    imported directly, not reimplemented, so the hash can never drift from what a
    real run will actually compute. `strategy_name` and `seed` are read from the
    merged run_config exactly like `main()` reads them (`run_config.get(...)`), not
    tracked as separate sweep-runner state."""
    merged = {**defaults, **overrides}
    seed = int(merged.get("seed", 0))
    strategy_name = str(merged.get("strategy-name", "fedavg")).lower()
    resolved_config = {"run_config": merged, "strategy": strategy_name, "seed": seed}
    return make_run_id(resolved_config, seed)


DEFAULT_OUTPUT_DIR = "results/fl"


def resolve_output_dir(
    run: RunSpec, defaults: dict[str, Any], requested: str | Path
) -> tuple[str, dict[str, Any]]:
    """Where this run will *actually* write, and the overrides that make it do so.

    `run_sweep` checked `is_completed(run_id, output_dir)` against its `--output-dir`
    argument while `execute_run` never put `output-dir` in the run_config -- so the run wrote
    to pyproject's `output-dir` and the runner looked somewhere else. With the default
    (`results/fl` both sides) they coincided and nothing showed. Pass anything else -- and
    `run_sweep.py`'s own docstring says "`output-dir` is routinely outside: a Colab run
    writes to Drive, a Kaggle run to ..." -- and two things break at once:

    * every cell is recorded `status: "unknown"`, which this module reserves for the
      genuinely alarming case of `flwr run` exiting 0 while the round died;
    * **resume stops working.** `is_completed` never finds anything, so a sweep restarted
      after a Kaggle session timeout silently re-runs every cell it had already finished.
      For 576 cells that is the whole point of the manifest, lost without a symptom.

    The fix is to make the two agree by construction: whatever directory is returned here is
    both injected into the run_config and used for the completion check. A value already in
    the run's own overrides wins, since the experiment YAML is the more specific statement.

    Note the run_id legitimately depends on this: `make_run_id` hashes the whole resolved
    config, `output-dir` included. That is pre-existing and consistent -- `predict_run_id`
    merges the same overrides the app receives -- but it does mean a cell's id changes if its
    output directory does, which is why `run_sweep.py` matches results by config while
    ignoring path keys rather than by predicted id.
    """
    if "output-dir" in run.overrides:
        return str(run.overrides["output-dir"]), dict(run.overrides)
    requested = str(requested)
    if requested == str(defaults.get("output-dir", DEFAULT_OUTPUT_DIR)):
        return requested, dict(run.overrides)
    return requested, {**run.overrides, "output-dir": requested}


def is_completed(run_id: str, output_dir: str | Path) -> bool:
    path = Path(output_dir) / f"{run_id}.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "completed"


# ======================================================================================
# --run-config command-string construction (real `flwr run` CLI syntax, verified via
# `flwr run --help`: `--run-config '<k1>=<v1> <k2>=<v2>'`, TOML value syntax)
# ======================================================================================


def format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_run_config_arg(overrides: dict[str, Any]) -> str:
    return " ".join(f"{key}={format_toml_value(value)}" for key, value in overrides.items())


# ======================================================================================
# Lock files -- one per in-flight run_id; locks older than the timeout are treated as
# crashed and re-queued (plan §6.2).
# ======================================================================================


def lock_path(run_id: str, lock_dir: str | Path) -> Path:
    return Path(lock_dir) / f"{run_id}.lock"


def try_acquire_lock(run_id: str, lock_dir: str | Path, timeout_s: float = DEFAULT_LOCK_TIMEOUT_S) -> bool:
    """Returns True if the lock was acquired (a fresh lock file now exists, owned by
    this call). Returns False if another lock is live (younger than `timeout_s`) --
    a stale lock (older than `timeout_s`, presumably a crashed session) is replaced,
    not respected."""
    path = lock_path(run_id, lock_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        age_s = time.time() - path.stat().st_mtime
        if age_s < timeout_s:
            return False
    path.write_text(json.dumps({"pid": os.getpid(), "acquired_at": time.time()}))
    return True


def release_lock(run_id: str, lock_dir: str | Path) -> None:
    lock_path(run_id, lock_dir).unlink(missing_ok=True)


# ======================================================================================
# Execution + manifest
# ======================================================================================

Runner = Callable[[list[str]], "subprocess.CompletedProcess[bytes] | int"]


def _default_runner(cmd: list[str]) -> "subprocess.CompletedProcess[bytes]":
    return subprocess.run(cmd)


def execute_run(
    run_spec: RunSpec, flwr_bin: str = "flwr", runner: Runner = _default_runner
) -> int:
    """Builds and runs `flwr run . --stream --run-config "..."`. Returns the process
    return code (0 == flwr CLI accepted and completed the run without raising --
    still check the actual `results/fl/<run_id>.json` for `status: "completed"`,
    since a 0 exit code from the CLI doesn't by itself guarantee the FL round
    finished successfully, e.g. the heartbeat failure this repo has real history
    with, docs/OPEN_QUESTIONS.md)."""
    run_config_arg = build_run_config_arg(run_spec.overrides)
    cmd = [flwr_bin, "run", ".", "--stream", "--run-config", run_config_arg]
    result = runner(cmd)
    return result if isinstance(result, int) else result.returncode


def required_supernodes(defaults: dict[str, Any], overrides: dict[str, Any]) -> int:
    """This run's `num-clients`, resolved against pyproject's defaults.

    `num-clients` is this project's own key for how the *data* is partitioned. How many
    ClientApps the Simulation Runtime actually creates is Flower's `num_supernodes`, which
    defaults to 2 and is a property of the *federation*, not of a run config -- so it has
    to be set out-of-band, per cell, before `flwr run`.
    """
    return int({**defaults, **overrides}.get("num-clients", 2))


def gpu_fraction_problem(gpus_per_client: float | None, num_clients: int) -> str | None:
    """Refuse to start a sweep that would train on CPU on a GPU box. None means clear.

    Ray hides accelerators from an actor requested with `num_gpus=0`. That is not a guess:
    `ray/_private/worker.py` (2.55.1) computes
    `override_on_zero = env_bool(RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO_ENV_VAR, True)` and warns
    that *future* versions "will no longer override accelerator visible devices env var if
    num_gpus=0 or num_gpus=None (default)" -- so the current version does. A ClientApp actor
    given no GPU fraction therefore sees `torch.cuda.is_available() == False` and trains on
    CPU.

    **Why this has to refuse rather than warn.** The ServerApp runs in the driver process, not
    a Ray actor, so it keeps the GPU: server-side evaluation, `test_macro_f1` and every logged
    metric look completely normal. The only symptom is wall-clock, and the plan's own cost
    projection is the thing you would check it against -- so the failure reads as "the estimate
    was optimistic", which is exactly what a 4-core box legitimately causes too. For
    `main.yaml` that is 576 cells x 100 rounds of client training at CPU speed: it does not
    finish inside any Kaggle quota, and nothing in the result files says why.

    An explicit `--gpus-per-client 0` is honoured -- someone who says they want CPU gets CPU.
    It is the *unset* case on a GPU box that is refused.
    """
    if gpus_per_client is not None:
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
    except Exception:  # pragma: no cover - no torch means no GPU training either
        return None

    suggested = round(1.0 / max(num_clients, 1), 4)
    return (
        "A GPU is present but --gpus-per-client was not set. Ray hides GPUs from ClientApp "
        "actors requested with num_gpus=0, so every client would train on CPU while the "
        "server kept the GPU -- the results would look correct and the sweep would simply "
        "never finish.\n"
        f"  Pass --gpus-per-client {suggested} to let all {num_clients} ClientApps share the "
        "card, or a larger fraction to run fewer concurrently.\n"
        "  Pass --gpus-per-client 0 if you really do want CPU training."
    )


def configure_federation(
    num_supernodes: int,
    cpus_per_client: int = 1,
    gpus_per_client: float = 0.0,
    runner: Runner = _default_runner,
) -> int:
    """Set the Simulation Runtime's supernode count and per-ClientApp resources.

    `gpus_per_client` is a *fraction* of a GPU's VRAM per ClientApp, so 0.2 lets five share
    one card. It has to be passed here rather than left to a separate manual call, because
    this function re-issues `simulation-config` whenever a cell's K changes and it is not
    established that unspecified options survive that. Left at 0 on a GPU box, Ray may
    allocate no GPU to the ClientApp actors and the whole sweep runs on CPU -- which for
    576 cells x 100 rounds does not finish, and shows up only as "this is slower than the
    projection" rather than as an error.
    """
    cmd = [
        "flwr", "federation", "simulation-config",
        "--num-supernodes", str(num_supernodes),
        "--client-resources-num-cpus", str(cpus_per_client),
    ]
    if gpus_per_client > 0:
        cmd += ["--client-resources-num-gpus", str(gpus_per_client)]
    result = runner(cmd)
    return result if isinstance(result, int) else result.returncode


def append_manifest(manifest_path: str | Path, entry: dict[str, Any]) -> None:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ======================================================================================
# Orchestration
# ======================================================================================


def run_sweep(
    runs: list[RunSpec],
    pyproject_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    manifest_path: str | Path = "results/manifest.jsonl",
    lock_dir: str | Path = "results/fl/_locks",
    lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_S,
    dry_run: bool = False,
    runner: Runner = _default_runner,
    cpus_per_client: int = 1,
    gpus_per_client: float = 0.0,
) -> list[dict[str, Any]]:
    """Runs `runs` in the given order (call `order_seed_first` beforehand if that
    ordering is wanted), skipping any whose predicted run_id already has a completed
    result, and skipping any currently locked by another in-flight process. Returns
    the list of manifest entries this call itself wrote (dry-run: what it *would*
    write, without touching locks, `flwr run`, or the manifest file)."""
    defaults = pyproject_flat_defaults(pyproject_path)
    entries: list[dict[str, Any]] = []
    # Reconfigured per cell, but only when the value actually changes -- these configs
    # VARY num-clients across cells (overhead.yaml sweeps 5 -> 150, main_client_scale
    # 10 -> 50, robustness_r5 likewise), and `num_supernodes` is a federation-level
    # setting that no --run-config key can carry. Left unset it defaults to 2: a cell
    # labelled K=100 would train on 2 clients while recording 100, and an overhead curve
    # measured that way comes out FLAT -- which reads as evidence for the O(K^2) cost
    # being negligible, the very claim (C3) these sweeps exist to test.
    configured_supernodes: int | None = None

    for run in runs:
        # Resolved before the id, because the id hashes the config this returns.
        cell_output_dir, overrides = resolve_output_dir(run, defaults, output_dir)
        run = RunSpec(label=run.label, group=run.group, overrides=overrides)
        run_id = predict_run_id(defaults, run.overrides)

        if is_completed(run_id, cell_output_dir):
            entry = {"run_id": run_id, "label": run.label, "group": run.group, "status": "skipped_completed"}
            entries.append(entry)
            if not dry_run:
                append_manifest(manifest_path, entry)
            continue

        if dry_run:
            entries.append({"run_id": run_id, "label": run.label, "group": run.group, "status": "planned"})
            continue

        if not try_acquire_lock(run_id, lock_dir, lock_timeout_s):
            entry = {"run_id": run_id, "label": run.label, "group": run.group, "status": "skipped_locked"}
            entries.append(entry)
            append_manifest(manifest_path, entry)
            continue

        wanted = required_supernodes(defaults, run.overrides)
        if wanted != configured_supernodes:
            code = configure_federation(
                wanted, cpus_per_client, gpus_per_client, runner=runner
            )
            if code != 0:
                # Refuse rather than run: a cell at the wrong supernode count either hangs
                # at round 0 (min-train-nodes above the count) or silently mislabels its
                # client count. Both are worse than stopping.
                entry = {
                    "run_id": run_id, "label": run.label, "group": run.group,
                    "status": "failed_federation_config", "return_code": code,
                }
                entries.append(entry)
                append_manifest(manifest_path, entry)
                release_lock(run_id, lock_dir)
                continue
            configured_supernodes = wanted

        try:
            return_code = execute_run(run, runner=runner)
            completed = is_completed(run_id, cell_output_dir)
            if completed:
                status = "completed"
            elif return_code != 0:
                status = "failed"
            else:
                # Exit code 0 but no completed results/fl/<run_id>.json -- an
                # ambiguous state worth a human look (this repo's real history
                # includes exactly this: `flwr run` returning cleanly while the FL
                # round itself later reported "No heartbeat", docs/OPEN_QUESTIONS.md).
                status = "unknown"
            entry = {
                "run_id": run_id,
                "label": run.label,
                "group": run.group,
                "status": status,
                "return_code": return_code,
            }
        finally:
            release_lock(run_id, lock_dir)

        entries.append(entry)
        append_manifest(manifest_path, entry)

    return entries
