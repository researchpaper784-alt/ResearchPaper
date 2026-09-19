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


def configure_federation(
    num_supernodes: int, cpus_per_client: int = 1, runner: Runner = _default_runner
) -> int:
    """Set the Simulation Runtime's supernode count. Returns the process return code."""
    cmd = [
        "flwr", "federation", "simulation-config",
        "--num-supernodes", str(num_supernodes),
        "--client-resources-num-cpus", str(cpus_per_client),
    ]
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
    output_dir: str | Path = "results/fl",
    manifest_path: str | Path = "results/manifest.jsonl",
    lock_dir: str | Path = "results/fl/_locks",
    lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_S,
    dry_run: bool = False,
    runner: Runner = _default_runner,
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
        run_id = predict_run_id(defaults, run.overrides)

        if is_completed(run_id, output_dir):
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
            code = configure_federation(wanted, runner=runner)
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
            completed = is_completed(run_id, output_dir)
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
