"""Phase 6 — the main sweep runner.

Expands a YAML sweep definition into (strategy x regime x seed) cells, runs each as one
`flwr run`, and indexes every completed run in `results/manifest.jsonl` so an interrupted
sweep resumes instead of recomputing. This is the piece that turns a pile of one-off
commands into the main table.

Four things here are not stylistic choices; each is a lesson from a failure recorded in
docs/OPEN_QUESTIONS.md, and removing any of them reintroduces a silent one:

**`--stream` is mandatory.** A bare `flwr run` submits the run and returns exit 0
*immediately*, while the simulation is still starting. A sweep that shells out without it
launches all 330 runs near-simultaneously, concludes every one failed, and leaves the
machine thrashing under orphaned simulations that keep executing after the sweep has
"finished". This was observed, not theorised.

**Exit codes are ignored.** `flwr run` also exits 0 when the simulation itself dies. Since
it lies in both directions, a cell counts as done only when a result file with
`status: "completed"` exists for it.

**Client resources are checked before the first run.** The Simulation Runtime assigns 2
CPUs per ClientApp by default, so K=20 requests 40 cores. Oversubscription does not
queue — it stalls at round 0 with idle actors and no error, which in a long sweep is
indistinguishable from a slow round.

**Nothing here reads a test metric.** Selection and analysis belong downstream; the sweep
only runs cells and records where their results landed.

Run:
    python scripts/run_sweep.py --config configs/experiment/main.yaml --dry-run
    python scripts/run_sweep.py --config configs/experiment/main.yaml
    python scripts/run_sweep.py --config configs/experiment/main.yaml --only fedaco,fedavg
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "results/manifest.jsonl"

# Measured on this project's own hardware: 48.1 s/round at K=2, 1 local epoch, CPU
# (docs/EXPERIMENT_LOG.md, 2026-09-17). Used only to project a sweep's cost in --dry-run,
# and deliberately labelled as what it is -- a single CPU datapoint at a K far below the
# sweep's, not a GPU measurement at K=20. Override with --seconds-per-round once a real
# run at the target shape exists.
MEASURED_SECONDS_PER_ROUND = 48.1
MEASURED_AT_K = 2


@dataclass
class Cell:
    strategy: str
    regime_name: str
    seed: int
    overrides: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.strategy}/{self.regime_name}/seed{self.seed}"

    def run_config(self, common: dict) -> dict:
        return {
            **common,
            **self.overrides,
            "strategy-name": self.strategy,
            "seed": self.seed,
        }


def load_sweep(path: Path) -> dict:
    spec = yaml.safe_load(path.read_text())
    for key in ("strategies", "regimes", "seeds"):
        if key not in spec:
            raise ValueError(f"{path} is missing required key {key!r}")
    return spec


def expand(spec: dict) -> list[Cell]:
    """Full factorial, in a stable order.

    Seed is the innermost loop so an interrupted sweep leaves whole (strategy, regime)
    groups finished rather than one seed of everything -- a partial sweep is then still
    analysable for the cells it covered, instead of having no complete group at all.
    """
    cells = []
    for strategy in spec["strategies"]:
        for regime in spec["regimes"]:
            overrides = {k: v for k, v in regime.items() if k != "name"}
            for seed in spec["seeds"]:
                cells.append(
                    Cell(
                        strategy=str(strategy),
                        regime_name=str(regime.get("name", regime.get("regime", "?"))),
                        seed=int(seed),
                        overrides=overrides,
                    )
                )
    return cells


def format_run_config(config: dict) -> str:
    parts = []
    for key, value in sorted(config.items()):
        parts.append(f"{key}='{value}'" if isinstance(value, str) else f"{key}={value}")
    return " ".join(parts)


# ======================================================================================
# Manifest -- the resume index CLAUDE.md specifies.
# ======================================================================================


def _record_path(path: Path) -> str:
    """Repo-relative when the path is inside the clone, absolute otherwise.

    `Path.relative_to` *raises* rather than falling back when the path is outside, and
    `output-dir` is routinely outside: a Colab run writes to Drive, a Kaggle run to
    /kaggle/working. An unguarded call crashed this runner one cell into its first real
    sweep -- after the run had completed but before its manifest entry was written, so
    the sweep aborted and the finished run went unindexed. Exactly the failure that
    would end an unattended 360-cell sweep on cell 1 and lose a night.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_manifest(path: Path) -> dict[str, dict]:
    """Manifest keyed by cell label. Written as JSON Lines so an interrupted sweep leaves
    a readable file rather than a truncated JSON document, and so appending a record is
    one atomic-ish write rather than a rewrite of the whole index."""
    if not path.exists():
        return {}
    entries: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn final line from a hard kill; the cell simply re-runs
        if "label" in record:
            entries[record["label"]] = record
    return entries


def append_manifest(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def find_result(output_dir: Path, cell: Cell, common: dict) -> Path | None:
    """The result file for this cell, matched on the resolved run_config the file itself
    records rather than on a filename convention.

    Matching must include the seed and the regime, not just the strategy: matching on too
    little is how a sweep silently reuses one seed's result for another and destroys the
    variance every error bar depends on (the same bug hit the Phase 5 search)."""
    wanted = cell.run_config(common)
    ignored = {"output-dir", "checkpoint-dir", "cache-dir", "manifest-path"}
    wanted = {k: v for k, v in wanted.items() if k not in ignored}
    for path in sorted(output_dir.glob("*.json")):
        try:
            result = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if result.get("status") != "completed":
            continue
        run_config = result.get("config", {}).get("run_config", {})
        if all(_same(run_config.get(k), v) for k, v in wanted.items()):
            return path
    return None


def _same(left, right) -> bool:
    if isinstance(right, float) or isinstance(left, float):
        try:
            return abs(float(left) - float(right)) < 1e-12
        except (TypeError, ValueError):
            return False
    return left == right


# ======================================================================================
# Preflight
# ======================================================================================


def preflight(num_clients: int, skip: bool) -> list[str]:
    """Problems worth refusing to start 330 runs over. Returns a list of warnings; an
    empty list means clear."""
    warnings = []
    if shutil.which("flwr") is None:
        warnings.append("`flwr` is not on PATH -- activate the venv (.venv/bin) first.")

    cores = os.cpu_count() or 1
    # 2 CPUs/ClientApp is the Simulation Runtime default; see the module docstring.
    needed = 2 * num_clients
    if needed > cores:
        warnings.append(
            f"{num_clients} clients request {needed} CPUs at the default 2/ClientApp, but "
            f"this machine has {cores}. Oversubscription stalls silently at round 0 rather "
            f"than queueing. Run:\n"
            f"      flwr federation simulation-config --client-resources-num-cpus 1\n"
            f"    and confirm a short run at this K actually completes before committing "
            f"the sweep."
        )
    if skip:
        return []
    return warnings


def project_cost(cells: int, rounds: int, seconds_per_round: float, k: int) -> str:
    total_h = cells * rounds * seconds_per_round / 3600
    note = (
        f"    basis: {seconds_per_round:.1f}s/round"
        + (
            f", measured at K={MEASURED_AT_K} on CPU -- this sweep is K={k}, so treat it "
            "as an order of magnitude, not a quote"
            if seconds_per_round == MEASURED_SECONDS_PER_ROUND
            else " (supplied)"
        )
    )
    return (
        f"    {cells} runs x {rounds} rounds = {cells * rounds:,} rounds\n"
        f"    projected {total_h:.1f} hours ({total_h / 24:.1f} days of continuous compute)\n"
        + note
    )


# ======================================================================================
# Running
# ======================================================================================


def run_cell(cell: Cell, common: dict, stream: bool) -> tuple[int, str]:
    config = format_run_config(cell.run_config(common))
    # --stream is required for the call to block; see the module docstring.
    cmd = ["flwr", "run", ".", "--stream", "--run-config", config]
    env = {**os.environ, "FEDSWARM_REPO_ROOT": str(REPO_ROOT)}
    if stream:
        completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
        return completed.returncode, ""
    completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


def diagnose(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if "arn" not in line.lower()]
    for marker in (
        "Invalid run configuration",
        "Dataset root does not exist",
        "No cache at",
        "FileNotFoundError",
        "Traceback",
        "Exit Code:",
        "Error",
    ):
        for line in lines:
            if marker in line:
                return line[:300]
    return "no diagnostic line found in output"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/main.yaml")
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--only", default=None, help="comma-separated strategies to run")
    parser.add_argument("--regimes", default=None, help="comma-separated regime names to run")
    parser.add_argument("--seeds", default=None, help="comma-separated seeds to run")
    parser.add_argument("--limit", type=int, default=None, help="stop after N cells (a pilot)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and cost, run nothing")
    parser.add_argument("--stream", action="store_true", help="echo each run's output live")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--seconds-per-round",
        type=float,
        default=MEASURED_SECONDS_PER_ROUND,
        help="for the --dry-run cost projection",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    spec = load_sweep(config_path)
    common = dict(spec.get("common", {}))

    cells = expand(spec)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        cells = [c for c in cells if c.strategy in wanted]
    if args.regimes:
        wanted = {s.strip() for s in args.regimes.split(",")}
        cells = [c for c in cells if c.regime_name in wanted]
    if args.seeds:
        wanted = {int(s.strip()) for s in args.seeds.split(",")}
        cells = [c for c in cells if c.seed in wanted]
    if args.limit is not None:
        cells = cells[: args.limit]

    if not cells:
        print("No cells selected -- check --only/--regimes/--seeds against the config.")
        return 2

    num_clients = int(common.get("num-clients", 20))
    rounds = int(common.get("num-rounds", 100))
    output_dir = REPO_ROOT / str(common.get("output-dir", "results/fl"))
    manifest_path = REPO_ROOT / args.manifest

    print(f"sweep: {spec.get('name', config_path.stem)}  ({len(cells)} cells)")
    print(project_cost(len(cells), rounds, args.seconds_per_round, num_clients))

    warnings = preflight(num_clients, args.skip_preflight)
    for warning in warnings:
        print(f"\n  ⚠️  {warning}")

    manifest = read_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # A completed result file is the authority on whether a cell is done; the manifest is
    # an index over those files, not the record of record. Gating on the manifest as well
    # means a cell whose entry was lost -- a torn line from a hard kill, a crash between
    # the run finishing and the append, a manifest deleted or not carried between
    # machines -- gets recomputed even though its result is sitting right there. That
    # happened on this runner's own first sweep and cost a full re-run of the cell.
    # Instead: decide from the result file, and backfill any missing index entry.
    todo = []
    backfilled = 0
    for cell in cells:
        existing = find_result(output_dir, cell, common)
        if existing is None:
            todo.append(cell)
            continue
        if cell.label not in manifest:
            append_manifest(
                manifest_path,
                {
                    "label": cell.label,
                    "status": "completed",
                    "strategy": cell.strategy,
                    "regime": cell.regime_name,
                    "seed": cell.seed,
                    "result_path": _record_path(existing),
                    "backfilled": True,
                    "ts": time.time(),
                },
            )
            backfilled += 1
    done = len(cells) - len(todo)
    print(
        f"\n  {done} already complete, {len(todo)} to run"
        + (f" ({backfilled} indexed from existing results)" if backfilled else "")
    )

    if args.dry_run:
        for cell in todo[:20]:
            print(f"    {cell.label}")
        if len(todo) > 20:
            print(f"    ... and {len(todo) - 20} more")
        return 0

    if warnings:
        print("\nRefusing to start with the warnings above. Fix them, or pass --skip-preflight.")
        return 1

    failures = []
    started = time.time()
    for i, cell in enumerate(todo, 1):
        elapsed = time.time() - started
        eta = (elapsed / max(i - 1, 1)) * (len(todo) - i + 1) if i > 1 else 0
        print(f"\n[{i}/{len(todo)}] {cell.label}" + (f"   (eta {eta / 3600:.1f}h)" if i > 1 else ""))
        try:
            returncode, output = run_cell(cell, common, args.stream)
            result_path = find_result(output_dir, cell, common)
        except Exception as exc:  # noqa: BLE001 -- one cell must never abort the sweep
            # An unattended 360-cell sweep that dies on cell 1 loses a night. Whatever
            # went wrong here is one cell's problem: record it, carry on, and let the
            # summary at the end say what failed. (This guard exists because an
            # unguarded Path.relative_to did exactly that on the first real sweep.)
            reason = f"{type(exc).__name__}: {exc}"
            print(f"  !! cell raised -- {reason}")
            failures.append((cell.label, reason))
            append_manifest(
                manifest_path,
                {"label": cell.label, "status": "failed", "reason": reason, "ts": time.time()},
            )
            continue
        if result_path is None:
            reason = diagnose(output) if output else f"exit {returncode}, output not captured"
            print(f"  !! no result file -- {reason}")
            failures.append((cell.label, reason))
            append_manifest(
                manifest_path,
                {"label": cell.label, "status": "failed", "reason": reason, "ts": time.time()},
            )
            continue

        result = json.loads(result_path.read_text())
        final = result.get("final", {})
        print(
            f"  -> {result_path.name}  val={final.get('best_val_macro_f1')}  "
            f"test={final.get('final_test_macro_f1')}  {final.get('wall_clock_s', 0):.0f}s"
        )
        append_manifest(
            manifest_path,
            {
                "label": cell.label,
                "status": "completed",
                "strategy": cell.strategy,
                "regime": cell.regime_name,
                "seed": cell.seed,
                "run_id": result.get("run_id"),
                "result_path": _record_path(result_path),
                "wall_clock_s": final.get("wall_clock_s"),
                "ts": time.time(),
            },
        )

    print(f"\n{len(todo) - len(failures)}/{len(todo)} cells completed in {(time.time() - started) / 3600:.1f}h")
    if failures:
        print(f"{len(failures)} failed:")
        for label, reason in failures[:10]:
            print(f"  {label}: {reason}")
        print("\nRe-run the same command to retry only the failures -- completed cells are skipped.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
