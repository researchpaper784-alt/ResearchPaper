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

from fedswarm.utils.runner import (
    diagnose,
    format_run_config,
    load_result_files,
    run_flwr,
    same_value,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "results/manifest.jsonl"

# Measured on this project's own hardware: 48.1 s/round at K=2, 1 local epoch, CPU
# (docs/EXPERIMENT_LOG.md, 2026-09-17). Used only to project a sweep's cost in --dry-run,
# and deliberately labelled as what it is -- a single CPU datapoint at a K far below the
# sweep's, not a GPU measurement at K=20. Override with --seconds-per-round once a real
# run at the target shape exists.
MEASURED_SECONDS_PER_ROUND = 48.1
MEASURED_AT_K = 2


DEFAULT_VARIANT = "default"


@dataclass
class Cell:
    strategy: str
    regime_name: str
    seed: int
    overrides: dict = field(default_factory=dict)
    variant: str = DEFAULT_VARIANT

    @property
    def label(self) -> str:
        # The variant is omitted from the label when it is the implicit default, so a
        # sweep config without a `variants:` axis produces exactly the labels it did
        # before that axis existed -- an existing manifest keeps matching and nothing
        # already computed gets re-run.
        middle = "" if self.variant == DEFAULT_VARIANT else f"/{self.variant}"
        return f"{self.strategy}/{self.regime_name}{middle}/seed{self.seed}"

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
    """Full factorial over strategies x regimes x variants x seeds, in a stable order.

    Seed is the innermost loop so an interrupted sweep leaves whole groups finished
    rather than one seed of everything -- a partial sweep is then still analysable for
    the cells it covered, instead of having no complete group at all.

    `variants` is the optional Phase 7 axis: named override sets layered on top of the
    regime's, so an ablation ("pheromone persistence off", "fitness mode = server_val")
    is a variant of one strategy rather than a separate sweep file with its own runner.
    Omit it and the sweep behaves exactly as it did before the axis existed.
    """
    variants = spec.get("variants") or [{"name": DEFAULT_VARIANT}]
    cells = []
    for strategy in spec["strategies"]:
        for regime in spec["regimes"]:
            regime_overrides = {k: v for k, v in regime.items() if k != "name"}
            for variant in variants:
                variant_overrides = {k: v for k, v in variant.items() if k != "name"}
                for seed in spec["seeds"]:
                    cells.append(
                        Cell(
                            strategy=str(strategy),
                            regime_name=str(regime.get("name", regime.get("regime", "?"))),
                            seed=int(seed),
                            # Variant wins on a key collision: it is the thing being
                            # ablated, and a regime that also set it would otherwise
                            # silently cancel the ablation.
                            overrides={**regime_overrides, **variant_overrides},
                            variant=str(variant.get("name", DEFAULT_VARIANT)),
                        )
                    )
    return cells


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


def find_result(
    output_dir: Path, cell: Cell, common: dict, spec_variants: list[dict] | None = None
) -> Path | None:
    """The result file for this cell, matched on the resolved run_config the file itself
    records rather than on a filename convention.

    Matching must include the seed and the regime, not just the strategy: matching on too
    little is how a sweep silently reuses one seed's result for another and destroys the
    variance every error bar depends on (the same bug hit the Phase 5 search)."""
    wanted = cell.run_config(common)
    ignored = {"output-dir", "checkpoint-dir", "cache-dir", "manifest-path"}
    wanted = {k: v for k, v in wanted.items() if k not in ignored}

    # Matching on `wanted` alone is not enough for a variant that sets no keys -- the
    # `default` control in ablation_all.yaml and `clean` in robustness.yaml both do. Their
    # `wanted` contains no variant key at all, so it would match *any* sibling variant's
    # result in the shared output dir: the control marked complete, never run, and every
    # ablation delta differenced against another ablation.
    #
    # The discriminator has to be by VALUE, not by key presence. Every variant key is
    # declared in pyproject (it has to be, or `flwr run` rejects it), so it appears in
    # every resolved run_config with its default -- rejecting candidates that merely
    # *carry* the key rejects the control's own result too. Asked for `clean` and handed
    # its own file, an earlier version of this check said "no result file" and re-ran the
    # cell forever.
    foreign_pairs = _foreign_variant_values(spec_variants, cell.variant, wanted)

    for path, result in load_result_files(output_dir, use_cache=True):
        if result.get("status") != "completed":
            continue
        run_config = result.get("config", {}).get("run_config", {})
        if not all(same_value(run_config.get(k), v) for k, v in wanted.items()):
            continue
        if any(same_value(run_config.get(k), v) for k, v in foreign_pairs):
            continue
        return path
    return None


def _foreign_variant_values(
    variants: list[dict] | None, own_variant: str, wanted: dict
) -> list[tuple[str, object]]:
    """(key, value) pairs that identify some *other* variant of this sweep.

    A candidate result carrying any of them belongs to that other variant, not to this
    cell. Keys this cell sets itself are excluded -- `wanted` already pins those.
    """
    pairs = []
    for variant in variants or []:
        if str(variant.get("name", DEFAULT_VARIANT)) == own_variant:
            continue
        for key, value in variant.items():
            if key != "name" and key not in wanted:
                pairs.append((key, value))
    return pairs


# ======================================================================================
# Preflight
# ======================================================================================


def configure_federation(num_clients: int, cpus_per_client: int = 1) -> tuple[bool, str]:
    """Set the Simulation Runtime's supernode count and per-client CPUs for this sweep.

    **This is not optional setup, it is a correctness requirement.** `num-clients` is this
    project's own run-config key -- it tells `partition_spec_from_run_config` how many ways
    to split the data. It has nothing to do with how many ClientApps the Flower Simulation
    Runtime actually creates, which is `num_supernodes` and defaults to **2**
    (flwr/supercore/constant.py). Nothing in this repository ever set it.

    Left unset, `make main` would run 360 cells that record `num-clients: 20` while only
    partition ids 0 and 1 ever receive a ClientApp: 18/20 of the data never trained on,
    every result file mislabelled. In `robustness.yaml` it compounds --
    `malicious_ids(20, 0.1) == {0, 1}`, so a cell labelled "10% malicious" would have had
    *the entire participating federation* compromised.

    It also explains a stall this project previously misdiagnosed: a K=4 run that sat at
    round 0 with idle actors and no error was recorded as CPU oversubscription. It was not.
    Only 2 supernodes existed while `min-train-nodes=4` waited for 4 that were never
    created. With `--num-supernodes 4` the same run completes fine on the same 4-core box
    (verified 2026-09-17).
    """
    cmd = [
        "flwr", "federation", "simulation-config",
        "--num-supernodes", str(num_clients),
        "--client-resources-num-cpus", str(cpus_per_client),
    ]
    completed = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        return False, (completed.stderr or completed.stdout or "").strip()[:400]
    return True, f"{num_clients} supernodes, {cpus_per_client} CPU each"


def preflight(num_clients: int, skip: bool) -> list[str]:
    """Problems worth refusing to start 360 runs over. Returns a list of warnings; an
    empty list means clear."""
    warnings = []
    if shutil.which("flwr") is None:
        warnings.append("`flwr` is not on PATH -- activate the venv (.venv/bin) first.")

    cores = os.cpu_count() or 1
    if num_clients > cores:
        # Not a correctness problem -- `configure_federation` pins 1 CPU per ClientApp and
        # Ray queues the excess -- but it does mean the cells run more serially than the
        # cost projection assumes, so the ETA will be optimistic.
        warnings.append(
            f"NOTE (not fatal): {num_clients} clients on {cores} cores, so ClientApps will "
            f"queue rather than run concurrently and the projected time above is optimistic."
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
    # `run_flwr` passes --stream unconditionally; see its docstring and this module's.
    return run_flwr(format_run_config(cell.run_config(common)), REPO_ROOT, stream=stream)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/main.yaml")
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--only", default=None, help="comma-separated strategies to run")
    parser.add_argument("--regimes", default=None, help="comma-separated regime names to run")
    parser.add_argument("--variants", default=None, help="comma-separated variant names to run")
    parser.add_argument("--seeds", default=None, help="comma-separated seeds to run")
    parser.add_argument("--limit", type=int, default=None, help="stop after N cells (a pilot)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and cost, run nothing")
    parser.add_argument("--stream", action="store_true", help="echo each run's output live")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--cpus-per-client", type=int, default=1,
        help="CPUs per ClientApp; the Simulation Runtime's own default is 2",
    )
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
    if args.variants:
        wanted = {s.strip() for s in args.variants.split(",")}
        cells = [c for c in cells if c.variant in wanted]
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
        existing = find_result(output_dir, cell, common, spec.get('variants'))
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
                    "variant": cell.variant,
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

    fatal = [w for w in warnings if not w.startswith("NOTE")]
    if fatal:
        print("\nRefusing to start with the problems above. Fix them, or pass --skip-preflight.")
        return 1

    # Correctness, not convenience: without this the Simulation Runtime creates 2
    # ClientApps regardless of `num-clients`, and every result file is mislabelled.
    ok, detail = configure_federation(num_clients, args.cpus_per_client)
    if not ok:
        print(f"\nCould not configure the federation ({detail}).")
        print("Every cell would silently run 2 clients regardless of num-clients. Refusing.")
        return 1
    print(f"  federation: {detail}")

    failures = []
    started = time.time()
    for i, cell in enumerate(todo, 1):
        elapsed = time.time() - started
        eta = (elapsed / max(i - 1, 1)) * (len(todo) - i + 1) if i > 1 else 0
        print(f"\n[{i}/{len(todo)}] {cell.label}" + (f"   (eta {eta / 3600:.1f}h)" if i > 1 else ""))
        try:
            returncode, output = run_cell(cell, common, args.stream)
            result_path = find_result(output_dir, cell, common, spec.get('variants'))
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
                "variant": cell.variant,
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
