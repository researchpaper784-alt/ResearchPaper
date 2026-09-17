"""Phase 5 — an honest per-baseline hyperparameter search.

The plan's requirement is that every baseline gets "an honest hyperparameter search on
the val split with a budget matched to FedACO's". Two words in that sentence do the work:

**"on the val split"** — every trial is scored by `best_val_macro_f1` from its result
file, which comes from the server-held global val split (735 images, never partitioned,
disjoint from test). This script never reads `final_test_macro_f1`, and refuses to, so a
baseline cannot be tuned against the number the paper reports. Before 2026-09-17 result
files carried no val metric at all, which is why this script could not have been written
honestly then.

**"a budget matched to FedACO's"** — every strategy gets the same number of trials
(`--budget`, default 8). A baseline with fewer distinct settings than that (FedAvg has
none at all; FedMedian and FedNova have none this project exposes) uses fewer, and the
summary records the shortfall explicitly rather than quietly giving FedACO a larger
search than its competitors. That asymmetry is the single easiest way to make a new
method look good, so it is reported as a first-class field, not a footnote.

Each trial is a real `flwr run` subprocess. Trials are resumable: a trial whose exact
override set already has a result file in `--output-dir` is skipped and its recorded
score reused, so an interrupted search resumes without recomputing.

Run:
    python scripts/run_hparam_search.py --strategy fedprox --regime dirichlet --seed 0
    python scripts/run_hparam_search.py --strategy all --budget 8
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Per-strategy search grids. Values are `--run-config` keys, so they must also be
# declared in pyproject.toml's [tool.flwr.app.config] -- `flwr run` rejects undeclared
# keys with a bare "[code: 15]" naming nothing (docs/OPEN_QUESTIONS.md).
#
# Ranges are centred on each strategy's own published default (the value in
# strategies/factory.py) and span the region that paper actually explores, rather than an
# arbitrary sweep: a grid that never includes a baseline's recommended setting is not an
# honest search.
GRIDS: dict[str, dict[str, list]] = {
    # No aggregation hyperparameter at all -- the reference point, 1 trial by definition.
    "fedavg": {},
    "fedprox": {"fedprox-mu": [0.001, 0.01, 0.1, 1.0]},
    # Per-strategy eta keys, not a shared `fedopt-eta` -- see strategies/factory.py for
    # why one shared key would silently overwrite FedYogi's published default.
    "fedadam": {"fedadam-eta": [0.01, 0.1], "fedadam-eta-l": [0.01, 0.1, 0.316]},
    "fedyogi": {"fedyogi-eta": [0.001, 0.01, 0.1], "fedyogi-eta-l": [0.01, 0.0316, 0.1]},
    "krum": {"num-malicious-nodes": [0, 1, 2]},
    "trimmed-mean": {"trim-beta": [0.05, 0.1, 0.2, 0.3]},
    "median": {},
    "fednova": {},
    "loss-based": {"lossweight-temperature": [0.25, 0.5, 1.0, 2.0, 5.0]},
    "scaffold": {"scaffold-server-lr": [0.25, 0.5, 1.0, 2.0]},
    "fedlaw": {"fedlaw-steps": [10, 20, 50], "fedlaw-lr": [0.01, 0.1]},
    # FedACO is searched over the same budget as everything else. gamma_dispersion is
    # included deliberately: normalization made it co-equal with alignment and it was
    # left at its specified 1.0 rather than retuned (docs/EXPERIMENT_LOG.md, 2026-09-17),
    # so this is where that choice gets tested rather than assumed.
    "fedaco": {
        "aco-gamma-dispersion": [0.5, 1.0, 2.0],
        "aco-q0": [0.7, 0.9],
        "aco-gamma-entropy": [0.1, 0.5],
    },
}


def trial_overrides(strategy: str, budget: int) -> list[dict]:
    """Full factorial of the strategy's grid, truncated to `budget` trials.

    Truncation is deterministic (itertools.product order) rather than random so a search
    is reproducible and two strategies are never compared across different samplings of
    their own grids.
    """
    grid = GRIDS[strategy]
    if not grid:
        return [{}]
    keys = sorted(grid)
    combos = [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]
    return combos[:budget]


def format_run_config(overrides: dict) -> str:
    parts = []
    for key, value in sorted(overrides.items()):
        parts.append(f"{key}='{value}'" if isinstance(value, str) else f"{key}={value}")
    return " ".join(parts)


def load_results(output_dir: Path) -> list[dict]:
    results = []
    for path in sorted(output_dir.glob("*.json")):
        try:
            results.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return results


def find_existing(results: list[dict], overrides: dict, base: dict | None = None) -> dict | None:
    """A previous trial with exactly this configuration, for resume. Matches on the
    resolved run_config the result file records, not on a filename convention.

    `base` must be included in the match, not just `overrides`. Matching on the swept
    keys alone means a search at seed 1 happily reuses seed 0's results from the same
    output directory -- every trial "resumes" instantly and the second seed is silently
    a copy of the first. The same applies to regime and round count.
    """
    wanted = {**(base or {}), **overrides}
    # Paths and bookkeeping say nothing about what was computed, and differ harmlessly
    # between environments (a Colab cache lives elsewhere than a local one).
    ignored = {"output-dir", "checkpoint-dir", "cache-dir", "manifest-path"}
    wanted = {k: v for k, v in wanted.items() if k not in ignored}
    for result in results:
        run_config = result.get("config", {}).get("run_config", {})
        if all(_same(run_config.get(k), v) for k, v in wanted.items()):
            if result.get("status") == "completed":
                return result
    return None


def _same(left, right) -> bool:
    if isinstance(right, float) or isinstance(left, float):
        try:
            return abs(float(left) - float(right)) < 1e-12
        except (TypeError, ValueError):
            return False
    return left == right


def score(result: dict) -> float | None:
    """The one number a search is allowed to rank on.

    Deliberately reads `best_val_macro_f1` and nothing else. `final_test_macro_f1` is
    present in the same dict and is intentionally not consulted -- selecting on it would
    tune each baseline against the metric the paper reports, which is the contamination
    this whole script exists to avoid.
    """
    return result.get("final", {}).get("best_val_macro_f1")


def run_trial(strategy: str, overrides: dict, base: dict, stream: bool) -> tuple[int, str]:
    """Returns (returncode, captured output).

    `--stream` is passed unconditionally, and that is load-bearing rather than cosmetic:
    **a bare `flwr run` is asynchronous.** It submits the run to the SuperLink, prints
    "Successfully started run <id>", and returns immediately with exit 0 while the
    simulation is still starting. A search that shells out without `--stream` therefore
    checks for each trial's result file microseconds after launching it, finds nothing,
    and reports every trial as failed -- which is exactly what the first end-to-end run
    of this script did. `--stream` makes the call block until the run finishes.

    `flwr run` also exits 0 even when the simulation itself dies (a missing image cache
    surfaces as an in-run traceback and "Exit Code: 700" while the outer process still
    returns success), so the return code cannot decide whether a trial worked either.
    Both facts together are why success is judged by "did a result file appear", not by
    the subprocess's own verdict.

    `stream=True` additionally echoes the run's output live instead of capturing it.
    """
    run_config = format_run_config({"strategy-name": strategy, **base, **overrides})
    cmd = ["flwr", "run", ".", "--stream", "--run-config", run_config]
    env = {**os.environ, "FEDSWARM_REPO_ROOT": str(REPO_ROOT)}
    print(f"  $ flwr run . --stream --run-config \"{run_config}\"", flush=True)
    if stream:
        completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
        return completed.returncode, ""
    completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


def diagnose(output: str) -> str:
    """The one line from a failed trial's output most likely to explain it.

    Markers are tried in priority order and each is scanned across the whole output, so
    a specific cause wins over an incidental match anywhere in the log. Scanning
    line-by-line against a flat marker set instead let Ray's `FutureWarning: ... turn off
    this error message` be reported as the reason a trial failed, purely because it
    appeared first and contains the word "error".
    """
    lines = [line.strip() for line in output.splitlines()]
    interesting = [
        line
        for line in lines
        if "Warning" not in line and "warn" not in line.lower()
    ]
    for marker in (
        "Invalid run configuration",
        "Dataset root does not exist",
        "No cache at",
        "FileNotFoundError",
        "Traceback",
        "Exit Code:",
        "ValueError",
        "KeyError",
        "Error",
    ):
        for line in interesting:
            if marker in line:
                return line[:300]
    return "no diagnostic line found in output"


def search(strategy: str, base: dict, output_dir: Path, budget: int, stream: bool) -> dict:
    combos = trial_overrides(strategy, budget)
    grid_size = len(trial_overrides(strategy, budget=10**6))
    print(f"\n=== {strategy}: {len(combos)} trial(s) (grid has {grid_size}, budget {budget})")

    trials = []
    for i, overrides in enumerate(combos, 1):
        print(f"[{strategy} {i}/{len(combos)}] {overrides or '(no hyperparameters)'}")
        existing = find_existing(load_results(output_dir), overrides, base)
        if existing is not None:
            print(f"  -> reusing existing result, val_macro_f1={score(existing)}")
            trials.append({"overrides": overrides, "val_macro_f1": score(existing), "reused": True})
            continue

        returncode, output = run_trial(strategy, overrides, base, stream)
        result = find_existing(load_results(output_dir), overrides, base)

        # Three outcomes, kept distinct. Collapsing them into a bare "None" is how a
        # search reports a clean table having actually produced nothing at all.
        if result is None:
            reason = diagnose(output) if output else f"exit {returncode}, output not captured"
            print(f"  !! trial produced no result file -- {reason}")
            trials.append(
                {"overrides": overrides, "val_macro_f1": None, "failed": True, "reason": reason}
            )
            continue

        value = score(result)
        if value is None:
            print("  !! trial completed but carries no val metric (result file predates val logging)")
            trials.append(
                {
                    "overrides": overrides,
                    "val_macro_f1": None,
                    "failed": True,
                    "reason": "no best_val_macro_f1 in result file",
                }
            )
            continue

        print(f"  -> val_macro_f1={value:.4f}")
        trials.append({"overrides": overrides, "val_macro_f1": value, "reused": False})

    scored = [t for t in trials if t["val_macro_f1"] is not None]
    best = max(scored, key=lambda t: t["val_macro_f1"]) if scored else None
    return {
        "strategy": strategy,
        "budget": budget,
        "grid_size": grid_size,
        "trials_run": len(combos),
        # Made explicit so an unequal search is visible in the artifact rather than
        # inferred: a strategy whose grid is smaller than the budget genuinely had fewer
        # chances to win, and that has to be stated wherever its number is quoted.
        "budget_shortfall": budget - len(combos),
        "trials_scored": len(scored),
        "selection_metric": "best_val_macro_f1",
        "base_config": base,
        "trials": trials,
        "best": best,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="all", help="strategy name, or 'all'")
    parser.add_argument("--budget", type=int, default=8, help="max trials per strategy")
    parser.add_argument("--regime", default="iid")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/fl/hparam")
    parser.add_argument("--cache-dir", default=None, help="override the image cache location")
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--summary-dir", default="results/hparam_search")
    parser.add_argument("--stream", action="store_true", help="echo each run's output live instead of capturing it")
    parser.add_argument("--dry-run", action="store_true", help="print the trial plan and exit")
    args = parser.parse_args()

    strategies = sorted(GRIDS) if args.strategy == "all" else [args.strategy]
    unknown = [s for s in strategies if s not in GRIDS]
    if unknown:
        parser.error(f"unknown strategy/strategies: {unknown}. Known: {sorted(GRIDS)}")

    base = {
        "regime": args.regime,
        "num-clients": args.num_clients,
        "min-train-nodes": args.num_clients,
        "min-evaluate-nodes": args.num_clients,
        "min-available-nodes": args.num_clients,
        "num-rounds": args.num_rounds,
        "local-epochs": args.local_epochs,
        "seed": args.seed,
        "output-dir": args.output_dir,
    }
    if args.alpha is not None:
        base["alpha"] = args.alpha
    if args.cache_dir is not None:
        base["cache-dir"] = args.cache_dir
    if args.manifest_path is not None:
        base["manifest-path"] = args.manifest_path

    if args.dry_run:
        total = 0
        for strategy in strategies:
            combos = trial_overrides(strategy, args.budget)
            total += len(combos)
            print(f"{strategy:14} {len(combos):2} trial(s)  {[c for c in combos]}")
        print(f"\ntotal: {total} runs x {args.num_rounds} rounds")
        return 0

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = REPO_ROOT / args.summary_dir
    summary_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for strategy in strategies:
        summary = search(strategy, base, output_dir, args.budget, args.stream)
        summaries.append(summary)
        (summary_dir / f"{strategy}.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'strategy':14} {'trials':>7} {'short':>6} {'best val_macro_f1':>18}  best config")
    print("-" * 88)
    for summary in summaries:
        best = summary["best"]
        value = f"{best['val_macro_f1']:.4f}" if best else "n/a"
        print(
            f"{summary['strategy']:14} {summary['trials_run']:>7} "
            f"{summary['budget_shortfall']:>6} {value:>18}  {best['overrides'] if best else '-'}"
        )
    combined = summary_dir / "all.json"
    combined.write_text(json.dumps({"budget": args.budget, "summaries": summaries}, indent=2))
    print(f"\nWrote {combined}")

    scored = sum(s["trials_scored"] for s in summaries)
    attempted = sum(s["trials_run"] for s in summaries)
    if scored == 0:
        # Exiting 0 here would report a clean run of a search that produced no usable
        # trial at all -- the failure mode this script hit on its own first end-to-end
        # test, where every trial died on a missing cache and nothing said so.
        print(f"\nFAILED: 0 of {attempted} trials produced a val score. Nothing was selected.")
        return 1
    if scored < attempted:
        print(f"\nWARNING: only {scored} of {attempted} trials scored; the rest failed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
