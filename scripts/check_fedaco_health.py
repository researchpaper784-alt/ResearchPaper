"""Does FedACO's mechanism actually do anything on real data?

Every federated run so far has been on a synthetic pixel cache, which verifies plumbing
and nothing else. This script reads the result files from the first real-data run and
answers the three questions that decide whether Phase 6 is worth building as designed.
They are not accuracy questions -- a method can post a respectable macro-F1 while its
stated mechanism sits inert.

**1. Is the colony searching at all?**
    `pheromone_entropy` is the mean per-client entropy of tau across the L multiplier
    levels. Its maximum is log(L) = log(11) = 2.398, reached exactly when tau is uniform.
    Uniform tau means `tau^a * eta^b` reduces to `eta^b`: no colony search, no cross-round
    stigmergy, just deterministic heuristic-greedy weighting. The "Adaptive"/"swarm" claim
    in the paper's title is then decorative, which `aco/pheromone.py`'s own docstring
    names as the failure to watch for. In the 3-round synthetic run entropy sat ~1% below
    maximum, which at K=2 over 3 rounds proves nothing -- this is the real check.

**2. Has the deposit floor engaged?**
    `colony.py` deposits `rho * Q * max(F, 0)`. If F is negative for every candidate in a
    round, every deposit is zero and tau can only decay toward uniform -- question 1's
    failure, by a specific cause. Normalizing dispersion by trace(G)/K removed the
    *scale*-driven path into this (docs/EXPERIMENT_LOG.md, 2026-09-17), but F can still
    go negative on its own terms when client updates have no shared direction and
    alignment sits near zero. That residual is recorded as open in docs/OPEN_QUESTIONS.md
    and this is how it gets settled.

**3. Does FedACO actually beat FedAvg, and is the fallback carrying it?**
    `fallback_used=1` means the colony lost to the FedAvg point that round and FedACO
    silently *was* FedAvg. A high fallback rate with a good final score means the score
    belongs to FedAvg. Note the fallback compares both sides under the same fitness, so
    it cannot detect a degenerate fitness -- which is exactly why questions 1 and 2 are
    asked separately rather than inferred from it.

Also reports measured per-round wall-clock, which docs/EXPERIMENT_LOG.md's compute-budget
table explicitly says to substitute for its estimates ("Do not cite these numbers until
measured").

Run:
    python scripts/check_fedaco_health.py --results-dir results/fl
    python scripts/check_fedaco_health.py --results-dir results/fl --num-levels 11
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from fedswarm.utils.runner import load_result_files

REPO_ROOT = Path(__file__).resolve().parents[1]

# How close to log(L) counts as "uniform". Entropy is a log-scale quantity and the top of
# its range is flat, so a small absolute gap is a large behavioural one: at L=11, 2% below
# maximum still leaves tau nearly flat across levels. This threshold is a judgement call,
# not a figure from the plan -- stated here rather than buried, and adjustable via --flat.
DEFAULT_FLAT_FRACTION = 0.02


def load_results(results_dir: Path, strategy: str | None = None) -> list[dict]:
    """Every result under `results_dir` that has per-round records, optionally one
    strategy's. A run still in progress or killed mid-write has no `rounds` and is
    skipped; this check reads round-by-round behaviour, so it has nothing to say about a
    result that has none."""
    out = []
    for path, result in load_result_files(results_dir, recursive=True):
        if "rounds" not in result or "config" not in result:
            continue
        if strategy and str(result["config"].get("strategy", "")).lower() != strategy:
            continue
        result["_path"] = str(path)
        out.append(result)
    return out


def _round_metric(result: dict, key: str) -> list[float]:
    """Per-round values of an aggregate_train metric.

    FedACO's metrics travel in the MetricRecord returned by `aggregate_train`, which is
    not the same place as the evaluation metrics in `rounds`. Different result-writing
    paths have put them in different spots, so look in both rather than assume one.
    """
    values = []
    for entry in result.get("rounds", []):
        # `merge_train_metrics` folds aggregate_train's MetricRecord in under a `train_`
        # prefix, to keep it clear of the test_/val_ evaluation metrics. Bare keys are
        # accepted too so this still reads files written by other paths.
        for candidate in (f"train_{key}", key):
            if entry.get(candidate) is not None:
                values.append(float(entry[candidate]))
                break
    return values


def check_colony_searching(result: dict, num_levels: int, flat_fraction: float) -> dict:
    entropies = _round_metric(result, "pheromone_entropy")
    ceiling = math.log(num_levels)
    if not entropies:
        return {
            "question": "Is the colony searching?",
            "verdict": "NO DATA",
            "detail": (
                "no `pheromone_entropy` in the result file. Either this is not a FedACO "
                "run, or it predates the Phase 4 close-out that started logging it."
            ),
        }

    gaps = [(ceiling - e) / ceiling for e in entropies]
    mean_gap = statistics.fmean(gaps)
    final_gap = gaps[-1]
    trend = "falling" if len(gaps) > 2 and gaps[-1] > gaps[0] else "flat or rising"

    if mean_gap < flat_fraction:
        verdict = "INERT"
        detail = (
            f"tau is within {mean_gap:.2%} of uniform on average (entropy "
            f"{statistics.fmean(entropies):.4f} vs ceiling {ceiling:.4f}). Pheromone is "
            "carrying essentially no signal, so selection is driven by eta alone -- the "
            "colony is not searching and cross-round stigmergy is not happening."
        )
    elif final_gap < flat_fraction:
        verdict = "DEGRADING"
        detail = (
            f"tau started structured but ended within {final_gap:.2%} of uniform. "
            "Pheromone is washing out as rounds progress."
        )
    else:
        verdict = "SEARCHING"
        detail = (
            f"tau is {mean_gap:.2%} below uniform on average, {final_gap:.2%} by the "
            f"final round ({trend} entropy). Pheromone carries real structure."
        )
    return {
        "question": "Is the colony searching?",
        "verdict": verdict,
        "detail": detail,
        "entropy_ceiling": ceiling,
        "entropy_per_round": entropies,
        "mean_gap_fraction": mean_gap,
    }


def check_deposit_floor(result: dict) -> dict:
    fitness = _round_metric(result, "best_fitness")
    scales = _round_metric(result, "delta_mean_sq_norm")
    if not fitness:
        return {
            "question": "Has the deposit floor engaged?",
            "verdict": "NO DATA",
            "detail": "no `best_fitness` in the result file.",
        }

    negative = [i + 1 for i, f in enumerate(fitness) if f <= 0.0]
    scale_note = (
        f" delta_mean_sq_norm ranged {min(scales):.4f}–{max(scales):.4f}." if scales else ""
    )
    if not negative:
        verdict = "CLEAR"
        detail = (
            f"best_fitness stayed positive every round "
            f"({min(fitness):.4f}–{max(fitness):.4f}), so every round deposited."
            + scale_note
        )
    elif len(negative) == len(fitness):
        verdict = "ENGAGED"
        detail = (
            f"best_fitness was <= 0 in all {len(fitness)} rounds "
            f"({min(fitness):.4f}–{max(fitness):.4f}): every deposit was zeroed by the "
            "max(F, 0) floor, which fully explains an INERT verdict above." + scale_note
        )
    else:
        verdict = "PARTIAL"
        detail = (
            f"best_fitness was <= 0 in {len(negative)} of {len(fitness)} rounds "
            f"(rounds {negative[:8]}{'...' if len(negative) > 8 else ''}); those rounds "
            "deposited nothing." + scale_note
        )
    return {
        "question": "Has the deposit floor engaged?",
        "verdict": verdict,
        "detail": detail,
        "best_fitness_per_round": fitness,
        "negative_rounds": negative,
    }


def check_beats_fedavg(fedaco: dict, fedavg: dict | None) -> dict:
    fallbacks = _round_metric(fedaco, "fallback_used")
    rate = statistics.fmean(fallbacks) if fallbacks else None
    aco_score = fedaco.get("final", {}).get("final_test_macro_f1")

    if fedavg is None:
        detail = "no FedAvg run found to compare against -- run one with the same seed and partition."
        verdict = "NO BASELINE"
        margin = None
    else:
        avg_score = fedavg.get("final", {}).get("final_test_macro_f1")
        if aco_score is None or avg_score is None:
            return {
                "question": "Does FedACO beat FedAvg?",
                "verdict": "NO DATA",
                "detail": "one of the runs has no final_test_macro_f1.",
            }
        margin = aco_score - avg_score
        verdict = "AHEAD" if margin > 0 else "BEHIND"
        detail = f"FedACO {aco_score:.4f} vs FedAvg {avg_score:.4f} (margin {margin:+.4f})."

    if rate is not None:
        detail += f" Fallback fired in {rate:.0%} of rounds."
        if rate >= 0.5:
            detail += (
                " At that rate FedACO *was* FedAvg for most of the run, so any margin"
                " above is largely FedAvg's own number, not the colony's."
            )
    return {
        "question": "Does FedACO beat FedAvg?",
        "verdict": verdict,
        "detail": detail,
        "fallback_rate": rate,
        "margin": margin,
    }


def wall_clock_note(result: dict) -> str:
    final = result.get("final", {})
    total = final.get("wall_clock_s")
    rounds = final.get("num_rounds_completed")
    if not total or not rounds:
        return "no wall-clock recorded."
    per_round = total / rounds
    run_config = result.get("config", {}).get("run_config", {})
    clients = run_config.get("num-clients", "?")
    epochs = run_config.get("local-epochs", "?")
    return (
        f"{per_round:.1f}s per round ({total:.0f}s over {rounds} rounds, K={clients}, "
        f"{epochs} local epoch(s)). A 100-round run at this shape would take "
        f"{per_round * 100 / 3600:.2f} GPU-hours. docs/EXPERIMENT_LOG.md's compute table "
        "is estimates only and says to replace it with exactly this number."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/fl")
    parser.add_argument("--num-levels", type=int, default=11, help="aco-num-levels used in the run")
    parser.add_argument(
        "--flat",
        type=float,
        default=DEFAULT_FLAT_FRACTION,
        help="fraction below max entropy that counts as uniform tau (default 0.02)",
    )
    parser.add_argument("--out", default=None, help="optional path to write the report JSON")
    parser.add_argument("--strict", action="store_true", help="exit nonzero unless the colony is SEARCHING")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = REPO_ROOT / results_dir

    fedaco_runs = load_results(results_dir, "fedaco")
    fedavg_runs = load_results(results_dir, "fedavg")
    if not fedaco_runs:
        print(f"No FedACO result files under {results_dir}.")
        print("Run one first -- see `make validate-fedaco` or notebooks/colab_fl_smoke.ipynb.")
        return 2

    fedaco = max(fedaco_runs, key=lambda r: r.get("final", {}).get("num_rounds_completed", 0))
    fedavg = (
        max(fedavg_runs, key=lambda r: r.get("final", {}).get("num_rounds_completed", 0))
        if fedavg_runs
        else None
    )

    print(f"FedACO health check\n  run: {fedaco['_path']}")
    if fedavg:
        print(f"  baseline: {fedavg['_path']}")
    print(f"  timing: {wall_clock_note(fedaco)}\n")

    checks = [
        check_colony_searching(fedaco, args.num_levels, args.flat),
        check_deposit_floor(fedaco),
        check_beats_fedavg(fedaco, fedavg),
    ]
    for i, check in enumerate(checks, 1):
        print(f"{i}. {check['question']}")
        print(f"   {check['verdict']}: {check['detail']}\n")

    searching = checks[0]["verdict"] == "SEARCHING"
    if not searching:
        print(
            "The colony is not searching. Before Phase 6 spends GPU-hours, this needs\n"
            "resolving -- a sweep built on an inert mechanism produces results that say\n"
            "nothing about the method's contribution. Start with check 2: if the deposit\n"
            "floor is ENGAGED, the fix is the deposit rule (a rank-based or\n"
            "baseline-centred deposit), not the fitness weights."
        )
    else:
        print("The mechanism is live on real data. Phase 6 can be built against it.")

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"run": fedaco["_path"], "checks": checks}, indent=2))
        print(f"\nWrote {out_path}")

    return 1 if (args.strict and not searching) else 0


if __name__ == "__main__":
    sys.exit(main())
