"""Is the pheromone health check asking a question its own run can answer?

`check_fedaco_health.py` question 1 calls the colony INERT when tau's entropy sits within
`--flat` (default 2%) of its ceiling log(L). That threshold is absolute; tau's distance
from uniform is not. tau starts *at* the ceiling and walks away from it at a rate fixed by
`rho`, `q_deposit`, the fitness magnitude and the iteration budget, so a short run has
barely moved however well the colony is searching.

This script puts a number on that. `fedswarm.aco.diagnostics` computes the best case -- a
colony in which one level wins at every single iteration, the fastest the deposit rule
permits -- and the useful quantity is the *fraction of it* a run must realize to clear the
threshold. On the four-round smoke run that produced the project's only INERT verdict,
that fraction is 64%: to be called "searching", the colony had to concentrate tau at
nearly two thirds the rate of one that had stopped exploring altogether.

It simulates nothing about data. It drives the real pheromone update with a stated,
degenerate input to bound one quantity; the fitness landscape, the partition and the model
play no part.

Run:
    python scripts/analyze_pheromone_dynamics.py                      # budget table
    python scripts/analyze_pheromone_dynamics.py --result results/fl/<run>.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from fedswarm.aco.diagnostics import MEASURED_FITNESS, best_case_gaps

REPO_ROOT = Path(__file__).resolve().parents[1]

# Budgets worth quoting: the shape of the only FedACO run measured so far, and the plan's
# defaults at several run lengths.
BUDGETS = {
    "health smoke run (6->4, R=4)": (6, 4, 4),
    "smoke shape, R=20": (6, 4, 20),
    "plan default (10->4, R=4)": (10, 4, 4),
    "plan default (10->4, R=20)": (10, 4, 20),
    "plan default (10->4, R=100)": (10, 4, 100),
}


def observed_gaps(result: dict, num_levels: int) -> list[float]:
    """Per-round entropy gaps recorded by a real FedACO run."""
    ceiling = math.log(num_levels)
    return [
        (ceiling - log["train_pheromone_entropy"]) / ceiling
        for log in result.get("rounds", [])
        if "train_pheromone_entropy" in log
    ]


def _aco(result: dict, key: str, default):
    return result.get("config", {}).get("run_config", {}).get(key, default)


def _percent(values) -> str:
    return "[" + ", ".join(f"{100 * v:.2f}%" for v in values) + "]"


def report_budget_table(threshold: float, fitness: float) -> None:
    print("Best case -- one level wins at every iteration. No colony concentrates faster.")
    print(f"Deposit driver: rho * Q * F with F = {fitness} (fedswarm.aco.diagnostics).\n")
    header = (
        f"  {'budget':<30} {'mean gap':>9} {'final':>7} "
        f"{'reaches ' + f'{100 * threshold:.0f}%':>12} {'must realize':>13}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, (iters_start, iters_end, rounds) in BUDGETS.items():
        gaps = best_case_gaps(rounds, iters_start, iters_end, fitness=fitness)
        mean_gap = statistics.fmean(gaps)
        crossed = next((i + 1 for i, g in enumerate(gaps) if g >= threshold), None)
        required = threshold / mean_gap if mean_gap else float("inf")
        print(
            f"  {label:<30} {100 * mean_gap:>8.2f}% {100 * gaps[-1]:>6.2f}% "
            f"{('round ' + str(crossed)) if crossed else 'never':>12} "
            f"{(f'{100 * required:.0f}%' if required <= 1 else 'impossible'):>13}"
        )
    print(
        "\n  'must realize' is the fraction of the best case a run has to reach before\n"
        "  question 1 calls it SEARCHING. Above ~60% the test is asking the colony to\n"
        "  concentrate tau nearly as fast as one that has stopped exploring, so at those\n"
        "  budgets an INERT verdict is a statement about the run length, not the mechanism."
    )


def report_against_result(path: Path, threshold: float) -> int:
    result = json.loads(path.read_text())
    num_levels = int(_aco(result, "aco-num-levels", 11))
    observed = observed_gaps(result, num_levels)
    if not observed:
        print(f"No per-round `train_pheromone_entropy` in {path} -- not a FedACO run?")
        return 1

    fitnesses = [r["train_best_fitness"] for r in result["rounds"] if "train_best_fitness" in r]
    best = best_case_gaps(
        len(observed),
        iters_start=int(_aco(result, "aco-iters-start", 10)),
        iters_end=int(_aco(result, "aco-iters-end", 4)),
        fitness=statistics.fmean(fitnesses) if fitnesses else MEASURED_FITNESS,
        num_levels=num_levels,
        num_clients=int(_aco(result, "num-clients", 4)),
        rho=float(_aco(result, "aco-rho", 0.1)),
        tau0=float(_aco(result, "aco-tau0", 1.0)),
        rho_round=float(_aco(result, "aco-rho-round", 0.1)),
        q_deposit=float(_aco(result, "aco-q-deposit", 1.0)),
        global_best_every=int(_aco(result, "aco-global-best-every", 5)),
    )

    observed_mean, best_mean = statistics.fmean(observed), statistics.fmean(best)
    realized = observed_mean / best_mean if best_mean else 0.0
    required = threshold / best_mean if best_mean else float("inf")

    print(f"Run: {path.name}  ({len(observed)} rounds, K={_aco(result, 'num-clients', '?')})\n")
    print(f"  observed gap  {_percent(observed)}  mean {100 * observed_mean:.2f}%")
    print(f"  best case     {_percent(best)}  mean {100 * best_mean:.2f}%")
    print(f"\n  realized {100 * realized:.0f}% of the concentration this budget allows.")
    print(f"  question 1 requires {100 * required:.0f}% of it to report SEARCHING.")

    if required > 1.0:
        print(
            f"\n  VERDICT UNINFORMATIVE: even the best case averages only "
            f"{100 * best_mean:.2f}%, below the\n  {100 * threshold:.0f}% threshold. A colony "
            "concentrating tau as fast as the rule allows would\n  also be called INERT here."
        )
    elif required > 0.6:
        print(
            f"\n  VERDICT UNDERPOWERED: passing needs {100 * required:.0f}% of the best case, "
            "and the best case is a\n  colony that picks the same level every iteration -- one "
            "that has stopped exploring.\n  A colony still exploring cannot clear that bar at "
            "this run length."
        )
    else:
        trend = "rising" if len(observed) > 1 and observed[-1] > observed[0] else "flat or falling"
        print("\n  Threshold is comfortably reachable here, so the verdict means something.")
        print(f"  Observed concentration trend: {trend}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=None, help="a FedACO result JSON")
    parser.add_argument(
        "--flat",
        type=float,
        default=0.02,
        help="the INERT threshold to test against; match check_fedaco_health.py --flat",
    )
    parser.add_argument("--fitness", type=float, default=MEASURED_FITNESS)
    args = parser.parse_args()

    if args.result is not None:
        if not args.result.exists():
            print(f"No such result file: {args.result}")
            return 1
        return report_against_result(args.result, args.flat)

    report_budget_table(args.flat, args.fitness)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
