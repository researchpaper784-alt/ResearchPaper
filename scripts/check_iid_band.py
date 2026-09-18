"""Phase 5 — the IID narrow-band acceptance check.

The argument: FedSwarm's claim is that *searching* aggregation weights beats FedAvg's
fixed n_k weighting **under heterogeneity**. Under a genuinely IID partition there is
almost no heterogeneity to exploit — every client's update is an estimate of the same
gradient — so every aggregation rule should land in a narrow band. If FedACO shows a
large gain under IID, that is not good news for the method; it is evidence that
something other than heterogeneity-handling is driving the number (a bug, a partition
that is not actually IID, an unequal search budget, or an evaluation leak). This script
is the check that turns that reasoning into a gate.

It reads result files, keeps the IID runs, and reports the spread of per-strategy means
against two references:

  * `--band` — the absolute macro-F1 spread treated as acceptable.
  * seed noise — the pooled per-strategy standard deviation across seeds. A spread that
    is inside seed noise is not a real difference regardless of the band, and a spread
    that exceeds the band but not the noise is reported as INCONCLUSIVE rather than
    FAIL, because with few seeds the two are genuinely not distinguishable.

⚠️ `--band`'s default of 0.05 macro-F1 is **this script's choice, not a number taken from
the implementation plan** (which specifies the check but, in the material available in
this repo, no threshold). It is recorded here rather than silently baked in so it can be
replaced with the plan's own figure if one exists. Per CLAUDE.md: not fabricated as if it
were sourced.

Run:
    python scripts/check_iid_band.py --results-dir results/fl
    python scripts/check_iid_band.py --results-dir results/fl --band 0.03 --metric val
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from fedswarm.utils.runner import load_result_files

REPO_ROOT = Path(__file__).resolve().parents[1]

METRIC_KEYS = {
    # The reported metric. Legitimate here: this is a sanity gate on the published
    # number, not a selection step, so it is not the val/test contamination that
    # run_hparam_search.py guards against.
    "test": "final_test_macro_f1",
    "val": "best_val_macro_f1",
}


def load_iid_results(results_dir: Path) -> list[dict]:
    results = []
    for _, result in load_result_files(results_dir, recursive=True):
        config = result.get("config", {})
        run_config = config.get("run_config", {})
        if str(run_config.get("regime", "")).lower() != "iid":
            continue
        if result.get("status") != "completed":
            continue
        results.append(result)
    return results


def group_by_strategy(results: list[dict], metric_key: str) -> dict[str, list[tuple[int, float]]]:
    grouped: dict[str, list[tuple[int, float]]] = {}
    for result in results:
        strategy = str(result.get("config", {}).get("strategy", "unknown"))
        value = result.get("final", {}).get(metric_key)
        if value is None:
            continue
        seed = int(result.get("config", {}).get("seed", result.get("seed", 0)))
        grouped.setdefault(strategy, []).append((seed, float(value)))
    return grouped


def summarize(grouped: dict[str, list[tuple[int, float]]]) -> list[dict]:
    rows = []
    for strategy, pairs in sorted(grouped.items()):
        values = [v for _, v in pairs]
        rows.append(
            {
                "strategy": strategy,
                "n_seeds": len(values),
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "seeds": sorted(s for s, _ in pairs),
            }
        )
    return rows


def evaluate_band(rows: list[dict], band: float) -> dict:
    means = [r["mean"] for r in rows]
    spread = max(means) - min(means) if means else 0.0
    # Pooled across strategies: the typical seed-to-seed wobble of a single strategy,
    # which is the floor below which a between-strategy difference means nothing.
    with_seeds = [r["std"] for r in rows if r["n_seeds"] > 1]
    noise = statistics.fmean(with_seeds) if with_seeds else 0.0
    single_seed = [r["strategy"] for r in rows if r["n_seeds"] < 2]

    if spread <= band:
        verdict = "PASS"
        reason = f"spread {spread:.4f} <= band {band:.4f}"
    elif noise > 0.0 and spread <= 2 * noise:
        verdict = "INCONCLUSIVE"
        reason = (
            f"spread {spread:.4f} exceeds band {band:.4f} but is within 2x seed noise "
            f"({noise:.4f}) -- add seeds before drawing a conclusion"
        )
    else:
        verdict = "FAIL"
        reason = (
            f"spread {spread:.4f} exceeds band {band:.4f}"
            + (f" and 2x seed noise ({noise:.4f})" if noise > 0.0 else " (no seed noise estimate)")
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "spread": spread,
        "band": band,
        "seed_noise": noise,
        "leader": max(rows, key=lambda r: r["mean"])["strategy"] if rows else None,
        "laggard": min(rows, key=lambda r: r["mean"])["strategy"] if rows else None,
        "single_seed_strategies": single_seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/fl")
    parser.add_argument("--band", type=float, default=0.05, help="acceptable macro-F1 spread")
    parser.add_argument("--metric", choices=sorted(METRIC_KEYS), default="test")
    parser.add_argument("--out", default=None, help="optional path to write the report JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero on INCONCLUSIVE as well as FAIL (for use as a CI gate)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = REPO_ROOT / results_dir

    results = load_iid_results(results_dir)
    if not results:
        print(f"No completed IID results found under {results_dir}.")
        print("Run the IID arm of the sweep first; this check has nothing to assert on.")
        return 2

    metric_key = METRIC_KEYS[args.metric]
    rows = summarize(group_by_strategy(results, metric_key))
    if not rows:
        print(f"Found {len(results)} IID result(s) but none carry '{metric_key}'.")
        return 2

    report = evaluate_band(rows, args.band)

    print(f"IID narrow-band check — metric: {metric_key} ({len(results)} runs)\n")
    print(f"{'strategy':14} {'seeds':>5} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
    print("-" * 56)
    for row in rows:
        print(
            f"{row['strategy']:14} {row['n_seeds']:>5} {row['mean']:>8.4f} "
            f"{row['std']:>8.4f} {row['min']:>8.4f} {row['max']:>8.4f}"
        )
    print(
        f"\nspread of means : {report['spread']:.4f} "
        f"({report['laggard']} -> {report['leader']})"
    )
    print(f"band            : {report['band']:.4f}")
    print(f"seed noise      : {report['seed_noise']:.4f}")
    if report["single_seed_strategies"]:
        print(
            f"⚠️  single-seed  : {', '.join(report['single_seed_strategies'])} "
            "-- these contribute no noise estimate"
        )
    print(f"\n{report['verdict']}: {report['reason']}")
    if report["verdict"] != "PASS":
        print(
            "\nUnder IID there is little heterogeneity for a weighting rule to exploit, so a\n"
            "large spread points at something other than the method: check the partition is\n"
            "actually IID (Phase 1.4 measured JS divergence 0.007), that search budgets were\n"
            "matched, and that no run selected on the test split."
        )

    payload = {"metric": metric_key, "rows": rows, **report}
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {out_path}")

    if report["verdict"] == "FAIL":
        return 1
    if report["verdict"] == "INCONCLUSIVE" and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
