"""Phase 9, Step 9.1 -- aggregation and statistics CLI. All real logic lives in
`fedswarm.analysis` (pure pandas/scipy, no Flower dependency, unit-tested against
synthetic result-JSON fixtures matching the real schema); this script points it at
the real `results/` directory and prints/saves the summary + comparison tables.

Run:
  .venv/bin/python scripts/aggregate_results.py --results-dir results/fl \
      --method fedaco --baselines fedavg fedprox krum trimmed_mean \
      --partitions iid dirichlet_0.3 dirichlet_0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedswarm.analysis import compare_against_baselines, load_results, summarize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/fl")
    parser.add_argument("--metric", default="final_test_macro_f1")
    parser.add_argument("--method", default="fedaco", help="The strategy every baseline is compared against")
    parser.add_argument("--baselines", nargs="+", required=True)
    parser.add_argument("--partitions", nargs="+", required=True)
    parser.add_argument("--out-summary", default="results/summary.csv")
    parser.add_argument("--out-comparison", default="results/comparison.csv")
    args = parser.parse_args()

    df = load_results(args.results_dir)
    if df.empty:
        print(f"No completed results found under {args.results_dir} -- nothing to aggregate.")
        return

    summary = summarize(df, metric=args.metric)
    print("Per-(strategy, partition) summary:")
    print(summary.to_string(index=False))

    comparison = compare_against_baselines(df, args.method, args.baselines, args.partitions, metric=args.metric)
    print(f"\n{args.method} vs. baselines (Wilcoxon + Holm-Bonferroni + Cohen's d):")
    if comparison.empty:
        print("  (no overlapping-seed comparisons found)")
    else:
        print(comparison.to_string(index=False))

    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_summary, index=False)
    comparison.to_csv(args.out_comparison, index=False)
    print(f"\nWrote {args.out_summary} and {args.out_comparison}")


if __name__ == "__main__":
    main()
