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

from fedswarm.analysis import (  # noqa: E402
    compare_against_baselines,
    load_results,
    summarize,
    with_partition_column,
)


def unmatched_names(df, args) -> list[str]:
    """Names on the command line that no result carries. Empty means every name matched.

    **Why this has to be fatal.** Holm-Bonferroni corrects each p-value by the size of the
    comparison family. A mistyped partition or baseline silently removes comparisons from that
    family, so every surviving p-value is corrected *less* aggressively and the results come
    out looking **more** significant -- the exact failure plan §9.1 warns about ("With 5 seeds
    and a dozen comparisons, uncorrected p-values manufacture significance. A reviewer who
    checks will find it."), arriving through a typo rather than through a missing correction.

    Before this check, `--partitions totally_bogus --baselines nonexistent` printed
    "(no overlapping-seed comparisons found)", wrote both CSVs and exited **0**. The
    partition labels in particular invite it: the regime is `dirichlet` in the config and
    `dirichlet_0.3` in the analysis frame, so `--partitions dirichlet` matches nothing.
    """
    problems = []
    # `load_results` returns regime/alpha, not the combined label; `with_partition_column`
    # builds it, and both `summarize` and `compare_against_baselines` call it internally.
    # Using it here is what makes the names checked the same ones the comparison will match.
    labelled = with_partition_column(df)
    known_strategies = sorted(labelled["strategy"].dropna().unique())
    known_partitions = sorted(labelled["partition"].dropna().unique())

    if args.method not in known_strategies:
        problems.append(
            f"--method {args.method!r} matches no result. Present: {known_strategies}"
        )
    for baseline in args.baselines:
        if baseline not in known_strategies:
            problems.append(
                f"--baselines {baseline!r} matches no result. Present: {known_strategies}"
            )
    for partition in args.partitions:
        if partition not in known_partitions:
            problems.append(
                f"--partitions {partition!r} matches no result. Present: {known_partitions}"
            )
    return problems


def main() -> int:
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
        return 2

    problems = unmatched_names(df, args)
    if problems:
        print("Refusing to aggregate:")
        for problem in problems:
            print(f"  - {problem}")
        print(
            "\nEvery name must match, because Holm-Bonferroni corrects by family size: a "
            "name that matches nothing shrinks the family and inflates the significance of "
            "every comparison that remains."
        )
        return 1

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
