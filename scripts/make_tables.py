"""Phase 9, Step 9.3 -- tables CLI. Real formatting logic lives in `fedswarm.tables`
(pure string building, `tests/test_tables.py`); this script loads real results
(`fedswarm.analysis`) and writes `paper/tables/*.tex`.

Run:
  .venv/bin/python scripts/make_tables.py --results-dir results/fl \
      --method fedaco --baselines fedavg fedprox krum trimmed_mean \
      --partitions iid dirichlet_0.3 dirichlet_0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedswarm.analysis import compare_against_baselines, load_results, summarize  # noqa: E402
from fedswarm.tables import comparison_table, main_results_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/fl")
    parser.add_argument("--out-dir", default="paper/tables")
    parser.add_argument("--method", default="fedaco")
    parser.add_argument("--baselines", nargs="+", required=True)
    parser.add_argument("--partitions", nargs="+", required=True)
    args = parser.parse_args()

    df = load_results(args.results_dir)
    if df.empty:
        print(f"No completed results found under {args.results_dir} -- nothing to tabulate.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize(df)
    comparison = compare_against_baselines(df, args.method, args.baselines, args.partitions)

    (out_dir / "main_results.tex").write_text(main_results_table(summary, comparison, method=args.method))
    print(f"Wrote {out_dir / 'main_results.tex'}")

    if not comparison.empty:
        (out_dir / "comparisons.tex").write_text(comparison_table(comparison))
        print(f"Wrote {out_dir / 'comparisons.tex'}")


if __name__ == "__main__":
    main()
