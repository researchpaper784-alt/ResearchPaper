"""Phase 9, Step 9.2 -- figures CLI. Real plotting logic lives in `fedswarm.figures`
(pure matplotlib, testable against synthetic data, `tests/test_figures.py`); this
script loads real results (`fedswarm.analysis`) and saves whichever figures the
loaded data actually supports.

Only figures 1 (convergence) and 2 (final bar chart) are produced from generic
`results/*.json` alone -- figures 3 (alpha heatmap), 4 (entropy), 5 (overhead vs K),
6 (robustness), 7 (sensitivity), 9 (gain vs heterogeneity) each need a specific
sweep's output (e.g. figure 5 needs `configs/experiment/overhead.yaml`'s results) and
are better built with a small dedicated script once that data exists, reusing
`fedswarm.figures`'s functions directly -- not duplicated here as speculative code
against data that doesn't exist yet. Figure 8 (partition diagnostics) is
`scripts/make_partition_figures.py`, already built in Phase 1.

Run:
  .venv/bin/python scripts/make_figures.py --results-dir results/fl --out-dir paper/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedswarm.analysis import load_results, summarize, with_partition_column  # noqa: E402
from fedswarm.figures import plot_convergence_curves, plot_final_bar_chart, save_figure  # noqa: E402


def _rounds_to_long_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, run in with_partition_column(df).iterrows():
        for entry in run["rounds"]:
            if "test_macro_f1" not in entry:
                continue
            rows.append(
                {
                    "strategy": run["strategy"],
                    "partition": run["partition"],
                    "seed": run["seed"],
                    "round": entry["round"],
                    "test_macro_f1": entry["test_macro_f1"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/fl")
    parser.add_argument("--out-dir", default="paper/figures")
    args = parser.parse_args()

    df = load_results(args.results_dir)
    if df.empty:
        print(f"No completed results found under {args.results_dir} -- nothing to plot.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df = _rounds_to_long_df(df)
    if not long_df.empty:
        save_figure(plot_convergence_curves(long_df), str(out_dir / "convergence_curves.pdf"))
        print(f"Wrote {out_dir / 'convergence_curves.pdf'}")

    summary = summarize(df)
    save_figure(plot_final_bar_chart(summary), str(out_dir / "final_macro_f1_bars.pdf"))
    print(f"Wrote {out_dir / 'final_macro_f1_bars.pdf'}")


if __name__ == "__main__":
    main()
