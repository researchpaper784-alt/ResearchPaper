"""Phase 9, Step 9.3 -- fedswarm.tables, against synthetic data. Checks structural
correctness of the generated LaTeX (booktabs markers present, right row/column
counts, best-value bolding, significance markers) -- not that it typesets a
specific way, which needs a real LaTeX run to verify."""

from __future__ import annotations

import pandas as pd

from fedswarm.tables import comparison_table, main_results_table, overhead_table


def test_main_results_table_bolds_the_best_strategy_per_partition() -> None:
    summary = pd.DataFrame(
        [
            {"strategy": "fedavg", "partition": "iid", "mean": 0.6, "std": 0.02, "n": 3},
            {"strategy": "fedaco", "partition": "iid", "mean": 0.7, "std": 0.01, "n": 3},
        ]
    )
    latex = main_results_table(summary)
    assert "\\textbf{0.700" in latex
    assert "\\textbf{0.600" not in latex
    assert "\\toprule" in latex and "\\bottomrule" in latex
    assert "\\begin{tabular}{lc}" in latex  # 1 partition -> 1 data column


def test_main_results_table_includes_significance_markers_from_comparison() -> None:
    summary = pd.DataFrame(
        [
            {"strategy": "fedavg", "partition": "iid", "mean": 0.5, "std": 0.01, "n": 3},
            {"strategy": "fedaco", "partition": "iid", "mean": 0.7, "std": 0.01, "n": 3},
        ]
    )
    comparison = pd.DataFrame(
        [{"partition": "iid", "baseline": "fedavg", "p_value_holm": 0.0001, "cohens_d": 3.0,
          "method_mean": 0.7, "baseline_mean": 0.5}]
    )
    latex = main_results_table(summary, comparison, method="fedaco")
    assert "0.500" in latex and "***" in latex


def test_main_results_table_escapes_underscores() -> None:
    summary = pd.DataFrame(
        [{"strategy": "trimmed_mean", "partition": "dirichlet_0.3", "mean": 0.5, "std": 0.01, "n": 3}]
    )
    latex = main_results_table(summary)
    assert "trimmed\\_mean" in latex
    assert "dirichlet\\_0.3" in latex


def test_main_results_table_handles_a_missing_strategy_partition_cell() -> None:
    summary = pd.DataFrame(
        [
            {"strategy": "fedavg", "partition": "iid", "mean": 0.5, "std": 0.01, "n": 3},
            {"strategy": "fedaco", "partition": "dirichlet_0.3", "mean": 0.6, "std": 0.02, "n": 3},
        ]
    )
    latex = main_results_table(summary)  # fedavg has no dirichlet_0.3 row, fedaco no iid row
    assert "--" in latex


def test_comparison_table_has_one_row_per_input_row() -> None:
    comparison = pd.DataFrame(
        [
            {"partition": "iid", "baseline": "fedavg", "p_value_holm": 0.02, "cohens_d": 1.5,
             "method_mean": 0.7, "baseline_mean": 0.5},
            {"partition": "dirichlet_0.3", "baseline": "krum", "p_value_holm": float("nan"), "cohens_d": 0.1,
             "method_mean": 0.6, "baseline_mean": 0.59},
        ]
    )
    latex = comparison_table(comparison)
    assert latex.count("\\\\") >= 2 + 1  # 2 data rows + header row, at least
    assert "n/a" in latex  # the nan p-value row


def test_overhead_table_lists_every_k_value() -> None:
    latex = overhead_table([10, 20, 50], [12.0, 45.0, 280.0])
    assert "K=10" in latex and "K=20" in latex and "K=50" in latex
    assert "12.0" in latex and "280.0" in latex
