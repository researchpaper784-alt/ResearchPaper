"""Phase 9, Step 9.3 -- LaTeX tables (plan §9.3): booktabs, best-in-bold,
significance markers. Pure string formatting from a `fedswarm.analysis`-shaped
DataFrame -- no Flower dependency, fully testable against synthetic frames.
"""

from __future__ import annotations

import pandas as pd


def _escape_latex(text: str) -> str:
    return str(text).replace("_", r"\_")


def _significance_marker(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def main_results_table(
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame | None = None,
    method: str = "fedaco",
    caption: str = "Main results: test macro-F1 (mean $\\pm$ std over seeds).",
    label: str = "tab:main-results",
) -> str:
    """Strategies x partitions, mean +/- std, best (per partition) in bold,
    significance markers from `comparison_df` (fedswarm.analysis.
    compare_against_baselines' output, `p_value_holm`) where `method`'s row is
    the one the marker describes ("method vs. this baseline, at this partition")."""
    partitions = sorted(summary_df["partition"].unique())
    strategies = sorted(summary_df["strategy"].unique())

    best_by_partition = {}
    for partition in partitions:
        subset = summary_df[summary_df["partition"] == partition]
        if not subset.empty:
            best_by_partition[partition] = subset.loc[subset["mean"].idxmax(), "strategy"]

    marker_by_cell: dict[tuple[str, str], str] = {}
    if comparison_df is not None and not comparison_df.empty:
        for _, row in comparison_df.iterrows():
            marker_by_cell[(row["partition"], row["baseline"])] = _significance_marker(row["p_value_holm"])

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "c" * len(partitions) + "}",
        "\\toprule",
        "Strategy & " + " & ".join(_escape_latex(p) for p in partitions) + " \\\\",
        "\\midrule",
    ]
    for strategy in strategies:
        cells = []
        for partition in partitions:
            row = summary_df[(summary_df["strategy"] == strategy) & (summary_df["partition"] == partition)]
            if row.empty:
                cells.append("--")
                continue
            mean, std = float(row["mean"].iloc[0]), float(row["std"].iloc[0])
            cell = f"{mean:.3f} $\\pm$ {std:.3f}"
            if strategy != method:
                cell += marker_by_cell.get((partition, strategy), "")
            if best_by_partition.get(partition) == strategy:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(f"{_escape_latex(strategy)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def comparison_table(
    comparison_df: pd.DataFrame,
    caption: str = "Paired comparisons against FedACO (Wilcoxon signed-rank, Holm-Bonferroni corrected).",
    label: str = "tab:comparisons",
) -> str:
    """Every row of `comparison_df` (fedswarm.analysis.compare_against_baselines'
    output) as its own table row: partition, baseline, means, Cohen's d, corrected
    p-value."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Partition & Baseline & Method mean & Baseline mean & Cohen's $d$ & $p$ (Holm) \\\\",
        "\\midrule",
    ]
    for _, row in comparison_df.iterrows():
        p_holm = row["p_value_holm"]
        p_str = "n/a" if pd.isna(p_holm) else f"{p_holm:.3g}{_significance_marker(p_holm)}"
        lines.append(
            f"{_escape_latex(row['partition'])} & {_escape_latex(row['baseline'])} & "
            f"{row['method_mean']:.3f} & {row['baseline_mean']:.3f} & "
            f"{row['cohens_d']:.2f} & {p_str} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def overhead_table(
    k_values: list[int],
    overhead_ms: list[float],
    caption: str = "FedACO aggregation overhead (ms) vs. number of clients K.",
    label: str = "tab:overhead",
) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{" + "c" * len(k_values) + "}",
        "\\toprule",
        " & ".join(f"K={k}" for k in k_values) + " \\\\",
        "\\midrule",
        " & ".join(f"{v:.1f}" for v in overhead_ms) + " \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)
