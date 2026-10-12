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
    power_warning: str | None = None,
) -> str:
    """Strategies x partitions, mean +/- std, best (per partition) in bold,
    significance markers from `comparison_df` (fedswarm.analysis.
    compare_against_baselines' output, `p_value_holm`) where `method`'s row is
    the one the marker describes ("method vs. this baseline, at this partition").

    **An `expected_n` column marks cells built from fewer seeds than they should have, and
    the marking goes in the caption.** Without it this emitter printed `0.510 $\\pm$ 0.010`
    for a cell averaging 3 of 8 seeds, character-for-character identical to a complete one --
    while the markdown table beside it showed `3 ⚠️` and two paragraphs of warning. The
    markdown is read by whoever ran the sweep; the LaTeX is what goes in the paper, and the
    main sweep runs over many sessions so a partially-filled table is its normal state for
    weeks. A caption is also the only thing that travels with a table someone copies into a
    draft, which is why the count goes there and not only into a symbol.

    `power_warning` does the same job for the significance legend. At small n the signed-rank
    test's smallest achievable p-value can exceed the thresholds the legend advertises, so a
    caption promising $^{*}p<0.05$ can promise something the test cannot deliver at any data.
    Passing the warning replaces the legend with the truth.
    """
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

    incomplete_cells: list[tuple[str, str, int, int]] = []
    body: list[str] = []
    _table_head = [
        "\\begin{table}[t]",
        "\\centering",
        "__CAPTION__",
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
            if "expected_n" in row and not pd.isna(row["expected_n"].iloc[0]):
                actual, expected = int(row["n"].iloc[0]), int(row["expected_n"].iloc[0])
                if actual < expected:
                    cell += "$^{\\dagger}$"
                    incomplete_cells.append((strategy, partition, actual, expected))
            if strategy != method:
                cell += marker_by_cell.get((partition, strategy), "")
            if best_by_partition.get(partition) == strategy:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        body.append(f"{_escape_latex(strategy)} & " + " & ".join(cells) + " \\\\")

    # The caption is assembled last, because what it has to say depends on what the body
    # turned out to contain. Both additions below are things the markdown table already said
    # and this one silently dropped.
    full_caption = caption
    if incomplete_cells:
        listed = ", ".join(
            f"{_escape_latex(st)}/{_escape_latex(pa)} ({a} of {e})"
            for st, pa, a, e in incomplete_cells
        )
        full_caption += (
            f" $^{{\\dagger}}$~{len(incomplete_cells)} cell(s) are built from fewer seeds "
            f"than planned and are not comparable to the rest: {listed}. "
            "Do not quote these numbers."
        )
    if power_warning:
        # Replaces the legend rather than sitting beside it. A caption that advertises
        # $^{*}p<0.05$ while the test cannot reach 0.05 at this n is worse than one that
        # advertises nothing, because the reader has no way to know.
        full_caption += f" {power_warning}"

    lines = [line.replace("__CAPTION__", f"\\caption{{{full_caption}}}") for line in _table_head]
    lines += body
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
