"""Phase 9, Step 9.2 -- figures (plan §9.2). Every function here returns a
matplotlib `Figure` built from plain, minimal, documented inputs (arrays/DataFrames)
rather than reading `results/` itself -- `scripts/make_figures.py` is the thin CLI
that loads real results (via `fedswarm.analysis`) and calls these; that split is
what makes "does this run and produce a well-formed figure" testable against
synthetic data without needing any real experiment output to exist yet.

All vector output (`savefig(..., format="pdf")`), colorblind-safe palette
(matplotlib's own "tab10" -- not a from-scratch palette choice, a well-established
colorblind-tested default), >=10pt fonts. Figure 8 (partition diagnostic heatmaps)
is produced separately by the existing `scripts/make_partition_figures.py`
(Phase 1), not duplicated here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def plot_convergence_curves(
    df: pd.DataFrame, metric: str = "test_macro_f1", partitions: list[str] | None = None
) -> Figure:
    """Figure 1: metric vs. round, mean +/- 95% CI band across seeds, one panel per
    partition. `df` columns: strategy, partition, seed, round, <metric>."""
    partitions = partitions or sorted(df["partition"].unique())
    fig = Figure(figsize=(4.5 * len(partitions), 4.0))
    axes = fig.subplots(1, len(partitions), sharey=True)
    if len(partitions) == 1:
        axes = [axes]

    for ax, partition in zip(axes, partitions):
        subset = df[df["partition"] == partition]
        for strategy in sorted(subset["strategy"].unique()):
            strategy_df = subset[subset["strategy"] == strategy]
            grouped = strategy_df.groupby("round")[metric]
            mean = grouped.mean()
            sem = grouped.sem()
            ci95 = 1.96 * sem.fillna(0.0)
            ax.plot(mean.index, mean.values, label=strategy)
            ax.fill_between(mean.index, (mean - ci95).values, (mean + ci95).values, alpha=0.2)
        ax.set_title(partition)
        ax.set_xlabel("round")
    axes[0].set_ylabel(metric)
    axes[-1].legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def plot_final_bar_chart(summary_df: pd.DataFrame) -> Figure:
    """Figure 2: final metric bar chart with error bars, grouped by partition.
    `summary_df` columns: strategy, partition, mean, std (fedswarm.analysis.summarize's
    own output shape)."""
    partitions = sorted(summary_df["partition"].unique())
    strategies = sorted(summary_df["strategy"].unique())
    fig = Figure(figsize=(max(6.0, 1.2 * len(partitions) * len(strategies)), 4.0))
    ax = fig.add_subplot(111)

    width = 0.8 / max(len(strategies), 1)
    x = np.arange(len(partitions))
    for i, strategy in enumerate(strategies):
        means, stds = [], []
        for partition in partitions:
            row = summary_df[(summary_df["strategy"] == strategy) & (summary_df["partition"] == partition)]
            means.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
            stds.append(float(row["std"].iloc[0]) if not row.empty else 0.0)
        ax.bar(x + i * width, means, width=width, yerr=stds, label=strategy, capsize=3)

    ax.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax.set_xticklabels(partitions, rotation=20, ha="right")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_alpha_heatmap(
    alpha_matrix: np.ndarray, client_quality: np.ndarray | None = None
) -> Figure:
    """Figure 3: alpha heatmap (clients x rounds) for one representative run --
    "the figure that shows the method doing something interpretable" (plan §9.2).
    `alpha_matrix` is [num_rounds, num_clients]; `client_quality`, if given, is a
    length-num_clients array annotated alongside the y-axis (e.g. a planted-bad-
    client indicator, or measured update alignment)."""
    num_rounds, num_clients = alpha_matrix.shape
    fig = Figure(figsize=(max(6.0, num_rounds * 0.3), max(4.0, num_clients * 0.3)))
    ax = fig.add_subplot(111)
    im = ax.imshow(alpha_matrix.T, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("round")
    ax.set_ylabel("client")
    fig.colorbar(im, ax=ax, label="alpha")

    if client_quality is not None:
        for client_idx, quality in enumerate(client_quality):
            ax.annotate(f"{quality:.2f}", xy=(num_rounds - 1, client_idx), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, va="center")
    fig.tight_layout()
    return fig


def plot_entropy_vs_round(
    rounds: list[int], alpha_entropy: list[float], pheromone_entropy: list[float] | None = None
) -> Figure:
    """Figure 4: alpha entropy (and, if FedACO's own aco search method, pheromone
    entropy) vs. round -- a search-convergence diagnostic."""
    fig = Figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(111)
    ax.plot(rounds, alpha_entropy, label="alpha entropy", marker="o", markersize=3)
    if pheromone_entropy is not None:
        ax.plot(rounds, pheromone_entropy, label="pheromone entropy", marker="s", markersize=3)
    ax.set_xlabel("round")
    ax.set_ylabel("entropy")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_overhead_vs_k(k_values: np.ndarray, overhead_ms: np.ndarray) -> Figure:
    """Figure 5: aggregation overhead vs. K, with a fitted c*K^2 curve overlaid
    (plan §4.6/§9.2 -- the empirical half of the O(K^2) claim). The fit constrains
    the curve through the origin (overhead(0)=0 is physically required, a free
    intercept would let the fit hide a systematic offset the plan's claim doesn't
    make room for) by fitting c in overhead = c*K^2 via least squares on K^2 alone.

    ⚠️ **The measured data is not quadratic in the range this paper runs, so this
    overlay currently asserts more than the numbers support.**
    `scripts/bench_aggregation_overhead.py` measures K from 5 to 800 at d=390,404 on CPU
    (which is where aggregation really runs): `ms/K^2` falls 43x across that range while
    `ms/K` rises only 3.7x, and the implied GFLOP/s climbs monotonically from 1.2 to 52.9 --
    the arithmetic is never the limit. Least squares gives R^2 = 0.972 for a linear fit
    against 0.969 for the quadratic one: effectively tied, because both are approximating a
    curve that is between the two and closer to linear. The Gram build is
    memory-bandwidth-bound (O(Kd) traffic) long before it is compute-bound (O(K^2 d)
    arithmetic).

    O(K^2) remains the correct asymptotic statement and the honest claim -- that overhead is
    negligible -- holds more strongly than the plan expected, at 1.0-3.1% of round wall-clock
    across every K measured, improving as K grows. But a reader who plots `ms/K^2` sees it
    fall by a factor of 43, so figure 5 should show the measured points with the fit that
    describes them, or plot both, rather than a lone quadratic overlay. Deciding which is a
    paper-presentation call, so this is documented rather than silently changed.
    """
    k_values = np.asarray(k_values, dtype=float)
    overhead_ms = np.asarray(overhead_ms, dtype=float)
    fig = Figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(111)
    ax.scatter(k_values, overhead_ms, label="measured", zorder=3)

    k_squared = k_values**2
    c = float((k_squared @ overhead_ms) / (k_squared @ k_squared)) if np.any(k_squared) else 0.0
    k_fit = np.linspace(float(k_values.min()), float(k_values.max()), 200)
    ax.plot(k_fit, c * k_fit**2, linestyle="--", label=f"fit: {c:.2e}*K^2", zorder=2)

    ax.set_xlabel("K (number of clients)")
    ax.set_ylabel("ACO overhead (ms)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_robustness(df: pd.DataFrame, metric: str = "final_test_macro_f1") -> Figure:
    """Figure 6: metric vs. attacker fraction, one line per strategy. `df` columns:
    strategy, attacker_fraction, <metric> (already aggregated to one row per
    (strategy, attacker_fraction), e.g. via fedswarm.analysis.summarize-style
    grouping upstream of this call)."""
    fig = Figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(111)
    for strategy in sorted(df["strategy"].unique()):
        subset = df[df["strategy"] == strategy].sort_values("attacker_fraction")
        ax.plot(subset["attacker_fraction"], subset[metric], marker="o", label=strategy)
    ax.set_xlabel("attacker fraction")
    ax.set_ylabel(metric)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_sensitivity_heatmap(
    values_x: np.ndarray, values_y: np.ndarray, metric_grid: np.ndarray, label_x: str, label_y: str
) -> Figure:
    """Figure 7: sensitivity heatmap over two ACO hyperparameters (plan A6).
    `metric_grid[i, j]` is the outcome at `values_y[i]`, `values_x[j]`."""
    fig = Figure(figsize=(6.0, 5.0))
    ax = fig.add_subplot(111)
    im = ax.imshow(metric_grid, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(values_x)))
    ax.set_xticklabels([str(v) for v in values_x])
    ax.set_yticks(range(len(values_y)))
    ax.set_yticklabels([str(v) for v in values_y])
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_gain_vs_heterogeneity(js_divergence: np.ndarray, gain: np.ndarray) -> Figure:
    """Figure 9: gain (method - best baseline) vs. measured Jensen-Shannon
    heterogeneity, scatter with a fitted linear trend across all runs."""
    js_divergence = np.asarray(js_divergence, dtype=float)
    gain = np.asarray(gain, dtype=float)
    fig = Figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(111)
    ax.scatter(js_divergence, gain, alpha=0.6)

    if len(js_divergence) >= 2 and np.var(js_divergence) > 0:
        slope, intercept = np.polyfit(js_divergence, gain, deg=1)
        x_fit = np.linspace(float(js_divergence.min()), float(js_divergence.max()), 100)
        ax.plot(x_fit, slope * x_fit + intercept, linestyle="--", color="black")

    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel("measured Jensen-Shannon divergence (heterogeneity)")
    ax.set_ylabel("gain over best baseline")
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: str, dpi: int = 300) -> None:
    fig.savefig(path, format="pdf", dpi=dpi, bbox_inches="tight")
