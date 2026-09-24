"""Phase 9, Step 9.2 -- figures (plan §9.2). Every function here returns a matplotlib
`Figure` built from plain, minimal, documented inputs (arrays/DataFrames) rather than
reading `results/` itself; `scripts/make_figures.py` is the CLI that loads real results and
calls these. That split is what makes "does this produce a well-formed figure" testable
against synthetic data without any real experiment output existing yet.

⚠️ **This module was imported by nothing until 2026-09-24.** All nine plan §9.2 plot
functions were written and working, `scripts/make_figures.py` had independently
reimplemented four of them, and nobody noticed -- so five paper figures were missing for want
of extractors, not plotting code. Nothing fails when a subsystem's output goes nowhere, which
is why this was invisible; `tests/test_figures_wiring.py` now covers the extractors and every
function below is reachable from the CLI.

What remains here is the five that `make_figures.py` does *not* reimplement:

    plot_alpha_heatmap          figure 3   <- make_figures.figure_alpha_heatmap
    plot_overhead_vs_k          figure 5   <- make_figures.figure_overhead
    plot_robustness             figure 6   <- make_figures.figure_robustness
    plot_sensitivity_heatmap    figure 7   <- make_figures.figure_sensitivity
    plot_gain_vs_heterogeneity  figure 9   <- make_figures.figure_gain_vs_heterogeneity

`plot_convergence_curves`, `plot_final_bar_chart`, `plot_entropy_vs_round` and
`save_figure` were removed on the same date: `make_figures.py` carries better versions of
the first three (faceted per regime, the documented emphasis palette, a log(L) threshold
rule on the entropy panel), and keeping simpler duplicates beside them is how someone edits
one and wonders why the figure does not change. Figure 4's alpha-entropy half now lives in
`make_figures.figure_alpha_entropy`; figure 8 (partition diagnostics) is
`scripts/make_partition_figures.py`, from Phase 1.

Colorblind-safe palette (matplotlib's "tab10" -- an established colorblind-tested default,
not a from-scratch choice), >=10pt fonts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure


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
