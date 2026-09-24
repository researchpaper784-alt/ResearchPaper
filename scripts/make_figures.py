"""Phase 9 — the paper's figures.

Four figures, each chosen by what its data has to say rather than by what looks full:

1. **Convergence** (line, faceted by regime) — change over time.
2. **Final comparison** (dot plot with error bars, faceted by regime) — magnitude.
3. **Ablation deltas** (diverging bars) — polarity: better or worse than the control.
4. **Colony health** (line with a threshold rule) — whether the mechanism ran at all.

Design decisions worth stating, because each is a choice that a default would get wrong:

**Emphasis, not twelve hues.** The sweep has 12 strategies, and a categorical palette
carries at most ~8 before adjacent hues blur — past that the honest move is to fold the
rest, not to generate more colors. The paper's claim is FedACO against FedAvg, so those
two get the first two categorical slots and every other baseline renders as one muted
gray band. That reads as what it is: the two series under discussion, against the field.
The palette is the validated default (`#2a78d6` / `#eb6834`); both slots clear every
check all-pairs in light mode, including the 3:1 contrast floor, so no relief rule is
owed.

**No dual axis, anywhere.** Macro-F1 and pheromone entropy are different scales and get
different figures, never twinned y-axes on one plot.

**Dashing is reserved for the one real threshold.** Gridlines and axes are solid
hairlines; the only dashed rule in these figures is log(L), the pheromone-entropy
ceiling, which genuinely is a threshold rather than decoration.

**These are static figures for a paper**, so there is no hover layer — the interactive
complement is the table view `scripts/make_tables.py` already emits, which carries every
value plotted here plus the ones the figures deliberately leave to the axis. They are
stepped for the light print surface only; a dark variant would need its own steps from
the same ramps rather than an inverted flip, and paper figures do not need one.

Run:
    python scripts/make_figures.py --results-dir results/fl/main
    python scripts/make_figures.py --results-dir results/fl/ablation --only ablation
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Reused rather than reimplemented: the figures and the tables must group, label and
# judge completeness identically, or a figure and its table will disagree about what a
# cell is.
from make_tables import add_deltas, cell_key, load_results, summarize  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# The nine plot functions plan §9.2 specifies already exist here. Four were reimplemented
# in this file before anyone noticed; these five are wired by the extractors below.
from fedswarm.figures import (  # noqa: E402
    plot_alpha_heatmap,
    plot_gain_vs_heterogeneity,
    plot_overhead_vs_k,
    plot_robustness,
    plot_sensitivity_heatmap,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# The validated default palette, light surface. Categorical slots 1 and 2 carry the two
# series under discussion; everything else is chrome. See references/palette.md.
SERIES_1 = "#2a78d6"  # FedACO — the method
SERIES_2 = "#eb6834"  # FedAvg — the reference every claim is relative to
DIVERGING_BETTER = "#2a78d6"
DIVERGING_WORSE = "#e34948"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
FIELD = "#b8b7b0"  # the un-highlighted baselines, as one visual group

HIGHLIGHT = {"fedaco": SERIES_1, "fedavg": SERIES_2}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            # Solid hairlines. Dashed gridlines read as "threshold" when they are just a
            # grid, and the only genuine threshold in these figures is log(L).
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "legend.labelcolor": INK_SECONDARY,
        }
    )


def _despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _rounds(curve: list[float]) -> range:
    """x positions for a round curve: 1-based, matching the round numbers everywhere else.

    `build_evaluate_fn` logs `"round": server_round + round_offset` and flwr numbers
    server_round from 1, so a 100-round run logs rounds 1..100. Plotting against
    `range(len(curve))` put the first logged round at x=0, so every figure's round axis
    was off by one against the round numbers in the result JSONs and in
    `paper/tables/*.csv`, and a 100-round run's last point read as round 99.
    """
    return range(1, len(curve) + 1)


def _integer_rounds(ax) -> None:
    """Round numbers are integers, so the round axis must tick as integers.

    Matplotlib's default locator picks fractional ticks when the range is short -- a
    2-round smoke run came out labelled 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, i.e. five rounds
    that do not exist. Harmless-looking on a 100-round sweep, nonsense on the short runs
    that get looked at first.
    """
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))


def _regimes_of(results: list[dict]) -> list[str]:
    return sorted({cell_key(r)[1] for r in results})


# ======================================================================================
# 1. Convergence
# ======================================================================================


def figure_convergence(results: list[dict], out: Path) -> Path | None:
    """Test macro-F1 against round, one panel per regime.

    Curves are averaged over seeds so a panel shows one line per strategy rather than
    five overlapping ones -- with 12 strategies x 5 seeds, per-seed curves would be 60
    lines and legible as none of them.
    """
    regimes = _regimes_of(results)
    if not regimes:
        return None

    by_cell: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    for result in results:
        strategy, regime, variant = cell_key(result)
        if variant != "default":
            continue
        curve = [
            entry.get("test_macro_f1")
            for entry in result.get("rounds", [])
            if entry.get("test_macro_f1") is not None
        ]
        if curve:
            by_cell[(strategy, regime)].append(curve)
    if not by_cell:
        return None

    cols = min(len(regimes), 3)
    rows = math.ceil(len(regimes) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)

    for index, regime in enumerate(regimes):
        ax = axes[index // cols][index % cols]
        _despine(ax)
        ax.set_title(regime)
        ax.set_xlabel("round")
        ax.set_ylabel("test macro-F1")
        _integer_rounds(ax)

        # Field first, so the two series under discussion draw on top of it.
        for (strategy, cell_regime), curves in sorted(by_cell.items()):
            if cell_regime != regime or strategy in HIGHLIGHT:
                continue
            mean_curve = _mean_curve(curves)
            ax.plot(_rounds(mean_curve), mean_curve, color=FIELD, linewidth=1.0, zorder=1)

        endpoints = []
        for strategy, color in HIGHLIGHT.items():
            curves = by_cell.get((strategy, regime))
            if not curves:
                continue
            mean_curve = _mean_curve(curves)
            ax.plot(
                _rounds(mean_curve),
                mean_curve,
                color=color,
                linewidth=2.0,
                zorder=3,
                label=strategy,
            )
            endpoints.append((len(mean_curve), mean_curve[-1], color))

        # Direct-label the endpoint only -- a value on every point is unreadable and goes
        # unread; the axis and the table view carry the rest. FedACO and FedAvg converge
        # to within a hair of each other in exactly the regimes that matter, so the two
        # labels are pushed apart when they would otherwise overprint, and the x-limit is
        # extended so neither runs off the panel.
        _label_endpoints(ax, endpoints)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            field_line = plt.Line2D([], [], color=FIELD, linewidth=1.0)
            ax.legend(
                handles + [field_line],
                labels + ["other baselines"],
                loc="lower right",
            )

    for index in range(len(regimes), rows * cols):
        axes[index // cols][index % cols].set_visible(False)

    fig.suptitle("Convergence by partition regime", color=INK_PRIMARY, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _label_endpoints(ax, endpoints: list[tuple[float, float, str]]) -> None:
    """Annotate each series' last value, nudged apart if two would collide."""
    if not endpoints:
        return
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    gap = 0.045 * span  # below this the two labels visibly overprint
    ordered = sorted(endpoints, key=lambda e: e[1])
    adjusted = []
    for x, y, color in ordered:
        if adjusted and y - adjusted[-1][1] < gap:
            y = adjusted[-1][1] + gap
        adjusted.append((x, y, color))
    for (x, y_label, color), (_, y_true, _) in zip(adjusted, ordered):
        ax.annotate(
            f"{y_true:.3f}",
            xy=(x, y_label),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=8,
            va="center",
            annotation_clip=False,
        )
    # Room on the right for the labels, so they sit beside the plot rather than over the
    # spine or clipped by the panel edge.
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min, x_max + 0.10 * (x_max - x_min))


def _mean_curve(curves: list[list[float]]) -> list[float]:
    """Mean across seeds, truncated to the shortest run.

    Truncating rather than padding matters: a seed that stopped early would otherwise
    drag the tail of the mean toward its last value and invent a plateau that no run
    actually produced.
    """
    length = min(len(c) for c in curves)
    return [sum(c[i] for c in curves) / len(curves) for i in range(length)]


# ======================================================================================
# 2. Final comparison
# ======================================================================================


def figure_comparison(rows: list[dict], out: Path) -> Path | None:
    """Final macro-F1 per strategy, one panel per regime, sorted within panel.

    A dot plot with error bars rather than bars: the values sit in a narrow band well
    away from zero, and bars would either start at zero (compressing every difference
    into nothing) or be truncated (exaggerating them).
    """
    rows = [r for r in rows if r["variant"] == "default"]
    regimes = sorted({r["regime"] for r in rows})
    if not regimes:
        return None

    cols = min(len(regimes), 3)
    panel_rows = math.ceil(len(regimes) / cols)
    height = max(3.0, 0.32 * max(len(set(r["strategy"] for r in rows)), 1) + 1.2)
    fig, axes = plt.subplots(
        panel_rows, cols, figsize=(4.4 * cols, height * panel_rows), squeeze=False
    )

    for index, regime in enumerate(regimes):
        ax = axes[index // cols][index % cols]
        _despine(ax)
        ax.set_title(regime)
        ax.set_xlabel("final test macro-F1")
        ax.grid(axis="y", visible=False)

        subset = sorted(
            [r for r in rows if r["regime"] == regime], key=lambda r: r["mean"]
        )
        labels, means, errs, colors = [], [], [], []
        for row in subset:
            mark = f"{row['strategy']} (n={row['n']})" if row["incomplete"] else row["strategy"]
            labels.append(mark)
            means.append(row["mean"])
            errs.append(row["std"])
            colors.append(HIGHLIGHT.get(row["strategy"], FIELD))

        positions = range(len(subset))
        for pos, mean, err, color in zip(positions, means, errs, colors):
            ax.errorbar(
                mean, pos, xerr=err, fmt="o", color=color,
                markersize=6, elinewidth=1.2, capsize=3, zorder=3,
            )
        ax.set_yticks(list(positions))
        ax.set_yticklabels(labels)
        for tick, row in zip(ax.get_yticklabels(), subset):
            # Incomplete cells are called out on the axis itself, not only in the table:
            # a reader comparing a 5-seed mean to a 2-seed mean should see it here.
            tick.set_color(INK_PRIMARY if row["strategy"] in HIGHLIGHT else INK_SECONDARY)

    for index in range(len(regimes), panel_rows * cols):
        axes[index // cols][index % cols].set_visible(False)

    fig.suptitle(
        "Final macro-F1 by strategy (mean ± std over seeds)",
        color=INK_PRIMARY,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# ======================================================================================
# 3. Ablation deltas
# ======================================================================================


def figure_ablations(rows: list[dict], out: Path) -> Path | None:
    """Each ablation's distance from the FedACO control, per regime.

    Diverging because the question is polarity -- did removing this help or hurt -- so
    the two arms take the diverging pair (blue/red) around a neutral zero rule, never a
    single hue whose length would read as magnitude alone.
    """
    control = {
        r["regime"]: r["mean"]
        for r in rows
        if r["strategy"] == "fedaco" and r["variant"] == "default"
    }
    ablated = [r for r in rows if r["strategy"] == "fedaco" and r["variant"] != "default"]
    if not ablated or not control:
        return None

    regimes = sorted({r["regime"] for r in ablated if r["regime"] in control})
    if not regimes:
        return None

    cols = min(len(regimes), 3)
    panel_rows = math.ceil(len(regimes) / cols)
    variants = sorted({r["variant"] for r in ablated})
    height = max(3.0, 0.34 * len(variants) + 1.2)
    fig, axes = plt.subplots(
        panel_rows, cols, figsize=(4.6 * cols, height * panel_rows), squeeze=False
    )

    for index, regime in enumerate(regimes):
        ax = axes[index // cols][index % cols]
        _despine(ax)
        ax.set_title(regime)
        ax.set_xlabel("Δ macro-F1 vs FedACO default")
        ax.grid(axis="y", visible=False)

        subset = sorted(
            [r for r in ablated if r["regime"] == regime],
            key=lambda r: r["mean"] - control[r["regime"]],
        )
        deltas = [r["mean"] - control[regime] for r in subset]
        colors = [DIVERGING_BETTER if d >= 0 else DIVERGING_WORSE for d in deltas]
        positions = range(len(subset))
        ax.barh(list(positions), deltas, color=colors, height=0.55, zorder=3)
        ax.axvline(0, color=AXIS, linewidth=1.0, zorder=2)
        # Deltas are small numbers, so the default tick density puts five 6-character
        # labels in a narrow panel and they run together into one unreadable string.
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, prune=None))
        ax.set_yticks(list(positions))
        ax.set_yticklabels([r["variant"] for r in subset], color=INK_SECONDARY)

    for index in range(len(regimes), panel_rows * cols):
        axes[index // cols][index % cols].set_visible(False)

    fig.suptitle(
        "Ablation effect on FedACO (negative = removing it hurt)",
        color=INK_PRIMARY,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# ======================================================================================
# 4. Colony health
# ======================================================================================


def figure_colony_health(results: list[dict], out: Path, num_levels: int) -> Path | None:
    """Pheromone entropy per round against its log(L) ceiling, one panel per regime.

    The one figure that answers whether the method's stated mechanism ran. Entropy pinned
    at the ceiling means tau is uniform, `tau^a * eta^b` collapses to `eta^b`, and the
    colony was not searching -- a fact invisible in every other figure here, all of which
    would look perfectly healthy.

    Small multiples rather than one panel with a line per regime: the regimes' curves sit
    close together, so overlaying them needs a hue per regime (past the categorical cap
    once regimes grow) or an alpha ramp, which is a weak channel that collapses the
    moment two curves coincide -- the first draft did exactly that and the three labels
    printed on top of each other. A panel each costs space and reads unambiguously.
    """
    ceiling = math.log(num_levels)
    series: dict[str, list[list[float]]] = defaultdict(list)
    for result in results:
        strategy, regime, variant = cell_key(result)
        if strategy != "fedaco" or variant != "default":
            continue
        curve = [
            entry.get("train_pheromone_entropy")
            for entry in result.get("rounds", [])
            if entry.get("train_pheromone_entropy") is not None
        ]
        if curve:
            series[regime].append(curve)
    if not series:
        return None

    regimes = sorted(series)
    cols = min(len(regimes), 3)
    rows = math.ceil(len(regimes) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)

    for index, regime in enumerate(regimes):
        ax = axes[index // cols][index % cols]
        _despine(ax)
        ax.set_title(regime)
        ax.set_xlabel("round")
        ax.set_ylabel("pheromone entropy")
        _integer_rounds(ax)

        mean_curve = _mean_curve(series[regime])
        ax.plot(_rounds(mean_curve), mean_curve, color=SERIES_1, linewidth=2.0, zorder=3)
        # Dashed *here only*: this is a genuine threshold, not chrome.
        ax.axhline(ceiling, color=DIVERGING_WORSE, linewidth=1.2, linestyle="--", zorder=2)

        # The y-range is what makes this figure readable or useless. Anchored to the
        # ceiling with a floor of a few percent below it, so a curve sitting a hair under
        # uniform does not get auto-scaled into looking like a dramatic descent.
        low = min(min(mean_curve), ceiling * 0.95)
        ax.set_ylim(low - 0.01, ceiling + 0.01)

        gap = (ceiling - mean_curve[-1]) / ceiling
        ax.annotate(
            f"{gap:.1%} below uniform",
            xy=(0.03, 0.06), xycoords="axes fraction",
            color=INK_SECONDARY, fontsize=8,
        )
        if index == 0:
            # Below the rule, not on it: at 0.93 the text sat across the dashed line.
            ax.annotate(
                f"uniform τ = log {num_levels} = {ceiling:.3f}",
                xy=(0.03, 0.86), xycoords="axes fraction",
                color=DIVERGING_WORSE, fontsize=8,
            )

    for index in range(len(regimes), rows * cols):
        axes[index // cols][index % cols].set_visible(False)

    fig.suptitle("Is the colony searching?", color=INK_PRIMARY, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# ======================================================================================


# ======================================================================================
# Plan §9.2 figures 3, 5, 6, 7 and 9 -- src/fedswarm/figures.py had them all along
# ======================================================================================
#
# Those five plot functions were written, complete, and imported by nothing: not this
# script, not any test. This script independently reimplemented the other four. So the
# module was dead code and the paper was five figures short for want of the extractors
# below -- result JSON in, the arrays each function already accepts out.
#
# Each returns None when the results present cannot feed it, so `main`'s existing
# "skipped (no data)" line names it rather than the figure vanishing silently.


def _aco_runs(results: list[dict]) -> list[dict]:
    return [r for r in results
            if str((r.get("config") or {}).get("run_config", {}).get("strategy-name", "")) == "fedaco"]


def _round_series(result: dict, key: str) -> list:
    return [r[key] for r in result.get("rounds", []) if r.get(key) is not None]


def figure_alpha_heatmap(results: list[dict], out: Path) -> Path | None:
    """Figure 3 -- alpha over (clients x rounds) for one representative run.

    The plan calls this "the figure that shows the method doing something interpretable"
    and "the one reviewers remember". Representative is defined as the FedACO run whose
    final macro-F1 is closest to the median of all FedACO runs -- deliberately not the best
    one, since picking the best run for the interpretability figure is how a method looks
    more consistent than it is.
    """
    runs = [r for r in _aco_runs(results) if _round_series(r, "train_alpha")]
    if not runs:
        return None

    scored = sorted(
        (r for r in runs if r["final"].get("final_test_macro_f1") is not None),
        key=lambda r: r["final"]["final_test_macro_f1"],
    )
    chosen = scored[len(scored) // 2] if scored else runs[0]

    rows = _round_series(chosen, "train_alpha")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        # K changed mid-run (a cold-start or straggler config). A ragged matrix cannot be
        # imaged, and padding it would draw clients that were not there.
        return None

    alpha = np.asarray(rows, dtype=float)
    fig = plot_alpha_heatmap(alpha)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_overhead(results: list[dict], out: Path) -> Path | None:
    """Figure 5 -- ACO + Gram time against K, with the fitted c*K^2 overlaid.

    The empirical half of claim C3. Needs at least three distinct K to be worth plotting a
    quadratic through; `overhead.yaml` supplies them.
    """
    by_k: dict[int, list[float]] = defaultdict(list)
    for result in _aco_runs(results):
        k = (result.get("config") or {}).get("run_config", {}).get("num-clients")
        if k is None:
            continue
        aco = _round_series(result, "train_aco_time_ms")
        gram = _round_series(result, "train_gram_time_ms")
        if not aco:
            continue
        # Total aggregation overhead: the colony plus the Gram build it depends on. Timing
        # only the colony would understate the K^2 term, which lives in the Gram matrix.
        per_round = [a + g for a, g in zip(aco, gram)] if gram else aco
        by_k[int(k)].append(sum(per_round) / len(per_round))

    if len(by_k) < 3:
        return None

    ks = np.array(sorted(by_k), dtype=float)
    overhead = np.array([sum(by_k[int(k)]) / len(by_k[int(k)]) for k in ks], dtype=float)
    fig = plot_overhead_vs_k(ks, overhead)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_robustness(results: list[dict], out: Path) -> Path | None:
    """Figure 6 -- macro-F1 against attacker fraction, one line per strategy.

    Reads `attack-fraction` from the run config, so a sweep whose cells differ only by
    attack type (R2's gaussian vs sign_flip at one fraction each) yields one point per
    strategy and is not plotted -- a line through one x value says nothing.
    """
    points: dict[tuple[str, float], list[float]] = defaultdict(list)
    for result in results:
        run_config = (result.get("config") or {}).get("run_config", {})
        score = result["final"].get("final_test_macro_f1")
        if score is None:
            continue
        strategy = str(run_config.get("strategy-name", "unknown"))
        # A clean cell is attacker fraction 0, which is the anchor every line needs.
        fraction = 0.0 if str(run_config.get("attack", "none")) == "none" else float(
            run_config.get("attack-fraction", 0.0)
        )
        points[(strategy, fraction)].append(float(score))

    fractions = {f for _, f in points}
    if len(fractions) < 2:
        return None

    frame = pd.DataFrame(
        [{"strategy": s, "attacker_fraction": f,
          "final_test_macro_f1": sum(v) / len(v)} for (s, f), v in sorted(points.items())]
    )
    fig = plot_robustness(frame)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# The ACO knobs A6 varies, in the flat run_config spelling. Used to find a pair that was
# actually crossed rather than assuming which two the sweep chose.
SENSITIVITY_KEYS = (
    "aco-q0", "aco-rho", "aco-pheromone-exp", "aco-heuristic-exp",
    "aco-gamma-entropy", "aco-num-levels", "aco-ants-start", "aco-iters-start",
)


def figure_sensitivity(results: list[dict], out: Path) -> Path | None:
    """Figure 7 -- sensitivity over the two most-varied ACO hyperparameters.

    ⚠️ `ablation_a6.yaml` is a **one-at-a-time** sweep: it moves one knob per arm around a
    shared centre, so its 270 cells contain no filled 2-D grid. This function therefore
    finds the two keys that were genuinely crossed and returns None when none were, rather
    than imaging a grid that is one row and one column of real cells around a hole.
    Producing that figure needs a factorial sweep A6 does not currently declare.
    """
    varied: dict[str, set] = {}
    for key in SENSITIVITY_KEYS:
        values = {(r.get("config") or {}).get("run_config", {}).get(key) for r in results}
        values.discard(None)
        if len(values) > 1:
            varied[key] = values

    if len(varied) < 2:
        return None

    key_y, key_x = sorted(varied, key=lambda k: (-len(varied[k]), k))[:2]
    values_y = sorted(varied[key_y])
    values_x = sorted(varied[key_x])

    cells: dict[tuple, list[float]] = defaultdict(list)
    for result in results:
        run_config = (result.get("config") or {}).get("run_config", {})
        score = result["final"].get("final_test_macro_f1")
        if score is None:
            continue
        cells[(run_config.get(key_y), run_config.get(key_x))].append(float(score))

    grid = np.full((len(values_y), len(values_x)), np.nan)
    for i, vy in enumerate(values_y):
        for j, vx in enumerate(values_x):
            got = cells.get((vy, vx))
            if got:
                grid[i, j] = sum(got) / len(got)

    # Distinguishing a factorial grid from a one-at-a-time cross is structural, not a
    # matter of how full the grid is. A KxK cross fills 2K-1 cells, which is 56% at K=3 and
    # 44% at K=4 -- so any fixed fill threshold accepts the cross at some K and rejects a
    # partly-resumed factorial at another. The signature of a cross is that every filled
    # cell shares a coordinate with the centre; a factorial sweep has cells that differ
    # from the centre in *both* coordinates.
    centre_y = max(values_y, key=lambda v: sum(1 for (y, _) in cells if y == v))
    centre_x = max(values_x, key=lambda v: sum(1 for (_, x) in cells if x == v))
    off_cross = sum(
        1
        for i, vy in enumerate(values_y)
        for j, vx in enumerate(values_x)
        if vy != centre_y and vx != centre_x and not np.isnan(grid[i, j])
    )
    if off_cross == 0:
        return None

    filled = int(np.count_nonzero(~np.isnan(grid)))
    if filled < grid.size * 0.8:
        # Enough of the grid is missing that the colour scale would be read across holes.
        return None

    fig = plot_sensitivity_heatmap(np.array(values_x), np.array(values_y), grid, key_x, key_y)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_gain_vs_heterogeneity(results: list[dict], out: Path) -> Path | None:
    """Figure 9 -- FedACO's gain over the best baseline against measured JS divergence.

    The x-axis is each run's own `partition_stats["js_divergence"]`, not the dirichlet
    `alpha` that was requested: alpha is the parameter, js_divergence is the skew the draw
    actually produced, and the plan's claim is about the latter. That field was null in
    every result file until `fl/app.py` started passing it, which is why this figure had no
    source and could not be wired.
    """
    by_regime_seed: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    js: dict[tuple[str, int], float] = {}
    for result in results:
        run_config = (result.get("config") or {}).get("run_config", {})
        score = result["final"].get("final_test_macro_f1")
        stats = result.get("partition_stats") or {}
        divergence = stats.get("js_divergence")
        if score is None or divergence is None:
            continue
        key = (str(run_config.get("regime", "unknown")), int((result.get("config") or {}).get("seed", 0)))
        by_regime_seed[key][str(run_config.get("strategy-name", "unknown"))] = float(score)
        js[key] = float(divergence)

    xs, gains = [], []
    for key, scores in by_regime_seed.items():
        if "fedaco" not in scores or len(scores) < 2:
            continue
        best_baseline = max(v for s, v in scores.items() if s != "fedaco")
        xs.append(js[key])
        gains.append(scores["fedaco"] - best_baseline)

    if len(xs) < 3:
        return None

    fig = plot_gain_vs_heterogeneity(np.array(xs), np.array(gains))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_alpha_entropy(results: list[dict], out: Path) -> Path | None:
    """Plan §9.2 figure 4's other half: alpha entropy per round, with pheromone entropy
    alongside it where the run is FedACO's own colony.

    `figure_colony_health` already plots pheromone entropy against its log(L) ceiling, which
    answers "did tau move". This answers a different question -- did the *weights* actually
    concentrate -- and the two can disagree: the local screens found the winning search
    method was also the least concentrated arm (highest alpha entropy, lowest alpha_max), so
    whatever this fitness rewards on heterogeneous data, it is not concentration.

    Averaged across seeds per strategy, because a single run's curve is noise at these
    scales and the plan asks for a diagnostic, not an anecdote.
    """
    by_strategy: dict[str, list[list[float]]] = defaultdict(list)
    pheromone: dict[str, list[list[float]]] = defaultdict(list)
    for result in results:
        strategy = str((result.get("config") or {}).get("run_config", {}).get("strategy-name", "?"))
        alpha = _round_series(result, "train_alpha_entropy")
        if alpha:
            by_strategy[strategy].append(alpha)
        tau = _round_series(result, "train_pheromone_entropy")
        if tau:
            pheromone[strategy].append(tau)

    if not by_strategy:
        return None

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for strategy in sorted(by_strategy):
        curves = by_strategy[strategy]
        mean = _mean_curve(curves)
        ax.plot(_rounds(mean), mean, marker="o", markersize=3, label=f"{strategy} alpha")
        if pheromone.get(strategy):
            tau_mean = _mean_curve(pheromone[strategy])
            ax.plot(_rounds(tau_mean), tau_mean, marker="s", markersize=3,
                    linestyle=":", label=f"{strategy} pheromone")
    ax.set_xlabel("round")
    ax.set_ylabel("entropy (nats)")
    _integer_rounds(ax)
    _despine(ax)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/fl/main")
    parser.add_argument("--out", default="paper/figures")
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--num-levels", type=int, default=11, help="aco-num-levels used")
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated: convergence,comparison,ablation,health,alpha,overhead,robustness,sensitivity,heterogeneity,entropy",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = REPO_ROOT / results_dir
    results = load_results(results_dir)
    if not results:
        print(f"No completed result files under {results_dir}.")
        print("Run the sweep first: make main  (or make main-plan to see the cost)")
        return 2

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    style()
    rows = summarize(results, args.expected_seeds)
    add_deltas(rows, "fedavg")

    wanted = {w.strip() for w in args.only.split(",")} if args.only else None
    jobs = {
        "convergence": lambda: figure_convergence(results, out_dir / "convergence.png"),
        "comparison": lambda: figure_comparison(rows, out_dir / "comparison.png"),
        "ablation": lambda: figure_ablations(rows, out_dir / "ablations.png"),
        "health": lambda: figure_colony_health(
            results, out_dir / "colony_health.png", args.num_levels
        ),
        "alpha": lambda: figure_alpha_heatmap(results, out_dir / "alpha_heatmap.png"),
        "entropy": lambda: figure_alpha_entropy(results, out_dir / "alpha_entropy.png"),
        "overhead": lambda: figure_overhead(results, out_dir / "overhead_vs_k.png"),
        "robustness": lambda: figure_robustness(results, out_dir / "robustness.png"),
        "sensitivity": lambda: figure_sensitivity(results, out_dir / "sensitivity.png"),
        "heterogeneity": lambda: figure_gain_vs_heterogeneity(
            results, out_dir / "gain_vs_heterogeneity.png"
        ),
    }

    written, skipped = [], []
    for name, job in jobs.items():
        if wanted and name not in wanted:
            continue
        path = job()
        (written if path else skipped).append(name)
        if path:
            print(f"  wrote {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")

    if skipped:
        # Named rather than silently omitted: a missing figure usually means the sweep
        # that feeds it has not run, not that the figure was not wanted.
        print(f"  skipped (no data): {', '.join(skipped)}")
    if not written:
        print("\nNothing plotted -- the results present do not cover any figure.")
        return 2
    print(f"\n{len(written)} figure(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
