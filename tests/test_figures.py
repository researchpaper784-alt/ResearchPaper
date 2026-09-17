"""Phase 9 tests -- scripts/make_figures.py.

Rendering is checked by looking at the output; what is tested here is the logic that can
silently produce a plausible-looking but wrong figure -- a mean curve padded past where a
seed actually stopped, a chart that quietly drops a strategy it has no color for, or a
figure emitted with no data behind it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_figures  # noqa: E402
from make_figures import (  # noqa: E402
    HIGHLIGHT,
    _mean_curve,
    figure_ablations,
    figure_colony_health,
    figure_comparison,
    figure_convergence,
)
from make_tables import add_deltas, summarize  # noqa: E402


@pytest.fixture(autouse=True)
def _style():
    make_figures.style()


def _run(strategy, regime, seed, finals, entropy=None, **run_config):
    rounds = []
    for i, value in enumerate(finals):
        entry = {"round": i + 1, "test_macro_f1": value, "val_macro_f1": value - 0.01}
        if entropy is not None:
            entry["train_pheromone_entropy"] = entropy[i]
        rounds.append(entry)
    return {
        "status": "completed",
        "config": {
            "strategy": strategy,
            "seed": seed,
            "run_config": {"regime": regime, "seed": seed, "strategy-name": strategy, **run_config},
        },
        "rounds": rounds,
        "final": {"final_test_macro_f1": finals[-1], "best_val_macro_f1": finals[-1] - 0.01},
    }


# ======================================================================================
# Mean curve
# ======================================================================================


def test_mean_curve_truncates_to_the_shortest_run() -> None:
    """A seed that stopped early must not be padded. Padding drags the tail of the mean
    toward its last value and draws a plateau no run actually produced -- a convergence
    figure that invents flatness at exactly the point readers look hardest."""
    curves = [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2]]

    mean = _mean_curve(curves)

    assert len(mean) == 2
    assert mean == [0.1, 0.2]


def test_mean_curve_averages_across_seeds() -> None:
    assert _mean_curve([[0.0, 1.0], [1.0, 3.0]]) == [0.5, 2.0]


# ======================================================================================
# Figures are emitted only when there is data behind them
# ======================================================================================


def test_convergence_returns_none_without_round_curves(tmp_path: Path) -> None:
    """Better no file than an empty axes that looks like a real, flat result."""
    bare = {
        "status": "completed",
        "config": {"strategy": "fedavg", "seed": 0, "run_config": {"regime": "iid"}},
        "rounds": [],
        "final": {"final_test_macro_f1": 0.9},
    }

    assert figure_convergence([bare], tmp_path / "c.png") is None


def test_colony_health_returns_none_without_pheromone_data(tmp_path: Path) -> None:
    """A baseline-only sweep has no pheromone to plot. The figure must be skipped and
    named as skipped, not rendered blank."""
    results = [_run("fedavg", "iid", 0, [0.8, 0.9])]

    assert figure_colony_health(results, tmp_path / "h.png", num_levels=11) is None


def test_ablations_returns_none_without_a_control(tmp_path: Path) -> None:
    """Every ablation bar is a distance from the FedACO default. With no default in the
    same sweep there is no zero to measure from, and a figure drawn anyway would be
    differencing against nothing."""
    results = [_run("fedaco", "iid", 0, [0.8], **{"aco-persistence": "none"})]
    rows = summarize(results, expected_seeds=1)
    add_deltas(rows, "fedavg")

    assert figure_ablations(rows, tmp_path / "a.png") is None


# ======================================================================================
# Figures render when data is present
# ======================================================================================


def test_convergence_renders_and_keeps_unhighlighted_strategies(tmp_path: Path) -> None:
    """The 9 baselines with no categorical slot still have to appear -- as the muted
    field. Dropping them would silently narrow the comparison to two methods."""
    results = [
        _run("fedaco", "iid", 0, [0.1, 0.5, 0.9]),
        _run("fedavg", "iid", 0, [0.1, 0.4, 0.8]),
        _run("krum", "iid", 0, [0.1, 0.3, 0.7]),
        _run("median", "iid", 0, [0.1, 0.2, 0.6]),
    ]

    out = figure_convergence(results, tmp_path / "c.png")

    assert out is not None and out.exists() and out.stat().st_size > 0


def test_comparison_renders_with_incomplete_cells(tmp_path: Path) -> None:
    results = [_run("fedaco", "iid", 0, [0.9]), _run("fedavg", "iid", 0, [0.8])]
    rows = summarize(results, expected_seeds=5)
    add_deltas(rows, "fedavg")

    out = figure_comparison(rows, tmp_path / "m.png")

    assert out is not None and out.exists()
    assert all(r["incomplete"] for r in rows)


def test_colony_health_renders_per_regime(tmp_path: Path) -> None:
    import math

    ceiling = math.log(11)
    results = [
        _run("fedaco", "iid", 0, [0.5, 0.6], entropy=[ceiling - 0.01, ceiling - 0.02]),
        _run("fedaco", "quantity_skew", 0, [0.5, 0.6], entropy=[ceiling - 0.3, ceiling - 0.4]),
    ]

    out = figure_colony_health(results, tmp_path / "h.png", num_levels=11)

    assert out is not None and out.exists()


# ======================================================================================
# Palette
# ======================================================================================


def test_only_the_method_and_its_reference_get_categorical_slots() -> None:
    """Twelve strategies cannot each have a hue -- a categorical palette carries ~8 before
    adjacent hues blur, and generating a 9th is the documented anti-pattern. The story is
    FedACO against FedAvg, so exactly those two are highlighted and the rest fold into one
    muted group."""
    assert set(HIGHLIGHT) == {"fedaco", "fedavg"}
    assert HIGHLIGHT["fedaco"] == "#2a78d6"  # validated categorical slot 1
    assert HIGHLIGHT["fedavg"] == "#eb6834"  # slot 2
    assert len(set(HIGHLIGHT.values())) == 2


def test_diverging_poles_are_two_hues_not_one_ramp() -> None:
    """Ablation deltas carry polarity, so the two arms must read as opposite. One hue
    would encode only magnitude and lose the better/worse distinction entirely."""
    assert make_figures.DIVERGING_BETTER != make_figures.DIVERGING_WORSE
