"""The five plan §9.2 figures that `src/fedswarm/figures.py` had all along.

Those plot functions were written, complete, and imported by nothing -- not by
`scripts/make_figures.py`, which independently reimplemented the other four, and not by any
test. Dead code in a module named `figures.py` while the paper was five figures short.

So these tests are not about the plotting, which was already fine. They cover the extractors
that turn result JSON into what each function accepts, and in particular the **refusals**:
each figure must return None rather than draw something misleading when the results cannot
support it. Two of those refusals encode real facts about this project's sweeps:

* A6 is a one-at-a-time sweep, so it contains no filled 2-D grid to make a sensitivity
  heatmap from -- half a grid imaged as a whole one would invent structure in the holes.
* An attack sweep whose cells differ only by attack *type* gives one x value per strategy,
  and a line through one point is not a degradation curve.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from make_figures import (  # noqa: E402
    figure_alpha_heatmap,
    figure_gain_vs_heterogeneity,
    figure_overhead,
    figure_robustness,
    figure_sensitivity,
)


def _result(
    *,
    strategy: str = "fedaco",
    seed: int = 0,
    regime: str = "dirichlet",
    f1: float = 0.3,
    num_clients: int = 10,
    rounds: int = 4,
    alpha_width: int | None = None,
    times: bool = True,
    js: float | None = None,
    extra_config: dict | None = None,
) -> dict:
    width = alpha_width if alpha_width is not None else num_clients
    round_records = []
    for i in range(rounds):
        record = {"train_alpha": [1.0 / width] * width}
        if times:
            record["train_aco_time_ms"] = 10.0 * num_clients
            record["train_gram_time_ms"] = 2.0 * num_clients**2
        round_records.append(record)
    return {
        "config": {
            "run_config": {
                "strategy-name": strategy,
                "regime": regime,
                "num-clients": num_clients,
                **(extra_config or {}),
            },
            "seed": seed,
        },
        "partition_stats": ({"js_divergence": js} if js is not None else None),
        "rounds": round_records,
        "final": {"final_test_macro_f1": f1},
        "status": "completed",
    }


# ======================================================================================
# Figure 3 -- alpha heatmap
# ======================================================================================


def test_alpha_heatmap_is_written_from_train_alpha(tmp_path: Path) -> None:
    out = tmp_path / "a.png"
    assert figure_alpha_heatmap([_result()], out) == out
    assert out.stat().st_size > 0


def test_alpha_heatmap_picks_the_median_run_not_the_best(tmp_path: Path) -> None:
    """Choosing the best run for the interpretability figure makes the method look more
    consistent than it is. The median is the honest representative, so a run with a
    conspicuously better score must not be the one drawn."""
    import numpy as np

    import make_figures

    captured = {}
    original = make_figures.plot_alpha_heatmap

    def spy(alpha_matrix, client_quality=None):
        captured["width"] = alpha_matrix.shape[1]
        return original(alpha_matrix, client_quality)

    make_figures.plot_alpha_heatmap = spy
    try:
        runs = [
            _result(f1=0.10, num_clients=3),
            _result(f1=0.20, num_clients=5),   # median score -> 5 clients
            _result(f1=0.90, num_clients=9),
        ]
        figure_alpha_heatmap(runs, tmp_path / "a.png")
    finally:
        make_figures.plot_alpha_heatmap = original

    assert captured["width"] == 5, "the median-scoring run must be the one drawn"
    assert np is not None


def test_alpha_heatmap_refuses_a_ragged_matrix(tmp_path: Path) -> None:
    """K changing mid-run is a cold-start or straggler config. Padding to a rectangle would
    draw clients that were not there."""
    run = _result(rounds=3)
    run["rounds"][1]["train_alpha"] = [0.5, 0.5]
    assert figure_alpha_heatmap([run], tmp_path / "a.png") is None


def test_alpha_heatmap_skips_non_aco_runs(tmp_path: Path) -> None:
    assert figure_alpha_heatmap([_result(strategy="fedavg")], tmp_path / "a.png") is None


# ======================================================================================
# Figure 5 -- overhead vs K
# ======================================================================================


def test_overhead_needs_three_distinct_k(tmp_path: Path) -> None:
    """A quadratic through two points is not a measurement of a quadratic."""
    two = [_result(num_clients=k) for k in (5, 10)]
    assert figure_overhead(two, tmp_path / "o.png") is None

    three = [_result(num_clients=k) for k in (5, 10, 20)]
    assert figure_overhead(three, tmp_path / "o.png") == tmp_path / "o.png"


def test_overhead_sums_colony_and_gram_time(tmp_path: Path) -> None:
    """The K^2 term lives in the Gram build. Timing only the colony understates the very
    curve the figure exists to fit."""
    import make_figures

    captured = {}
    original = make_figures.plot_overhead_vs_k

    def spy(k_values, overhead_ms):
        captured["overhead"] = list(overhead_ms)
        return original(k_values, overhead_ms)

    make_figures.plot_overhead_vs_k = spy
    try:
        figure_overhead([_result(num_clients=k) for k in (5, 10, 20)], tmp_path / "o.png")
    finally:
        make_figures.plot_overhead_vs_k = original

    # _result sets aco = 10*K and gram = 2*K^2 per round.
    assert captured["overhead"] == [10 * 5 + 2 * 25, 10 * 10 + 2 * 100, 10 * 20 + 2 * 400]


def test_overhead_skips_runs_with_no_timing(tmp_path: Path) -> None:
    runs = [_result(num_clients=k, times=False) for k in (5, 10, 20)]
    assert figure_overhead(runs, tmp_path / "o.png") is None


# ======================================================================================
# Figure 6 -- robustness
# ======================================================================================


def test_robustness_needs_more_than_one_attacker_fraction(tmp_path: Path) -> None:
    """R2's cells differ by attack *type* at one fraction each. One x value per strategy is
    not a degradation curve, and drawing it would imply a trend from a single point."""
    one_fraction = [
        _result(strategy=s, extra_config={"attack": a, "attack-fraction": 0.1})
        for s in ("fedaco", "krum")
        for a in ("gaussian", "sign_flip")
    ]
    assert figure_robustness(one_fraction, tmp_path / "r.png") is None


def test_robustness_treats_a_clean_cell_as_fraction_zero(tmp_path: Path) -> None:
    """Every line needs its unattacked anchor, and a clean cell carries no attack-fraction
    key at all -- reading it as missing rather than as zero would drop the anchor."""
    runs = [
        _result(strategy="fedaco", extra_config={"attack": "none"}),
        _result(strategy="fedaco", extra_config={"attack": "gaussian", "attack-fraction": 0.3}),
    ]
    assert figure_robustness(runs, tmp_path / "r.png") == tmp_path / "r.png"


# ======================================================================================
# Figure 7 -- sensitivity heatmap
# ======================================================================================


def test_sensitivity_refuses_a_one_at_a_time_sweep(tmp_path: Path) -> None:
    """This is A6's actual shape: each arm moves one knob off a shared centre, so the cells
    form a cross, not a grid. Imaging it would present interpolated holes as measurements.
    """
    centre = {"aco-q0": 0.7, "aco-rho": 0.1}
    runs = [_result(extra_config=centre)]
    for q0 in (0.5, 0.9):
        runs.append(_result(extra_config={**centre, "aco-q0": q0}))
    for rho in (0.05, 0.3):
        runs.append(_result(extra_config={**centre, "aco-rho": rho}))

    assert figure_sensitivity(runs, tmp_path / "s.png") is None


def test_sensitivity_draws_a_genuine_factorial_grid(tmp_path: Path) -> None:
    runs = [
        _result(extra_config={"aco-q0": q0, "aco-rho": rho})
        for q0 in (0.5, 0.7, 0.9)
        for rho in (0.05, 0.1, 0.3)
    ]
    assert figure_sensitivity(runs, tmp_path / "s.png") == tmp_path / "s.png"


def test_sensitivity_needs_two_varied_keys(tmp_path: Path) -> None:
    runs = [_result(extra_config={"aco-q0": q0}) for q0 in (0.5, 0.7, 0.9)]
    assert figure_sensitivity(runs, tmp_path / "s.png") is None


# ======================================================================================
# Figure 9 -- gain vs measured heterogeneity
# ======================================================================================


def test_gain_vs_heterogeneity_uses_measured_js_not_the_dirichlet_alpha(tmp_path: Path) -> None:
    """`alpha` is what was requested; `js_divergence` is the skew the draw produced. The
    plan's claim is about the latter, and the field was null in every result file until
    `fl/app.py` started passing partition_stats."""
    runs = []
    for i, js in enumerate((0.2, 0.35, 0.5)):
        runs.append(_result(strategy="fedaco", regime=f"r{i}", seed=0, f1=0.4 + i * 0.02, js=js))
        runs.append(_result(strategy="fedavg", regime=f"r{i}", seed=0, f1=0.35, js=js))
    assert figure_gain_vs_heterogeneity(runs, tmp_path / "g.png") == tmp_path / "g.png"


def test_gain_vs_heterogeneity_returns_none_without_partition_stats(tmp_path: Path) -> None:
    """The regression guard for the bug that hid this figure: every result file written
    before today has `partition_stats: null`, and the figure must say "no data" rather than
    fall back to the dirichlet parameter."""
    runs = []
    for i in range(3):
        runs.append(_result(strategy="fedaco", regime=f"r{i}", f1=0.4, js=None))
        runs.append(_result(strategy="fedavg", regime=f"r{i}", f1=0.35, js=None))
    assert figure_gain_vs_heterogeneity(runs, tmp_path / "g.png") is None


def test_gain_vs_heterogeneity_needs_a_baseline_to_difference_against(tmp_path: Path) -> None:
    runs = [_result(strategy="fedaco", regime=f"r{i}", js=0.2 + 0.1 * i) for i in range(3)]
    assert figure_gain_vs_heterogeneity(runs, tmp_path / "g.png") is None


@pytest.mark.parametrize("figure", [
    figure_alpha_heatmap, figure_overhead, figure_robustness,
    figure_sensitivity, figure_gain_vs_heterogeneity,
])
def test_every_figure_handles_no_results(figure, tmp_path: Path) -> None:
    assert figure([], tmp_path / "x.png") is None
