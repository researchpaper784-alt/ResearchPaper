"""Phase 9, Step 9.2 -- fedswarm.figures, against synthetic data. Not a check of
visual correctness (not unit-testable) -- a check that every figure function runs
without crashing and produces a well-formed matplotlib Figure with the expected
structure (right number of axes, non-empty, savable to PDF)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from fedswarm.figures import (
    plot_alpha_heatmap,
    plot_convergence_curves,
    plot_entropy_vs_round,
    plot_final_bar_chart,
    plot_gain_vs_heterogeneity,
    plot_overhead_vs_k,
    plot_robustness,
    plot_sensitivity_heatmap,
    save_figure,
)


def _convergence_df() -> pd.DataFrame:
    rows = []
    for strategy in ("fedavg", "fedaco"):
        for partition in ("iid", "dirichlet_0.3"):
            for seed in range(3):
                for round_ in range(1, 6):
                    rows.append(
                        {
                            "strategy": strategy,
                            "partition": partition,
                            "seed": seed,
                            "round": round_,
                            "test_macro_f1": 0.5 + 0.05 * round_ + 0.01 * seed,
                        }
                    )
    return pd.DataFrame(rows)


def test_plot_convergence_curves_one_panel_per_partition() -> None:
    fig = plot_convergence_curves(_convergence_df())
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2  # iid, dirichlet_0.3


def test_plot_final_bar_chart_runs() -> None:
    summary = pd.DataFrame(
        [
            {"strategy": "fedavg", "partition": "iid", "mean": 0.6, "std": 0.02, "n": 3},
            {"strategy": "fedaco", "partition": "iid", "mean": 0.7, "std": 0.01, "n": 3},
        ]
    )
    fig = plot_final_bar_chart(summary)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_plot_alpha_heatmap_shape() -> None:
    alpha_matrix = np.random.default_rng(0).random((10, 5))  # 10 rounds, 5 clients
    fig = plot_alpha_heatmap(alpha_matrix, client_quality=np.array([0.1, 0.9, 0.2, 0.8, 0.5]))
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2  # heatmap + colorbar


def test_plot_entropy_vs_round_with_and_without_pheromone() -> None:
    rounds = list(range(1, 11))
    alpha_entropy = [1.5 - 0.05 * r for r in rounds]
    fig1 = plot_entropy_vs_round(rounds, alpha_entropy)
    assert isinstance(fig1, Figure)

    pheromone_entropy = [2.0 - 0.03 * r for r in rounds]
    fig2 = plot_entropy_vs_round(rounds, alpha_entropy, pheromone_entropy)
    assert isinstance(fig2, Figure)
    assert len(fig2.axes[0].lines) == 2


def test_plot_overhead_vs_k_fits_through_origin() -> None:
    k_values = np.array([10.0, 20.0, 50.0, 100.0])
    overhead_ms = 0.002 * k_values**2  # exact K^2 relationship, c=0.002
    fig = plot_overhead_vs_k(k_values, overhead_ms)
    assert isinstance(fig, Figure)
    # The fitted curve at K=100 should closely match the true c*K^2 relationship.
    fitted_line = fig.axes[0].lines[0]
    x_data, y_data = fitted_line.get_data()
    assert y_data[-1] == pytest.approx(0.002 * 100.0**2, rel=0.05)


def test_plot_robustness_one_line_per_strategy() -> None:
    df = pd.DataFrame(
        [
            {"strategy": "fedavg", "attacker_fraction": f, "final_test_macro_f1": 0.6 - 0.3 * f}
            for f in (0.0, 0.1, 0.2, 0.3)
        ]
        + [
            {"strategy": "fedaco", "attacker_fraction": f, "final_test_macro_f1": 0.7 - 0.1 * f}
            for f in (0.0, 0.1, 0.2, 0.3)
        ]
    )
    fig = plot_robustness(df)
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].lines) == 2


def test_plot_sensitivity_heatmap_runs() -> None:
    values_x = [0.5, 1.0, 2.0]
    values_y = [0.05, 0.10, 0.30]
    metric_grid = np.random.default_rng(0).random((3, 3))
    fig = plot_sensitivity_heatmap(values_x, values_y, metric_grid, "a", "rho")
    assert isinstance(fig, Figure)


def test_plot_gain_vs_heterogeneity_fits_a_trend_line() -> None:
    rng = np.random.default_rng(0)
    js_divergence = rng.uniform(0, 1, 20)
    gain = 0.1 * js_divergence + rng.normal(0, 0.01, 20)
    fig = plot_gain_vs_heterogeneity(js_divergence, gain)
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].lines) == 2  # trend line + zero-reference line


def test_save_figure_writes_a_real_pdf(tmp_path: Path) -> None:
    fig = plot_entropy_vs_round([1, 2, 3], [1.0, 0.9, 0.8])
    out_path = tmp_path / "test.pdf"
    save_figure(fig, str(out_path))
    assert out_path.exists()
    assert out_path.read_bytes()[:4] == b"%PDF"
