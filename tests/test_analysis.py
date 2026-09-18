"""Phase 9, Step 9.1 -- fedswarm.analysis, tested against synthetic result-JSON
fixtures built with the REAL write_result() (utils/results.py), so the schema
these tests exercise is exactly what a live run would actually produce."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fedswarm.analysis import (
    bootstrap_rounds_to_target,
    cohens_d,
    compare_against_baselines,
    holm_bonferroni,
    load_results,
    paired_values,
    rounds_to_target,
    summarize,
    wilcoxon_paired_test,
)
from fedswarm.utils.results import write_result


def _write_run(
    results_dir: Path,
    strategy: str,
    seed: int,
    regime: str,
    final_macro_f1: float,
    alpha: float | None = None,
    status: str = "completed",
    rounds: list[dict] | None = None,
) -> None:
    run_config = {"regime": regime, "num-clients": 20}
    if alpha is not None:
        run_config["alpha"] = alpha
    config = {"run_config": run_config, "strategy": strategy, "seed": seed}
    result = write_result(
        results_dir / f"{strategy}_{regime}_{seed}.json",
        config=config,
        rounds=rounds or [{"round": 1, "test_macro_f1": final_macro_f1}],
        final={"final_test_macro_f1": final_macro_f1, "wall_clock_s": 12.3, "num_rounds_completed": 1},
        seed=seed,
        status=status,
    )
    assert result["status"] == status


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    d = tmp_path / "fl"
    d.mkdir()
    return d


# ======================================================================================
# load_results / summarize
# ======================================================================================


def test_load_results_excludes_non_completed_runs(results_dir: Path) -> None:
    _write_run(results_dir, "fedavg", 0, "iid", 0.5, status="completed")
    _write_run(results_dir, "fedavg", 1, "iid", 0.9, status="failed")

    df = load_results(results_dir)
    assert len(df) == 1
    assert df.iloc[0]["seed"] == 0


def test_summarize_computes_mean_std_per_strategy_partition(results_dir: Path) -> None:
    for seed, f1 in enumerate([0.4, 0.5, 0.6]):
        _write_run(results_dir, "fedavg", seed, "iid", f1)
    for seed, f1 in enumerate([0.6, 0.7, 0.8]):
        _write_run(results_dir, "fedaco", seed, "iid", f1)

    df = load_results(results_dir)
    summary = summarize(df)

    fedavg_row = summary[summary["strategy"] == "fedavg"].iloc[0]
    fedaco_row = summary[summary["strategy"] == "fedaco"].iloc[0]
    assert fedavg_row["mean"] == pytest.approx(0.5)
    assert fedaco_row["mean"] == pytest.approx(0.7)
    assert fedavg_row["n"] == 3


def test_summarize_separates_dirichlet_alphas_as_distinct_partitions(results_dir: Path) -> None:
    _write_run(results_dir, "fedavg", 0, "dirichlet", 0.3, alpha=0.1)
    _write_run(results_dir, "fedavg", 1, "dirichlet", 0.6, alpha=0.5)

    df = load_results(results_dir)
    summary = summarize(df)
    partitions = set(summary["partition"])
    assert "dirichlet_0.1" in partitions
    assert "dirichlet_0.5" in partitions


# ======================================================================================
# paired_values -- only overlapping seeds
# ======================================================================================


def test_paired_values_keeps_only_seeds_present_in_both_strategies(results_dir: Path) -> None:
    _write_run(results_dir, "fedaco", 0, "iid", 0.7)
    _write_run(results_dir, "fedaco", 1, "iid", 0.8)
    _write_run(results_dir, "fedavg", 0, "iid", 0.5)
    _write_run(results_dir, "fedavg", 2, "iid", 0.4)  # seed 2 has no fedaco counterpart

    df = load_results(results_dir)
    a, b = paired_values(df, "fedaco", "fedavg", "iid", "final_test_macro_f1")
    assert len(a) == 1 and len(b) == 1
    assert a[0] == pytest.approx(0.7)  # seed 0's fedaco value
    assert b[0] == pytest.approx(0.5)  # seed 0's fedavg value


# ======================================================================================
# cohens_d
# ======================================================================================


def test_cohens_d_zero_when_samples_are_identical() -> None:
    a = np.array([0.5, 0.6, 0.7])
    assert cohens_d(a, a.copy()) == 0.0


def test_cohens_d_matches_hand_computation() -> None:
    a = np.array([0.8, 0.85, 0.9])
    b = np.array([0.5, 0.55, 0.6])
    n_a, n_b = len(a), len(b)
    pooled_std = np.sqrt(((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / (n_a + n_b - 2))
    expected = (a.mean() - b.mean()) / pooled_std
    assert cohens_d(a, b) == pytest.approx(expected)


def test_cohens_d_nan_with_fewer_than_two_samples() -> None:
    assert np.isnan(cohens_d(np.array([0.5]), np.array([0.4, 0.5])))


# ======================================================================================
# wilcoxon_paired_test
# ======================================================================================


def test_wilcoxon_detects_a_real_consistent_difference() -> None:
    a = np.array([0.8, 0.82, 0.85, 0.9, 0.88])
    b = np.array([0.5, 0.52, 0.55, 0.6, 0.58])
    result = wilcoxon_paired_test(a, b)
    # n=5, every pair the same sign -> the smallest two-sided p-value achievable at
    # all is 2*(1/2)^5 = 0.0625 (exactly what scipy returns here) -- not < 0.05, a
    # real statistical fact about n=5, not a bug. This is itself the plan's own
    # point (§9.1) about why 5 seeds + Wilcoxon needs effect sizes alongside it.
    assert result["p_value"] == pytest.approx(0.0625)
    assert result["n_pairs"] == 5


def test_wilcoxon_returns_nan_for_identical_paired_samples() -> None:
    a = np.array([0.5, 0.6, 0.7])
    result = wilcoxon_paired_test(a, a.copy())
    assert np.isnan(result["p_value"])


def test_wilcoxon_returns_nan_with_zero_pairs() -> None:
    result = wilcoxon_paired_test(np.array([]), np.array([]))
    assert np.isnan(result["p_value"])
    assert result["n_pairs"] == 0


# ======================================================================================
# holm_bonferroni
# ======================================================================================


def test_holm_bonferroni_matches_hand_computed_example() -> None:
    # 3 p-values: 0.01, 0.02, 0.20 -- Holm: sorted [0.01,0.02,0.20], m=3
    # adjusted = max-so-far of (m-i)*p_sorted[i]: [3*0.01, 2*0.02, 1*0.20] = [0.03, 0.04, 0.20]
    # running max (already increasing here): [0.03, 0.04, 0.20]
    p_values = [0.01, 0.02, 0.20]
    adjusted = holm_bonferroni(p_values)
    assert adjusted == pytest.approx([0.03, 0.04, 0.20])


def test_holm_bonferroni_is_monotonic_non_decreasing_in_sorted_order() -> None:
    p_values = [0.5, 0.001, 0.3, 0.02, 0.04]
    adjusted = holm_bonferroni(p_values)
    order = np.argsort(p_values)
    sorted_adjusted = np.array(adjusted)[order]
    assert np.all(np.diff(sorted_adjusted) >= -1e-12)


def test_holm_bonferroni_never_exceeds_one() -> None:
    adjusted = holm_bonferroni([0.9, 0.8, 0.99])
    assert all(p <= 1.0 for p in adjusted)


def test_holm_bonferroni_preserves_nan_and_does_not_corrupt_others() -> None:
    adjusted = holm_bonferroni([0.01, float("nan"), 0.02])
    assert np.isnan(adjusted[1])
    assert not np.isnan(adjusted[0])
    assert not np.isnan(adjusted[2])


# ======================================================================================
# compare_against_baselines -- end to end
# ======================================================================================


def test_compare_against_baselines_produces_one_row_per_partition_baseline_pair(
    results_dir: Path,
) -> None:
    for seed in range(5):
        _write_run(results_dir, "fedaco", seed, "iid", 0.7 + 0.01 * seed)
        _write_run(results_dir, "fedavg", seed, "iid", 0.5 + 0.01 * seed)
        _write_run(results_dir, "fedaco", seed, "dirichlet", 0.65 + 0.01 * seed, alpha=0.3)
        _write_run(results_dir, "fedavg", seed, "dirichlet", 0.45 + 0.01 * seed, alpha=0.3)

    df = load_results(results_dir)
    table = compare_against_baselines(df, "fedaco", ["fedavg"], ["iid", "dirichlet_0.3"])

    assert len(table) == 2
    assert set(table["partition"]) == {"iid", "dirichlet_0.3"}
    assert "p_value_holm" in table.columns
    # fedaco is consistently higher in this synthetic fixture.
    assert (table["method_mean"] > table["baseline_mean"]).all()


def test_compare_against_baselines_skips_partitions_with_no_overlap(results_dir: Path) -> None:
    _write_run(results_dir, "fedaco", 0, "iid", 0.7)
    df = load_results(results_dir)
    table = compare_against_baselines(df, "fedaco", ["fedavg"], ["iid"])
    assert table.empty


# ======================================================================================
# rounds_to_target / bootstrap_rounds_to_target
# ======================================================================================


def test_rounds_to_target_returns_first_round_reaching_it() -> None:
    rounds = [
        {"round": 1, "test_macro_f1": 0.3},
        {"round": 2, "test_macro_f1": 0.6},
        {"round": 3, "test_macro_f1": 0.9},
    ]
    assert rounds_to_target(rounds, target=0.5) == 2


def test_rounds_to_target_returns_last_round_as_censored_bound_when_never_reached() -> None:
    rounds = [{"round": 1, "test_macro_f1": 0.1}, {"round": 2, "test_macro_f1": 0.2}]
    assert rounds_to_target(rounds, target=0.99) == 2


def test_rounds_to_target_handles_empty_rounds() -> None:
    assert rounds_to_target([], target=0.5) == 0


def test_bootstrap_rounds_to_target_ci_contains_the_mean() -> None:
    rounds_per_seed = [
        [{"round": r, "test_macro_f1": r / 10} for r in range(1, 11)] for _ in range(5)
    ]  # every seed reaches 0.5 at round 5
    result = bootstrap_rounds_to_target(rounds_per_seed, target=0.5, rng=np.random.default_rng(0))
    assert result["mean"] == pytest.approx(5.0)
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]


def test_bootstrap_rounds_to_target_empty_input() -> None:
    result = bootstrap_rounds_to_target([], target=0.5)
    assert np.isnan(result["mean"])
