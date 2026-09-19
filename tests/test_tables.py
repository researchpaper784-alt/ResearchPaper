"""Phase 9 tests -- scripts/make_tables.py.

A results table misleads by omission far more easily than by arithmetic. These test the
omissions: a cell quietly averaged over fewer seeds than its neighbours, two different
experimental conditions pooled into one row, and a FedACO score presented without the
two numbers that say whether its mechanism did anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_tables import (  # noqa: E402
    add_significance,
    power_note,
    _variant_of,
    add_deltas,
    cell_key,
    summarize,
    to_markdown,
)


def _result(strategy, regime, seed, test_f1, val_f1=None, rounds=None, **run_config):
    config = {"regime": regime, "seed": seed, "strategy-name": strategy, **run_config}
    return {
        "status": "completed",
        "config": {"strategy": strategy, "seed": seed, "run_config": config},
        "rounds": rounds or [],
        "final": {"final_test_macro_f1": test_f1, "best_val_macro_f1": val_f1},
    }


# ======================================================================================
# Grouping
# ======================================================================================


def test_dirichlet_alphas_are_never_pooled() -> None:
    """dirichlet@0.1 and dirichlet@1.0 are different experimental conditions -- Phase 1.4
    measured JS divergence 0.541 vs 0.217. Pooling them into one `dirichlet` row would
    average across a factor-of-two difference in heterogeneity."""
    a = cell_key(_result("fedavg", "dirichlet", 0, 0.8, alpha=0.1))
    b = cell_key(_result("fedavg", "dirichlet", 0, 0.8, alpha=1.0))

    assert a != b
    assert a[1] == "dirichlet_0.1" and b[1] == "dirichlet_1.0"


def test_variant_is_recovered_from_the_resolved_config() -> None:
    """A result file records the resolved config, not the sweep's variant name, so the
    ablation has to be reconstructed from the knobs that differ from default."""
    assert _variant_of({}) == "default"
    assert "persistence=none" in _variant_of({"aco-persistence": "none"})
    assert "norm=batchnorm" in _variant_of({"model-norm": "batchnorm"})
    assert "no-fallback" in _variant_of({"aco-safety-fallback": False})
    assert "s=0.8" in _variant_of({"aco-target-sum": 0.8})


def test_ablation_variants_do_not_pool_with_the_control() -> None:
    control = cell_key(_result("fedaco", "iid", 0, 0.8))
    ablated = cell_key(_result("fedaco", "iid", 0, 0.8, **{"aco-persistence": "none"}))

    assert control != ablated


# ======================================================================================
# Incompleteness
# ======================================================================================


def test_a_cell_with_missing_seeds_is_marked_not_dropped() -> None:
    """Silently averaging whatever finished is how one strategy's mean over 5 seeds ends
    up next to another's over 2, presented identically."""
    results = [_result("fedavg", "iid", s, 0.9) for s in (0, 1)]

    rows = summarize(results, expected_seeds=5)

    assert len(rows) == 1
    assert rows[0]["n"] == 2
    assert rows[0]["incomplete"] is True


def test_a_complete_cell_is_not_marked() -> None:
    results = [_result("fedavg", "iid", s, 0.9) for s in range(5)]

    rows = summarize(results, expected_seeds=5)

    assert rows[0]["n"] == 5 and rows[0]["incomplete"] is False


def test_markdown_warns_about_incomplete_cells() -> None:
    rows = summarize([_result("fedavg", "iid", 0, 0.9)], expected_seeds=5)
    add_deltas(rows, "fedavg")

    markdown = to_markdown(rows, "fedavg")

    assert "fewer seeds than expected" in markdown
    assert "⚠️" in markdown


# ======================================================================================
# Deltas
# ======================================================================================


def test_delta_is_paired_by_seed() -> None:
    """Both methods saw the same seeds, so the per-seed difference has far less variance
    than the difference of the means."""
    results = [_result("fedavg", "iid", s, 0.80 + 0.01 * s) for s in range(3)]
    results += [_result("fedaco", "iid", s, 0.85 + 0.01 * s) for s in range(3)]

    rows = summarize(results, expected_seeds=3)
    add_deltas(rows, "fedavg")

    aco = next(r for r in rows if r["strategy"] == "fedaco")
    assert aco["delta_paired"] is True
    assert abs(aco["delta"] - 0.05) < 1e-9


def test_delta_is_none_when_no_shared_seeds_exist() -> None:
    """Better an explicit gap than a mean-to-mean number quietly presented as if it were
    the paired comparison."""
    results = [_result("fedavg", "iid", 0, 0.80), _result("fedaco", "iid", 7, 0.85)]

    rows = summarize(results, expected_seeds=1)
    add_deltas(rows, "fedavg")

    aco = next(r for r in rows if r["strategy"] == "fedaco")
    assert aco["delta"] is None and aco["delta_paired"] is False


def test_baseline_row_has_zero_delta() -> None:
    rows = summarize([_result("fedavg", "iid", 0, 0.9)], expected_seeds=1)
    add_deltas(rows, "fedavg")

    assert rows[0]["delta"] == 0.0


# ======================================================================================
# FedACO health columns
# ======================================================================================


def test_fedaco_rows_carry_fallback_and_entropy() -> None:
    """A headline macro-F1 says nothing about whether the colony searched or whether the
    fallback quietly turned FedACO into FedAvg. Both must sit next to the score."""
    rounds = [
        {"round": 1, "train_fallback_used": 1, "train_pheromone_entropy": 2.39},
        {"round": 2, "train_fallback_used": 0, "train_pheromone_entropy": 2.37},
    ]
    rows = summarize([_result("fedaco", "iid", 0, 0.9, rounds=rounds)], expected_seeds=1)

    assert rows[0]["fallback_rate"] == 0.5
    assert abs(rows[0]["pheromone_entropy"] - 2.38) < 1e-9


def test_markdown_explains_how_to_read_the_health_columns() -> None:
    rounds = [{"round": 1, "train_fallback_used": 1, "train_pheromone_entropy": 2.39}]
    rows = summarize([_result("fedaco", "iid", 0, 0.9, rounds=rounds)], expected_seeds=1)
    add_deltas(rows, "fedavg")

    markdown = to_markdown(rows, "fedavg")

    assert "silently *was* FedAvg" in markdown
    assert "not searching" in markdown


def test_rows_without_health_metrics_render_as_dashes_not_zeros() -> None:
    """A baseline has no pheromone. Rendering that as 0.000 would read as 'the colony was
    maximally concentrated', the opposite of 'not applicable'."""
    rows = summarize([_result("fedavg", "iid", 0, 0.9)], expected_seeds=1)
    add_deltas(rows, "fedavg")

    assert rows[0]["pheromone_entropy"] is None
    assert "| — | — |" in to_markdown(rows, "fedavg")


# ======================================================================================
# Significance, and whether the table can support the claim at all
# ======================================================================================


def _best_case(n_seeds: int, regimes=("iid",)):
    """FedACO ahead of FedAvg on every seed in every regime -- the most favourable data
    that can exist. Any "not significant" here is about the sample size."""
    results = []
    for regime in regimes:
        for seed in range(n_seeds):
            results.append(_result("fedavg", regime, seed, 0.880 + 0.002 * seed))
            results.append(_result("fedaco", regime, seed, 0.905 + 0.002 * seed))
    return results


def _rows(results, expected_seeds, alternative="two-sided"):
    rows = summarize(results, expected_seeds)
    add_deltas(rows, "fedavg")
    add_significance(rows, "fedavg", alternative)
    return rows


def test_the_table_reports_a_test_not_just_a_margin() -> None:
    """Mean +/- std invites a reader to eyeball two overlapping error bars and conclude
    whatever they arrived believing. At n=5 the seed spread is comparable to the margins
    these methods differ by, which is exactly when that goes wrong."""
    rows = _rows(_best_case(5), 5)
    aco = next(r for r in rows if r["strategy"] == "fedaco")

    assert aco["p_value"] is not None
    assert aco["p_value_holm"] is not None
    assert aco["cohens_d"] > 0


def test_the_baseline_is_not_tested_against_itself() -> None:
    rows = _rows(_best_case(5), 5)
    fedavg = next(r for r in rows if r["strategy"] == "fedavg")

    assert fedavg["p_value"] is None and fedavg["cohens_d"] is None


def test_holm_corrects_across_the_whole_table_not_per_regime() -> None:
    """Correcting inside each regime and then reporting six regimes under-corrects by
    exactly the factor the correction exists to supply."""
    one_regime = _rows(_best_case(8, ("iid",)), 8)
    six_regimes = _rows(_best_case(8, tuple(f"r{i}" for i in range(6))), 8)

    single = next(r for r in one_regime if r["strategy"] == "fedaco")["p_value_holm"]
    family = next(r for r in six_regimes if r["strategy"] == "fedaco")["p_value_holm"]

    assert family > single, "a larger family must not get the same adjusted p"


def test_a_table_that_cannot_reach_alpha_says_so() -> None:
    """The check that makes the statistics honest rather than decorative.

    At 5 seeds the signed-rank test's smallest possible two-sided p is 0.0625, so the
    main sweep's planned seed count cannot produce a significant result at alpha=0.05 --
    with Cohen's d near 8 and every seed favouring the method. Reporting "not
    significant" there without saying why would be the single most misleading number the
    paper could contain.
    """
    rows = _rows(_best_case(5), 5)
    note = power_note(rows, alpha=0.05, alternative="two-sided")

    assert note is not None
    assert "cannot reach" in note
    assert "6 seeds" in note, "it has to say what would be enough for this family"

    # And the requirement scales with the family -- asserted as the relationship rather
    # than a hard-coded count, because the count depends on how many claims the table
    # makes and is exactly the thing that is easy to miscount by hand.
    from fedswarm.analysis import seeds_needed_for

    assert seeds_needed_for(0.05, 6) > seeds_needed_for(0.05, 1)
    assert seeds_needed_for(0.05, 66) > seeds_needed_for(0.05, 6)
    # And the effect it is failing to detect is enormous, which is the point.
    assert next(r for r in rows if r["strategy"] == "fedaco")["cohens_d"] > 3


def test_the_warning_is_silent_when_the_test_can_actually_fire() -> None:
    """The other half: a check that always warns is noise, not a safeguard."""
    rows = _rows(_best_case(8), 8)

    assert power_note(rows, alpha=0.05, alternative="two-sided") is None
    assert next(r for r in rows if r["strategy"] == "fedaco")["p_value_holm"] < 0.05


def test_a_one_sided_test_is_available_but_not_the_default() -> None:
    """Halving the achievable p is legitimate for a directional hypothesis and illegitimate
    if chosen after seeing the two-sided result, so it must be opt-in."""
    two_sided = _rows(_best_case(6), 6, "two-sided")
    one_sided = _rows(_best_case(6), 6, "greater")

    p_two = next(r for r in two_sided if r["strategy"] == "fedaco")["p_value"]
    p_one = next(r for r in one_sided if r["strategy"] == "fedaco")["p_value"]
    assert p_one < p_two
