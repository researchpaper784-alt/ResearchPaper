"""Phase 5 tests -- the honest hyperparameter search (scripts/run_hparam_search.py) and
the IID narrow-band acceptance check (scripts/check_iid_band.py).

Both scripts orchestrate real `flwr run` subprocesses, so what is tested here is their
decision logic against hand-built result files: which trials a budget produces, which
metric a search is allowed to rank on, and what verdict a given spread earns. The
subprocess plumbing itself is exercised by actually running the search end-to-end
(docs/EXPERIMENT_LOG.md, 2026-09-17).
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_iid_band import evaluate_band, group_by_strategy, load_iid_results, summarize  # noqa: E402
from run_hparam_search import (  # noqa: E402
    GRIDS,
    diagnose,
    find_existing,
    score,
    trial_overrides,
)


def _result(strategy: str, seed: int, test_f1: float, val_f1: float, regime: str = "iid", **run_config):
    return {
        "run_id": f"{strategy}_{seed}",
        "status": "completed",
        "seed": seed,
        # The real run_config carries `seed` too, which is what makes resume able to
        # tell one seed's trials from another's.
        "config": {
            "strategy": strategy,
            "seed": seed,
            "run_config": {"regime": regime, "seed": seed, **run_config},
        },
        "rounds": [],
        "final": {"final_test_macro_f1": test_f1, "best_val_macro_f1": val_f1},
    }


# ======================================================================================
# run_hparam_search -- budget and selection metric
# ======================================================================================


def test_budget_truncates_the_grid_deterministically() -> None:
    first = trial_overrides("fedyogi", budget=8)
    second = trial_overrides("fedyogi", budget=8)
    assert first == second, "truncation must be reproducible, not a fresh sample each run"
    assert len(first) == 8
    assert len(trial_overrides("fedyogi", budget=10**6)) == 9  # 3 x 3 full factorial


def test_strategies_without_hyperparameters_yield_exactly_one_trial() -> None:
    """FedAvg is the reference point and has no aggregation hyperparameter at all.
    It must produce one trial, not zero (which would drop it from the comparison) and
    not `budget` copies of the same config (which would waste the budget)."""
    for strategy in ("fedavg", "median", "fednova"):
        assert trial_overrides(strategy, budget=8) == [{}]


def test_every_grid_key_is_declared_in_pyproject() -> None:
    """`flwr run` rejects any --run-config key absent from [tool.flwr.app.config] with a
    bare "[code: 15]" that names nothing (docs/OPEN_QUESTIONS.md). A grid referencing an
    undeclared key would therefore fail every trial at runtime, with no clue why."""
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_bytes()
    declared = set(tomllib.loads(pyproject.decode())["tool"]["flwr"]["app"]["config"])
    for strategy, grid in GRIDS.items():
        for key in grid:
            assert key in declared, f"{strategy}'s grid key {key!r} is not declared in pyproject.toml"


def test_search_ranks_on_val_and_never_on_test() -> None:
    """The core honesty property. A result whose test score is excellent and whose val
    score is poor must score poorly -- otherwise every baseline would be tuned against
    the number the paper reports."""
    tuned_on_test = _result("fedprox", 0, test_f1=0.99, val_f1=0.10)
    tuned_on_val = _result("fedprox", 0, test_f1=0.10, val_f1=0.90)

    assert score(tuned_on_test) == 0.10
    assert score(tuned_on_val) == 0.90
    assert score(tuned_on_val) > score(tuned_on_test)


def test_score_is_none_when_no_val_metric_is_present() -> None:
    """Result files written before per-round val evaluation existed carry no val metric.
    Those must score None (and be excluded from selection) rather than silently falling
    back to the test metric."""
    legacy = {"final": {"final_test_macro_f1": 0.95}}
    assert score(legacy) is None


def test_resume_matches_a_trial_by_its_override_values() -> None:
    existing = [
        _result("fedprox", 0, 0.5, 0.6, **{"fedprox-mu": 0.01}),
        _result("fedprox", 0, 0.5, 0.8, **{"fedprox-mu": 0.1}),
    ]
    found = find_existing(existing, {"fedprox-mu": 0.1})
    assert found is not None and score(found) == 0.8
    assert find_existing(existing, {"fedprox-mu": 1.0}) is None


def test_resume_ignores_incomplete_runs() -> None:
    """A crashed or interrupted run must be retried, not reused as a real score."""
    crashed = _result("fedprox", 0, 0.5, 0.6, **{"fedprox-mu": 0.01})
    crashed["status"] = "failed"
    assert find_existing([crashed], {"fedprox-mu": 0.01}) is None


# ======================================================================================
# check_iid_band -- verdicts
# ======================================================================================


def test_tight_spread_passes() -> None:
    rows = summarize(
        group_by_strategy(
            [
                _result("fedavg", 0, 0.900, 0.88),
                _result("fedavg", 1, 0.910, 0.89),
                _result("fedaco", 0, 0.905, 0.89),
                _result("fedaco", 1, 0.915, 0.90),
            ],
            "final_test_macro_f1",
        )
    )
    report = evaluate_band(rows, band=0.05)
    assert report["verdict"] == "PASS"
    assert report["spread"] < 0.05


def test_large_spread_under_iid_fails() -> None:
    """The case this check exists to catch: a big IID gap means something other than
    heterogeneity-handling is driving the number."""
    rows = summarize(
        group_by_strategy(
            [
                _result("fedavg", 0, 0.70, 0.70),
                _result("fedavg", 1, 0.71, 0.71),
                _result("fedaco", 0, 0.90, 0.90),
                _result("fedaco", 1, 0.91, 0.91),
            ],
            "final_test_macro_f1",
        )
    )
    report = evaluate_band(rows, band=0.05)
    assert report["verdict"] == "FAIL"
    assert report["leader"] == "fedaco" and report["laggard"] == "fedavg"


def test_spread_within_seed_noise_is_inconclusive_not_a_failure() -> None:
    """With noisy seeds, a spread slightly over the band is not evidence of anything.
    Calling that a FAIL would manufacture a problem; calling it a PASS would hide one."""
    rows = summarize(
        group_by_strategy(
            [
                _result("fedavg", 0, 0.60, 0.60),
                _result("fedavg", 1, 0.90, 0.90),
                _result("fedaco", 0, 0.68, 0.68),
                _result("fedaco", 1, 0.98, 0.98),
            ],
            "final_test_macro_f1",
        )
    )
    report = evaluate_band(rows, band=0.05)
    assert report["verdict"] == "INCONCLUSIVE"
    assert report["seed_noise"] > 0.05


def test_single_seed_strategies_are_flagged() -> None:
    rows = summarize(
        group_by_strategy(
            [_result("fedavg", 0, 0.90, 0.90), _result("fedaco", 0, 0.91, 0.91)],
            "final_test_macro_f1",
        )
    )
    report = evaluate_band(rows, band=0.05)
    assert set(report["single_seed_strategies"]) == {"fedavg", "fedaco"}
    assert report["seed_noise"] == 0.0


def test_only_iid_completed_runs_are_loaded(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps(_result("fedavg", 0, 0.9, 0.9, regime="iid")))
    (tmp_path / "b.json").write_text(json.dumps(_result("fedavg", 0, 0.5, 0.5, regime="dirichlet")))
    crashed = _result("fedaco", 0, 0.9, 0.9, regime="iid")
    crashed["status"] = "failed"
    (tmp_path / "c.json").write_text(json.dumps(crashed))
    (tmp_path / "d.json").write_text("{ not json")

    loaded = load_iid_results(tmp_path)

    assert len(loaded) == 1
    assert loaded[0]["config"]["strategy"] == "fedavg"


@pytest.mark.parametrize("metric", ["final_test_macro_f1", "best_val_macro_f1"])
def test_band_check_works_on_either_metric(metric: str) -> None:
    rows = summarize(
        group_by_strategy([_result("fedavg", 0, 0.90, 0.40), _result("fedaco", 0, 0.91, 0.85)], metric)
    )
    report = evaluate_band(rows, band=0.05)
    # test spread is 0.01 (passes); val spread is 0.45 (fails) -- the metric argument
    # genuinely selects which number is checked.
    assert report["verdict"] == ("PASS" if metric == "final_test_macro_f1" else "FAIL")


def test_diagnose_surfaces_the_explanatory_line() -> None:
    """`flwr run` exits 0 even when the simulation dies, so a failed trial is only
    diagnosable from its captured output. The first end-to-end run of this search had
    every trial fail on a missing image cache and reported a clean table of `None`s --
    this is what stops that recurring."""
    assert "Invalid run configuration" in diagnose(
        "some noise\n[code: 15] Invalid run configuration.\nmore noise"
    )
    assert "Dataset root does not exist" in diagnose(
        "INFO: starting\nFileNotFoundError: Dataset root does not exist: data/raw/brain-tumor-mri"
    )
    assert diagnose("") == "no diagnostic line found in output"
    assert diagnose("everything was fine") == "no diagnostic line found in output"

    # Ray prints "FutureWarning: ... turn off this error message" on every run. Matching
    # bare "error" line-by-line reported that as the cause of a failed trial.
    noisy = (
        "ray/_private/worker.py: FutureWarning: ... turn off this error message, set X\n"
        "[code: 15] Invalid run configuration."
    )
    assert "Invalid run configuration" in diagnose(noisy)
    assert "FutureWarning" not in diagnose(noisy)


def test_resume_does_not_reuse_a_different_seeds_result() -> None:
    """Resume must match the whole configuration, not just the swept keys. Matching on
    overrides alone means a search at seed 1 instantly "resumes" seed 0's results and the
    second seed is silently a duplicate of the first -- which would quietly destroy the
    seed variance every error bar in the paper depends on."""
    seed0 = _result("fedprox", 0, 0.5, 0.6, **{"fedprox-mu": 0.01})
    base_seed1 = {"seed": 1, "regime": "iid"}

    assert find_existing([seed0], {"fedprox-mu": 0.01}, base_seed1) is None
    assert find_existing([seed0], {"fedprox-mu": 0.01}, {"seed": 0, "regime": "iid"}) is not None


def test_resume_ignores_path_only_differences() -> None:
    """A cache or output directory that moved between environments (Colab vs local) says
    nothing about what was computed, so it must not defeat resume."""
    existing = _result(
        "fedprox", 0, 0.5, 0.6, **{"fedprox-mu": 0.01, "cache-dir": "/colab/cache"}
    )
    base = {"seed": 0, "regime": "iid", "cache-dir": "/somewhere/else"}
    assert find_existing([existing], {"fedprox-mu": 0.01}, base) is not None
