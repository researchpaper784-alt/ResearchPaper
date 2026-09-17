"""Tests for scripts/check_fedaco_health.py.

The script's job is to notice when FedACO's mechanism is inert on real data -- a state
that produces no error, no failed run, and possibly a respectable macro-F1. So the tests
that matter are the ones proving it does not return a clean bill of health for a dead
colony.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_fedaco_health import (  # noqa: E402
    check_beats_fedavg,
    check_colony_searching,
    check_deposit_floor,
    load_results,
    wall_clock_note,
)

CEILING = math.log(11)


def _run(strategy: str, rounds: list[dict], final: dict | None = None) -> dict:
    return {
        "run_id": f"{strategy}_0",
        "status": "completed",
        "config": {"strategy": strategy, "seed": 0, "run_config": {"num-clients": 10, "local-epochs": 1}},
        "rounds": rounds,
        "final": final or {},
        "_path": f"/tmp/{strategy}.json",
    }


# ======================================================================================
# 1. Is the colony searching?
# ======================================================================================


def test_uniform_pheromone_is_reported_inert() -> None:
    """The failure this script exists for: tau pinned at its ceiling means tau^a * eta^b
    collapses to eta^b, so there is no colony search and no cross-round stigmergy -- the
    paper's central claim -- while the run completes normally and reports a score."""
    rounds = [{"round": i, "pheromone_entropy": CEILING - 0.001} for i in range(1, 11)]

    check = check_colony_searching(_run("fedaco", rounds), num_levels=11, flat_fraction=0.02)

    assert check["verdict"] == "INERT"
    assert "not searching" in check["detail"]


def test_structured_pheromone_is_reported_searching() -> None:
    rounds = [{"round": i, "pheromone_entropy": 1.6} for i in range(1, 11)]

    check = check_colony_searching(_run("fedaco", rounds), num_levels=11, flat_fraction=0.02)

    assert check["verdict"] == "SEARCHING"


def test_pheromone_washing_out_over_rounds_is_caught_separately() -> None:
    """Averaging alone would hide this: a colony that starts structured and flattens has a
    healthy mean but is inert by the end, which is what the later rounds' weights are
    actually built from."""
    rounds = [{"round": 1, "pheromone_entropy": 1.2}, {"round": 2, "pheromone_entropy": 1.5}]
    rounds += [{"round": i, "pheromone_entropy": CEILING - 0.002} for i in range(3, 9)]

    check = check_colony_searching(_run("fedaco", rounds), num_levels=11, flat_fraction=0.02)

    assert check["verdict"] in ("DEGRADING", "INERT")


def test_missing_pheromone_metric_is_no_data_not_a_pass() -> None:
    """Result files predating the Phase 4 close-out carry no pheromone_entropy. Treating
    that absence as healthy would be the worst possible default."""
    check = check_colony_searching(_run("fedaco", [{"round": 1}]), num_levels=11, flat_fraction=0.02)

    assert check["verdict"] == "NO DATA"


def test_num_levels_changes_the_ceiling() -> None:
    """The ceiling is log(L), so a run with a different aco-num-levels must be judged
    against its own ceiling -- an entropy of 1.6 is near-uniform at L=5 and structured
    at L=11."""
    rounds = [{"round": i, "pheromone_entropy": 1.60} for i in range(1, 6)]

    at_11 = check_colony_searching(_run("fedaco", rounds), num_levels=11, flat_fraction=0.02)
    at_5 = check_colony_searching(_run("fedaco", rounds), num_levels=5, flat_fraction=0.02)

    assert at_11["verdict"] == "SEARCHING"
    assert at_5["verdict"] == "INERT"


# ======================================================================================
# 2. Has the deposit floor engaged?
# ======================================================================================


def test_all_negative_fitness_is_reported_engaged() -> None:
    rounds = [{"round": i, "best_fitness": -2.5, "delta_mean_sq_norm": 40.0} for i in range(1, 6)]

    check = check_deposit_floor(_run("fedaco", rounds))

    assert check["verdict"] == "ENGAGED"
    assert "every deposit was zeroed" in check["detail"]


def test_some_negative_rounds_are_reported_partial() -> None:
    rounds = [
        {"round": 1, "best_fitness": 0.4},
        {"round": 2, "best_fitness": -0.1},
        {"round": 3, "best_fitness": 0.6},
    ]

    check = check_deposit_floor(_run("fedaco", rounds))

    assert check["verdict"] == "PARTIAL"
    assert check["negative_rounds"] == [2]


def test_positive_fitness_throughout_is_clear() -> None:
    rounds = [{"round": i, "best_fitness": 0.5} for i in range(1, 6)]

    assert check_deposit_floor(_run("fedaco", rounds))["verdict"] == "CLEAR"


# ======================================================================================
# 3. Does FedACO beat FedAvg?
# ======================================================================================


def test_high_fallback_rate_is_called_out_even_when_ahead() -> None:
    """A margin earned while the fallback fired most rounds is largely FedAvg's own
    number. Reporting only the margin would credit the colony for it."""
    rounds = [{"round": i, "fallback_used": 1} for i in range(1, 9)]
    rounds += [{"round": i, "fallback_used": 0} for i in range(9, 11)]
    fedaco = _run("fedaco", rounds, {"final_test_macro_f1": 0.88})
    fedavg = _run("fedavg", [], {"final_test_macro_f1": 0.86})

    check = check_beats_fedavg(fedaco, fedavg)

    assert check["verdict"] == "AHEAD"
    assert check["fallback_rate"] == 0.8
    assert "largely FedAvg's own number" in check["detail"]


def test_behind_fedavg_is_reported() -> None:
    fedaco = _run("fedaco", [{"round": 1, "fallback_used": 0}], {"final_test_macro_f1": 0.80})
    fedavg = _run("fedavg", [], {"final_test_macro_f1": 0.86})

    check = check_beats_fedavg(fedaco, fedavg)

    assert check["verdict"] == "BEHIND"
    assert check["margin"] < 0


def test_missing_baseline_is_flagged_not_assumed() -> None:
    fedaco = _run("fedaco", [{"round": 1, "fallback_used": 0}], {"final_test_macro_f1": 0.88})

    assert check_beats_fedavg(fedaco, None)["verdict"] == "NO BASELINE"


# ======================================================================================
# Wall clock and loading
# ======================================================================================


def test_wall_clock_note_projects_a_hundred_round_run() -> None:
    """docs/EXPERIMENT_LOG.md's compute table is explicitly unmeasured and says to
    replace it with a real per-round number; this is that number."""
    fedaco = _run("fedaco", [], {"wall_clock_s": 360.0, "num_rounds_completed": 12})

    note = wall_clock_note(fedaco)

    assert "30.0s per round" in note
    assert "0.83 GPU-hours" in note  # 30s * 100 / 3600


def test_load_results_filters_by_strategy(tmp_path: Path) -> None:
    import json

    (tmp_path / "a.json").write_text(json.dumps(_run("fedaco", [{"round": 1}])))
    (tmp_path / "b.json").write_text(json.dumps(_run("fedavg", [{"round": 1}])))
    (tmp_path / "c.json").write_text("{ not json")

    assert len(load_results(tmp_path, "fedaco")) == 1
    assert len(load_results(tmp_path, "fedavg")) == 1
    assert len(load_results(tmp_path)) == 2
