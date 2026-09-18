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
    check_degenerate_optimum,
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


def _smoke_run(entropies: list[float]) -> dict:
    """A run shaped like the only real FedACO run so far: K=2, 4 rounds, a reduced colony
    budget (iters 6->4), best_fitness ~0.87. Its ceiling on tau concentration is ~3%, so
    the 2% threshold sits at 64% of what the budget allows."""
    run = _run(
        "fedaco",
        [
            {"round": i + 1, "pheromone_entropy": e, "best_fitness": 0.87}
            for i, e in enumerate(entropies)
        ],
    )
    run["config"]["run_config"] = {
        "num-clients": 2,
        "aco-iters-start": 6,
        "aco-iters-end": 4,
        "aco-global-best-every": 5,
    }
    return run


def test_a_run_too_short_to_answer_is_not_called_inert() -> None:
    """The verdict this check originally got wrong. These are the real entropies from the
    four-round smoke run: tau moved monotonically away from uniform every round, but only
    to 1.57% -- and the threshold is 2%. Calling that INERT ("the colony is not searching")
    reported a property of the run length as a property of the mechanism, and nearly sent
    the project to rewrite a deposit rule that was working."""
    check = check_colony_searching(
        _smoke_run([2.39339, 2.38436, 2.36516, 2.36035]), num_levels=11, flat_fraction=0.02
    )

    assert check["verdict"] == "UNDERPOWERED"
    assert check["realized_fraction"] > 0.15  # tau did move
    assert check["required_fraction"] > 0.6  # but the bar was out of reach


def test_a_genuinely_dead_colony_is_still_inert_in_a_short_run() -> None:
    """The other half, and the one that matters more: making the check budget-aware must
    not make it unfalsifiable. Same short run, same ceiling -- but tau that never leaves
    uniform is inert whatever the budget was."""
    check = check_colony_searching(
        _smoke_run([CEILING - 0.0005] * 4), num_levels=11, flat_fraction=0.02
    )

    assert check["verdict"] == "INERT"


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


# ======================================================================================
# 3. Is the fitness optimum degenerate?
# ======================================================================================


def _margin_run(margins: list[float], num_clients: int = 4, alpha_max: float = 0.5) -> dict:
    run = _run(
        "fedaco",
        [
            {"round": i + 1, "corner_margin": m, "alpha_max": alpha_max}
            for i, m in enumerate(margins)
        ],
    )
    run["config"]["run_config"] = {"num-clients": num_clients}
    return run


def test_a_fitness_that_prefers_one_client_is_flagged() -> None:
    """The failure nothing else in this report can see. A colony that searches well finds
    the corner; `fallback_used` stays 0, `best_fitness` beats `fedavg_fitness`, alpha looks
    decisive -- every signal reads as success while the method returns "use one client"."""
    check = check_degenerate_optimum(_margin_run([0.05, 0.03, 0.06, 0.04], alpha_max=0.98))

    assert check["verdict"] == "DEGENERATE"
    assert "discard the rest" in check["detail"]


def test_a_corner_that_barely_loses_is_not_a_comfortable_pass() -> None:
    """At the sweep's K under high heterogeneity the margin measured on synthetic deltas
    is about -0.01: negative, but close enough that a different partition could flip it.
    Reporting that as CLEAR would be the wrong shape of reassurance."""
    check = check_degenerate_optimum(_margin_run([-0.011, -0.008, -0.015, -0.009], num_clients=20))

    assert check["verdict"] == "MARGINAL"


def test_a_healthy_margin_is_clear() -> None:
    check = check_degenerate_optimum(_margin_run([-0.20, -0.19, -0.22, -0.18], num_clients=20))

    assert check["verdict"] == "CLEAR"


def test_a_run_without_the_metric_is_no_data_not_a_pass() -> None:
    """Every result file written before 2026-09-18 lacks `corner_margin`. Treating the
    absence as healthy is how a known failure mode gets silently reintroduced."""
    check = check_degenerate_optimum(_run("fedaco", [{"round": 1, "alpha_max": 0.9}]))

    assert check["verdict"] == "NO DATA"
