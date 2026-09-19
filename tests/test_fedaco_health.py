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

REPO_ROOT = Path(__file__).resolve().parents[1]

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


# ======================================================================================
# The gate must name the fix, and --strict must gate A1 on it
# ======================================================================================


def test_the_degenerate_verdict_names_the_gamma_that_closes_it() -> None:
    """The first real GPU run reported the corner winning 15/15 rounds and offered only
    "raise `aco-gamma-entropy`" -- no value. Closing it therefore meant booking another GPU
    session purely to measure a quantity that is a closed form over a Gram matrix the round
    had already built. The number belongs in the verdict that diagnoses the problem."""
    run = _margin_run([0.45, 0.80, 0.55], num_clients=10)
    for i, entry in enumerate(run["rounds"]):
        entry["required_gamma_entropy"] = [0.30, 0.45, 0.34][i]
    run["config"]["run_config"]["aco-gamma-entropy"] = 0.1

    check = check_degenerate_optimum(run)

    assert check["verdict"] == "DEGENERATE"
    # The MAX, not the mean: gamma_entropy is set once for the whole run, so a value that
    # clears the average round leaves the worst rounds degenerate.
    assert check["required_gamma_entropy"] == 0.45
    assert "0.450" in check["detail"]


def test_an_infinite_requirement_does_not_swallow_the_reported_value() -> None:
    """`required_gamma_entropy` returns `inf` for a round whose base weights were a single
    client -- the penalty then charges the corner and the reference identically and no value
    separates them. Taking a naive max over the rounds would report `inf` and read as "this
    is unfixable" when 14 of 15 rounds had a perfectly ordinary answer."""
    run = _margin_run([0.45, 0.80], num_clients=10)
    run["rounds"][0]["required_gamma_entropy"] = 0.30
    run["rounds"][1]["required_gamma_entropy"] = float("inf")

    check = check_degenerate_optimum(run)

    assert check["required_gamma_entropy"] == 0.30


def test_an_older_result_says_the_value_is_missing_rather_than_guessing() -> None:
    """Results written before 2026-09-19 have no `required_gamma_entropy`. Saying so beats
    inventing a number, and beats silently dropping the only actionable line in the check."""
    check = check_degenerate_optimum(_margin_run([0.45, 0.80], num_clients=10))

    assert check["required_gamma_entropy"] is None
    assert "predates" in check["detail"]


def test_strict_fails_on_a_degenerate_fitness_even_when_the_colony_searches(tmp_path: Path) -> None:
    """What lets the notebook gate A1's 80 cells on this check instead of on someone reading
    the text. A1's four controls optimize the SAME fitness the colony does, so on a
    degenerate landscape a tie between them says nothing about ACO -- and the plan reframes
    the paper on exactly that result. `--strict` previously passed whenever the colony was
    searching, which is the one case where a degenerate optimum is most convincingly hidden:
    the colony is working, and working toward the corner.
    """
    import json
    import subprocess
    import sys

    rounds = [
        {
            "round": i + 1,
            # A structured pheromone, so check 1 reports SEARCHING and cannot be the reason
            # the exit code is nonzero.
            "train_pheromone_entropy": 0.4,
            "train_best_fitness": 0.5,
            "train_corner_margin": 0.6,
            "train_alpha_max": 0.35,
            "train_required_gamma_entropy": 0.45,
            "train_fallback_used": 0.0,
        }
        for i in range(15)
    ]
    payload = {
        "status": "completed",
        "rounds": rounds,
        "config": {"strategy": "fedaco", "seed": 0, "run_config": {"num-clients": 10}},
        "final": {"final_test_macro_f1": 0.2, "num_rounds_completed": 15, "wall_clock_s": 100.0},
    }
    (tmp_path / "r.json").write_text(json.dumps(payload))

    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_fedaco_health.py"),
         "--results-dir", str(tmp_path), "--strict"],
        capture_output=True, text=True,
    )

    assert "DEGENERATE" in completed.stdout
    assert completed.returncode == 1, (
        "a degenerate fitness must fail --strict, or the notebook cannot stop A1 from "
        f"running on it. stdout:\n{completed.stdout}"
    )


def test_the_check_lists_which_strategies_it_actually_found(tmp_path: Path) -> None:
    """Check 4's "NO BASELINE" cannot distinguish a FedAvg run that has not finished from one
    whose result file the check failed to match -- and the first GPU gate printed it while a
    FedAvg run was visibly starting in the same cell. Listing the directory settles it in the
    output rather than in the next GPU session."""
    import json
    import subprocess
    import sys

    for strategy in ("fedaco", "fedavg"):
        payload = {
            "status": "completed",
            "rounds": [{"round": 1, "train_corner_margin": -0.5, "train_pheromone_entropy": 0.4}],
            "config": {"strategy": strategy, "seed": 0, "run_config": {"num-clients": 10}},
            "final": {"final_test_macro_f1": 0.2, "num_rounds_completed": 1, "wall_clock_s": 10.0},
        }
        (tmp_path / f"{strategy}.json").write_text(json.dumps(payload))

    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_fedaco_health.py"),
         "--results-dir", str(tmp_path)],
        capture_output=True, text=True,
    )

    assert "strategies present" in completed.stdout
    assert "fedaco, fedavg" in completed.stdout


def _strict_exit(tmp_path: Path, colony_entropy: float, fitness: float, margin: float) -> int:
    """Both knobs matter for check 1's verdict, which is why they are both parameters.

    The verdict is not a function of tau's entropy alone -- it is tau's movement as a
    FRACTION of what this run's own deposit budget makes reachable, and the deposit is
    `rho * Q * max(F, 0)`. So a near-uniform tau reads INERT at a healthy fitness (the
    budget was there and went unused) and UNDERPOWERED at a small one (the budget was never
    there). Verified by probing the real function rather than assumed.
    """
    import json
    import subprocess
    import sys

    rounds = [
        {"round": i + 1, "train_pheromone_entropy": colony_entropy, "train_best_fitness": fitness,
         "train_corner_margin": margin, "train_alpha_max": 0.3,
         "train_required_gamma_entropy": 0.48, "train_fallback_used": 0.0}
        for i in range(15)
    ]
    (tmp_path / "r.json").write_text(json.dumps({
        "status": "completed", "rounds": rounds,
        "config": {"strategy": "fedaco", "seed": 0,
                   "run_config": {"num-clients": 10, "aco-gamma-entropy": 0.6}},
        "final": {"final_test_macro_f1": 0.2, "num_rounds_completed": 15, "wall_clock_s": 100.0},
    }))
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_fedaco_health.py"),
         "--results-dir", str(tmp_path), "--strict"],
        capture_output=True, text=True,
    ).returncode


def test_an_inconclusive_colony_check_does_not_block_a_fixed_corner(tmp_path: Path) -> None:
    """UNDERPOWERED means "this run cannot tell", not "the colony failed" -- the gate is 15
    rounds by design and tau has not had time to leave uniform by a margin the threshold can
    see. Blocking on it would leave someone who has just fixed the corner still refused, with
    the only remedy being to lengthen a gate whose whole purpose is to be short. A1 itself
    runs at 100 rounds, where the question is answerable.

    A gate that cannot distinguish inconclusive from failing is a gate nobody can pass.
    """
    # tau moving but the deposit too small to settle it -- the shape of the real gate run,
    # whose best_fitness averaged ~0.19. Corner comfortably lost.
    assert _strict_exit(tmp_path, colony_entropy=2.0, fitness=0.2, margin=-0.35) == 0


def test_a_degenerate_corner_blocks_even_when_the_colony_is_searching(tmp_path: Path) -> None:
    """The case the flag exists for, and the one a human reading the report most easily
    talks themselves past: every mechanism signal looks healthy, and the colony is healthily
    searching for the wrong thing."""
    assert _strict_exit(tmp_path, colony_entropy=1.5, fitness=0.8, margin=0.6) == 1


def test_a_genuinely_inert_colony_still_blocks(tmp_path: Path) -> None:
    """The other side of not blocking on UNDERPOWERED. INERT is a real negative verdict --
    the deposit budget was there and tau did not move -- so it must still stop A1 even with
    the corner fixed, or making UNDERPOWERED non-blocking would have quietly disabled the
    colony half of the gate."""
    assert _strict_exit(tmp_path, colony_entropy=math.log(11) - 0.001, fitness=0.8, margin=-0.35) == 1


def test_the_freshest_run_is_reported_not_the_first_one_found(tmp_path: Path) -> None:
    """Changing `aco-gamma-entropy` changes the config hash, so a re-run at a new value
    writes a NEW file beside the old one -- both completed, both 15 rounds. `max` returns the
    first maximal element, so without a tiebreak the check could read the previous attempt
    and report the old verdict against the new setting: you change the value, re-run, and the
    report tells you nothing changed."""
    import json
    import os
    import subprocess
    import sys
    import time

    def write(name: str, gamma: float, margin: float, mtime: float) -> None:
        rounds = [
            {"round": i + 1, "train_corner_margin": margin, "train_alpha_max": 0.3,
             "train_required_gamma_entropy": 0.48, "train_pheromone_entropy": 0.4,
             "train_best_fitness": 0.2, "train_fallback_used": 0.0}
            for i in range(15)
        ]
        path = tmp_path / name
        path.write_text(json.dumps({
            "status": "completed", "rounds": rounds,
            "config": {"strategy": "fedaco", "seed": 0,
                       "run_config": {"num-clients": 10, "aco-gamma-entropy": gamma}},
            "final": {"final_test_macro_f1": 0.13, "num_rounds_completed": 15,
                      "wall_clock_s": 113.0},
        }))
        os.utime(path, (mtime, mtime))

    now = time.time()
    write("a_old.json", 0.1, 0.62, now - 3600)   # sorts first by name; the stale attempt
    write("z_new.json", 0.6, -0.35, now)         # the fresh one

    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_fedaco_health.py"),
         "--results-dir", str(tmp_path)],
        capture_output=True, text=True,
    ).stdout

    assert "z_new.json" in out
    assert "aco-gamma-entropy of the run above: 0.6" in out
    assert "CLEAR" in out
