"""Tests for fedswarm.aco.diagnostics -- the bound that makes a pheromone-entropy
verdict readable.

The project's only FedACO run was declared INERT ("the colony is not searching") on a
four-round smoke test. The bound here is what showed that verdict was forced: at that
run's budget, *no* colony could have cleared the threshold without concentrating tau at
64% of the rate of one that picks the same level every iteration. If the bound is wrong,
so is that conclusion, so it is checked two ways -- against an independent closed-form
derivation, and against real colonies which must never exceed it.
"""

from __future__ import annotations

import math
import statistics

import pytest
import torch

from fedswarm.aco.colony import ColonyConfig, run_colony
from fedswarm.aco.diagnostics import best_case_gaps, closed_form_entropy
from fedswarm.aco.pheromone import Pheromone, PheromoneConfig
from fedswarm.aco.schedules import colony_budget, level_set

CEILING = math.log(11)


# ======================================================================================
# The closed form and the implementation agree
# ======================================================================================


@pytest.mark.parametrize("iterations", [1, 2, 4, 6, 10, 20, 40])
def test_closed_form_matches_the_implemented_update(iterations: int) -> None:
    """Derivation against code. The closed form says a losing level is a geometric decay
    and the winner is that decay plus a geometric series of deposits; `run_colony` is the
    rule the strategy actually runs. They are written independently and must agree, or one
    of them has the update wrong.

    `q0=1.0` plus a constant fitness makes every ant identical and selection fully greedy,
    so the same level wins every iteration -- the situation the closed form describes. The
    global-best deposit is disabled here because the closed form has no term for it; the
    bound in `best_case_gaps` does include it, and `test_global_best_deposit_raises_the_
    bound` covers that.
    """
    num_clients, num_levels, fitness = 4, 11, 0.78
    levels = level_set(num_levels)
    eta = 1.0 / (1.0 + (levels.unsqueeze(0) - torch.ones(num_clients).unsqueeze(1)).abs())
    config = ColonyConfig(q0=1.0, global_best_every=10**9, stagnation_patience=10**9)

    result = run_colony(
        torch.full((num_clients, num_levels), 1.0),
        eta,
        levels,
        torch.full((num_clients,), 1.0 / num_clients),
        lambda _alpha: fitness,
        num_ants=1,
        num_iterations=iterations,
        config=config,
    )

    assert result.final_entropy == pytest.approx(
        closed_form_entropy(iterations, fitness=fitness), abs=1e-5
    )


# ======================================================================================
# It is actually an upper bound
# ======================================================================================


def _real_colony_gaps(rounds: int, seed: int, num_clients: int = 4, num_levels: int = 11):
    """A colony with a real (non-degenerate) fitness landscape and the default q0=0.9."""
    torch.manual_seed(seed)
    levels = level_set(num_levels)
    base_weights = torch.full((num_clients,), 1.0 / num_clients)
    target = torch.empty(num_clients).uniform_(0.5, 1.8)
    eta = 1.0 / (1.0 + (levels.unsqueeze(0) - target.unsqueeze(1)).abs())
    pheromone = Pheromone(num_levels, PheromoneConfig())
    generator = torch.Generator().manual_seed(seed)

    gaps = []
    for round_idx in range(rounds):
        ants, iterations = colony_budget(round_idx, rounds)
        result = run_colony(
            pheromone.begin_round([f"c{i}" for i in range(num_clients)]),
            eta,
            levels,
            base_weights,
            lambda alpha: float(0.85 - 0.3 * (alpha * num_clients - target).abs().mean()),
            ants,
            iterations,
            ColonyConfig(),
            1.0,
            generator,
        )
        pheromone.end_round([f"c{i}" for i in range(num_clients)], result.tau_final)
        gaps.append(
            (math.log(num_levels) - pheromone.entropy([f"c{i}" for i in range(num_clients)]))
            / math.log(num_levels)
        )
    return gaps


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_no_real_colony_beats_the_bound(seed: int) -> None:
    """The claim the whole analysis rests on. If a real colony can concentrate tau faster
    than the 'best case', then 'the run realized 29% of what was available' means nothing
    and the UNDERPOWERED verdict is unfounded.

    An earlier version of the bound *was* beatable: it disabled the global-best deposit,
    which lands on top of the iteration-best one every T iterations, so a real colony with
    the default T=5 concentrated faster than its own supposed ceiling.
    """
    rounds = 10
    observed = _real_colony_gaps(rounds, seed)
    bound = best_case_gaps(rounds, fitness=0.85)

    for round_idx, (got, limit) in enumerate(zip(observed, bound)):
        assert got <= limit + 1e-9, f"round {round_idx}: {got:.4%} exceeds bound {limit:.4%}"


def test_global_best_deposit_raises_the_bound() -> None:
    """Why the parameter is threaded through rather than defaulted away: at the project's
    T=5 it moves the four-round smoke-run ceiling by most of a percentage point, which is
    the difference between two verdicts."""
    without = best_case_gaps(4, 6, 4, fitness=0.87, num_clients=2, global_best_every=10**9)
    with_gb = best_case_gaps(4, 6, 4, fitness=0.87, num_clients=2, global_best_every=5)

    assert sum(with_gb) > sum(without)
    assert statistics.fmean(with_gb) > 0.03 > statistics.fmean(without)


# ======================================================================================
# The bound behaves the way the argument uses it
# ======================================================================================


def test_tau_starts_at_the_ceiling_and_walks_away() -> None:
    """The premise: an absolute entropy threshold is not scale-free in the run length,
    because tau begins uniform by construction (tau0 on every level)."""
    gaps = best_case_gaps(6, 10, 4)

    assert gaps[0] < 0.03, "round 1 is necessarily close to uniform"
    assert gaps[-1] > gaps[0], "and tau moves away from it"


def test_a_weaker_deposit_lowers_the_ceiling() -> None:
    """The deposit is `rho * Q * max(F, 0)`, so the reachable gap scales with the fitness
    magnitude. A run whose fitness happens to sit low is held to a lower ceiling, which is
    why the bound reads F from the run rather than assuming one."""
    strong = best_case_gaps(4, fitness=1.5)
    weak = best_case_gaps(4, fitness=0.3)

    assert statistics.fmean(weak) < statistics.fmean(strong)


def test_more_iterations_reach_further() -> None:
    assert statistics.fmean(best_case_gaps(4, 10, 4)) > statistics.fmean(best_case_gaps(4, 6, 4))


def test_the_smoke_run_budget_cannot_clear_the_threshold_honestly() -> None:
    """The finding, pinned as a regression test. At the smoke run's budget (iters 6->4 over
    4 rounds, F~0.87) the health check's 2% threshold demands most of a best case that has
    stopped exploring, so INERT there was a statement about the run length."""
    bound = statistics.fmean(best_case_gaps(4, 6, 4, fitness=0.87, num_clients=2))
    required = 0.02 / bound

    assert required > 0.6, f"threshold needs {required:.0%} of the best case"
    assert bound < 0.05, "and the best case itself is only a few percent"
