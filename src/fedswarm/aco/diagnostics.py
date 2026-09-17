"""How fast *can* tau move? A bound for reading pheromone-entropy diagnostics.

`check_fedaco_health.py` asks "is the colony searching?" by measuring how far tau's
entropy sits below its ceiling log(L). The threshold it compares against is absolute; the
quantity is not. tau starts *at* the ceiling -- `tau0` on every level of every row -- and
walks away from it at a rate fixed by `rho`, `q_deposit`, the fitness magnitude and the
iteration budget. A four-round smoke test has barely begun to move, however well the
colony is searching, so an absolute threshold applied to it measures the run length.

This module computes the ceiling on that movement: the trajectory of a colony in which
one level wins at *every* iteration, which is the fastest the deposit rule permits. Any
real colony realizes some fraction of it, and that fraction -- not the raw gap -- is what
says whether pheromone is carrying signal.

Driven through the real `run_colony` and `Pheromone` rather than a reimplementation, so
the cross-round pull-back toward tau0 in `end_round` (which is what stops tau reaching its
within-round fixed point) is included exactly as the strategy applies it.
"""

from __future__ import annotations

import math

import torch

from fedswarm.aco.colony import ColonyConfig, run_colony
from fedswarm.aco.pheromone import Pheromone, PheromoneConfig
from fedswarm.aco.schedules import colony_budget, level_set

# The only FedACO run measured end-to-end so far reported best_fitness in 0.78-0.92 across
# its four rounds (docs/OPEN_QUESTIONS.md, "Residual 1"). 0.8 is the low-middle of that
# range, so a bound computed from it is if anything conservative. It is a measured number
# from this project, not a guess -- and it matters, because the deposit is
# `rho * Q * max(F, 0)`: halve F and you halve how fast tau can move.
MEASURED_FITNESS = 0.8


def closed_form_entropy(
    iterations: int,
    rho: float = 0.1,
    tau0: float = 1.0,
    fitness: float = MEASURED_FITNESS,
    num_levels: int = 11,
    q_deposit: float = 1.0,
    tau_min: float = 0.01,
    tau_max: float = 10.0,
) -> float:
    """Entropy of one tau row after `iterations` iterations in which the same level wins
    every time -- derived, not simulated.

    Each iteration evaporates every entry by (1-rho) and adds `rho * Q * F` to the winner,
    so a losing level is a plain geometric decay and the winner is that decay plus a
    geometric series of deposits. This exists to be disagreed with: the tests check it
    against what `run_colony` actually does, and a divergence means one of the two has the
    implemented rule wrong.
    """
    decay = 1.0 - rho
    loser = min(max(tau0 * decay**iterations, tau_min), tau_max)
    winner = tau0 * decay**iterations
    for i in range(iterations):
        winner += decay ** (iterations - 1 - i) * rho * q_deposit * fitness
    winner = min(max(winner, tau_min), tau_max)

    row = [loser] * (num_levels - 1) + [winner]
    total = sum(row)
    return -sum((x / total) * math.log(x / total) for x in row)


def best_case_gaps(
    rounds: int,
    iters_start: int = 10,
    iters_end: int = 4,
    fitness: float = MEASURED_FITNESS,
    num_levels: int = 11,
    num_clients: int = 4,
    rho: float = 0.1,
    tau0: float = 1.0,
    rho_round: float = 0.1,
    q_deposit: float = 1.0,
    global_best_every: int = 5,
) -> list[float]:
    """Per-round entropy gap, as a fraction of log(L), for a colony concentrating tau as
    fast as the rule allows.

    `q0=1.0` makes selection fully greedy and a constant fitness makes every ant
    identical, so one level wins every iteration by construction. The early-stop rules are
    disabled: stagnation is guaranteed here (the fitness never improves), and stopping
    would measure the stop rule instead of the deposit rule.

    `global_best_every` is included, not disabled, and that correction matters: the
    global-best deposit lands on top of the iteration-best one every T iterations, so a
    bound that left it out is not a bound. At the project's default T=5 it raises the
    four-round smoke-run ceiling from 2.41% to 3.13% -- enough to move a verdict.

    `num_clients` is accepted for symmetry with a real run and does not change the answer:
    every row evolves identically under this input, and the entropy reported is their mean.
    """
    ceiling = math.log(num_levels)
    levels = level_set(num_levels)
    base_weights = torch.full((num_clients,), 1.0 / num_clients)
    peak = torch.ones(num_clients)
    eta = 1.0 / (1.0 + (levels.unsqueeze(0) - peak.unsqueeze(1)).abs())

    colony = ColonyConfig(
        q0=1.0,
        rho=rho,
        q_deposit=q_deposit,
        global_best_every=global_best_every,
        # Stagnation is guaranteed under a constant fitness, and stopping early would
        # measure the stop rule rather than the deposit rule's ceiling.
        stagnation_patience=10**9,
    )
    pheromone = Pheromone(
        num_levels, PheromoneConfig(tau0=tau0, rho_round=rho_round, persistence="decayed")
    )
    client_ids = [f"c{i}" for i in range(num_clients)]

    gaps = []
    for round_idx in range(rounds):
        _, iterations = colony_budget(
            round_idx, rounds, iters_start=iters_start, iters_end=iters_end
        )
        result = run_colony(
            pheromone.begin_round(client_ids),
            eta,
            levels,
            base_weights,
            lambda _alpha: fitness,
            num_ants=1,
            num_iterations=iterations,
            config=colony,
        )
        pheromone.end_round(client_ids, result.tau_final)
        gaps.append((ceiling - pheromone.entropy(client_ids)) / ceiling)
    return gaps
