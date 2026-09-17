"""Unit tests for aco/colony.py, aco/pheromone.py, aco/heuristics.py, aco/schedules.py --
pure tensor math, no Flower runtime needed."""

from __future__ import annotations

import torch

from fedswarm.aco.colony import ColonyConfig, run_colony
from fedswarm.aco.fitness import DataFreeFitness, DataFreeFitnessConfig
from fedswarm.aco.gram import precompute_gram
from fedswarm.aco.heuristics import HeuristicWeights, desirability_matrix, desirability_scores
from fedswarm.aco.pheromone import Pheromone, PheromoneConfig
from fedswarm.aco.schedules import colony_budget, level_set


def test_level_set_contains_exactly_one_and_is_sorted() -> None:
    levels = level_set(num_levels=11, low=0.0, high=2.5)
    assert levels.numel() == 11
    assert torch.any(levels == 1.0)
    assert torch.equal(levels, torch.sort(levels).values)
    assert float(levels.min()) == 0.0
    assert float(levels.max()) <= 2.5 + 1e-6


def test_colony_budget_decays_from_start_to_end() -> None:
    ants0, iters0 = colony_budget(0, 10, ants_start=30, ants_end=10, iters_start=10, iters_end=4)
    ants_last, iters_last = colony_budget(9, 10, ants_start=30, ants_end=10, iters_start=10, iters_end=4)
    assert ants0 == 30 and iters0 == 10
    assert ants_last == 10 and iters_last == 4


def test_simplex() -> None:
    """test_simplex (plan §4.8 acceptance #5): the colony's returned alpha is
    non-negative and sums to the configured target within 1e-6."""
    torch.manual_seed(0)
    num_clients, dim = 6, 200
    deltas = torch.randn(num_clients, dim)
    gram = precompute_gram(deltas)
    fitness = DataFreeFitness(gram, DataFreeFitnessConfig())
    levels = level_set(num_levels=7)
    tau0 = torch.full((num_clients, levels.numel()), 1.0)
    eta = torch.ones(num_clients, levels.numel())
    base_weights = torch.full((num_clients,), 1.0 / num_clients)

    for target_sum in (1.0, 0.8):
        result = run_colony(
            tau0, eta, levels, base_weights, fitness.evaluate,
            num_ants=10, num_iterations=5, config=ColonyConfig(), target_sum=target_sum,
        )
        assert torch.all(result.alpha_best >= 0.0)
        assert abs(float(result.alpha_best.sum()) - target_sum) < 1e-6


def test_pheromone_persists() -> None:
    """test_pheromone_persists (plan §4.8 acceptance #4): pheromone state at round t+1
    is a decayed function of round t and is keyed by client id under a changing
    participation set."""
    config = PheromoneConfig(tau0=1.0, rho_round=0.2, persistence="decayed")
    pheromone = Pheromone(num_levels=5, config=config)

    tau_round1 = pheromone.begin_round(["a", "b"])
    assert torch.all(tau_round1 == 1.0)  # fresh rows, all at tau0

    boosted = tau_round1.clone()
    boosted[:, 2] = 5.0  # simulate a colony run depositing pheromone on level index 2
    pheromone.end_round(["a", "b"], boosted)

    # Client "c" is new; "b" drops out this round; "a" returns.
    tau_round2 = pheromone.begin_round(["a", "c"])
    row_a = tau_round2[0]
    row_c = tau_round2[1]

    expected_a = (1.0 - config.rho_round) * boosted[0] + config.rho_round * config.tau0
    assert torch.allclose(row_a, expected_a, atol=1e-6)
    assert torch.all(row_c == config.tau0)  # never-seen client starts fresh at tau0

    # "b" is absent, not forgotten: its row still exists internally and has decayed
    # toward tau0 rather than being reset outright (persistence="decayed", not "none").
    assert "b" in pheromone._rows
    assert not torch.equal(pheromone._rows["b"], boosted[1])
    assert not torch.all(pheromone._rows["b"] == config.tau0)


def test_pheromone_none_persistence_resets_every_round() -> None:
    config = PheromoneConfig(tau0=1.0, persistence="none")
    pheromone = Pheromone(num_levels=4, config=config)
    tau = pheromone.begin_round(["x"])
    boosted = tau.clone()
    boosted[:, 0] = 9.0
    pheromone.end_round(["x"], boosted)

    tau_next = pheromone.begin_round(["x"])
    assert torch.all(tau_next == config.tau0)


def test_desirability_prefers_well_aligned_high_data_client() -> None:
    torch.manual_seed(2)
    dim = 100
    consensus_direction = torch.randn(dim)
    aligned = consensus_direction + 0.01 * torch.randn(dim)
    misaligned = -consensus_direction + 0.01 * torch.randn(dim)
    deltas = torch.stack([aligned, misaligned, aligned * 0.9], dim=0)

    gram = precompute_gram(deltas, trim_fraction=0.0)
    num_examples = torch.tensor([100.0, 100.0, 100.0])
    d_k = desirability_scores(gram, num_examples, weights=HeuristicWeights())

    assert d_k[0] > d_k[1]  # aligned client scores higher desirability than the negated one

    levels = level_set(num_levels=5)
    eta = desirability_matrix(d_k, levels)
    # The misaligned client's peak desirability should sit at a lower multiplier level
    # than the well-aligned client's.
    assert levels[eta[0].argmax()] >= levels[eta[1].argmax()]
