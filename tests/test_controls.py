"""Phase 7, ablation A1 -- equal-fitness-budget controls (aco/controls.py). Each
control must (1) never exceed its evaluation budget, (2) return a valid point on the
simplex, and (3) actually be capable of finding a good solution on an easy objective
(a basic competence check, not the real ACO-vs-control empirical comparison, which
needs live data)."""

from __future__ import annotations


import pytest
import torch

from fedswarm.aco.controls import (
    GAConfig,
    PSOConfig,
    coordinate_grid_search,
    genetic_algorithm_search,
    pso_search,
    random_search,
)
from fedswarm.aco.fitness import BudgetedFitness, EvaluationBudgetExceeded
from fedswarm.aco.schedules import level_set


class _FavorFirstClient:
    """Toy fitness: rewards weight concentrated on client 0. Any competent search
    should find it easily given a reasonable budget."""

    def evaluate(self, alpha: torch.Tensor) -> float:
        return float(alpha[0]) - 0.1 * float(alpha[1:].sum())


@pytest.fixture
def toy_setup():
    torch.manual_seed(0)
    levels = level_set(num_levels=7, low=0.0, high=2.5)
    base_weights = torch.full((4,), 0.25)
    return levels, base_weights


def test_budgeted_fitness_raises_once_exhausted() -> None:
    budgeted = BudgetedFitness(_FavorFirstClient(), budget=2)
    alpha = torch.tensor([0.25, 0.25, 0.25, 0.25])
    budgeted.evaluate(alpha)
    budgeted.evaluate(alpha)
    assert budgeted.remaining() == 0
    with pytest.raises(EvaluationBudgetExceeded):
        budgeted.evaluate(alpha)


@pytest.mark.parametrize(
    "search_fn",
    [random_search, coordinate_grid_search],
)
def test_discrete_controls_never_exceed_budget(toy_setup, search_fn) -> None:
    levels, base_weights = toy_setup
    budget = 15
    fitness = BudgetedFitness(_FavorFirstClient(), budget=budget)
    result = search_fn(levels, base_weights, fitness, budget)
    assert result.evaluations_used <= budget
    assert fitness.calls_used == result.evaluations_used


def test_pso_never_exceeds_budget(toy_setup) -> None:
    levels, base_weights = toy_setup
    budget = 17  # deliberately not a multiple of num_particles
    fitness = BudgetedFitness(_FavorFirstClient(), budget=budget)
    result = pso_search(levels, base_weights, fitness, budget, config=PSOConfig(num_particles=5))
    assert result.evaluations_used <= budget
    assert fitness.calls_used == result.evaluations_used


def test_ga_never_exceeds_budget(toy_setup) -> None:
    levels, base_weights = toy_setup
    budget = 23  # deliberately not a multiple of population_size
    fitness = BudgetedFitness(_FavorFirstClient(), budget=budget)
    result = genetic_algorithm_search(
        levels, base_weights, fitness, budget, config=GAConfig(population_size=6)
    )
    assert result.evaluations_used <= budget
    assert fitness.calls_used == result.evaluations_used


@pytest.mark.parametrize(
    "search_fn,kwargs",
    [
        (random_search, {}),
        (coordinate_grid_search, {}),
        (pso_search, {}),
        (genetic_algorithm_search, {}),
    ],
)
def test_all_controls_return_a_valid_simplex_point(toy_setup, search_fn, kwargs) -> None:
    levels, base_weights = toy_setup
    target_sum = 0.9
    budget = 20
    fitness = BudgetedFitness(_FavorFirstClient(), budget=budget)
    result = search_fn(levels, base_weights, fitness, budget, target_sum=target_sum, **kwargs)
    assert torch.all(result.alpha_best >= 0.0)
    assert abs(float(result.alpha_best.sum()) - target_sum) < 1e-5


@pytest.mark.parametrize(
    "search_fn",
    [random_search, coordinate_grid_search, pso_search, genetic_algorithm_search],
)
def test_all_controls_are_competent_on_an_easy_objective(toy_setup, search_fn) -> None:
    """Not the real ablation (that needs FedACO's own results) -- just a sanity check
    that none of these controls are strawmen: given 200 evaluations on an objective
    that plainly rewards client 0, each must land closer to "all weight on client 0"
    than to uniform FedAvg weighting."""
    levels, base_weights = toy_setup
    budget = 200
    fitness = BudgetedFitness(_FavorFirstClient(), budget=budget)
    result = search_fn(levels, base_weights, fitness, budget)
    assert result.alpha_best[0] > base_weights[0]


def test_random_search_is_capped_even_with_generous_budget_request(toy_setup) -> None:
    """Regression guard for the loop-condition bug class: requesting a budget larger
    than what BudgetedFitness itself allows must still stop at the wrapper's limit,
    not the (possibly larger) `budget` argument passed to the search function."""
    levels, base_weights = toy_setup
    fitness = BudgetedFitness(_FavorFirstClient(), budget=5)
    result = random_search(levels, base_weights, fitness, budget=1000)
    assert result.evaluations_used == 5
    assert fitness.calls_used == 5
