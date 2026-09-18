"""Phase 7, ablation A1 -- equal-fitness-budget controls (plan §7): random search,
coordinate grid search, PSO, and a GA, each searching against the *identical*
`Fitness` object the real colony (`aco/colony.py`) uses, capped at the *identical*
number of evaluations via `BudgetedFitness` (`aco/fitness.py`).

**Why this file matters more than its size suggests.** Per the plan: this is "the
make-or-break experiment (claim C2). If ACO ties random search at equal budget, you
do not have an ACO paper." Every control here must be a genuinely competent
implementation of its own metaheuristic -- a strawman random search would make the
comparison meaningless in the other direction.

The two discrete controls (random search, coordinate grid search) reuse
`aco/colony.py::levels_to_alpha` -- the exact same discretized construction the real
colony uses -- so they search literally the same space, not an approximation of it.
PSO's particles operate in continuous multiplier space directly (no natural discrete
analogue for velocity), which is standard for PSO and not a departure the ablation
needs to control for; the comparison is about evaluation count, not discretization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from fedswarm.aco.colony import levels_to_alpha
from fedswarm.aco.fitness import BudgetedFitness


@dataclass
class ControlResult:
    alpha_best: torch.Tensor
    best_fitness: float
    evaluations_used: int


def _to_alpha_continuous(
    position: torch.Tensor, base_weights: torch.Tensor, target_sum: float, low: float, high: float
) -> torch.Tensor:
    tilde = position.clamp(low, high) * base_weights
    total = tilde.sum()
    if total <= 1e-12:
        return base_weights * target_sum
    return tilde * (target_sum / total)


def random_search(
    levels: torch.Tensor,
    base_weights: torch.Tensor,
    fitness: BudgetedFitness,
    budget: int,
    target_sum: float = 1.0,
    generator: torch.Generator | None = None,
) -> ControlResult:
    """Uniform random sampling over the same discretized construction graph the
    colony uses -- the simplest possible baseline, and the one ACO absolutely must
    beat at equal budget for claim C2 to hold at all."""
    num_clients = base_weights.numel()
    num_levels = levels.numel()
    best_alpha: torch.Tensor | None = None
    best_fitness = -math.inf
    used = 0

    while fitness.remaining() > 0 and used < budget:
        level_idx = torch.randint(0, num_levels, (num_clients,), generator=generator)
        alpha = levels_to_alpha(level_idx, levels, base_weights, target_sum)
        value = fitness.evaluate(alpha)
        used += 1
        if value > best_fitness:
            best_fitness, best_alpha = value, alpha

    assert best_alpha is not None
    return ControlResult(best_alpha, best_fitness, used)


def coordinate_grid_search(
    levels: torch.Tensor,
    base_weights: torch.Tensor,
    fitness: BudgetedFitness,
    budget: int,
    target_sum: float = 1.0,
) -> ControlResult:
    """Coordinate descent over the discretized grid: cycle through clients one at a
    time, exhaustively try every level for that client with everyone else held at
    their current best, keep whatever wins, move to the next client. Starts from
    the FedAvg point (lambda=1 everywhere), the natural, zero-cost starting point --
    not a random one, so this control gets it for free before spending any budget."""
    num_clients = base_weights.numel()
    num_levels = levels.numel()

    one_idx = int(torch.argmin(torch.abs(levels - 1.0)))
    level_idx = torch.full((num_clients,), one_idx, dtype=torch.long)
    best_alpha = levels_to_alpha(level_idx, levels, base_weights, target_sum)
    best_fitness = fitness.evaluate(best_alpha)
    used = 1

    client = 0
    while fitness.remaining() > 0 and used < budget:
        current_best_l = int(level_idx[client])
        current_best_f = best_fitness
        for level in range(num_levels):
            if fitness.remaining() <= 0 or used >= budget:
                break
            if level == int(level_idx[client]):
                continue
            trial_idx = level_idx.clone()
            trial_idx[client] = level
            trial_alpha = levels_to_alpha(trial_idx, levels, base_weights, target_sum)
            value = fitness.evaluate(trial_alpha)
            used += 1
            if value > current_best_f:
                current_best_f, current_best_l = value, level
        if current_best_f > best_fitness:
            best_fitness = current_best_f
            level_idx[client] = current_best_l
            best_alpha = levels_to_alpha(level_idx, levels, base_weights, target_sum)
        client = (client + 1) % num_clients

    return ControlResult(best_alpha, best_fitness, used)


@dataclass
class PSOConfig:
    num_particles: int = 10
    inertia: float = 0.5
    cognitive: float = 1.5
    social: float = 1.5


def pso_search(
    levels: torch.Tensor,
    base_weights: torch.Tensor,
    fitness: BudgetedFitness,
    budget: int,
    target_sum: float = 1.0,
    config: PSOConfig | None = None,
    generator: torch.Generator | None = None,
) -> ControlResult:
    """Standard particle swarm optimization (inertia + cognitive + social velocity
    update) directly in continuous multiplier space, clamped to [levels.min(),
    levels.max()]. No discretization: particles are free to land anywhere in that
    continuous range, which is the natural PSO representation."""
    config = config or PSOConfig()
    num_clients = base_weights.numel()
    low, high = float(levels.min()), float(levels.max())

    positions = low + (high - low) * torch.rand(config.num_particles, num_clients, generator=generator)
    velocities = torch.zeros(config.num_particles, num_clients)
    personal_best_pos = positions.clone()
    personal_best_fit = torch.full((config.num_particles,), -math.inf)
    global_best_pos: torch.Tensor | None = None
    global_best_fit = -math.inf
    used = 0

    while fitness.remaining() > 0 and used < budget:
        for i in range(config.num_particles):
            if fitness.remaining() <= 0 or used >= budget:
                break
            alpha = _to_alpha_continuous(positions[i], base_weights, target_sum, low, high)
            value = fitness.evaluate(alpha)
            used += 1
            if value > personal_best_fit[i]:
                personal_best_fit[i] = value
                personal_best_pos[i] = positions[i].clone()
            if value > global_best_fit:
                global_best_fit = value
                global_best_pos = positions[i].clone()

        if global_best_pos is None or used >= budget:
            break

        r1 = torch.rand(config.num_particles, num_clients, generator=generator)
        r2 = torch.rand(config.num_particles, num_clients, generator=generator)
        velocities = (
            config.inertia * velocities
            + config.cognitive * r1 * (personal_best_pos - positions)
            + config.social * r2 * (global_best_pos.unsqueeze(0) - positions)
        )
        positions = (positions + velocities).clamp(low, high)

    assert global_best_pos is not None
    best_alpha = _to_alpha_continuous(global_best_pos, base_weights, target_sum, low, high)
    return ControlResult(best_alpha, global_best_fit, used)


@dataclass
class GAConfig:
    population_size: int = 10
    mutation_rate: float = 0.1
    elite_fraction: float = 0.2
    tournament_size: int = 3


def genetic_algorithm_search(
    levels: torch.Tensor,
    base_weights: torch.Tensor,
    fitness: BudgetedFitness,
    budget: int,
    target_sum: float = 1.0,
    config: GAConfig | None = None,
    generator: torch.Generator | None = None,
) -> ControlResult:
    """Standard generational GA over the discretized level-index representation:
    tournament selection, uniform crossover, per-gene mutation (reassign to a random
    level), with elitism carrying the top performers forward unchanged."""
    config = config or GAConfig()
    num_clients = base_weights.numel()
    num_levels = levels.numel()
    num_elite = max(1, int(config.elite_fraction * config.population_size))

    def to_alpha(level_idx: torch.Tensor) -> torch.Tensor:
        return levels_to_alpha(level_idx, levels, base_weights, target_sum)

    def tournament_pick(pool: torch.Tensor, fitnesses: torch.Tensor) -> torch.Tensor:
        candidates = torch.randint(0, pool.shape[0], (config.tournament_size,), generator=generator)
        winner = candidates[torch.argmax(fitnesses[candidates])]
        return pool[winner]

    population = torch.randint(
        0, num_levels, (config.population_size, num_clients), generator=generator
    )
    used = 0
    best_alpha: torch.Tensor | None = None
    best_fitness = -math.inf

    while fitness.remaining() > 0 and used < budget:
        fitnesses = torch.full((config.population_size,), -math.inf)
        for i in range(config.population_size):
            if fitness.remaining() <= 0 or used >= budget:
                break
            alpha = to_alpha(population[i])
            value = fitness.evaluate(alpha)
            used += 1
            fitnesses[i] = value
            if value > best_fitness:
                best_fitness, best_alpha = value, alpha

        if fitness.remaining() <= 0 or used >= budget:
            break

        order = torch.argsort(fitnesses, descending=True)
        next_generation = [population[order[i]].clone() for i in range(num_elite)]
        while len(next_generation) < config.population_size:
            parent_a = tournament_pick(population, fitnesses)
            parent_b = tournament_pick(population, fitnesses)
            mask = torch.rand(num_clients, generator=generator) < 0.5
            child = torch.where(mask, parent_a, parent_b).clone()
            mutate_mask = torch.rand(num_clients, generator=generator) < config.mutation_rate
            num_mutations = int(mutate_mask.sum())
            if num_mutations > 0:
                child[mutate_mask] = torch.randint(0, num_levels, (num_mutations,), generator=generator)
            next_generation.append(child)
        population = torch.stack(next_generation[: config.population_size], dim=0)

    assert best_alpha is not None
    return ControlResult(best_alpha, best_fitness, used)
