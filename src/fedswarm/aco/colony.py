"""Ant Colony System construction graph and run loop (plan §4.2, §4.7): K decision
stations (one per client), an ant picks one multiplier level per station, MAX-MIN-style
evaporation/deposit updates tau after every iteration, with an early-stop on stagnation
or low pheromone entropy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class ColonyConfig:
    pheromone_exp: float = 1.0  # 'a' in tau^a * eta^b
    heuristic_exp: float = 2.0  # 'b' in tau^a * eta^b
    q0: float = 0.9  # exploitation probability (ACS pseudo-random-proportional rule)
    rho: float = 0.1  # within-colony evaporation rate
    q_deposit: float = 1.0  # deposit scale Q
    tau_min: float = 0.01
    tau_max: float = 10.0
    global_best_every: int = 5  # deposit the global-best every T iterations, on top of the iteration-best
    stagnation_patience: int = 5  # stop early if the global best hasn't improved for this many iterations
    entropy_stop_threshold: float | None = None


@dataclass
class ColonyResult:
    alpha_best: torch.Tensor
    best_fitness: float
    tau_final: torch.Tensor
    realized_ants: int
    realized_iterations: int
    final_entropy: float


def _pheromone_entropy(tau: torch.Tensor) -> float:
    probs = (tau / tau.sum(dim=1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-12)
    row_entropy = -(probs * probs.log()).sum(dim=1)
    return float(row_entropy.mean())


def _select_levels(
    tau: torch.Tensor,
    eta: torch.Tensor,
    config: ColonyConfig,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Pseudo-random-proportional transition rule (ACS, plan §4.2): with probability
    q0, greedily pick the highest-scoring level at each station; otherwise sample
    proportionally to tau^a * eta^b. `q0 >= 1.0` is special-cased to skip the random
    draw entirely, so a fully-exploitative colony is exactly reproducible regardless of
    RNG state -- this is what `test_fedavg_recoverable` depends on."""
    num_clients = tau.shape[0]
    scores = tau.clamp_min(1e-12).pow(config.pheromone_exp) * eta.clamp_min(1e-12).pow(
        config.heuristic_exp
    )
    greedy = scores.argmax(dim=1)
    if config.q0 >= 1.0:
        return greedy

    probs = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
    explore = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
    use_greedy = torch.rand(num_clients, generator=generator) < config.q0
    return torch.where(use_greedy, greedy, explore)


def levels_to_alpha(
    level_idx: torch.Tensor, levels: torch.Tensor, base_weights: torch.Tensor, target_sum: float
) -> torch.Tensor:
    """Public (not colony-internal) because `aco/controls.py`'s discrete A1 controls
    (plan §7) reuse this exact construction -- reimplementing it there would risk
    silent drift between what the colony and its equal-budget controls actually
    search over, which would invalidate the comparison the ablation exists to make."""
    tilde = levels[level_idx] * base_weights
    total = tilde.sum()
    if total <= 1e-12:
        # Every station happened to pick lambda=0 (possible, if rare, under the ACS
        # rule -- e.g. a fully uniform tau/eta with ties broken by argmax). Normalizing
        # a near-zero vector up to target_sum would still leave it at ~0, silently
        # violating "alpha sums to target_sum" (plan §4.8 acceptance #5, test_simplex)
        # -- fall back to the FedAvg point instead of returning an invalid candidate.
        return base_weights * target_sum
    return tilde * (target_sum / total)


def run_colony(
    tau0: torch.Tensor,
    eta: torch.Tensor,
    levels: torch.Tensor,
    base_weights: torch.Tensor,
    fitness_fn: Callable[[torch.Tensor], float],
    num_ants: int,
    num_iterations: int,
    config: ColonyConfig | None = None,
    target_sum: float = 1.0,
    generator: torch.Generator | None = None,
) -> ColonyResult:
    """Runs the colony for up to `num_iterations` iterations of `num_ants` ants each,
    starting from pheromone matrix `tau0` ([K, num_levels], from `Pheromone.begin_round`).
    Returns the best (alpha, fitness) found, the post-run tau (to hand back to
    `Pheromone.end_round`), and the realized budget (may be less than requested, on
    early stop)."""
    config = config or ColonyConfig()
    num_clients = tau0.shape[0]
    tau = tau0.clone()

    best_alpha: torch.Tensor | None = None
    best_fitness = -float("inf")
    best_levels: torch.Tensor | None = None
    stagnant_iterations = 0
    iterations_run = 0
    entropy = _pheromone_entropy(tau)

    for iteration in range(num_iterations):
        iterations_run = iteration + 1
        iter_best_fitness = -float("inf")
        iter_best_alpha: torch.Tensor | None = None
        iter_best_levels: torch.Tensor | None = None

        for _ in range(num_ants):
            level_idx = _select_levels(tau, eta, config, generator)
            candidate = levels_to_alpha(level_idx, levels, base_weights, target_sum)
            fitness = fitness_fn(candidate)
            if fitness > iter_best_fitness:
                iter_best_fitness, iter_best_alpha, iter_best_levels = fitness, candidate, level_idx

        assert iter_best_levels is not None and iter_best_alpha is not None

        tau = (1.0 - config.rho) * tau
        deposit = config.q_deposit * max(iter_best_fitness, 0.0)
        tau[torch.arange(num_clients), iter_best_levels] += config.rho * deposit

        if best_levels is not None and (iteration + 1) % config.global_best_every == 0:
            best_deposit = config.q_deposit * max(best_fitness, 0.0)
            tau[torch.arange(num_clients), best_levels] += config.rho * best_deposit

        tau = tau.clamp(config.tau_min, config.tau_max)

        if iter_best_fitness > best_fitness:
            best_fitness, best_alpha, best_levels = iter_best_fitness, iter_best_alpha, iter_best_levels
            stagnant_iterations = 0
        else:
            stagnant_iterations += 1

        entropy = _pheromone_entropy(tau)
        if config.entropy_stop_threshold is not None and entropy < config.entropy_stop_threshold:
            break
        if stagnant_iterations >= config.stagnation_patience:
            break

    assert best_alpha is not None
    return ColonyResult(
        alpha_best=best_alpha,
        best_fitness=best_fitness,
        tau_final=tau,
        realized_ants=num_ants,
        realized_iterations=iterations_run,
        final_entropy=entropy,
    )
