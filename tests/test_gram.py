"""test_gram_equivalence (plan §4.8 acceptance #1): Gram-computed fitness must match a
naive full-vector computation to within 1e-4, for random deltas and random alpha."""

from __future__ import annotations

import math

import torch

from fedswarm.aco.fitness import DataFreeFitness, DataFreeFitnessConfig
from fedswarm.aco.gram import precompute_gram, trimmed_mean


def _naive_fitness(
    deltas: torch.Tensor, alpha: torch.Tensor, config: DataFreeFitnessConfig, trim_fraction: float = 0.2
) -> float:
    robust_mean = trimmed_mean(deltas, trim_fraction=trim_fraction)
    combined = alpha @ deltas  # Delta(alpha), materialized in full -- what the Gram trick avoids

    alignment = torch.dot(combined, robust_mean) / (
        combined.norm() * robust_mean.norm() + 1e-12
    )
    dispersion = sum(
        float(alpha[k]) * (deltas[k] - combined).pow(2).sum() for k in range(deltas.shape[0])
    )
    if config.normalize_dispersion:
        # Same scale the Gram path divides by, computed the naive way: mean ||delta_k||^2
        # over clients, which is exactly trace(G)/K.
        dispersion = dispersion / max(float(deltas.pow(2).sum(dim=1).mean()), 1e-12)
    total = alpha.sum().clamp_min(1e-12)
    probs = (alpha / total).clamp_min(1e-12)
    entropy = float(-(probs * probs.log()).sum())
    penalty = math.log(alpha.numel()) - entropy

    f = (
        config.gamma_alignment * float(alignment)
        - config.gamma_dispersion * float(dispersion)
        - config.gamma_entropy * penalty
    )
    return f


def test_gram_equivalence() -> None:
    torch.manual_seed(0)
    num_clients, dim = 12, 500
    deltas = torch.randn(num_clients, dim)
    alpha = torch.rand(num_clients)
    alpha = alpha / alpha.sum()  # a valid point on the simplex, like a colony candidate

    config = DataFreeFitnessConfig()
    gram = precompute_gram(deltas, trim_fraction=0.2)
    gram_fitness = DataFreeFitness(gram, config).evaluate(alpha)
    naive_fitness = _naive_fitness(deltas, alpha, config)

    assert abs(gram_fitness - naive_fitness) < 1e-4


def test_gram_equivalence_across_several_random_points() -> None:
    torch.manual_seed(1)
    num_clients, dim = 8, 300
    deltas = torch.randn(num_clients, dim)
    config = DataFreeFitnessConfig()
    gram = precompute_gram(deltas, trim_fraction=0.25)
    fitness = DataFreeFitness(gram, config)

    for _ in range(5):
        alpha = torch.rand(num_clients)
        alpha = alpha / alpha.sum()
        naive = _naive_fitness(deltas, alpha, config, trim_fraction=0.25)
        assert abs(fitness.evaluate(alpha) - naive) < 1e-4
