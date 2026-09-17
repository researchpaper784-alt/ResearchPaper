"""Heuristic desirability eta_{k,l} (plan §4.2) from server-side signals that cost no
extra privacy beyond what FedAvg already sees: each client's update and a handful of
scalars it already reports. Reuses `GramPrecompute` for the two update-geometry terms
(a_k, r_k) so this stays O(K), not O(K*d)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from fedswarm.aco.gram import GramPrecompute


@dataclass
class HeuristicWeights:
    beta_alignment: float = 2.0  # beta_1
    beta_drift: float = 1.0  # beta_2
    beta_val_improvement: float = 1.0  # beta_3
    beta_data_size: float = 0.5  # beta_4


def desirability_scores(
    gram: GramPrecompute,
    num_examples: torch.Tensor,
    val_improvement: torch.Tensor | None = None,
    weights: HeuristicWeights | None = None,
) -> torch.Tensor:
    """d_k = sigmoid(beta_1*a_k - beta_2*|log r_k| + beta_3*v_k + beta_4*q_k), plan §4.2.

    - a_k: cosine alignment of client k's update with the robust consensus direction.
    - r_k: update-magnitude ratio against the median client (drift/anomaly signal).
    - v_k: client-reported local validation/training-loss improvement (0 if unavailable).
    - q_k: data-size prior, n_k / max_j n_j.
    """
    weights = weights or HeuristicWeights()
    eps = 1e-12

    diag = torch.diagonal(gram.gram).clamp_min(eps)
    norms = diag.sqrt()
    a_k = gram.g_rob / (norms * math.sqrt(max(gram.rob_norm_sq, eps)) + eps)

    median_norm = norms.median().clamp_min(eps)
    r_k = (norms / median_norm).clamp_min(eps)

    v_k = val_improvement if val_improvement is not None else torch.zeros_like(norms)

    q_k = num_examples / num_examples.max().clamp_min(eps)

    z = (
        weights.beta_alignment * a_k
        - weights.beta_drift * torch.log(r_k).abs()
        + weights.beta_val_improvement * v_k
        + weights.beta_data_size * q_k
    )
    return torch.sigmoid(z)


def desirability_matrix(d_k: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """eta_{k,l} = (1 + |lambda_l - d_hat_k|)^-1, where d_hat_k rescales the sigmoid
    output d_k in (0, 1) linearly into [lambda_1, lambda_L]."""
    lam_min, lam_max = levels.min(), levels.max()
    d_hat = lam_min + d_k * (lam_max - lam_min)
    diff = (levels.unsqueeze(0) - d_hat.unsqueeze(1)).abs()  # [K, L]
    return 1.0 / (1.0 + diff)
