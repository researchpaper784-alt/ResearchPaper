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
    # How d_k maps onto the level range. See `desirability_matrix` -- "absolute" is plan
    # §4.2 as written and the default; "standardized" is the screenable candidate fix for
    # the 0.04-span problem that anchors the colony at the FedAvg point.
    scaling: str = "absolute"


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


# How d_k is mapped onto the level range before eta is formed. "absolute" is plan §4.2 as
# written and stays the default; "standardized" is a candidate fix for the measured problem
# below, screenable without touching the spec.
DESIRABILITY_SCALINGS = ("absolute", "standardized")


def desirability_matrix(
    d_k: torch.Tensor, levels: torch.Tensor, scaling: str = "absolute"
) -> torch.Tensor:
    """eta_{k,l} = (1 + |lambda_l - d_hat_k|)^-1, with d_hat_k the sigmoid output d_k mapped
    onto [lambda_1, lambda_L].

    **The measured problem.** Across K in {10, 20} and noise in {1.5, 3, 6}, the raw `d_k`
    spans about **0.04** (e.g. 0.7685-0.8073) against a level spacing of **0.25**. Every
    client's d_hat therefore lands in the same level bin, `argmax_l eta_{k,l}` returns the
    same level for every client, and a uniform level assignment normalises back to
    `base_weights` *exactly* (`levels_to_alpha`). So the greedy branch of the ACS rule
    reconstructs the FedAvg point, and at the default q0 that is 70% of station decisions.
    The colony is anchored at its own reference point by its exploitation rule. See
    docs/OPEN_QUESTIONS.md, "the greedy rule constructs the FedAvg point".

    `scaling`:

    * **"absolute"** -- plan §4.2 as specified: a linear map of (0, 1) onto the level range.
      `d_k` keeps its absolute meaning, so a federation whose clients genuinely are equally
      good produces a uniform assignment, which is the correct answer for that federation.
      It also produces one when they differ by 0.04, which is the bug.
    * **"standardized"** -- z-score `d_k` across clients, then map +/-2 sigma onto the level
      range with clipping. The *ranking* survives and the spread always fills the range, so
      small real differences become distinguishable levels.

      ⚠️ It is also outlier-sensitive: one extreme client inflates the standard deviation
      and compresses the rest back into a single bin, which is the original failure reached
      by another route. `test_standardized_scaling_is_outlier_sensitive` pins that.

      ⚠️ This discards absolute scale by construction. Under "standardized" the weakest
      client is always pushed toward lambda_1 and the strongest toward lambda_L **even when
      every client is equally good**, because a z-score of near-identical values is
      dominated by noise. That is a different method, not a bug fix, and which behaviour is
      wanted is a modelling decision -- hence a flag and a screen rather than a change to
      the default. A federation of identical clients is exactly where it misbehaves, and
      `regime: iid` is that federation.
    """
    if scaling not in DESIRABILITY_SCALINGS:
        raise ValueError(
            f"desirability scaling {scaling!r} is not one of {DESIRABILITY_SCALINGS}"
        )

    lam_min, lam_max = levels.min(), levels.max()

    if scaling == "standardized":
        centred = d_k - d_k.mean()
        spread = d_k.std(unbiased=False)
        # A single client, or K identical scores, has no spread to standardize by. Falling
        # through to the uniform midpoint is the honest answer there: there is no ranking
        # information in the input, so inventing one from floating-point dust would be worse
        # than the anchoring this option exists to fix.
        if float(spread) <= 1e-12:
            unit = torch.full_like(d_k, 0.5)
        else:
            # +/-2 sigma spans the level range and beyond it is clipped, so d_hat stays
            # inside [lambda_1, lambda_L] instead of being extrapolated past the grid.
            #
            # The clamp bounds an outlier's own position; it does NOT stop that outlier
            # inflating sigma. With d_k = [0.50, 0.51, 0.52, 0.53, 50.0] the four ordinary
            # clients span 0.0009 in d_hat against a level spacing of 0.25, so they collapse
            # into one bin exactly as under "absolute". Resisting that needs a robust
            # centre and scale (median/MAD), which is a third variant and a further
            # departure from §4.2 -- not something to add without the screen asking for it.
            unit = ((centred / spread).clamp(-2.0, 2.0) + 2.0) / 4.0
        d_hat = lam_min + unit * (lam_max - lam_min)
    else:
        d_hat = lam_min + d_k * (lam_max - lam_min)

    diff = (levels.unsqueeze(0) - d_hat.unsqueeze(1)).abs()  # [K, L]
    return 1.0 / (1.0 + diff)
