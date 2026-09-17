"""Gram-matrix trick (plan §4.6): every term in the data-free fitness (§4.5) depends on
client updates only through inner products, so precomputing G once per round turns each
of the A*I colony fitness evaluations from O(d) (d ~= 11M model parameters) into O(K)
(K ~= number of participating clients) -- the difference between negligible and
unpublishable per the plan's own framing.

Also holds the flatten/unflatten helpers for turning a list of per-client state dicts
into the stacked [K, d] delta matrix `compute_gram_matrix` expects, and back into a real
state dict once a winning alpha is chosen -- kept here rather than in strategies/fedaco.py
because both directions are pure tensor bookkeeping, not Flower-specific.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import torch


def flatten_state_dicts(
    base: "OrderedDict[str, torch.Tensor]",
    others: list["OrderedDict[str, torch.Tensor]"],
) -> tuple[torch.Tensor, list[tuple[str, torch.Size]]]:
    """Stack `others[k] - base` as flattened float32 rows -> deltas [K, d]. `shapes`
    records each parameter's key and shape in flatten order, the only thing needed to
    unflatten a combined delta back into a real state dict later."""
    shapes: list[tuple[str, torch.Size]] = [(k, v.shape) for k, v in base.items()]
    base_flat = torch.cat([v.reshape(-1).float() for v in base.values()])
    rows = [
        torch.cat([other[k].reshape(-1).float() for k, _ in shapes]) - base_flat
        for other in others
    ]
    return torch.stack(rows, dim=0), shapes


def unflatten_delta(
    vec: torch.Tensor, shapes: list[tuple[str, torch.Size]]
) -> "OrderedDict[str, torch.Tensor]":
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    offset = 0
    for key, shape in shapes:
        numel = math.prod(shape)
        out[key] = vec[offset : offset + numel].reshape(shape)
        offset += numel
    return out


def apply_delta(
    base: "OrderedDict[str, torch.Tensor]",
    delta_flat: torch.Tensor,
    shapes: list[tuple[str, torch.Size]],
) -> "OrderedDict[str, torch.Tensor]":
    """`base` + the flattened combined delta, cast back to each parameter's own dtype
    (deltas are computed/aggregated in float32 regardless of the model's own dtype)."""
    delta_sd = unflatten_delta(delta_flat, shapes)
    return OrderedDict((k, base[k] + delta_sd[k].to(base[k].dtype)) for k in base)


def compute_gram_matrix(deltas: torch.Tensor, chunk_size: int | None = None) -> torch.Tensor:
    """G[i, j] = <delta_i, delta_j>. `chunk_size` sums partial Gram matrices over
    slices of the parameter dimension when the full [K, d] matrix doesn't fit in
    memory at once; mathematically identical to the unchunked matmul."""
    if chunk_size is None:
        return deltas @ deltas.T
    num_clients, dim = deltas.shape
    gram = torch.zeros(num_clients, num_clients, dtype=deltas.dtype, device=deltas.device)
    for start in range(0, dim, chunk_size):
        chunk = deltas[:, start : start + chunk_size]
        gram += chunk @ chunk.T
    return gram


def trimmed_mean(deltas: torch.Tensor, trim_fraction: float) -> torch.Tensor:
    """Coordinate-wise trimmed mean across the client axis (dim 0) -- the robust
    consensus direction bar-Delta_rob used throughout §4.5. Not expressible as a fixed
    linear combination of the K deltas (trimming differs per coordinate), so it is
    materialized as a real d-dim vector once per round (O(K*d), the same order as the
    Gram pass itself) rather than approximated -- see `precompute_gram`'s docstring for
    how this stays out of the O(K) per-evaluation inner loop despite that."""
    num_clients = deltas.shape[0]
    trim_k = int(num_clients * trim_fraction)
    if 2 * trim_k >= num_clients:
        trim_k = max((num_clients - 1) // 2, 0)
    sorted_vals, _ = torch.sort(deltas, dim=0)
    if trim_k > 0:
        sorted_vals = sorted_vals[trim_k : num_clients - trim_k]
    return sorted_vals.mean(dim=0)


@dataclass
class GramPrecompute:
    """Everything a colony run needs from this round's deltas, each computed exactly
    once (O(K^2*d) worst case, one pass) so every subsequent fitness evaluation is
    O(K) -- independent of model size."""

    gram: torch.Tensor  # [K, K]
    g_rob: torch.Tensor  # [K] -- <delta_k, robust_mean> for each client k
    rob_norm_sq: float  # ||robust_mean||^2


def precompute_gram(deltas: torch.Tensor, trim_fraction: float = 0.2) -> GramPrecompute:
    """`g_rob[k] = <delta_k, robust_mean>` is computed once here (one extra O(K*d)
    matmul, negligible next to the O(K^2*d) Gram pass) specifically so that
    `<Delta(alpha), bar_Delta_rob> = alpha @ g_rob` during colony search -- an O(K) op,
    not O(K*d) -- without needing bar_Delta_rob to be a fixed linear combination of the
    deltas (see `trimmed_mean`)."""
    gram = compute_gram_matrix(deltas)
    robust_mean = trimmed_mean(deltas, trim_fraction)
    g_rob = deltas @ robust_mean
    rob_norm_sq = float(robust_mean @ robust_mean)
    return GramPrecompute(gram=gram, g_rob=g_rob, rob_norm_sq=rob_norm_sq)


def weighted_norm_sq(alpha: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
    """||Delta(alpha)||^2 = alpha^T G alpha, plan eq. in §4.6."""
    return alpha @ (gram @ alpha)


def weighted_dispersion(alpha: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
    """sum_k alpha_k * ||delta_k - Delta(alpha)||^2, expanded per §4.6 so it only ever
    touches G, never a materialized d-dim vector."""
    g_alpha = gram @ alpha
    a_g_a = alpha @ g_alpha
    diag = torch.diagonal(gram)
    return (alpha * (diag - 2 * g_alpha + a_g_a)).sum()
