"""Level set and adaptive colony-budget schedules (plan §4.2, §4.7)."""

from __future__ import annotations

import math

import torch


def level_set(num_levels: int = 11, low: float = 0.0, high: float = 2.5) -> torch.Tensor:
    """L multiplier levels, log-spaced over [low, high] (plan §4.2's default: L=11 over
    [0, 2.5]) with lambda=1 always present exactly -- the level that makes an ant's
    choice at every station recover the FedAvg weight, so FedAvg stays reachable inside
    the search space.

    Pure log-spacing is undefined at 0, so the positive part [max(low, eps), high] is
    log-spaced and 0 is kept as an explicit floor level when `low <= 0`. Exact inclusion
    of 1.0 is enforced by snapping the single closest generated level to 1.0 rather than
    hoping a log-spaced grid lands on it exactly.
    """
    if num_levels < 2:
        raise ValueError("level_set needs at least 2 levels (e.g. {0, 1})")
    eps = 1e-3
    positive_low = max(low, eps)
    levels = torch.logspace(
        math.log10(positive_low), math.log10(high), steps=num_levels, dtype=torch.float64
    )
    if low <= 0.0:
        levels[0] = 0.0
    closest = torch.argmin(torch.abs(levels - 1.0))
    levels[closest] = 1.0
    return torch.sort(levels).values.float()


def colony_budget(
    round_idx: int,
    num_rounds: int,
    ants_start: int = 30,
    ants_end: int = 10,
    iters_start: int = 10,
    iters_end: int = 4,
) -> tuple[int, int]:
    """Ants A_t and iterations I_t decay linearly with the round index (plan §4.7:
    "aggregation choice matters most early") from (ants_start, iters_start) at round 0
    to (ants_end, iters_end) at the final round."""
    denom = max(num_rounds - 1, 1)
    frac = min(max(round_idx, 0), denom) / denom
    ants = round(ants_start + (ants_end - ants_start) * frac)
    iters = round(iters_start + (iters_end - iters_start) * frac)
    return max(ants, 1), max(iters, 1)
