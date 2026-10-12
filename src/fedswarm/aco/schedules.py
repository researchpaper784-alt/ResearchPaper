"""Level set and adaptive colony-budget schedules (plan §4.2, §4.7)."""

from __future__ import annotations

import math

import torch


LEVEL_SPACINGS = ("linear", "log")


def level_set(
    num_levels: int = 11,
    low: float = 0.0,
    high: float = 2.5,
    spacing: str = "linear",
) -> torch.Tensor:
    """L multiplier levels over [low, high], with lambda=1 always present exactly -- the
    level that makes an ant's choice at every station recover the FedAvg weight, so FedAvg
    stays reachable inside the search space.

    **"linear" is the default because it is what the plan specifies.** §14's hyperparameter
    table writes the level set out explicitly:

        [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]

    which is exactly `linspace(0, 2.5, 11)`. This function was log-spaced until 2026-09-19
    -- a deviation nobody could catch, because the plan was not in the repository until the
    same day (it is now `docs/IMPLEMENTATION_PLAN.md`).

    The two are not a cosmetic difference. Over [0, 2.5] with L=11:

    | | linear (plan) | log (what shipped) |
    |---|---|---|
    | levels | 0, 0.25, ... 2.5 | 0, 0.0022, ... 0.523, 1.0, 2.5 |
    | max ratio between positive levels | 10x | **1143x** |
    | levels below 1.0 | 4 of 11 | **9 of 11** |

    Log spacing puts nine of eleven levels below the FedAvg point and lets a single ant
    construct alpha ratios of over 1000:1, so a near-vertex weighting like [0.99, 0.01] is
    one greedy draw away. Linear spacing caps the ratio at 10:1. That directly changes how
    reachable the degenerate single-client optimum is (`fitness.corner_margin`,
    docs/EXPERIMENT_LOG.md 2026-09-18), so a conclusion about the fitness drawn under log
    spacing is not transferable to the plan's search space.

    "log" is kept, not deleted: it is what every result produced before 2026-09-19 used, and
    A6 (§7's hyperparameter-sensitivity ablation, which owns L) is where the choice belongs.

    Pure log spacing is undefined at 0, so under `spacing="log"` the positive part
    [max(low, eps), high] is log-spaced and 0 is kept as an explicit floor level when
    `low <= 0`. Under "linear" no special case is needed -- 0 is simply the first point.
    Exact inclusion of 1.0 is enforced by snapping the closest generated level to it rather
    than hoping the grid lands there; at the plan's defaults linear already contains 1.0
    exactly, so the snap is a no-op.
    """
    if num_levels < 2:
        raise ValueError("level_set needs at least 2 levels (e.g. {0, 1})")
    if spacing not in LEVEL_SPACINGS:
        raise ValueError(f"Unknown level spacing {spacing!r} (expected one of {LEVEL_SPACINGS})")

    if spacing == "linear":
        levels = torch.linspace(low, high, steps=num_levels, dtype=torch.float64)
    else:
        eps = 1e-3
        levels = torch.logspace(
            math.log10(max(low, eps)), math.log10(high), steps=num_levels, dtype=torch.float64
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
