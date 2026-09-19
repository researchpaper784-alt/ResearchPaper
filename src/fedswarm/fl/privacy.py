"""Phase 8, R4 -- the Gaussian mechanism on every client's update.

Deliberately *not* in `fl/attacks.py`, and the distinction is the point: an attack is
something an adversary does to a subset of clients, and this is something every honest
client does to itself. Conflating them would make R4's results uninterpretable -- a
robustness sweep cannot tell "survives 20% adversaries" from "survives noise everyone
added on purpose" if both arrive through the same switch.

R4 answers the question every FL paper that inspects raw updates gets asked: *does your
method still work when the updates are noised for privacy?* FedACO is more exposed to it
than FedAvg is, because its fitness reads update geometry -- alignment against the robust
consensus direction, and the norm ratio against the median client. Both are computed from
exactly the quantities the noise corrupts, so "does the colony still find anything" is a
real question rather than a formality.

⚠️ **This is the Gaussian mechanism, not a privacy guarantee.** It clips and noises, which
is the shape of DP-SGD's per-round step, but it does no (epsilon, delta) accounting and
composes nothing across rounds. It answers "does the method survive noise of this
magnitude", which is what R4 asks. Reporting an epsilon from these runs would be
fabricating a number the code never computed.
"""

from __future__ import annotations

from collections import OrderedDict

import torch


def clip_and_noise(
    base: "OrderedDict[str, torch.Tensor]",
    local: "OrderedDict[str, torch.Tensor]",
    sigma: float,
    clip_norm: float,
    generator: torch.Generator | None = None,
) -> "OrderedDict[str, torch.Tensor]":
    """`base` + (the clipped update + Gaussian noise), the standard two-step mechanism.

    Clipping comes first and is not optional: noise calibrated to a bound means nothing
    if the thing it is added to is unbounded, so a "DP" implementation that only adds
    noise is noise, not a mechanism. The bound is on the L2 norm of the *whole* update
    across all parameters, not per-tensor -- per-tensor clipping would bound a different
    quantity and scale differently with model depth.

    Noise is `N(0, sigma * clip_norm)` per coordinate, so `sigma` is the noise multiplier
    in units of the clipping bound -- which is what makes a single `dp-noise-sigma` value
    comparable across models and rounds.

    ⚠️ **`sigma == 0` turns the whole mechanism off, clipping included, and that is a
    judgement call with a consequence.** It makes R4's `sigma=0.0_control` cell identical
    to an ordinary run, so it is directly comparable to the main sweep's numbers -- which
    is what a reader will assume a "control" means. The cost is that every sigma>0 cell
    then differs from the control by *two* things, clipping and noise, so a drop at
    sigma=0.01 cannot be attributed to noise alone. Separating them needs a clip-only
    cell (`dp-noise-sigma=0` with clipping forced on), which R4 does not currently have.
    Recorded in docs/OPEN_QUESTIONS.md rather than left for the analysis to trip over.

    Non-float and unmatched tensors pass through untouched, the same rule
    `attacks.poison_update` follows: integer buffers and SCAFFOLD control variates are
    protocol state, not the model.
    """
    if sigma < 0:
        raise ValueError(f"dp-noise-sigma must be >= 0, got {sigma}")
    if clip_norm <= 0:
        raise ValueError(f"dp-clip-norm must be > 0, got {clip_norm}")
    if sigma == 0:
        return local  # mechanism off -- see the docstring

    updatable = [
        key
        for key in local
        if key in base and local[key].is_floating_point() and base[key].shape == local[key].shape
    ]
    if not updatable:
        return local

    total_sq = sum(float((local[k] - base[k]).pow(2).sum()) for k in updatable)
    update_norm = total_sq**0.5
    # Scale down if over the bound, never up -- an update already inside it is untouched,
    # which is what "clip" means and what keeps the sigma=0 control a true no-op.
    scale = min(1.0, clip_norm / update_norm) if update_norm > 0 else 1.0

    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key, value in local.items():
        if key not in updatable:
            out[key] = value
            continue
        clipped = (value - base[key]) * scale
        if sigma > 0:
            noise = torch.normal(
                mean=0.0,
                std=sigma * clip_norm,
                size=clipped.shape,
                generator=generator,
                dtype=clipped.dtype,
            )
            clipped = clipped + noise
        out[key] = base[key] + clipped
    return out


def dp_from_run_config(run_config: dict) -> tuple[float, float]:
    """`(sigma, clip_norm)` from a run config. `sigma == 0` means the mechanism is off.

    Raises rather than clamping on a negative sigma: a typo'd `dp-noise-sigma=-0.1` that
    quietly ran clean would be recorded as a privacy result showing perfect robustness,
    which is the same failure shape `attacks.attack_from_run_config` guards against.
    """
    sigma = float(run_config.get("dp-noise-sigma", 0.0))
    clip_norm = float(run_config.get("dp-clip-norm", 1.0))
    if sigma < 0:
        raise ValueError(f"dp-noise-sigma must be >= 0, got {sigma}")
    if clip_norm <= 0:
        raise ValueError(f"dp-clip-norm must be > 0, got {clip_norm}")
    return sigma, clip_norm
