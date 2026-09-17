"""Phase 8 — adversarial client behaviour.

Until this module existed there was no way to make a client behave maliciously anywhere
in the codebase, which left three of the project's own baselines untestable on the only
thing they are for: Krum, FedTrimmedAvg and FedMedian are robust-aggregation rules whose
entire justification is surviving Byzantine clients, and nothing could put a Byzantine
client in front of them. (`num-malicious-nodes` is *Krum's own hyperparameter* — how many
Byzantine clients Krum should assume are present — not a mechanism that makes any client
actually misbehave. The two are easy to confuse and were.)

FedACO has a stake here too. Its desirability heuristic (`aco/heuristics.py`) scores
clients on alignment with the robust consensus direction and on update magnitude relative
to the median — both explicitly anomaly signals. `tests/test_strategy.py::
test_planted_bad_client` already asserts that a client sending a negated, amplified copy
of the consensus gets a weight materially below its FedAvg share. That is exactly the
`sign_flip` attack below, at unit scale; this module is what lets the same claim be made
about a real federated run.

**Two kinds of attack, deliberately kept apart:**

- *Update poisoning* (`sign_flip`, `scaled`, `gaussian`) corrupts what the client sends
  after training honestly. This is what robust aggregation defends against.
- *Data poisoning* (`label_flip`) corrupts the client's labels before training, so the
  client honestly reports a model trained on a lie. Aggregation-level defences see a
  perfectly well-formed update and have much less to grip.

A defence that handles one says nothing about the other, so a robustness claim has to
name which it was tested against.

**Who is malicious is deterministic**, by partition id: the lowest `ceil(fraction * K)`
ids. Not sampled per round, and not random — an attacker set that changes every round
measures something quite different from a fixed compromised subset, and a seeded-random
choice would make the result depend on a seed the analysis never sees. Fixed and
inspectable means the result files identify exactly who attacked.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch

ATTACKS = ("none", "sign_flip", "scaled", "gaussian", "label_flip")

# Attacks that corrupt the update after honest training, as opposed to the data before it.
UPDATE_ATTACKS = ("sign_flip", "scaled", "gaussian")
DATA_ATTACKS = ("label_flip",)


def malicious_ids(num_clients: int, fraction: float) -> set[int]:
    """The compromised partition ids: the lowest `ceil(fraction * num_clients)`.

    Deterministic rather than sampled, so every run at the same settings compromises the
    same clients and the result file identifies them without needing a seed the analysis
    does not have. `ceil` rather than `round` so a nonzero fraction always yields at least
    one attacker -- a "20% malicious" run on 4 clients that silently compromised none
    would read as evidence of robustness.
    """
    if fraction <= 0.0 or num_clients <= 0:
        return set()
    count = min(num_clients, math.ceil(fraction * num_clients))
    return set(range(count))


def is_malicious(partition_id: int, num_clients: int, fraction: float) -> bool:
    return partition_id in malicious_ids(num_clients, fraction)


def poison_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Cyclic label shift: `y -> (y + 1) mod C`.

    A cyclic shift rather than random relabelling, because it is a *consistent* lie -- the
    client trains to convergence on a coherent but wrong mapping, which is the realistic
    compromised-data case and much harder for an aggregator to spot than noise. Random
    per-sample labels mostly just prevent that client from learning anything, which
    resembles a weak client more than an adversarial one.
    """
    return (labels + 1) % num_classes


def poison_update(
    global_state: "OrderedDict[str, torch.Tensor]",
    local_state: "OrderedDict[str, torch.Tensor]",
    attack: str,
    scale: float,
    generator: torch.Generator | None = None,
) -> "OrderedDict[str, torch.Tensor]":
    """Return the state dict a malicious client should send instead of its honest one.

    Every attack is expressed as a transform of the *delta* (`local - global`) rather than
    of the weights, because the delta is what aggregation actually combines -- and because
    a transform applied to raw weights would be swamped by the weights themselves and
    barely move the aggregate.
    """
    if attack not in UPDATE_ATTACKS:
        return local_state

    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key, local in local_state.items():
        base = global_state.get(key)
        if base is None or not torch.is_floating_point(local):
            # SCAFFOLD control variates and any integer buffer pass through untouched:
            # corrupting them would be a different attack (on the protocol rather than on
            # the model), and conflating the two would make the result uninterpretable.
            out[key] = local
            continue
        delta = local - base
        if attack == "sign_flip":
            # The classic: push directly against the consensus, amplified. This is the
            # attack test_planted_bad_client already models at unit scale.
            poisoned = -scale * delta
        elif attack == "scaled":
            # Magnitude-only: same direction, wrong size. Defeats naive averaging by sheer
            # weight while staying aligned with everyone else, so alignment-based
            # heuristics (FedACO's a_k) see nothing wrong -- only magnitude ones (r_k) do.
            poisoned = scale * delta
        else:  # gaussian
            # Direction and magnitude both meaningless, matched to this tensor's own scale
            # so it is not trivially detectable as an outlier by norm alone.
            std = float(delta.std()) if delta.numel() > 1 else 1.0
            noise = torch.randn(delta.shape, generator=generator, dtype=delta.dtype)
            poisoned = scale * std * noise
        out[key] = base + poisoned
    return out


def attack_from_run_config(run_config: dict) -> tuple[str, float, float]:
    """`(attack, fraction, scale)` from a flat run config, validated.

    An unknown attack name raises rather than silently degrading to "none": a typo'd
    `--run-config "attack='signflip'"` that quietly ran a clean sweep would be recorded as
    a robustness result showing perfect robustness.
    """
    attack = str(run_config.get("attack", "none")).lower()
    if attack not in ATTACKS:
        raise ValueError(f"Unknown attack {attack!r}; expected one of {ATTACKS}")
    fraction = float(run_config.get("attack-fraction", 0.0))
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"attack-fraction must be in [0, 1], got {fraction}")
    return attack, fraction, float(run_config.get("attack-scale", 1.0))
