"""Fitness.evaluate(alpha) -> float, three pluggable modes (plan §4.4). `data_free` is
the method as proposed and the only one that benefits from the Gram trick (§4.6) -- the
other two are deliberately more expensive and, per the plan, only ever run at a heavily
reduced colony budget in the Phase 7 A3 ablation, not at the full search budget.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol

import torch
from torch.utils.data import DataLoader

from fedswarm.aco.gram import GramPrecompute, apply_delta, weighted_dispersion, weighted_norm_sq
from fedswarm.eval.evaluator import evaluate as evaluate_model


class Fitness(Protocol):
    def evaluate(self, alpha: torch.Tensor) -> float: ...


def _entropy(alpha: torch.Tensor, eps: float = 1e-12) -> float:
    total = alpha.sum().clamp_min(eps)
    probs = (alpha / total).clamp_min(eps)
    return float(-(probs * probs.log()).sum())


@dataclass
class DataFreeFitnessConfig:
    gamma_alignment: float = 1.0  # gamma_1
    gamma_dispersion: float = 1.0  # gamma_2
    gamma_entropy: float = 0.1  # gamma_3
    # Divide the dispersion term by trace(G)/K (`GramPrecompute.mean_sq_norm`) so it is
    # dimensionless like the other two terms. Left configurable rather than hard-wired
    # purely so the pre-normalization behavior stays reproducible for anyone re-running
    # results produced before 2026-09-17; there is no experimental reason to set it
    # False. See docs/EXPERIMENT_LOG.md for the measured justification.
    normalize_dispersion: bool = True


class DataFreeFitness:
    """F(alpha) from plan §4.5, evaluated in O(K) via the Gram trick (§4.6) -- the
    default and the method as proposed. No client data or extra round-trip needed
    beyond what FedAvg already collects.

    The dispersion term is divided by `gram.mean_sq_norm` (trace(G)/K). Without that
    division the three terms are not on comparable scales: alignment is a cosine in
    [-1, 1] and the entropy penalty is in [0, log K], but dispersion is a raw squared
    norm whose magnitude tracks the local-training step size. Measured over the
    (lr, local_epochs) grid Phases 6-7 sweep, dispersion at the FedAvg point ranged
    0.40 to 266.7 -- a 660x swing -- which drove F negative for every candidate in 6 of
    9 configs. Because `colony.py` floors deposits at `max(F, 0)`, that silently zeroed
    every pheromone deposit, leaving tau uniform and collapsing `tau^a * eta^b` to
    `eta^b`: no colony search, no cross-round stigmergy, just deterministic
    heuristic-greedy weighting, while the run still reported `fallback_used=0` (the
    safety fallback compares F_best against F_fedavg under the *same* fitness, so it
    cannot detect this). With the division, the same sweep holds dispersion in
    [0.763, 0.933] and tau develops real spread in all 9 configs.
    """

    def __init__(self, gram: GramPrecompute, config: DataFreeFitnessConfig | None = None) -> None:
        self.gram = gram
        self.config = config or DataFreeFitnessConfig()

    def evaluate(self, alpha: torch.Tensor) -> float:
        eps = 1e-12
        a_g_a = weighted_norm_sq(alpha, self.gram.gram).clamp_min(eps)
        alignment = (alpha @ self.gram.g_rob) / (
            torch.sqrt(a_g_a) * math.sqrt(max(self.gram.rob_norm_sq, eps))
        )
        dispersion = weighted_dispersion(alpha, self.gram.gram)
        if self.config.normalize_dispersion:
            dispersion = dispersion / max(self.gram.mean_sq_norm, eps)
        concentration_penalty = math.log(alpha.numel()) - _entropy(alpha)

        f = (
            self.config.gamma_alignment * alignment
            - self.config.gamma_dispersion * dispersion
            - self.config.gamma_entropy * concentration_penalty
        )
        return float(f)


def corner_margin(
    gram: GramPrecompute,
    base_weights: torch.Tensor,
    config: DataFreeFitnessConfig | None = None,
) -> float:
    """F(best single-client vertex) - F(the FedAvg point), exactly and in O(K^2).

    Positive means the data-free fitness scores "discard every client but one" above the
    aggregation FedACO exists to improve on -- the colony is then working correctly and
    optimizing toward a degenerate answer, which looks in the logs exactly like a colony
    that is working well (`fallback_used=0`, `best_fitness` comfortably above
    `fedavg_fitness`, a confident alpha).

    The vertex side needs no search, because F at a vertex has a closed form. Dispersion is
    `sum_k alpha_k ||delta_k - Delta(alpha)||^2`, a weighted variance about the weighted
    mean; at `alpha = e_j` the mean *is* `delta_j`, so every term is zero. Alignment
    collapses to `cos(delta_j, robust_mean)` and the concentration penalty to its maximum
    `log K`. So, for any Gram matrix whatsoever:

        F(e_j) = gamma_1 * cos(delta_j, robust_mean) - gamma_3 * log K

    That identity is what makes this cheap and exact rather than a search. The penalty is
    the only thing standing against a term that vanishes outright, and it grows only as
    `log K` -- so the smaller the federation, the weaker the guard. Measured on synthetic
    deltas, `gamma_entropy` has to exceed ~0.17 at K=2 and ~0.13 at K=4 to keep the corner
    from winning, against the project's default of 0.1 (docs/EXPERIMENT_LOG.md,
    2026-09-18). A K=2 or K=4 validation run therefore sits in the regime where this bites
    and the sweep's K=20 does not, which is the opposite of what a smoke test is for.

    A positive margin does not prove the corner is the global maximum -- an interior point
    may score higher still. It proves the weaker and more useful thing: that a degenerate
    answer outscores the reference the method is judged against.
    """
    config = config or DataFreeFitnessConfig()
    eps = 1e-12
    norms = torch.diagonal(gram.gram).clamp_min(eps).sqrt()
    cosines = gram.g_rob / (norms * math.sqrt(max(gram.rob_norm_sq, eps)))

    num_clients = int(base_weights.numel())
    best_vertex = config.gamma_alignment * float(cosines.max()) - config.gamma_entropy * math.log(
        num_clients
    )
    fedavg = DataFreeFitness(gram, config).evaluate(base_weights)
    return best_vertex - fedavg


class ServerValFitness:
    """Macro-F1 of w(alpha) on a small server-held val set (plan §4.4) -- an
    upper-bound reference that assumes the server holds data, unlike `data_free`.
    Orders of magnitude more expensive: materializes a full model per candidate and
    runs a forward pass, so callers must use a heavily reduced colony budget."""

    def __init__(
        self,
        model: torch.nn.Module,
        global_state: "OrderedDict[str, torch.Tensor]",
        deltas: torch.Tensor,
        shapes: list[tuple[str, torch.Size]],
        val_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.model = model
        self.global_state = global_state
        self.deltas = deltas
        self.shapes = shapes
        self.val_loader = val_loader
        self.device = device

    def evaluate(self, alpha: torch.Tensor) -> float:
        combined_delta = alpha @ self.deltas
        candidate_state = apply_delta(self.global_state, combined_delta, self.shapes)
        self.model.load_state_dict(candidate_state)
        self.model.to(self.device)
        metrics = evaluate_model(self.model, self.val_loader, self.device)
        return float(metrics["macro_f1"])


def _alpha_key(alpha: torch.Tensor, ndigits: int = 6) -> tuple[float, ...]:
    return tuple(round(float(x), ndigits) for x in alpha.tolist())


class ClientProbeFitness:
    """Realistic middle ground (plan §4.4): a fixed menu of candidate alphas is
    broadcast in round t+1's config, clients report local-val loss for each, and
    fitness is applied with a one-round delay. The broadcast/collect round-trip is a
    strategy-level, FL-runtime concern (belongs to `strategies/fedaco.py`, exercised
    in the Phase 7 A3 ablation) -- this class only turns already-collected reports into
    a scalar score, via `record_reports`. Calling `evaluate` on an alpha with no
    recorded reports yet raises rather than silently returning a meaningless value.
    """

    def __init__(self, num_examples: torch.Tensor) -> None:
        self.num_examples = num_examples
        self._reports: dict[tuple[float, ...], torch.Tensor] = {}

    def record_reports(self, alpha: torch.Tensor, per_client_loss: torch.Tensor) -> None:
        self._reports[_alpha_key(alpha)] = per_client_loss

    def evaluate(self, alpha: torch.Tensor) -> float:
        key = _alpha_key(alpha)
        if key not in self._reports:
            raise KeyError(
                "No client-reported losses for this alpha yet -- client_probe fitness "
                "needs a prior round's broadcast/report round-trip (the strategy layer's "
                "job, Phase 7's A3 ablation), not a fresh colony evaluation."
            )
        per_client_loss = self._reports[key]
        weights = self.num_examples / self.num_examples.sum().clamp_min(1e-12)
        return -float((weights * per_client_loss).sum())
