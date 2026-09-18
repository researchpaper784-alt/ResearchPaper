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


class EvaluationBudgetExceeded(RuntimeError):
    pass


class BudgetedFitness:
    """Wraps any `Fitness`, hard-capping the number of `.evaluate()` calls it will
    ever perform. Plan §7, A1: "implement the budget cap inside `Fitness` itself so
    no control can cheat by accident" -- the real colony (`aco/colony.py`) and every
    equal-budget control (`aco/controls.py`) share this wrapper for exactly that
    reason. Raising past budget (rather than, say, silently returning the last
    value) is the actual safety net; well-behaved callers should check `remaining()`
    and stop cleanly before ever triggering it."""

    def __init__(self, fitness: Fitness, budget: int) -> None:
        self._fitness = fitness
        self.budget = budget
        self.calls_used = 0

    def remaining(self) -> int:
        return self.budget - self.calls_used

    def evaluate(self, alpha: torch.Tensor) -> float:
        if self.calls_used >= self.budget:
            raise EvaluationBudgetExceeded(
                f"budget of {self.budget} evaluations exhausted"
            )
        self.calls_used += 1
        return self._fitness.evaluate(alpha)


def _entropy(alpha: torch.Tensor, eps: float = 1e-12) -> float:
    total = alpha.sum().clamp_min(eps)
    probs = (alpha / total).clamp_min(eps)
    return float(-(probs * probs.log()).sum())


@dataclass
class DataFreeFitnessConfig:
    gamma_alignment: float = 1.0  # gamma_1
    gamma_dispersion: float = 1.0  # gamma_2
    gamma_entropy: float = 0.1  # gamma_3


class DataFreeFitness:
    """F(alpha) from plan §4.5, evaluated in O(K) via the Gram trick (§4.6) -- the
    default and the method as proposed. No client data or extra round-trip needed
    beyond what FedAvg already collects."""

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
        concentration_penalty = math.log(alpha.numel()) - _entropy(alpha)

        f = (
            self.config.gamma_alignment * alignment
            - self.config.gamma_dispersion * dispersion
            - self.config.gamma_entropy * concentration_penalty
        )
        return float(f)


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
