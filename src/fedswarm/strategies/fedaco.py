"""FedACO(Strategy) -- plan §4.8. A thin adapter: all ACO logic lives in `fedswarm.aco`
(pure tensor math, no Flower types) and unit-tests independently of any Flower runtime;
this file only translates between that and Flower's Message/ArrayRecord/MetricRecord
types, plus the safety fallback that compares against plain FedAvg every round.

Subclasses `FedAvg`, not `Strategy` directly, per the plan's own guidance to reuse a
clean override point if one exists -- confirmed one does (`docs/FLOWER_API_NOTES.md`):
`configure_train`/`configure_evaluate`/`aggregate_evaluate`/`summary` are unchanged
FedAvg behavior (node sampling, message construction), and only `aggregate_train` needs
genuinely different logic. `configure_train` is overridden only to snapshot the global
`ArrayRecord` `aggregate_train` needs to compute per-client deltas against -- `Strategy.
aggregate_train`'s real signature (verified against the installed package, not assumed)
takes `(server_round, replies)` only, with no `arrays` parameter, so the global model at
the start of the round has to be captured somewhere else.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Literal

import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from torch.utils.data import DataLoader

from fedswarm.aco.colony import ColonyConfig, run_colony
from fedswarm.aco.fitness import DataFreeFitness, DataFreeFitnessConfig, Fitness, ServerValFitness
from fedswarm.aco.gram import apply_delta, flatten_state_dicts, precompute_gram
from fedswarm.aco.heuristics import HeuristicWeights, desirability_matrix, desirability_scores
from fedswarm.aco.pheromone import Pheromone, PheromoneConfig
from fedswarm.aco.schedules import colony_budget, level_set

FitnessMode = Literal["data_free", "server_val", "client_probe"]


@dataclass
class FedACOConfig:
    num_levels: int = 11
    level_low: float = 0.0
    level_high: float = 2.5
    # Global shrinkage s (plan §4.1) is exposed as a fixed config value, not searched
    # per-ant as an extra decision variable -- a deliberate scope reduction to keep the
    # colony's construction graph exactly the K-station one §4.2 describes. Ablating it
    # (sweeping this value, not searching it) is Phase 7's A7; see docs/OPEN_QUESTIONS.md.
    target_sum: float = 1.0
    num_rounds: int = 1  # must match Strategy.start()'s num_rounds for the budget decay (§4.7) to track real progress
    ants_start: int = 30
    ants_end: int = 10
    iters_start: int = 10
    iters_end: int = 4
    trim_fraction: float = 0.2
    safety_fallback: bool = True
    # A3 ablation axis (plan §4.4). "server_val" needs `model`/`val_loader`/`device`
    # passed to FedACO's constructor; "client_probe" is not wired for live execution
    # yet (docs/OPEN_QUESTIONS.md's Phase 4 entry) -- selecting it raises, not silently
    # falls back to data_free.
    fitness_mode: FitnessMode = "data_free"
    colony: ColonyConfig = field(default_factory=ColonyConfig)
    pheromone: PheromoneConfig = field(default_factory=PheromoneConfig)
    fitness: DataFreeFitnessConfig = field(default_factory=DataFreeFitnessConfig)
    heuristics: HeuristicWeights = field(default_factory=HeuristicWeights)


class FedACO(FedAvg):
    def __init__(
        self,
        *args,
        aco_config: FedACOConfig | None = None,
        model: torch.nn.Module | None = None,
        val_loader: DataLoader | None = None,
        device: torch.device | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.aco_config = aco_config or FedACOConfig()
        if self.aco_config.fitness_mode == "server_val" and (
            model is None or val_loader is None or device is None
        ):
            raise ValueError(
                "fitness_mode='server_val' requires model, val_loader, and device"
            )
        if self.aco_config.fitness_mode == "client_probe":
            raise NotImplementedError(
                "fitness_mode='client_probe' needs the broadcast/collect round-trip "
                "documented as not-yet-wired in docs/OPEN_QUESTIONS.md's Phase 4 "
                "entry -- ClientProbeFitness only implements report-aggregation, "
                "not live execution through this strategy."
            )
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.levels = level_set(
            self.aco_config.num_levels, self.aco_config.level_low, self.aco_config.level_high
        )
        self.pheromone = Pheromone(self.aco_config.num_levels, self.aco_config.pheromone)
        self._current_arrays: ArrayRecord | None = None
        self.fallback_count = 0
        self.round_count = 0

    def _build_fitness(
        self,
        gram,
        global_state: dict,
        deltas: torch.Tensor,
        shapes: list[tuple[str, torch.Size]],
    ) -> Fitness:
        if self.aco_config.fitness_mode == "server_val":
            assert self.model is not None and self.val_loader is not None and self.device is not None
            return ServerValFitness(self.model, global_state, deltas, shapes, self.val_loader, self.device)
        return DataFreeFitness(gram, self.aco_config.fitness)

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        self._current_arrays = arrays
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        t_start = time.perf_counter()
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies or self._current_arrays is None:
            return None, None

        reply_contents = [msg.content for msg in valid_replies]
        client_arrays = [next(iter(rc.array_records.values())) for rc in reply_contents]
        client_metrics = [next(iter(rc.metric_records.values())) for rc in reply_contents]

        try:
            client_ids = [str(int(m["client_id"])) for m in client_metrics]
        except KeyError as exc:
            raise KeyError(
                "FedACO requires every client reply's MetricRecord to include "
                "'client_id' -- pheromone persistence (plan §4.3) is keyed by client "
                "identity, and there is no safe fallback (e.g. reply position) that "
                "wouldn't silently break persistence across rounds with a changing "
                "participation set."
            ) from exc

        num_examples = torch.tensor([float(m["num-examples"]) for m in client_metrics])
        val_improvement = torch.tensor(
            [
                float(m.get("train_loss_before", 0.0)) - float(m.get("train_loss_after", 0.0))
                for m in client_metrics
            ]
        )

        global_state = self._current_arrays.to_torch_state_dict()
        client_states = [ar.to_torch_state_dict() for ar in client_arrays]
        deltas, shapes = flatten_state_dicts(global_state, client_states)

        gram_start = time.perf_counter()
        gram = precompute_gram(deltas, trim_fraction=self.aco_config.trim_fraction)
        gram_time_ms = (time.perf_counter() - gram_start) * 1000.0

        d_k = desirability_scores(gram, num_examples, val_improvement, self.aco_config.heuristics)
        eta = desirability_matrix(d_k, self.levels)

        tau0 = self.pheromone.begin_round(client_ids)
        num_ants, num_iterations = colony_budget(
            self.round_count,
            self.aco_config.num_rounds,
            self.aco_config.ants_start,
            self.aco_config.ants_end,
            self.aco_config.iters_start,
            self.aco_config.iters_end,
        )
        base_weights = num_examples / num_examples.sum()
        fitness = self._build_fitness(gram, global_state, deltas, shapes)

        colony_result = run_colony(
            tau0,
            eta,
            self.levels,
            base_weights,
            fitness.evaluate,
            num_ants,
            num_iterations,
            self.aco_config.colony,
            target_sum=self.aco_config.target_sum,
        )

        fedavg_alpha = base_weights * self.aco_config.target_sum
        fedavg_fitness = fitness.evaluate(fedavg_alpha)

        fallback_used = self.aco_config.safety_fallback and colony_result.best_fitness <= fedavg_fitness
        if fallback_used:
            self.fallback_count += 1
            alpha_final = fedavg_alpha
        else:
            alpha_final = colony_result.alpha_best

        self.pheromone.end_round(client_ids, colony_result.tau_final)
        self.round_count += 1

        combined_delta = alpha_final @ deltas
        new_state = apply_delta(global_state, combined_delta, shapes)
        arrays_out = ArrayRecord(new_state)

        aco_time_ms = (time.perf_counter() - t_start) * 1000.0
        metrics_out = MetricRecord(
            {
                "alpha_entropy": _entropy(alpha_final),
                "alpha_max": float(alpha_final.max()),
                # MetricRecord's real value type is int | float | list[int] | list[float]
                # and explicitly rejects bool (verified: flwr/app/message/metricrecord.py's
                # is_valid() checks `isinstance(v, bool)` even though bool subclasses int in
                # Python) -- 0/1, not True/False.
                "fallback_used": int(fallback_used),
                "fallback_count": self.fallback_count,
                "aco_time_ms": aco_time_ms,
                "gram_time_ms": gram_time_ms,
                "best_fitness": colony_result.best_fitness,
                "fedavg_fitness": fedavg_fitness,
                "pheromone_entropy": self.pheromone.entropy(client_ids),
                "realized_ants": colony_result.realized_ants,
                "realized_iterations": colony_result.realized_iterations,
                "alpha": alpha_final.tolist(),
                # MetricRecord list values are int | float only (verified: no list[str]
                # support) -- client_ids are numeric partition ids under the hood, so
                # int(...) round-trips exactly; the str keys stay internal to Pheromone.
                "participating_client_ids": [int(c) for c in client_ids],
            }
        )
        return arrays_out, metrics_out


def _entropy(alpha: torch.Tensor, eps: float = 1e-12) -> float:
    total = alpha.sum().clamp_min(eps)
    probs = (alpha / total).clamp_min(eps)
    return float(-(probs * probs.log()).sum())
