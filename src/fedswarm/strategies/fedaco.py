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
from typing import Iterable

import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from torch.utils.data import DataLoader

from fedswarm.aco.colony import ColonyConfig, run_colony
from fedswarm.aco.fitness import DataFreeFitness, DataFreeFitnessConfig, ServerValFitness
from fedswarm.aco.gram import apply_delta, flatten_state_dicts, precompute_gram
from fedswarm.aco.heuristics import HeuristicWeights, desirability_matrix, desirability_scores
from fedswarm.aco.pheromone import Pheromone, PheromoneConfig
from fedswarm.aco.schedules import colony_budget, level_set


@dataclass
class FedACOConfig:
    num_levels: int = 11
    level_low: float = 0.0
    level_high: float = 2.5
    # Global shrinkage s (plan §4.1) is exposed as a fixed config value, not searched
    # per-ant as an extra decision variable -- a deliberate scope reduction to keep the
    # colony's construction graph exactly the K-station one §4.2 describes. Ablating it
    # (sweeping this value, not searching it) is tracked as Phase 7 work; see
    # docs/OPEN_QUESTIONS.md.
    target_sum: float = 1.0
    num_rounds: int = 1  # must match Strategy.start()'s num_rounds for the budget decay (§4.7) to track real progress
    ants_start: int = 30
    ants_end: int = 10
    iters_start: int = 10
    iters_end: int = 4
    trim_fraction: float = 0.2
    safety_fallback: bool = True
    # Seeds the colony's own torch.Generator, derived per round as
    # f(seed, server_round) -- see `aggregate_train`. Set from the run's `seed` so the
    # colony's stochastic exploration is reproducible under the same determinism
    # contract as the rest of the codebase (CLAUDE.md), rather than riding on whatever
    # state the ServerApp process happens to have left in global torch RNG.
    seed: int = 0
    # "data_free" (the method as proposed, the only mode wired into the normal round
    # loop) or "server_val" (upper-bound reference; requires model/val_loader/device on
    # the strategy and a heavily reduced colony budget). "client_probe" is deliberately
    # not selectable here: `ClientProbeFitness` implements the aggregation side only,
    # and its broadcast/collect round-trip is Phase 7 work -- see docs/OPEN_QUESTIONS.md.
    fitness_mode: str = "data_free"
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
        val_loader: "DataLoader | None" = None,
        device: torch.device | None = None,
        **kwargs,
    ) -> None:
        """`model`/`val_loader`/`device` are only needed for
        `aco_config.fitness_mode="server_val"` (same pattern as FedLAW, which needs the
        same three) -- the default `data_free` mode ignores them entirely, which is the
        whole point of the method as proposed."""
        super().__init__(*args, **kwargs)
        self.aco_config = aco_config or FedACOConfig()
        if self.aco_config.fitness_mode not in ("data_free", "server_val"):
            raise ValueError(
                f"Unknown fitness_mode {self.aco_config.fitness_mode!r} "
                "(expected 'data_free' or 'server_val'; 'client_probe' needs the "
                "broadcast/collect round-trip that is Phase 7 work)"
            )
        if self.aco_config.fitness_mode == "server_val" and (
            model is None or val_loader is None or device is None
        ):
            raise ValueError(
                "fitness_mode='server_val' requires model, val_loader, and device"
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
        # Set by fl/app.py when resuming. `Strategy.start()` renumbers its rounds from 1
        # on every invocation -- the same reason `build_evaluate_fn` needs an offset -- so
        # a run resumed after round 50 would otherwise re-seed the colony with round 1's
        # draws and restart the ant/iteration decay from its start-of-run budget. A
        # resumed cell would not be the same experiment as an uninterrupted one, which
        # breaks the determinism contract in CLAUDE.md.
        self.round_offset = 0

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
            # True round, not a process-local counter: on a resumed run `round_count`
            # restarts at 0 while `num_rounds` is still the total, so the budget decay
            # would replay from the start-of-run ant count.
            self.round_count + int(self.round_offset),
            self.aco_config.num_rounds,
            self.aco_config.ants_start,
            self.aco_config.ants_end,
            self.aco_config.iters_start,
            self.aco_config.iters_end,
        )
        base_weights = num_examples / num_examples.sum()
        if self.aco_config.fitness_mode == "server_val":
            assert self.model is not None and self.val_loader is not None and self.device is not None
            fitness = ServerValFitness(
                self.model, global_state, deltas, shapes, self.val_loader, self.device
            )
        else:
            fitness = DataFreeFitness(gram, self.aco_config.fitness)

        # Derived per round rather than once per run, so a resumed run (fl/app.py
        # checkpoints every round) reproduces the same colony trajectory it would have
        # taken uninterrupted -- `server_round` is authoritative for that, not a
        # process-local counter.
        true_round = int(server_round) + int(self.round_offset)
        generator = torch.Generator().manual_seed(
            (int(self.aco_config.seed) & 0xFFFF_FFFF) * 1_000_003 + true_round
        )

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
            generator=generator,
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
        # Flower stores exactly what aggregate_train returns as the round's client
        # metrics, so a hand-built record silently drops everything the clients reported:
        # `is_malicious` (Phase 8 -- for the strategy under test in the robustness
        # sweep), `update_norm`, `train_loss_before/after`, `num-examples`. Every baseline
        # carries those and FedACO would not, which is exactly backwards. Folded in first
        # so the ACO-specific keys below still win on any name collision.
        aggregated = _weighted_client_metrics(client_metrics, num_examples)
        metrics_out = MetricRecord(
            {
                **aggregated,
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
                # trace(G)/K, the dispersion scale. Logged per round because a run whose
                # client updates are far larger than the fitness terms assume is the
                # failure mode the normalization in `DataFreeFitness` exists to prevent
                # -- and `fallback_used` cannot detect it (both sides of that comparison
                # use the same fitness). Pair it with `pheromone_entropy`: entropy pinned
                # at log(num_levels) means tau stayed uniform and the colony degenerated
                # to heuristic-greedy selection.
                "delta_mean_sq_norm": gram.mean_sq_norm,
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


def _weighted_client_metrics(
    client_metrics: list, num_examples: torch.Tensor
) -> dict[str, float]:
    """Data-size-weighted mean of every scalar the clients reported.

    Matches how Flower's own FedAvg aggregates client metrics, so FedACO's rows carry the
    same fields as every baseline's and a result file can be read the same way regardless
    of strategy.
    """
    total = float(num_examples.sum()) or 1.0
    weights = [float(n) / total for n in num_examples]
    out: dict[str, float] = {}
    keys = {k for record in client_metrics for k in dict(record)}
    for key in sorted(keys):
        values = []
        for weight, record in zip(weights, client_metrics):
            value = dict(record).get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(weight * float(value))
        if len(values) == len(client_metrics):
            out[key] = sum(values)
    return out


def _entropy(alpha: torch.Tensor, eps: float = 1e-12) -> float:
    total = alpha.sum().clamp_min(eps)
    probs = (alpha / total).clamp_min(eps)
    return float(-(probs * probs.log()).sum())
