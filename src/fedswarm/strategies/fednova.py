"""FedNova (Wang et al., 2020, https://arxiv.org/abs/2007.07481): normalizes each
client's update by its own local-step count before averaging, then rescales the
combination by the (data-size-weighted) average step count -- removes the objective-
inconsistency bias plain FedAvg introduces when clients take different numbers of
local SGD steps (e.g. under label-skewed partitions with unequal local dataset sizes
but the same local-epochs config, or genuinely heterogeneous client compute).

    normalized_i = Delta_i / tau_i                          (per-step-average direction)
    p_i          = n_i / sum_j n_j                           (data-size weight)
    tau_eff      = sum_i p_i * tau_i                         (weighted-average step count)
    w_{t+1}      = w_t + tau_eff * sum_i p_i * normalized_i

`tau_i` is this project's own client's already-reported `num_batches` (the local
optimizer here is plain SGD, matching the paper's simplified single-learning-rate
case where this reduces exactly to the paper's Algorithm 1 update).
"""

from __future__ import annotations

from typing import Iterable

import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from fedswarm.aco.gram import apply_delta, flatten_state_dicts


class FedNova(FedAvg):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._current_arrays: ArrayRecord | None = None

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        self._current_arrays = arrays
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies or self._current_arrays is None:
            return None, None

        reply_contents = [msg.content for msg in valid_replies]
        client_arrays = [next(iter(rc.array_records.values())) for rc in reply_contents]
        client_metrics = [next(iter(rc.metric_records.values())) for rc in reply_contents]

        global_state = self._current_arrays.to_torch_state_dict()
        client_states = [ar.to_torch_state_dict() for ar in client_arrays]
        deltas, shapes = flatten_state_dicts(global_state, client_states)

        num_examples = torch.tensor([float(m["num-examples"]) for m in client_metrics])
        # num_batches is always >= 1 in practice (local_train runs at least one batch
        # per epoch); clamp defensively so a client reporting 0 can't divide by zero.
        tau = torch.tensor(
            [float(m.get("num_batches", 1.0)) for m in client_metrics]
        ).clamp_min(1.0)

        weights = num_examples / num_examples.sum()
        tau_eff = float((weights * tau).sum())
        normalized = deltas / tau.unsqueeze(1)
        combined = tau_eff * (weights @ normalized)

        arrays_out = ArrayRecord(apply_delta(global_state, combined, shapes))
        metrics_out = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)
        return arrays_out, metrics_out
