"""Loss-based weight-selection baseline (plan §5: "loss-based weighting (FedNolowe-
style normalized-loss weights)").

⚠️ "FedNolowe" is the plan's own descriptive name for this class of baseline, not a
paper title this project verified a canonical formula against -- no such reference
was found to check numbers or a formula against (per CLAUDE.md: never fabricate an
API/formula attribution; record what can't be verified). What is implemented here is
a documented, reasonable interpretation of "weight clients by their own local loss,"
not a reproduction of a specific paper's reported results: alpha_i ~ softmax(-loss_i /
T) -- a client whose local model fits its own data better (lower post-training loss)
gets more say, tempered by T (T -> inf recovers uniform/FedAvg-shape weighting purely
from softmax's own limit, not from n_i; T -> 0 approaches winner-take-all on the
single lowest-loss client). See docs/OPEN_QUESTIONS.md.
"""

from __future__ import annotations

from typing import Iterable

import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from fedswarm.aco.gram import apply_delta, flatten_state_dicts


class LossBasedWeighting(FedAvg):
    def __init__(self, *args, temperature: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.temperature = temperature
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

        loss = torch.tensor(
            [float(m.get("train_loss_after", 0.0)) for m in client_metrics]
        )
        alpha = torch.softmax(-loss / max(self.temperature, 1e-6), dim=0)

        combined = alpha @ deltas
        arrays_out = ArrayRecord(apply_delta(global_state, combined, shapes))
        metrics_out = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)
        return arrays_out, metrics_out
