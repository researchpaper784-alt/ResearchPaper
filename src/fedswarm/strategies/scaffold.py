"""SCAFFOLD (Karimireddy et al., 2020, arxiv.org/abs/1910.06378): each client keeps a
persistent local control variate c_i across rounds and corrects its local gradient by
`c - c_i` (global minus local control variate) every local step, which is what
actually fixes FedAvg's client-drift problem under non-IID data -- unlike every other
baseline in this module, this needs genuine client-side protocol support, not just a
different server-side aggregation rule.

Two real, verified mechanisms this leans on (checked against the installed package's
own source, not assumed):

- `Context.state` is a real, durable per-node `RecordDict` -- confirmed via
  `flwr/supernode/start_client_internal.py`'s `context.state = existing_context.state`
  and the equivalent line in `flwr/server/superlink/linkstate/linkstate.py`, both
  restoring a node's previous state before a fresh handler invocation. This is where
  `fl/app.py`'s SCAFFOLD branch of `train_handler` persists `c_i`.
- `FedAvg`'s own `_construct_messages`/`_check_and_log_replies` are the same
  reusable, private-but-inheritable extension points `strategies/fedaco.py` already
  relies on (`docs/FLOWER_API_NOTES.md`) -- used here to inject the global control
  variate into the outgoing message without needing a third top-level `RecordDict`
  key FedAvg's own `configure_train` doesn't expose a way to add.

The global control variate `c` and each client's control-variate delta `dc_i` are
model-shaped tensors, so both travel inside the *same* ArrayRecord as the model
weights, under prefixed keys (`scaffold_c/<name>` outgoing, `scaffold_dc/<name>`
incoming) -- `ArrayRecord` is a flat named-tensor container, so this needs no schema
beyond the prefix convention itself, which the client half of this contract
(`fl/app.py::train_handler`) must match exactly.

**Documented deviation from the paper**: the model update itself is data-size-weighted
(`weighted_by_key`, matching every other baseline in this module for a fair
comparison), not the paper's uniform `1/|S_t|` average -- see docs/OPEN_QUESTIONS.md.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import torch
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import sample_nodes

CONTROL_PREFIX = "scaffold_c/"
DELTA_PREFIX = "scaffold_dc/"


class Scaffold(FedAvg):
    def __init__(self, *args, server_lr: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.server_lr = server_lr
        self._global_c: "OrderedDict[str, torch.Tensor] | None" = None
        self._num_total_nodes = 1

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        if self.fraction_train == 0.0:
            return []
        if self._global_c is None:
            self._global_c = OrderedDict(
                (name, torch.zeros_like(tensor))
                for name, tensor in arrays.to_torch_state_dict().items()
            )

        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
        node_ids, all_nodes = sample_nodes(grid, self.min_available_nodes, sample_size)
        self._num_total_nodes = max(len(all_nodes), 1)

        # Underscore. Spelled `server-round` this was a write nothing ever read --
        # `train_handler` looks for `server_round`. Redundant now that the factory sets it
        # for every strategy, and kept only so this class is correct when constructed
        # directly (its own tests do).
        config["server_round"] = server_round
        merged = ArrayRecord(arrays.to_torch_state_dict())
        for name, tensor in self._global_c.items():
            merged[f"{CONTROL_PREFIX}{name}"] = Array(tensor.numpy())

        record = RecordDict({self.arrayrecord_key: merged, self.configrecord_key: config})
        return self._construct_messages(record, node_ids, "train")

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        # validate=False: the standard consistency check expects the ArrayRecord to
        # carry only the model's own keys, but SCAFFOLD's replies also carry the
        # scaffold_dc/* keys merged into the same record.
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True, validate=False)
        if not valid_replies or self._global_c is None:
            return None, None

        reply_contents = [msg.content for msg in valid_replies]
        client_records = [next(iter(rc.array_records.values())) for rc in reply_contents]
        client_metrics = [next(iter(rc.metric_records.values())) for rc in reply_contents]

        param_names = list(self._global_c.keys())
        client_states = []
        client_dcs = []
        for record in client_records:
            state = record.to_torch_state_dict()
            client_states.append(OrderedDict((name, state[name]) for name in param_names))
            client_dcs.append(
                OrderedDict((name, state[f"{DELTA_PREFIX}{name}"]) for name in param_names)
            )

        num_examples = torch.tensor([float(m["num-examples"]) for m in client_metrics])
        weights = num_examples / num_examples.sum()

        new_state = OrderedDict()
        for name in param_names:
            stacked = torch.stack([s[name] for s in client_states], dim=0)
            broadcast_weights = weights.view(-1, *([1] * (stacked.dim() - 1)))
            new_state[name] = (broadcast_weights * stacked).sum(dim=0)

        scale = len(valid_replies) / self._num_total_nodes
        for name in param_names:
            stacked_dc = torch.stack([dc[name] for dc in client_dcs], dim=0)
            self._global_c[name] = self._global_c[name] + self.server_lr * scale * stacked_dc.mean(dim=0)

        arrays_out = ArrayRecord(new_state)
        metrics_out = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)
        return arrays_out, metrics_out
