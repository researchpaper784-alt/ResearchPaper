"""FedLAW-style learnable aggregation weights (plan §5; Li et al., "Revisiting
Weighted Aggregation in Federated Learning with Neural Networks", ICML 2023,
arxiv.org/abs/2302.10911).

⚠️ This reproduces the *idea* the plan asks for -- weights learned by gradient
descent against a server-held validation set, each round, from scratch -- not the
paper's exact learnable-global-scaling-factor formulation or its reported numbers;
no page/table from that paper was checked against this implementation. See
docs/OPEN_QUESTIONS.md.

alpha = softmax(theta) (theta unconstrained and learnable; softmax keeps alpha on the
simplex automatically, no manual projection needed). A handful of gradient steps
minimize the server val cross-entropy of w(alpha) = w_t + alpha @ deltas, using
`torch.func.functional_call` to run the model's forward pass differentiably against
candidate parameters without ever calling `load_state_dict` (which would detach the
graph) or materializing a second `nn.Module`.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from torch.func import functional_call
from torch.utils.data import DataLoader

from fedswarm.aco.gram import apply_delta, flatten_state_dicts


class FedLAW(FedAvg):
    def __init__(
        self,
        *args,
        model: torch.nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        num_steps: int = 20,
        lr: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.num_steps = num_steps
        self.lr = lr
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

        global_state = self._current_arrays.to_torch_state_dict()
        client_states = [ar.to_torch_state_dict() for ar in client_arrays]
        deltas, shapes = flatten_state_dicts(global_state, client_states)
        num_clients = deltas.shape[0]

        self.model.to(self.device)
        deltas_dev = deltas.to(self.device)
        global_state_dev = {k: v.to(self.device) for k, v in global_state.items()}
        images, labels = next(iter(self.val_loader))
        images, labels = images.to(self.device), labels.to(self.device)

        theta = torch.zeros(num_clients, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([theta], lr=self.lr)

        for _ in range(self.num_steps):
            optimizer.zero_grad()
            alpha = torch.softmax(theta, dim=0)
            combined = alpha @ deltas_dev
            candidate_state = apply_delta(global_state_dev, combined, shapes)
            logits = functional_call(self.model, candidate_state, (images,))
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

        alpha_final = torch.softmax(theta, dim=0).detach().cpu()
        combined = alpha_final @ deltas
        arrays_out = ArrayRecord(apply_delta(global_state, combined, shapes))

        metrics_out = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)
        metrics_out["fedlaw_alpha"] = alpha_final.tolist()
        metrics_out["fedlaw_final_val_loss"] = float(loss.detach())
        return arrays_out, metrics_out
