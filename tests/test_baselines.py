"""Phase 5 -- baseline strategies (strategies/factory.py, fednova.py, lossbased.py,
fedlaw.py) plus the server-side half of scaffold.py (its client-side half is tested in
tests/test_fl_app.py, since it needs Context/train_handler). Built-ins (FedProx,
FedAdam, FedYogi, Krum, FedTrimmedAvg, FedMedian) are exercised only through the
factory -- their own aggregation logic is Flower's, already covered by the installed
package's own test suite, not this project's to re-verify.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
import torch.nn as nn
from flwr.app import Array, ArrayRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAdam, FedAvg, FedMedian, FedProx, FedTrimmedAvg, FedYogi, Krum
from torch.utils.data import DataLoader, TensorDataset

from fedswarm.strategies.factory import strategy_from_run_config
from fedswarm.strategies.fedaco import FedACO
from fedswarm.strategies.fedlaw import FedLAW
from fedswarm.strategies.fednova import FedNova
from fedswarm.strategies.lossbased import LossBasedWeighting
from fedswarm.strategies.scaffold import Scaffold


def _state(dim: int = 20, seed: int = 0) -> "OrderedDict[str, torch.Tensor]":
    g = torch.Generator().manual_seed(seed)
    return OrderedDict(w=torch.randn(dim, generator=g))


def _make_reply(
    client_id: int,
    state: "OrderedDict[str, torch.Tensor]",
    num_examples: float,
    num_batches: int = 1,
    loss_before: float = 1.0,
    loss_after: float = 0.5,
    extra_arrays: "dict[str, torch.Tensor] | None" = None,
) -> Message:
    arrays = ArrayRecord(state)
    for name, tensor in (extra_arrays or {}).items():
        arrays[name] = Array(tensor.numpy())
    content = RecordDict(
        {
            "arrays": arrays,
            "metrics": MetricRecord(
                {
                    "client_id": client_id,
                    "num-examples": num_examples,
                    "num_batches": num_batches,
                    "train_loss_before": loss_before,
                    "train_loss_after": loss_after,
                }
            ),
        }
    )
    return Message(content=content, dst_node_id=client_id, message_type="train")


# ======================================================================================
# strategies/factory.py
# ======================================================================================


@pytest.mark.parametrize(
    "name,expected_cls",
    [
        ("fedavg", FedAvg),
        ("fedprox", FedProx),
        ("fedadam", FedAdam),
        ("fedyogi", FedYogi),
        ("krum", Krum),
        ("trimmed-mean", FedTrimmedAvg),
        ("median", FedMedian),
        ("fednova", FedNova),
        ("loss-based", LossBasedWeighting),
        ("scaffold", Scaffold),
        ("fedaco", FedACO),
    ],
)
def test_factory_selects_strategy_by_name(name: str, expected_cls: type) -> None:
    strategy = strategy_from_run_config({"strategy-name": name})
    assert isinstance(strategy, expected_cls)


def test_factory_defaults_to_fedavg_when_unset() -> None:
    assert isinstance(strategy_from_run_config({}), FedAvg)


def test_factory_rejects_unknown_strategy_name() -> None:
    with pytest.raises(ValueError):
        strategy_from_run_config({"strategy-name": "not-a-real-strategy"})


def test_factory_fedlaw_requires_model_val_loader_device() -> None:
    with pytest.raises(ValueError):
        strategy_from_run_config({"strategy-name": "fedlaw"})


# ======================================================================================
# strategies/fednova.py
# ======================================================================================


def test_fednova_matches_manual_normalized_averaging_formula() -> None:
    global_state = _state(seed=0)
    deltas = [torch.randn(20, generator=torch.Generator().manual_seed(s)) for s in (1, 2, 3)]
    client_states = [OrderedDict(w=global_state["w"] + d) for d in deltas]
    num_examples = [100.0, 50.0, 200.0]
    tau = [2, 4, 1]

    replies = [
        _make_reply(i, s, n, num_batches=t)
        for i, (s, n, t) in enumerate(zip(client_states, num_examples, tau))
    ]

    strategy = FedNova(min_train_nodes=2, min_evaluate_nodes=2)
    strategy._current_arrays = ArrayRecord(global_state)
    arrays_out, _ = strategy.aggregate_train(1, list(replies))
    assert arrays_out is not None

    weights = torch.tensor(num_examples) / sum(num_examples)
    tau_t = torch.tensor([float(t) for t in tau])
    tau_eff = float((weights * tau_t).sum())
    normalized = torch.stack([d / t for d, t in zip(deltas, tau_t)], dim=0)
    expected_delta = tau_eff * (weights @ normalized)
    expected = global_state["w"] + expected_delta

    got = arrays_out.to_torch_state_dict()["w"]
    assert torch.allclose(got, expected, atol=1e-5)


def test_fednova_equals_fedavg_when_all_clients_take_equal_steps() -> None:
    """tau_i all equal -> tau_eff = tau, normalized*tau_eff = raw delta -> reduces to
    plain weighted averaging, i.e. FedAvg."""
    global_state = _state(seed=0)
    deltas = [torch.randn(20, generator=torch.Generator().manual_seed(s)) for s in (1, 2, 3)]
    client_states = [OrderedDict(w=global_state["w"] + d) for d in deltas]
    num_examples = [100.0, 50.0, 200.0]
    replies = [
        _make_reply(i, s, n, num_batches=3) for i, (s, n) in enumerate(zip(client_states, num_examples))
    ]

    nova = FedNova(min_train_nodes=2, min_evaluate_nodes=2)
    nova._current_arrays = ArrayRecord(global_state)
    nova_out, _ = nova.aggregate_train(1, list(replies))

    fedavg_out, _ = FedAvg().aggregate_train(1, list(replies))

    got = nova_out.to_torch_state_dict()["w"]
    want = fedavg_out.to_torch_state_dict()["w"]
    assert torch.allclose(got, want, atol=1e-5)


# ======================================================================================
# strategies/lossbased.py
# ======================================================================================


def test_lossbased_weighting_favors_the_lower_loss_client() -> None:
    global_state = _state(seed=0)
    deltas = [torch.randn(20, generator=torch.Generator().manual_seed(s)) for s in (1, 2)]
    client_states = [OrderedDict(w=global_state["w"] + d) for d in deltas]
    # Equal data size -- FedAvg would weight them identically. Client 0 has much lower
    # post-training loss, so loss-based weighting must favor it.
    replies = [
        _make_reply(0, client_states[0], 100.0, loss_after=0.1),
        _make_reply(1, client_states[1], 100.0, loss_after=5.0),
    ]

    strategy = LossBasedWeighting(temperature=1.0)
    strategy._current_arrays = ArrayRecord(global_state)
    arrays_out, _ = strategy.aggregate_train(1, list(replies))
    assert arrays_out is not None

    got = arrays_out.to_torch_state_dict()["w"]
    # If client 0 dominates, the result should sit closer to global+delta[0] than to
    # the midpoint FedAvg (equal weights) would give.
    fedavg_mid = global_state["w"] + 0.5 * (deltas[0] + deltas[1])
    dist_to_client0_side = (got - (global_state["w"] + deltas[0])).norm()
    dist_to_fedavg_mid = (got - fedavg_mid).norm()
    assert dist_to_client0_side < dist_to_fedavg_mid


def test_lossbased_weighting_is_uniform_at_equal_losses() -> None:
    global_state = _state(seed=0)
    deltas = [torch.randn(20, generator=torch.Generator().manual_seed(s)) for s in (1, 2)]
    client_states = [OrderedDict(w=global_state["w"] + d) for d in deltas]
    replies = [
        _make_reply(0, client_states[0], 100.0, loss_after=1.0),
        _make_reply(1, client_states[1], 100.0, loss_after=1.0),
    ]

    strategy = LossBasedWeighting()
    strategy._current_arrays = ArrayRecord(global_state)
    arrays_out, _ = strategy.aggregate_train(1, list(replies))

    expected = global_state["w"] + 0.5 * (deltas[0] + deltas[1])
    got = arrays_out.to_torch_state_dict()["w"]
    assert torch.allclose(got, expected, atol=1e-5)


# ======================================================================================
# strategies/fedlaw.py
# ======================================================================================


class _Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_fedlaw_reduces_val_loss_relative_to_uniform_weighting() -> None:
    torch.manual_seed(0)
    model = _Linear()
    global_state = OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())

    # Client 0's delta genuinely improves val loss; client 1's is adversarial noise.
    images = torch.randn(16, 4)
    labels = torch.randint(0, 2, (16,))
    val_loader = DataLoader(TensorDataset(images, labels), batch_size=16)

    good_model = _Linear()
    good_model.load_state_dict(global_state)
    opt = torch.optim.SGD(good_model.parameters(), lr=0.5)
    for _ in range(5):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(good_model(images), labels)
        loss.backward()
        opt.step()
    good_state = OrderedDict((k, v.detach().clone()) for k, v in good_model.state_dict().items())
    bad_state = OrderedDict((k, v + 5.0 * torch.randn_like(v)) for k, v in global_state.items())

    replies = [
        _make_reply(0, good_state, 100.0),
        _make_reply(1, bad_state, 100.0),
    ]

    strategy = FedLAW(model=model, val_loader=val_loader, device=torch.device("cpu"), num_steps=30, lr=0.2)
    strategy._current_arrays = ArrayRecord(global_state)
    arrays_out, metrics_out = strategy.aggregate_train(1, list(replies))
    assert arrays_out is not None and metrics_out is not None

    alpha = metrics_out["fedlaw_alpha"]
    assert alpha[0] > alpha[1], alpha  # learned weights should favor the genuinely-good client

    eval_model = _Linear()
    eval_model.load_state_dict(arrays_out.to_torch_state_dict())
    fedlaw_loss = float(nn.functional.cross_entropy(eval_model(images), labels))

    uniform_state = OrderedDict(
        (k, global_state[k] + 0.5 * (good_state[k] - global_state[k]) + 0.5 * (bad_state[k] - global_state[k]))
        for k in global_state
    )
    eval_model.load_state_dict(uniform_state)
    uniform_loss = float(nn.functional.cross_entropy(eval_model(images), labels))

    assert fedlaw_loss < uniform_loss


# ======================================================================================
# strategies/scaffold.py -- server-side half only (client side: tests/test_fl_app.py)
# ======================================================================================


def test_scaffold_aggregate_train_consumes_dc_and_updates_global_c() -> None:
    global_state = _state(seed=0)
    deltas = [torch.randn(20, generator=torch.Generator().manual_seed(s)) for s in (1, 2)]
    client_states = [OrderedDict(w=global_state["w"] + d) for d in deltas]
    dcs = [torch.full((20,), 0.1), torch.full((20,), -0.1)]

    replies = [
        _make_reply(0, client_states[0], 100.0, extra_arrays={"scaffold_dc/w": dcs[0]}),
        _make_reply(1, client_states[1], 100.0, extra_arrays={"scaffold_dc/w": dcs[1]}),
    ]

    strategy = Scaffold(min_train_nodes=2, min_evaluate_nodes=2, server_lr=1.0)
    strategy._global_c = OrderedDict(w=torch.zeros(20))
    strategy._num_total_nodes = 2

    arrays_out, metrics_out = strategy.aggregate_train(1, list(replies))
    assert arrays_out is not None and metrics_out is not None

    expected_model = global_state["w"] + 0.5 * (deltas[0] + deltas[1])
    assert torch.allclose(arrays_out.to_torch_state_dict()["w"], expected_model, atol=1e-5)

    # scale = len(replies)/num_total_nodes = 2/2 = 1.0
    expected_c = 0.5 * (dcs[0] + dcs[1])
    assert torch.allclose(strategy._global_c["w"], expected_c, atol=1e-5)


def test_scaffold_configure_train_merges_global_c_into_outgoing_arrays() -> None:
    class _FakeGrid:
        def get_node_ids(self) -> list[int]:
            return [0, 1]

    from flwr.app import ConfigRecord

    strategy = Scaffold(min_train_nodes=2, min_evaluate_nodes=2, min_available_nodes=2)
    arrays = ArrayRecord(_state(seed=0))
    messages = list(strategy.configure_train(1, arrays, ConfigRecord(), _FakeGrid()))

    assert len(messages) == 2
    sent_keys = set(messages[0].content["arrays"].keys())
    assert "w" in sent_keys
    assert "scaffold_c/w" in sent_keys
