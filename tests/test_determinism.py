"""Phase 0.3 acceptance: two runs of a short training loop with the same seed must
produce identical (or near-identical, on non-deterministic ops) loss sequences."""

from __future__ import annotations

from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from fedswarm.models.simple_cnn import SimpleCNN
from fedswarm.utils.seed import seed_everything


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def _run_short_training_loop(seed: int, steps: int = 20) -> list[float]:
    deterministic = seed_everything(seed, deterministic=True)
    model = _TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    losses = []
    for _ in range(steps):
        x = torch.randn(8, 16)
        y = torch.randint(0, 4, (8,))
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses, deterministic


def test_same_seed_reproduces_loss_sequence() -> None:
    losses_a, det_a = _run_short_training_loop(seed=0)
    losses_b, det_b = _run_short_training_loop(seed=0)

    assert det_a == det_b
    if det_a:
        assert losses_a == losses_b
    else:
        assert all(abs(a - b) < 1e-5 for a, b in zip(losses_a, losses_b))


def test_different_seed_diverges() -> None:
    losses_a, _ = _run_short_training_loop(seed=0)
    losses_b, _ = _run_short_training_loop(seed=1)

    assert losses_a != losses_b


def test_seed_everything_enables_determinism_with_warn_only() -> None:
    """The actual bug this guards: the first Colab run crashed mid-epoch because
    determinism was enabled *without* warn_only=True, so AdaptiveAvgPool2d's backward
    (which SimpleCNN uses, and which historically lacks a deterministic CUDA kernel)
    raised instead of falling back. Never regress to a bare `True` call."""
    with patch("torch.use_deterministic_algorithms") as mock_enable:
        result = seed_everything(0, deterministic=True)

    assert result is True
    mock_enable.assert_called_once_with(True, warn_only=True)


def test_seed_everything_falls_back_on_old_torch_without_warn_only() -> None:
    """A torch old enough to not accept warn_only as a kwarg must not crash seeding --
    it should retry without the argument rather than propagate the TypeError."""
    with patch("torch.use_deterministic_algorithms") as mock_enable:
        mock_enable.side_effect = [TypeError("unexpected keyword argument 'warn_only'"), None]
        result = seed_everything(0, deterministic=True)

    assert result is True
    assert mock_enable.call_count == 2
    mock_enable.assert_any_call(True)


def test_seed_everything_returns_false_rather_than_raising_when_unsupported() -> None:
    with patch("torch.use_deterministic_algorithms", side_effect=RuntimeError("nope")):
        result = seed_everything(0, deterministic=True)

    assert result is False


def test_simple_cnn_trains_end_to_end_under_deterministic_mode() -> None:
    """SimpleCNN uses AdaptiveAvgPool2d -- exactly the layer whose backward pass
    historically lacks a deterministic CUDA kernel. This exercises the real model and
    real forward+backward path (not a mock) to confirm the combination that broke
    production doesn't crash locally either."""
    seed_everything(0, deterministic=True)
    model = SimpleCNN(num_classes=4, norm="groupnorm")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 4, (4,))

    optimizer.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


# ======================================================================================
# The client side of the seed contract -- unmet until 2026-09-19
# ======================================================================================


def test_the_client_handlers_seed_themselves() -> None:
    """`seed_everything` was called in `server_app.main()` and nowhere else.

    In the Simulation Runtime a ClientApp is a separate Ray actor process, so the server's
    seeding never reached it and every client's torch RNG started from OS entropy. The train
    loader shuffles with `shuffle=True` and no generator, which draws from exactly that RNG.

    The visible consequence, from two Kaggle gate runs at identical config and seed: mean
    corner margin 0.6252 vs 0.6184, the deposit floor engaging in 1 round vs 4, final
    macro-F1 0.1185 vs 0.1381. The `seed` recorded in every result file was honest about
    being recorded and wrong about what it controlled -- which makes it the worst kind of
    provenance, the kind that is checked and passes.
    """
    import inspect

    from fedswarm.fl import app

    for name in ("train_handler", "evaluate_handler"):
        source = inspect.getsource(getattr(app, name))
        assert "seed_everything(" in source, (
            f"{name} does not seed its own process; the server's seed does not reach a Ray "
            "actor, so this run is not reproducible"
        )


def test_the_client_seed_separates_clients_and_rounds() -> None:
    """Two properties, and missing either one is its own bug.

    Not varying by client would train every client on the same batch order. Not varying by
    round would train all 100 rounds on one fixed permutation -- which is not determinism,
    it is a much quieter defect that would still reproduce perfectly run to run.
    """
    from flwr.app import ConfigRecord

    from fedswarm.fl.app import _client_seed

    class _Ctx:
        run_config = {"seed": 0}

    r1 = ConfigRecord({"server_round": 1})
    r2 = ConfigRecord({"server_round": 2})

    assert _client_seed(_Ctx(), r1, 0) != _client_seed(_Ctx(), r1, 1), "clients collide"
    assert _client_seed(_Ctx(), r1, 0) != _client_seed(_Ctx(), r2, 0), "rounds collide"
    assert _client_seed(_Ctx(), r1, 0) == _client_seed(_Ctx(), r1, 0), "not a function"

    class _Ctx2:
        run_config = {"seed": 1}

    assert _client_seed(_Ctx(), r1, 0) != _client_seed(_Ctx2(), r1, 0), "run seed ignored"


def test_the_same_client_seed_gives_the_same_batch_order(tmp_path) -> None:
    """The property the seed is *for*: two loaders built with one seed iterate identically,
    and with different seeds they do not. Asserts against the real `DataLoader`, not against
    the presence of a `generator=` argument."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    def order(seed: int | None) -> list[int]:
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        dataset = TensorDataset(torch.arange(40).float().unsqueeze(1))
        loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=generator)
        return [int(x) for batch, in loader for x in batch.flatten()]

    assert order(7) == order(7)
    assert order(7) != order(8)
    assert sorted(order(7)) == list(range(40)), "shuffling must not drop or repeat samples"
    _ = np  # keep the import honest if the helper above is edited


def test_fedaco_orders_replies_by_client_id_not_by_arrival() -> None:
    """Ray does not fix reply order, and FedACO builds `deltas`, the Gram matrix, the
    desirability and the colony's stations positionally from it -- so a permuted arrival
    order hands the same pseudo-random draws to different clients and the search takes a
    different path. `participating_client_ids` in the first gate run shows the order
    changing between rounds of a single run.

    The aggregate was never *wrong*: alpha is matched to clients by id and the pheromone is
    keyed by id. It was not reproducible, which is a different property.
    """
    import inspect

    from fedswarm.strategies.fedaco import FedACO

    source = inspect.getsource(FedACO.aggregate_train)
    assert "sorted(" in source and "client_id" in source, (
        "aggregate_train must canonicalise reply order by client id"
    )
