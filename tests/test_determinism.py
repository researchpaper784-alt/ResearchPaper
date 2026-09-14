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
