"""Phase 0.3 acceptance: two runs of a short training loop with the same seed must
produce identical (or near-identical, on non-deterministic ops) loss sequences."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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
