"""SimpleCNN — the primary-config backbone (see docs/EXPERIMENT_LOG.md: the small model
at 112px is the primary configuration for the full main sweep, ablations, and robustness
tests; ResNet-18 @224 is a reduced secondary table). ~4 conv blocks, from scratch, <1M
parameters -- deliberately small, in the tradition of the original FedAvg paper's
two-conv CNN.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _norm_layer(norm: str, channels: int) -> nn.Module:
    if norm == "batchnorm":
        return nn.BatchNorm2d(channels)
    if norm == "groupnorm":
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unknown norm: {norm!r}, expected 'batchnorm' or 'groupnorm'")


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = _norm_layer(norm, out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.norm(self.conv(x))))


class SimpleCNN(nn.Module):
    """4 conv blocks (3->32->64->128->256), global average pool, linear head.

    BatchNorm running statistics aggregate badly under non-IID federated data (client
    batches have different label distributions, so client-local running means/variances
    diverge) -- this is a known FL pathology, not specific to this method. GroupNorm has
    no cross-batch running statistics and is the default for FL experiments; BatchNorm is
    offered for the A9 confound-control ablation.
    """

    def __init__(self, num_classes: int = 4, norm: str = "groupnorm", in_channels: int = 3) -> None:
        super().__init__()
        channels = (in_channels, 32, 64, 128, 256)
        self.blocks = nn.Sequential(
            *(_ConvBlock(channels[i], channels[i + 1], norm) for i in range(4))
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
