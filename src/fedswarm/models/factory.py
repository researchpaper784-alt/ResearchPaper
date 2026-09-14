"""Model factory — single entry point so strategy/training code never imports a specific
architecture module directly."""

from __future__ import annotations

import torch.nn as nn

from fedswarm.models.resnet import build_resnet18
from fedswarm.models.simple_cnn import SimpleCNN

_NORM_ALIASES = {"bn": "batchnorm", "gn": "groupnorm", "batchnorm": "batchnorm", "groupnorm": "groupnorm"}


def build_model(
    name: str,
    num_classes: int = 4,
    pretrained: bool = False,
    norm: str = "groupnorm",
) -> nn.Module:
    norm = _NORM_ALIASES.get(norm, norm)

    if name == "simple_cnn":
        if pretrained:
            raise ValueError("simple_cnn has no pretrained weights; pass pretrained=False")
        return SimpleCNN(num_classes=num_classes, norm=norm)

    if name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained, norm=norm)

    raise ValueError(f"Unknown model: {name!r}, expected 'simple_cnn' or 'resnet18'")


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    params = (p for p in model.parameters() if not trainable_only or p.requires_grad)
    return sum(p.numel() for p in params)
