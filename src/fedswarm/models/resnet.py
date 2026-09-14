"""ResNet-18 — the secondary "does it hold with a larger backbone" table (see
docs/EXPERIMENT_LOG.md). Both pretrained and from-scratch, both BatchNorm and GroupNorm.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def _replace_bn_with_gn(module: nn.Module) -> None:
    """Swap every BatchNorm2d for a GroupNorm with matching channel count, in place.

    When starting from ImageNet-pretrained weights, this keeps the pretrained conv
    filters and discards only the BatchNorm affine/running-stats parameters (which are FL
    problematic -- see simple_cnn.py's docstring), leaving GroupNorm's params to be
    learned fresh during (federated) training.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            channels = child.num_features
            groups = min(32, channels)
            while channels % groups != 0:
                groups -= 1
            setattr(module, name, nn.GroupNorm(groups, channels))
        else:
            _replace_bn_with_gn(child)


def build_resnet18(num_classes: int = 4, pretrained: bool = True, norm: str = "groupnorm") -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if norm == "groupnorm":
        _replace_bn_with_gn(model)
    elif norm != "batchnorm":
        raise ValueError(f"Unknown norm: {norm!r}, expected 'batchnorm' or 'groupnorm'")

    return model
