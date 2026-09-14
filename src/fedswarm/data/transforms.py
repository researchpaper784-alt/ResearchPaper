"""Phase 1.3 — preprocessing and augmentation.

Operates on the cached single-channel uint8 arrays, not on JPEG files. Geometric
augmentations come before photometric ones, and both come before normalization.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torchvision import transforms

from fedswarm.data.cache import CACHE_DIR

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_statistics(size: int, cache_dir: Path = CACHE_DIR) -> dict:
    meta_path = Path(cache_dir) / f"images_{size}_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No cache metadata at {meta_path}. Run `python -m fedswarm.data.cache "
            f"--size {size}` first."
        )
    return json.loads(meta_path.read_text())["statistics"]


def normalization(size: int, scheme: str, cache_dir: Path = CACHE_DIR) -> transforms.Normalize:
    """`imagenet` for pretrained backbones, `dataset` for from-scratch models."""
    if scheme == "imagenet":
        return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if scheme == "dataset":
        stats = load_statistics(size, cache_dir)
        mean, std = stats["mean"], stats["std"]
        return transforms.Normalize((mean,) * 3, (std,) * 3)
    raise ValueError(f"Unknown normalization scheme: {scheme!r}")


def build_transforms(
    train: bool,
    size: int = 112,
    scheme: str = "dataset",
    cache_dir: Path = CACHE_DIR,
) -> transforms.Compose:
    """Augmentation pipeline for cached uint8 images.

    No vertical flip: MRI orientation is anatomically meaningful, and a vertically
    mirrored brain is not a plausible scan. Rotation is limited to +/-10 degrees for the
    same reason.
    """
    steps: list = []

    if train:
        steps += [
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]

    steps += [
        ToFloatAndReplicate(),
        normalization(size, scheme, cache_dir),
    ]
    return transforms.Compose(steps)


class ToFloatAndReplicate:
    """uint8 (1, H, W) -> float32 (3, H, W) in [0, 1].

    Grayscale is replicated to three channels so ImageNet-pretrained backbones, which
    expect RGB, work unchanged.
    """

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 2:
            image = image.unsqueeze(0)
        return image.float().div_(255.0).expand(3, -1, -1).contiguous()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
