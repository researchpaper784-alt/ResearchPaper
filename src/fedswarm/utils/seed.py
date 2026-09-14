"""Global seeding for reproducible runs.

Deterministic mode can raise on ops without a deterministic kernel for the active
backend (e.g. AdaptiveAvgPool2d's backward historically lacks one on CUDA -- this is
exactly what broke the first Colab training run, mid-epoch, not at setup). Using
`warn_only=True` is what turns that crash into a warning-and-fallback instead, matching
this project's own design intent (see `provenance.py`) that a run should degrade to
"not fully deterministic" rather than dying.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> bool:
    """Seed random, numpy, and torch (if installed). Returns whether deterministic
    algorithms were requested and enabled (with warn_only=True: individual ops lacking a
    deterministic kernel fall back to their normal implementation with a warning rather
    than raising -- requesting determinism must never crash a training run)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return False

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if not deterministic:
        return False

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        return True
    except (RuntimeError, TypeError):
        # TypeError covers a torch old enough to lack the warn_only kwarg entirely.
        try:
            torch.use_deterministic_algorithms(True)
            return True
        except RuntimeError:
            return False
