"""Global seeding for reproducible runs.

Deterministic mode can raise on CUDA ops without a deterministic kernel; callers should
not crash on that -- see `provenance.py`, which records whether determinism held.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> bool:
    """Seed random, numpy, and torch (if installed). Returns whether deterministic
    algorithms were successfully enabled."""
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
        torch.use_deterministic_algorithms(True)
        return True
    except RuntimeError:
        return False
