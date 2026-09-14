"""Capture the full provenance of a run so results are traceable back to the exact
code, environment, and hardware that produced them."""

from __future__ import annotations

import platform
import socket
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any

_TRACKED_PACKAGES = (
    "flwr",
    "flwr-datasets",
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "omegaconf",
)


def _git_sha() -> tuple[str | None, bool]:
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        dirty = (
            subprocess.call(
                ["git", "diff", "--quiet", "--ignore-submodules", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            != 0
        )
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    return versions


def _gpu_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False}

    if torch.cuda.is_available():
        return {
            "available": True,
            "backend": "cuda",
            "name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
        }
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return {"available": True, "backend": "mps", "name": "Apple MPS"}
    return {"available": False, "backend": "cpu"}


def capture() -> dict[str, Any]:
    """Capture git SHA, dirty flag, resolved package versions, GPU, host, and UTC
    timestamp. Call once per run and embed the result in the run's result JSON."""
    git_sha, dirty = _git_sha()
    return {
        "git_sha": git_sha,
        "dirty": dirty,
        "packages": _package_versions(),
        "gpu": _gpu_info(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
