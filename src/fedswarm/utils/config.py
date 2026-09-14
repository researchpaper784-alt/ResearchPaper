"""Config loading: OmegaConf compose of base + data + model + strategy + experiment
YAML layers, with CLI dotlist overrides, plus a stable hash for run-id / sweep-resume."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(
    *layers: str | Path,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Merge one or more YAML files in order (later layers win), then apply CLI-style
    dotlist overrides (e.g. ["federated.num_rounds=5"])."""
    if not layers:
        raise ValueError("load_config requires at least one YAML layer")

    merged = OmegaConf.load(layers[0])
    for layer in layers[1:]:
        merged = OmegaConf.merge(merged, OmegaConf.load(layer))

    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))

    assert isinstance(merged, DictConfig)
    return merged


def config_hash(cfg: DictConfig, length: int = 10) -> str:
    """Stable hash of a fully-resolved config, used as part of run_id so sweeps can
    skip already-completed runs on resume."""
    resolved: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    canonical = json.dumps(resolved, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:length]
