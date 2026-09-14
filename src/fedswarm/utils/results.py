"""Result-file writer, conforming to the Phase 6 result contract (plan §6.1).

Built now, in Phase 2, rather than deferred: centralized results (this phase) and FL
results (Phase 3+) must share one schema so analysis scripts never need to know how a run
was produced. Phase 6 will extend `rounds` with FL-specific fields (alpha, fallback_used,
etc.); nothing here is FL-specific.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from fedswarm.utils.provenance import capture


def make_run_id(config: dict, seed: int, length: int = 10) -> str:
    canonical = json.dumps(config, sort_keys=True, default=str)
    config_hash = hashlib.sha256(canonical.encode()).hexdigest()[:length]
    return f"{config_hash}_{seed}"


def write_result(
    path: Path | str,
    config: dict[str, Any],
    rounds: list[dict[str, Any]],
    final: dict[str, Any],
    seed: int,
    status: str = "completed",
    partition_stats: dict[str, Any] | None = None,
) -> dict:
    result = {
        "run_id": make_run_id(config, seed),
        "config": config,
        "provenance": capture(),
        "partition_stats": partition_stats,
        "rounds": rounds,
        "final": final,
        "status": status,
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    return result
