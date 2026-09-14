"""Rich console logging plus an append-only JSONL file log per run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


class JsonlLogger:
    """Appends one JSON object per line to `path`. Safe to resume: opens in append
    mode, never truncates an existing log."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
