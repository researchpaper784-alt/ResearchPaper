"""Phase 3.2 -- checkpoint/resume for one FL run.

Distinct from `utils/checkpoint.py`, which mirrors whole pipeline *stages* (dataset zip,
image cache, centralized results) to Drive for the Colab notebook and predates this
module. This one saves a single in-flight FL run's global weights and round-by-round
log after every round, so a lost Colab runtime mid-sweep costs at most the round that
was in flight, not the whole run -- "Non-negotiable on Colab/Kaggle" per the plan's
Step 3.2, since free-tier sessions die mid-sweep as a matter of course.

`Strategy.start()` has no `start_round` parameter (verified, `docs/FLOWER_API_NOTES.md`)
-- resuming means calling it again with the checkpointed weights as `initial_arrays` and
`num_rounds` reduced by however many rounds already completed. `server_app.py` does that
remapping; this module only persists and restores the (round, rounds_log, state_dict)
triple.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def checkpoint_paths(checkpoint_dir: Path | str, run_id: str) -> tuple[Path, Path]:
    directory = Path(checkpoint_dir)
    return directory / f"{run_id}.json", directory / f"{run_id}.pt"


def save_checkpoint(
    checkpoint_dir: Path | str,
    run_id: str,
    round_number: int,
    rounds_log: list[dict],
    state_dict: dict,
) -> None:
    json_path, weights_path = checkpoint_paths(checkpoint_dir, run_id)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"round": round_number, "rounds_log": rounds_log}, default=str))
    torch.save(state_dict, weights_path)


def load_checkpoint(checkpoint_dir: Path | str, run_id: str) -> tuple[int, list[dict], dict] | None:
    """Returns (last_completed_round, rounds_log_so_far, state_dict), or None if no
    checkpoint exists for this run_id yet (the common case: first attempt at a run)."""
    json_path, weights_path = checkpoint_paths(checkpoint_dir, run_id)
    if not json_path.exists() or not weights_path.exists():
        return None
    payload = json.loads(json_path.read_text())
    state_dict = torch.load(weights_path, map_location="cpu")
    return payload["round"], payload["rounds_log"], state_dict


def clear_checkpoint(checkpoint_dir: Path | str, run_id: str) -> None:
    """Called once a run's final result is written -- a lingering checkpoint file
    always means "this run_id did not finish," so a completed run must not leave one
    behind for a later invocation to mistakenly resume from."""
    json_path, weights_path = checkpoint_paths(checkpoint_dir, run_id)
    json_path.unlink(missing_ok=True)
    weights_path.unlink(missing_ok=True)
