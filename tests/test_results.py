"""Tests for the Phase 6 result-contract writer (built in Phase 2 so centralized and
future FL results share one schema)."""

from __future__ import annotations

import json
from pathlib import Path

from fedswarm.utils.results import make_run_id, write_result


def test_make_run_id_is_deterministic_and_seed_specific() -> None:
    config = {"model": "simple_cnn", "lr": 0.001}

    assert make_run_id(config, seed=0) == make_run_id(config, seed=0)
    assert make_run_id(config, seed=0) != make_run_id(config, seed=1)
    assert make_run_id(config, seed=0).endswith("_0")


def test_make_run_id_changes_with_config() -> None:
    a = make_run_id({"lr": 0.001}, seed=0)
    b = make_run_id({"lr": 0.002}, seed=0)
    assert a != b


def test_write_result_produces_the_documented_schema(tmp_path: Path) -> None:
    out = tmp_path / "run.json"
    config = {"model": {"name": "simple_cnn"}, "seed": 0}
    rounds = [{"epoch": 1, "val_macro_f1": 0.5}]
    final = {"test_macro_f1": 0.6, "best_epoch": 1}

    result = write_result(out, config=config, rounds=rounds, final=final, seed=0)

    assert out.exists()
    on_disk = json.loads(out.read_text())
    for key in ("run_id", "config", "provenance", "partition_stats", "rounds", "final", "status"):
        assert key in on_disk

    assert on_disk["config"] == config
    assert on_disk["rounds"] == rounds
    assert on_disk["final"] == final
    assert on_disk["status"] == "completed"
    assert on_disk["provenance"]["git_sha"] is not None
    assert result == on_disk


def test_write_result_creates_parent_directories(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "dir" / "run.json"
    write_result(out, config={}, rounds=[], final={}, seed=0)
    assert out.exists()
