"""Phase 6, Step 6.2 -- fedswarm.sweep: config-grid expansion, run_id prediction,
resume-skip, locking, ordering, and command construction. All pure Python, no live
Flower runtime needed -- `run_sweep`'s subprocess call is injected via `runner`."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from fedswarm.sweep import (
    RunSpec,
    append_manifest,
    build_run_config_arg,
    execute_run,
    expand_grid,
    format_toml_value,
    is_completed,
    lock_path,
    order_seed_first,
    predict_run_id,
    pyproject_flat_defaults,
    release_lock,
    run_sweep,
    try_acquire_lock,
)
from fedswarm.utils.results import make_run_id


# ======================================================================================
# expand_grid / order_seed_first
# ======================================================================================


def test_expand_grid_is_a_full_factorial() -> None:
    strategies = [{"name": "fedavg", "overrides": {"strategy-name": "fedavg"}},
                  {"name": "fedaco", "overrides": {"strategy-name": "fedaco"}}]
    partitions = [{"name": "iid", "overrides": {"regime": "iid"}},
                  {"name": "dirichlet_0.3", "overrides": {"regime": "dirichlet", "alpha": 0.3}}]
    seeds = [0, 1]

    runs = expand_grid(strategies, partitions, seeds, base_overrides={"num-rounds": 5})

    assert len(runs) == 2 * 2 * 2
    labels = {r.label for r in runs}
    assert "fedaco/dirichlet_0.3/seed=1" in labels
    for run in runs:
        assert run.overrides["num-rounds"] == 5
        assert "seed" in run.overrides


def test_expand_grid_strategy_overrides_win_over_base_and_partition_on_key_collision() -> None:
    strategies = [{"name": "s", "overrides": {"regime": "overridden-by-strategy"}}]
    partitions = [{"name": "p", "overrides": {"regime": "from-partition"}}]
    runs = expand_grid(strategies, partitions, [0], base_overrides={"regime": "from-base"})
    assert runs[0].overrides["regime"] == "overridden-by-strategy"


def test_order_seed_first_groups_by_seed_preserving_within_seed_order() -> None:
    strategies = [{"name": "a", "overrides": {}}, {"name": "b", "overrides": {}}]
    partitions = [{"name": "p", "overrides": {}}]
    runs = expand_grid(strategies, partitions, [0, 1, 2])
    ordered = order_seed_first(runs)

    seeds_in_order = [r.overrides["seed"] for r in ordered]
    # Every seed=0 run appears before every seed=1 run, which appears before seed=2.
    assert seeds_in_order == sorted(seeds_in_order, key=lambda s: [0, 1, 2].index(s))
    # Within seed=0, strategy 'a' still comes before 'b' (original order preserved).
    seed0 = [r for r in ordered if r.overrides["seed"] == 0]
    assert seed0[0].label.startswith("a/")
    assert seed0[1].label.startswith("b/")


# ======================================================================================
# run_id prediction -- must match fl/app.py::main() exactly
# ======================================================================================


@pytest.fixture
def sample_pyproject(tmp_path: Path) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        """
[tool.flwr.app.config]
regime = "iid"
num-clients = 2
num-rounds = 2
seed = 0
strategy-name = "fedavg"
"""
    )
    return path


def test_pyproject_flat_defaults_reads_the_config_table(sample_pyproject: Path) -> None:
    defaults = pyproject_flat_defaults(sample_pyproject)
    assert defaults == {
        "regime": "iid",
        "num-clients": 2,
        "num-rounds": 2,
        "seed": 0,
        "strategy-name": "fedavg",
    }


def test_predict_run_id_matches_fl_app_mains_own_hashing(sample_pyproject: Path) -> None:
    """Reproduces fl/app.py::main()'s exact resolved_config construction by hand and
    checks predict_run_id agrees -- this is the one thing resume-skip depends on
    entirely; a mismatch here would mean the sweep runner can never recognize its
    own completed runs."""
    defaults = pyproject_flat_defaults(sample_pyproject)
    overrides = {"strategy-name": "fedaco", "seed": 3, "regime": "dirichlet", "alpha": 0.3}

    predicted = predict_run_id(defaults, overrides)

    merged_run_config = {**defaults, **overrides}
    seed = int(merged_run_config.get("seed", 0))
    strategy_name = str(merged_run_config.get("strategy-name", "fedavg")).lower()
    resolved_config = {"run_config": merged_run_config, "strategy": strategy_name, "seed": seed}
    expected = make_run_id(resolved_config, seed)

    assert predicted == expected


def test_predict_run_id_changes_when_a_relevant_key_changes(sample_pyproject: Path) -> None:
    defaults = pyproject_flat_defaults(sample_pyproject)
    id_a = predict_run_id(defaults, {"seed": 0})
    id_b = predict_run_id(defaults, {"seed": 1})
    assert id_a != id_b


def test_is_completed_true_only_for_a_real_completed_status(tmp_path: Path) -> None:
    output_dir = tmp_path / "fl"
    output_dir.mkdir()
    (output_dir / "abc_0.json").write_text(json.dumps({"status": "completed"}))
    (output_dir / "def_0.json").write_text(json.dumps({"status": "failed"}))

    assert is_completed("abc_0", output_dir) is True
    assert is_completed("def_0", output_dir) is False
    assert is_completed("missing_0", output_dir) is False


# ======================================================================================
# --run-config command-string construction
# ======================================================================================


def test_format_toml_value_covers_every_type() -> None:
    assert format_toml_value(True) == "true"
    assert format_toml_value(False) == "false"
    assert format_toml_value(3) == "3"
    assert format_toml_value(0.3) == "0.3"
    assert format_toml_value("dirichlet") == '"dirichlet"'
    assert format_toml_value('has "quotes"') == '"has \\"quotes\\""'


def test_build_run_config_arg_joins_key_value_pairs() -> None:
    arg = build_run_config_arg({"regime": "dirichlet", "alpha": 0.3, "seed": 2})
    assert arg == 'regime="dirichlet" alpha=0.3 seed=2'


def test_execute_run_builds_the_real_flwr_cli_invocation() -> None:
    captured = {}

    def fake_runner(cmd: list[str]) -> int:
        captured["cmd"] = cmd
        return 0

    run = RunSpec(label="x", group="g", overrides={"strategy-name": "fedaco", "seed": 0})
    code = execute_run(run, runner=fake_runner)

    assert code == 0
    assert captured["cmd"][:3] == ["flwr", "run", "."]
    assert "--stream" in captured["cmd"]
    assert "--run-config" in captured["cmd"]
    run_config_idx = captured["cmd"].index("--run-config")
    assert captured["cmd"][run_config_idx + 1] == 'strategy-name="fedaco" seed=0'


# ======================================================================================
# Locking
# ======================================================================================


def test_try_acquire_lock_then_second_attempt_fails(tmp_path: Path) -> None:
    assert try_acquire_lock("run1", tmp_path) is True
    assert try_acquire_lock("run1", tmp_path) is False  # still held


def test_try_acquire_lock_succeeds_after_release(tmp_path: Path) -> None:
    try_acquire_lock("run1", tmp_path)
    release_lock("run1", tmp_path)
    assert try_acquire_lock("run1", tmp_path) is True


def test_stale_lock_is_treated_as_crashed_and_replaced(tmp_path: Path) -> None:
    try_acquire_lock("run1", tmp_path, timeout_s=3600)
    # Backdate the lock file's mtime to simulate a crashed session from long ago.
    old_time = time.time() - 7200
    path = lock_path("run1", tmp_path)
    import os

    os.utime(path, (old_time, old_time))

    assert try_acquire_lock("run1", tmp_path, timeout_s=3600) is True


def test_release_lock_on_a_missing_lock_does_not_raise(tmp_path: Path) -> None:
    release_lock("never-acquired", tmp_path)  # must not raise


# ======================================================================================
# run_sweep orchestration
# ======================================================================================


def test_run_sweep_skips_already_completed_runs(tmp_path: Path, sample_pyproject: Path) -> None:
    output_dir = tmp_path / "fl"
    output_dir.mkdir()
    defaults = pyproject_flat_defaults(sample_pyproject)
    run = RunSpec(label="x", group="g", overrides={"seed": 0})
    run_id = predict_run_id(defaults, run.overrides)
    (output_dir / f"{run_id}.json").write_text(json.dumps({"status": "completed"}))

    calls = []
    entries = run_sweep(
        [run],
        pyproject_path=sample_pyproject,
        output_dir=output_dir,
        manifest_path=tmp_path / "manifest.jsonl",
        lock_dir=tmp_path / "locks",
        runner=lambda cmd: calls.append(cmd) or 0,
    )

    assert entries[0]["status"] == "skipped_completed"
    assert calls == []  # never actually invoked flwr


def test_run_sweep_dry_run_touches_nothing(tmp_path: Path, sample_pyproject: Path) -> None:
    run = RunSpec(label="x", group="g", overrides={"seed": 0})
    calls = []

    entries = run_sweep(
        [run],
        pyproject_path=sample_pyproject,
        output_dir=tmp_path / "fl",
        manifest_path=tmp_path / "manifest.jsonl",
        lock_dir=tmp_path / "locks",
        dry_run=True,
        runner=lambda cmd: calls.append(cmd) or 0,
    )

    assert entries[0]["status"] == "planned"
    assert calls == []
    assert not (tmp_path / "manifest.jsonl").exists()
    assert not (tmp_path / "locks").exists()


def test_run_sweep_executes_and_records_completed_status(tmp_path: Path, sample_pyproject: Path) -> None:
    output_dir = tmp_path / "fl"
    defaults = pyproject_flat_defaults(sample_pyproject)
    run = RunSpec(label="x", group="g", overrides={"seed": 0})
    run_id = predict_run_id(defaults, run.overrides)

    def fake_runner(cmd: list[str]) -> int:
        # Simulate the FL harness actually writing its result file.
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{run_id}.json").write_text(json.dumps({"status": "completed"}))
        return 0

    manifest_path = tmp_path / "manifest.jsonl"
    entries = run_sweep(
        [run],
        pyproject_path=sample_pyproject,
        output_dir=output_dir,
        manifest_path=manifest_path,
        lock_dir=tmp_path / "locks",
        runner=fake_runner,
    )

    assert entries[0]["status"] == "completed"
    assert manifest_path.exists()
    logged = json.loads(manifest_path.read_text().strip().splitlines()[0])
    assert logged["status"] == "completed"
    assert not lock_path(run_id, tmp_path / "locks").exists()  # released


def test_run_sweep_records_failed_when_return_code_nonzero_and_no_result(
    tmp_path: Path, sample_pyproject: Path
) -> None:
    run = RunSpec(label="x", group="g", overrides={"seed": 0})
    entries = run_sweep(
        [run],
        pyproject_path=sample_pyproject,
        output_dir=tmp_path / "fl",
        manifest_path=tmp_path / "manifest.jsonl",
        lock_dir=tmp_path / "locks",
        runner=lambda cmd: 1,
    )
    assert entries[0]["status"] == "failed"


def test_run_sweep_records_unknown_when_return_code_zero_but_no_result(
    tmp_path: Path, sample_pyproject: Path
) -> None:
    """The exact ambiguous case this repo has real history with: `flwr run` exits
    cleanly (return code 0) but the FL round itself never produced a completed
    result (docs/OPEN_QUESTIONS.md's heartbeat saga)."""
    run = RunSpec(label="x", group="g", overrides={"seed": 0})
    entries = run_sweep(
        [run],
        pyproject_path=sample_pyproject,
        output_dir=tmp_path / "fl",
        manifest_path=tmp_path / "manifest.jsonl",
        lock_dir=tmp_path / "locks",
        runner=lambda cmd: 0,
    )
    assert entries[0]["status"] == "unknown"


def test_run_sweep_skips_a_locked_run_without_releasing_the_other_holders_lock(
    tmp_path: Path, sample_pyproject: Path
) -> None:
    run = RunSpec(label="x", group="g", overrides={"seed": 0})
    defaults = pyproject_flat_defaults(sample_pyproject)
    run_id = predict_run_id(defaults, run.overrides)
    lock_dir = tmp_path / "locks"
    try_acquire_lock(run_id, lock_dir)  # simulate another in-flight sweep process

    calls = []
    entries = run_sweep(
        [run],
        pyproject_path=sample_pyproject,
        output_dir=tmp_path / "fl",
        manifest_path=tmp_path / "manifest.jsonl",
        lock_dir=lock_dir,
        runner=lambda cmd: calls.append(cmd) or 0,
    )

    assert entries[0]["status"] == "skipped_locked"
    assert calls == []
    assert lock_path(run_id, lock_dir).exists()  # the other holder's lock, untouched


def test_append_manifest_writes_one_json_line(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    append_manifest(manifest_path, {"run_id": "a"})
    append_manifest(manifest_path, {"run_id": "b"})
    lines = manifest_path.read_text().strip().splitlines()
    assert [json.loads(line)["run_id"] for line in lines] == ["a", "b"]


# ======================================================================================
# resolve_entry_overrides -- referencing standalone configs/strategy|data/*.yaml files
# ======================================================================================


def test_resolve_entry_overrides_inline() -> None:
    from fedswarm.sweep import resolve_entry_overrides

    entry = {"name": "fedavg", "overrides": {"strategy-name": "fedavg"}}
    assert resolve_entry_overrides(entry) == {"strategy-name": "fedavg"}


def test_resolve_entry_overrides_from_file(tmp_path: Path) -> None:
    from fedswarm.sweep import resolve_entry_overrides

    strategy_dir = tmp_path / "configs" / "strategy"
    strategy_dir.mkdir(parents=True)
    (strategy_dir / "fedaco.yaml").write_text("strategy-name: fedaco\nfedaco-target-sum: 0.9\n")

    entry = {"name": "fedaco", "file": "configs/strategy/fedaco.yaml"}
    overrides = resolve_entry_overrides(entry, base_dir=tmp_path)
    assert overrides == {"strategy-name": "fedaco", "fedaco-target-sum": 0.9}


def test_expand_grid_resolves_file_referenced_entries(tmp_path: Path) -> None:
    strategy_dir = tmp_path / "configs" / "strategy"
    strategy_dir.mkdir(parents=True)
    (strategy_dir / "fedavg.yaml").write_text("strategy-name: fedavg\n")

    strategies = [{"name": "fedavg", "file": "configs/strategy/fedavg.yaml"}]
    partitions = [{"name": "iid", "overrides": {"regime": "iid"}}]

    runs = expand_grid(strategies, partitions, [0], base_dir=tmp_path)
    assert runs[0].overrides["strategy-name"] == "fedavg"
    assert runs[0].overrides["regime"] == "iid"
