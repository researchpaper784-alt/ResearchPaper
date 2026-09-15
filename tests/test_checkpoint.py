"""Tests for stage-wise checkpointing, added after a Colab runtime recycle destroyed an
entire session's work (uploaded dataset, built cache, and finished runs) because nothing
was mirrored off the ephemeral disk until the sweep finished."""

from __future__ import annotations

import json
from pathlib import Path

from fedswarm.utils.checkpoint import ARTIFACTS, Checkpoint


def _working_tree(root: Path, stages: tuple[str, ...] = ("dataset", "cache", "results")) -> None:
    if "dataset" in stages:
        path = root / ARTIFACTS["dataset"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not really a zip")
    if "cache" in stages:
        cache = root / ARTIFACTS["cache"]
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "images_112.npy").write_bytes(b"pixels")
        (cache / "images_112_meta.json").write_text(json.dumps({"size": 112}))
    if "results" in stages:
        results = root / ARTIFACTS["results"]
        results.mkdir(parents=True, exist_ok=True)
        (results / "simple_cnn_groupnorm_0.json").write_text(json.dumps({"seed": 0}))


def test_round_trip_survives_losing_the_working_tree(tmp_path: Path) -> None:
    """The actual scenario: back everything up, lose /content entirely, restore."""
    root, backup = tmp_path / "repo", tmp_path / "drive"
    root.mkdir()
    _working_tree(root)

    assert Checkpoint(backup, root).save() == ["dataset", "cache", "results"]

    import shutil

    shutil.rmtree(root)
    root.mkdir()

    assert Checkpoint(backup, root).restore() == ["dataset", "cache", "results"]
    assert (root / ARTIFACTS["dataset"]).read_bytes() == b"not really a zip"
    assert (root / ARTIFACTS["cache"] / "images_112.npy").read_bytes() == b"pixels"
    assert (root / ARTIFACTS["results"] / "simple_cnn_groupnorm_0.json").exists()


def test_save_skips_stages_with_nothing_to_back_up(tmp_path: Path) -> None:
    """Checkpointing runs after every step, so early calls legitimately find later
    stages absent -- that must be a no-op, not an error."""
    root, backup = tmp_path / "repo", tmp_path / "drive"
    root.mkdir()
    _working_tree(root, stages=("dataset",))

    assert Checkpoint(backup, root).save() == ["dataset"]
    assert not (backup / ARTIFACTS["cache"]).exists()


def test_restore_of_an_empty_backup_is_a_no_op(tmp_path: Path) -> None:
    root, backup = tmp_path / "repo", tmp_path / "drive"
    root.mkdir()

    assert Checkpoint(backup, root).restore() == []


def test_save_named_stage_only(tmp_path: Path) -> None:
    root, backup = tmp_path / "repo", tmp_path / "drive"
    root.mkdir()
    _working_tree(root)

    assert Checkpoint(backup, root).save("results") == ["results"]
    assert (backup / ARTIFACTS["results"]).exists()
    assert not (backup / ARTIFACTS["cache"]).exists()


def test_restore_merges_into_an_existing_directory(tmp_path: Path) -> None:
    """A resumed session restores finished runs alongside any produced since, rather
    than failing because the destination directory already exists."""
    root, backup = tmp_path / "repo", tmp_path / "drive"
    root.mkdir()
    _working_tree(root, stages=("results",))
    Checkpoint(backup, root).save("results")

    (root / ARTIFACTS["results"] / "simple_cnn_groupnorm_1.json").write_text("{}")
    Checkpoint(backup, root).restore("results")

    names = sorted(p.name for p in (root / ARTIFACTS["results"]).glob("*.json"))
    assert names == ["simple_cnn_groupnorm_0.json", "simple_cnn_groupnorm_1.json"]


def test_status_and_report_reflect_the_filesystem(tmp_path: Path) -> None:
    root, backup = tmp_path / "repo", tmp_path / "drive"
    root.mkdir()
    _working_tree(root, stages=("cache", "results"))
    checkpoint = Checkpoint(backup, root)
    checkpoint.save("cache")

    status = checkpoint.status()
    assert status["cache"] == {"local": True, "backup": True}
    assert status["results"] == {"local": True, "backup": False}
    assert status["dataset"] == {"local": False, "backup": False}

    assert "simple_cnn_groupnorm_0" in checkpoint.report()
    assert "1 finished runs" in checkpoint.report()
