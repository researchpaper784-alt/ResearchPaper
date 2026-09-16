"""Phase 3.2 tests -- fl/checkpoint.py's save/load/clear round-checkpoint triple.

Pure filesystem + torch.save/load, no Flower dependency -- fully testable here."""

from __future__ import annotations

from pathlib import Path

from fedswarm.fl.checkpoint import checkpoint_paths, clear_checkpoint, load_checkpoint, save_checkpoint


def test_load_checkpoint_returns_none_when_absent(tmp_path: Path) -> None:
    assert load_checkpoint(tmp_path, "no-such-run") is None


def test_save_then_load_roundtrips(tmp_path: Path) -> None:
    rounds_log = [{"round": 1, "test_macro_f1": 0.5}, {"round": 2, "test_macro_f1": 0.6}]
    state_dict = {"w": [1.0, 2.0, 3.0]}

    save_checkpoint(tmp_path, "run-a", round_number=2, rounds_log=rounds_log, state_dict=state_dict)
    loaded_round, loaded_log, loaded_state = load_checkpoint(tmp_path, "run-a")

    assert loaded_round == 2
    assert loaded_log == rounds_log
    assert loaded_state == state_dict


def test_clear_checkpoint_removes_both_files(tmp_path: Path) -> None:
    save_checkpoint(tmp_path, "run-b", round_number=1, rounds_log=[], state_dict={})
    json_path, weights_path = checkpoint_paths(tmp_path, "run-b")
    assert json_path.exists() and weights_path.exists()

    clear_checkpoint(tmp_path, "run-b")

    assert not json_path.exists()
    assert not weights_path.exists()


def test_clear_checkpoint_is_a_noop_when_nothing_exists(tmp_path: Path) -> None:
    clear_checkpoint(tmp_path, "never-existed")  # must not raise


def test_two_run_ids_do_not_collide(tmp_path: Path) -> None:
    save_checkpoint(tmp_path, "run-a", round_number=5, rounds_log=[{"round": 5}], state_dict={"w": 1})
    save_checkpoint(tmp_path, "run-b", round_number=2, rounds_log=[{"round": 2}], state_dict={"w": 2})

    round_a, _, state_a = load_checkpoint(tmp_path, "run-a")
    round_b, _, state_b = load_checkpoint(tmp_path, "run-b")

    assert round_a == 5 and state_a == {"w": 1}
    assert round_b == 2 and state_b == {"w": 2}
