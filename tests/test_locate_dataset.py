"""`locate_or_fetch_kaggle_dataset` replaced the one manual step in the Kaggle notebooks.

A user ran Day 1 twice and stopped both times at "no dataset attached". These pin the behaviour
that removes the click, and the ways it must NOT succeed -- returning a results folder, or a
fetched path without the splits, would pass the dataset cell and fail an hour later.
"""

from __future__ import annotations

from pathlib import Path

from fedswarm.data.download import locate_or_fetch_kaggle_dataset


def _dataset(root: Path, nested: str = "") -> Path:
    base = root / nested if nested else root
    for split in ("Training", "Testing"):
        (base / split).mkdir(parents=True)
    return root


def _never(slug: str) -> str:
    raise AssertionError("fetched although the dataset was already attached")


def test_an_attached_dataset_is_used_without_fetching(tmp_path) -> None:
    mount = tmp_path / "input"
    _dataset(mount / "brain-tumor-mri-dataset")
    assert locate_or_fetch_kaggle_dataset(mount_root=mount, fetch=_never) == \
        mount / "brain-tumor-mri-dataset"


def test_a_previous_sessions_results_are_not_mistaken_for_the_dataset(tmp_path) -> None:
    """From session 2 a results folder is mounted too, and sorts first here."""
    mount = tmp_path / "input"
    (mount / "a-previous-output" / "results" / "fl").mkdir(parents=True)
    _dataset(mount / "brain-tumor-mri-dataset", nested="payload")
    assert locate_or_fetch_kaggle_dataset(mount_root=mount, fetch=_never) == \
        mount / "brain-tumor-mri-dataset"


def test_nothing_attached_fetches_it(tmp_path) -> None:
    fetched = _dataset(tmp_path / "cache" / "versions" / "1")
    calls = []

    def fetch(slug):
        calls.append(slug)
        return str(fetched)

    got = locate_or_fetch_kaggle_dataset(mount_root=tmp_path / "input", fetch=fetch)
    assert got == fetched
    assert calls == ["masoudnickparvar/brain-tumor-mri-dataset"]


def test_a_fetch_failure_returns_none_rather_than_raising(tmp_path, capsys) -> None:
    def fetch(slug):
        raise ConnectionError("no internet")

    assert locate_or_fetch_kaggle_dataset(mount_root=tmp_path / "input", fetch=fetch) is None
    assert "no internet" in capsys.readouterr().out


def test_a_fetched_path_without_the_splits_is_rejected(tmp_path) -> None:
    empty = tmp_path / "wrong"
    empty.mkdir()
    assert locate_or_fetch_kaggle_dataset(
        mount_root=tmp_path / "input", fetch=lambda s: str(empty)) is None
