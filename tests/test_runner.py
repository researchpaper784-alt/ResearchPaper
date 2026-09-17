"""Tests for fedswarm.utils.runner -- the plumbing shared by the scripts that drive
`flwr run`.

Two of these pin facts that no test covered while the helpers were duplicated, and that
each cost this project a wasted run when they were violated: that `--stream` is always
passed (without it `flwr run` returns exit 0 immediately and the caller concludes every
run failed), and that `diagnose` does not throw away the line that explains a failure.

The rest pin the shared-ness itself. The copies drifted once; a test that the scripts hold
the *same object* is what stops them drifting again.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_hparam_search  # noqa: E402
import run_sweep  # noqa: E402
from fedswarm.utils import runner  # noqa: E402


# ======================================================================================
# One definition, not two
# ======================================================================================


def test_both_runners_share_one_copy_of_each_helper() -> None:
    """`diagnose` had already diverged once: the sweep's copy filtered log noise with
    `"arn" not in line.lower()` while the search's filtered `Warning`/`warn`. The sweep
    therefore discarded any line containing *learn* and reported "no diagnostic line
    found" for failures the search would have explained. Identity, not equality -- two
    equal-but-separate definitions are exactly the state that drifted."""
    for name in ("diagnose", "format_run_config", "same_value"):
        shared = getattr(runner, name)
        assert getattr(run_sweep, name) is shared, name
        assert getattr(run_hparam_search, name) is shared, name
    assert run_sweep.run_flwr is runner.run_flwr
    assert run_hparam_search.run_flwr is runner.run_flwr


# ======================================================================================
# diagnose
# ======================================================================================


def test_diagnose_keeps_a_line_that_merely_contains_learn() -> None:
    """The regression the extraction fixes. A failed tuning cell's most likely
    explanatory line names a learning rate, and the sweep's filter dropped every line
    containing "arn" -- which "learning" does."""
    output = "ValueError: learning rate must be positive, got -0.1"

    assert "learning rate" in runner.diagnose(output)


def test_diagnose_still_ignores_warnings() -> None:
    noisy = "\n".join(
        [
            "FutureWarning: ray -- set RAY_DEDUP_LOGS=0 to turn off this error message",
            "Invalid run configuration [code: 15]",
        ]
    )

    assert "Invalid run configuration" in runner.diagnose(noisy)
    assert "FutureWarning" not in runner.diagnose(noisy)


def test_diagnose_prefers_the_specific_cause_over_a_generic_one() -> None:
    """Markers are scanned in priority order across the whole output, so a bare "Error"
    later in a log does not outrank the named cause earlier in it -- or vice versa."""
    output = "\n".join(["Error: run failed", "Dataset root does not exist: /data/brain"])

    assert "Dataset root does not exist" in runner.diagnose(output)


def test_diagnose_admits_when_it_has_nothing() -> None:
    assert runner.diagnose("") == "no diagnostic line found in output"
    assert runner.diagnose("all fine here") == "no diagnostic line found in output"


# ======================================================================================
# run_flwr
# ======================================================================================


class _FakeCompleted:
    returncode = 0
    stdout = "out"
    stderr = "err"


def test_run_flwr_always_passes_stream(monkeypatch) -> None:
    """Without `--stream`, `flwr run` submits the run and returns exit 0 while the
    simulation is still starting. A sweep that lost this flag would launch all 360 cells
    at once, report every one as failed, and leave the machine thrashing under orphaned
    simulations -- observed, not theorised."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _FakeCompleted()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    runner.run_flwr("strategy-name='fedaco'", Path("/repo"))

    assert "--stream" in captured["cmd"]
    assert captured["cmd"][:3] == ["flwr", "run", "."]
    assert captured["cmd"][-1] == "strategy-name='fedaco'"


def test_run_flwr_exports_the_repo_root(monkeypatch) -> None:
    """`flwr run` executes an *installed copy* of the app, whose `__file__` is not in this
    clone, so the app resolves data and cache paths against $FEDSWARM_REPO_ROOT. Dropping
    it makes every run fail to find the image cache -- from inside the simulation, where
    the outer process still exits 0."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        return _FakeCompleted()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    runner.run_flwr("x=1", Path("/repo"))

    assert captured["env"]["FEDSWARM_REPO_ROOT"] == "/repo"
    assert captured["cwd"] == Path("/repo")


def test_run_flwr_captures_output_unless_streaming(monkeypatch) -> None:
    monkeypatch.setattr(runner.subprocess, "run", lambda cmd, **kwargs: _FakeCompleted())

    code, output = runner.run_flwr("x=1", Path("/repo"))
    assert (code, output) == (0, "outerr")

    # Streaming echoes live, so there is nothing left to hand to `diagnose`.
    code, output = runner.run_flwr("x=1", Path("/repo"), stream=True)
    assert (code, output) == (0, "")


# ======================================================================================
# Reading result files
# ======================================================================================


def _write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def test_a_truncated_result_does_not_kill_the_read(tmp_path: Path) -> None:
    """A run killed mid-write leaves a half-written JSON document. One of those must not
    stop an analysis of the 300 runs that completed."""
    _write(tmp_path / "good.json", '{"status": "completed"}')
    _write(tmp_path / "torn.json", '{"status": "comp')

    loaded = runner.load_result_files(tmp_path)

    assert [path.name for path, _ in loaded] == ["good.json"]


def test_flat_by_default_recursive_on_request(tmp_path: Path) -> None:
    """The resume check must not match a result belonging to a different sweep, so it
    reads one flat output dir; the analysis scripts are pointed at `results/` as a whole
    and must see every sweep's subdirectory."""
    _write(tmp_path / "top.json", "{}")
    _write(tmp_path / "ablation" / "nested.json", "{}")

    assert len(runner.load_result_files(tmp_path)) == 1
    assert len(runner.load_result_files(tmp_path, recursive=True)) == 2


def test_the_cache_invalidates_when_a_new_result_lands(tmp_path: Path) -> None:
    """The sweep reads this directory after every completed cell. A cache that did not
    notice the file it just wrote would report the cell incomplete and re-run it, forever.
    """
    _write(tmp_path / "a.json", "{}")
    assert len(runner.load_result_files(tmp_path, use_cache=True)) == 1

    _write(tmp_path / "b.json", "{}")
    assert len(runner.load_result_files(tmp_path, use_cache=True)) == 2


# ======================================================================================
# Value comparison and rendering
# ======================================================================================


def test_same_value_tolerates_a_float_round_trip() -> None:
    """Values make a round trip through TOML and JSON between being requested and being
    read back out of a result file. An exact `==` that answers "different" for a config
    already run recomputes that cell forever."""
    assert runner.same_value(0.1, 0.1 + 1e-15)
    assert runner.same_value(3, 3.0)
    assert not runner.same_value(0.1, 0.2)
    assert not runner.same_value("fedaco", 0.3)  # unconvertible, not a crash


def test_booleans_render_as_lowercase_toml() -> None:
    """`str(False)` is "False", which tomli rejects -- and flwr rejects the whole config
    string with a bare "[code: 15]" naming nothing."""
    assert runner.format_run_config({"aco-safety-fallback": False}) == "aco-safety-fallback=false"
