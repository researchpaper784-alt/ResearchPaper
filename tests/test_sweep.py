"""Phase 6 tests -- scripts/run_sweep.py.

The sweep's job is to run 330+ cells unattended and resume correctly after an
interruption. Everything that can go wrong there goes wrong *quietly*: a cell silently
reused from the wrong seed, a torn manifest line swallowing a run, a machine that stalls
instead of erroring. These test the decisions, not the subprocess plumbing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_sweep import (  # noqa: E402
    REPO_ROOT,
    Cell,
    _record_path,
    append_manifest,
    expand,
    find_result,
    format_run_config,
    load_sweep,
    preflight,
    read_manifest,
)

SPEC = {
    "name": "test",
    "strategies": ["fedavg", "fedaco"],
    "regimes": [
        {"name": "iid", "regime": "iid"},
        {"name": "dirichlet_0.3", "regime": "dirichlet", "alpha": 0.3},
    ],
    "seeds": [0, 1, 2],
    "common": {"num-clients": 4, "num-rounds": 10},
}


# ======================================================================================
# Expansion
# ======================================================================================


def test_expand_is_the_full_factorial() -> None:
    cells = expand(SPEC)
    assert len(cells) == 2 * 2 * 3
    assert len({c.label for c in cells}) == 12  # every cell distinct


def test_seed_is_the_innermost_loop() -> None:
    """So an interruption leaves whole (strategy, regime) groups finished rather than one
    seed of everything -- a partial sweep is then still analysable for what it covered,
    instead of having no complete group at all."""
    labels = [c.label for c in expand(SPEC)]

    assert labels[:3] == ["fedavg/iid/seed0", "fedavg/iid/seed1", "fedavg/iid/seed2"]
    assert labels[3].startswith("fedavg/dirichlet_0.3/")


def test_regime_overrides_reach_the_run_config() -> None:
    cells = expand(SPEC)
    dirichlet = next(c for c in cells if c.regime_name == "dirichlet_0.3")

    config = dirichlet.run_config(SPEC["common"])

    assert config["regime"] == "dirichlet"
    assert config["alpha"] == 0.3
    assert config["strategy-name"] == "fedavg"
    assert config["num-clients"] == 4
    # `name` is the label, not a flwr run-config key -- passing it through would be
    # rejected as undeclared with a bare "[code: 15]".
    assert "name" not in config


def test_load_sweep_rejects_an_incomplete_spec(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"strategies": ["fedavg"], "seeds": [0]}))

    with pytest.raises(ValueError, match="regimes"):
        load_sweep(path)


def test_format_run_config_quotes_strings_only() -> None:
    formatted = format_run_config({"strategy-name": "fedaco", "alpha": 0.3, "num-clients": 4})

    assert "strategy-name='fedaco'" in formatted
    assert "alpha=0.3" in formatted
    assert "num-clients=4" in formatted


# ======================================================================================
# Manifest / resume
# ======================================================================================


def test_manifest_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    append_manifest(path, {"label": "fedavg/iid/seed0", "status": "completed"})
    append_manifest(path, {"label": "fedaco/iid/seed0", "status": "completed"})

    entries = read_manifest(path)

    assert set(entries) == {"fedavg/iid/seed0", "fedaco/iid/seed0"}


def test_manifest_survives_a_torn_final_line(tmp_path: Path) -> None:
    """A hard kill mid-write leaves a partial line. That must cost one cell (which simply
    re-runs), not the whole index -- which is the reason the manifest is JSON Lines rather
    than one JSON document."""
    path = tmp_path / "manifest.jsonl"
    append_manifest(path, {"label": "fedavg/iid/seed0", "status": "completed"})
    with path.open("a") as handle:
        handle.write('{"label": "fedaco/iid/seed0", "sta')

    entries = read_manifest(path)

    assert set(entries) == {"fedavg/iid/seed0"}


def test_missing_manifest_is_empty_not_an_error(tmp_path: Path) -> None:
    assert read_manifest(tmp_path / "nope.jsonl") == {}


# ======================================================================================
# Result matching -- the variance-destroying bug class
# ======================================================================================


def _write_result(path: Path, run_config: dict, status: str = "completed") -> None:
    path.write_text(
        json.dumps(
            {
                "run_id": path.stem,
                "status": status,
                "config": {"strategy": run_config.get("strategy-name"), "run_config": run_config},
                "rounds": [],
                "final": {},
            }
        )
    )


def test_find_result_will_not_reuse_another_seeds_run(tmp_path: Path) -> None:
    """Matching on too little is how a sweep silently reuses one seed's result for
    another and destroys the variance every error bar in the paper depends on. The same
    bug hit the Phase 5 search; this is the sweep's version of that guard."""
    common = {"num-clients": 4, "num-rounds": 10}
    seed0 = Cell("fedavg", "iid", 0, {"regime": "iid"})
    seed1 = Cell("fedavg", "iid", 1, {"regime": "iid"})
    _write_result(tmp_path / "a.json", seed0.run_config(common))

    assert find_result(tmp_path, seed0, common) is not None
    assert find_result(tmp_path, seed1, common) is None


def test_find_result_will_not_reuse_another_regimes_run(tmp_path: Path) -> None:
    common = {"num-clients": 4, "num-rounds": 10}
    iid = Cell("fedavg", "iid", 0, {"regime": "iid"})
    dirichlet = Cell("fedavg", "dirichlet_0.3", 0, {"regime": "dirichlet", "alpha": 0.3})
    _write_result(tmp_path / "a.json", iid.run_config(common))

    assert find_result(tmp_path, dirichlet, common) is None


def test_find_result_ignores_incomplete_runs(tmp_path: Path) -> None:
    """A crashed run must be retried, not counted as done."""
    common = {"num-clients": 4}
    cell = Cell("fedavg", "iid", 0, {"regime": "iid"})
    _write_result(tmp_path / "a.json", cell.run_config(common), status="failed")

    assert find_result(tmp_path, cell, common) is None


def test_find_result_ignores_path_only_differences(tmp_path: Path) -> None:
    """A cache or output directory that moved between machines says nothing about what
    was computed, so it must not defeat resume."""
    common = {"num-clients": 4, "cache-dir": "/local/cache"}
    cell = Cell("fedavg", "iid", 0, {"regime": "iid"})
    stored = cell.run_config({**common, "cache-dir": "/colab/cache"})
    _write_result(tmp_path / "a.json", stored)

    assert find_result(tmp_path, cell, common) is not None


# ======================================================================================
# Preflight
# ======================================================================================


def test_preflight_flags_cpu_oversubscription() -> None:
    """K=20 requests 40 CPUs at the Simulation Runtime's default 2/ClientApp. That does
    not queue -- it stalls at round 0 with idle actors and no error, which across a long
    sweep is indistinguishable from a slow round."""
    warnings = preflight(num_clients=1000, skip=False)

    assert any("stalls silently" in w for w in warnings)
    assert any("client-resources-num-cpus" in w for w in warnings)


def test_preflight_can_be_skipped() -> None:
    assert preflight(num_clients=1000, skip=True) == []


def test_preflight_is_quiet_when_resources_fit() -> None:
    warnings = preflight(num_clients=1, skip=False)

    assert not any("stalls silently" in w for w in warnings)


def test_record_path_falls_back_to_absolute_outside_the_repo(tmp_path: Path) -> None:
    """`Path.relative_to` raises rather than falling back, and `output-dir` is routinely
    outside the clone -- Colab writes to Drive, Kaggle to /kaggle/working. An unguarded
    call aborted this runner one cell into its first real sweep, *after* the run finished
    but before the manifest append, so the completed run went unindexed too."""
    outside = tmp_path / "elsewhere" / "result.json"

    assert _record_path(outside) == str(outside)
    assert Path(_record_path(outside)).is_absolute()

    inside = REPO_ROOT / "results" / "fl" / "x.json"
    assert _record_path(inside) == "results/fl/x.json"


def test_a_result_with_no_manifest_entry_counts_as_done(tmp_path: Path) -> None:
    """The result file is the authority; the manifest is an index over it. Gating resume
    on the manifest too means a lost entry -- a torn line, a crash between the run
    finishing and the append, a manifest not carried between machines -- silently
    recomputes a cell whose result is sitting right there."""
    common = {"num-clients": 4}
    cell = Cell("fedavg", "iid", 0, {"regime": "iid"})
    _write_result(tmp_path / "a.json", cell.run_config(common))

    # Nothing in the manifest, but the result exists and is completed.
    assert read_manifest(tmp_path / "missing.jsonl") == {}
    assert find_result(tmp_path, cell, common) is not None
