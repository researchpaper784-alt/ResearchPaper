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
import run_sweep  # noqa: E402,F401

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


def test_preflight_notes_but_does_not_block_on_more_clients_than_cores() -> None:
    """Superseded diagnosis. A K=4 run that stalled at round 0 was recorded as CPU
    oversubscription; it was not -- only 2 supernodes existed while `min-train-nodes=4`
    waited for 4 that were never created. With `--num-supernodes 4` the same run completes
    on the same 4-core box. More clients than cores is now a note about the ETA being
    optimistic, not a refusal."""
    warnings = preflight(num_clients=1000, skip=False)

    notes = [w for w in warnings if "queue rather than run concurrently" in w]
    assert notes, warnings
    assert all(w.startswith("NOTE") for w in notes)


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


def test_booleans_render_as_lowercase_toml() -> None:
    """flwr parses the assembled --run-config string with tomli, where `False` is not a
    valid value -- the whole string is rejected with a bare "[code: 15]" naming nothing.
    `ablation_all.yaml`'s no_safety_fallback variant is exactly such a cell, so this bug
    would have killed all 15 cells of the one ablation that separates "the colony helped"
    from "the fallback protected it"."""
    import tomllib

    rendered = format_run_config({"aco-safety-fallback": False, "attack": "sign_flip", "n": 3})

    assert "aco-safety-fallback=false" in rendered
    # The real check: flwr's own parser must accept it.
    tomllib.loads("\n".join(part.replace("=", " = ", 1) for part in rendered.split(" ")))


def test_a_variantless_control_does_not_match_another_variants_result(tmp_path: Path) -> None:
    """The `default` control in ablation_all.yaml and `clean` in robustness.yaml set no
    keys, so matching on their own overrides alone matches ANY sibling variant sharing the
    output dir. The control would be marked complete, never run, and every ablation delta
    differenced against another ablation."""
    common = {"num-clients": 4}
    variants = [{"name": "default"}, {"name": "p_none", "aco-persistence": "none"}]
    control = Cell("fedaco", "iid", 0, {}, variant="default")
    ablated = Cell("fedaco", "iid", 0, {"aco-persistence": "none"}, variant="p_none")

    _write_result(tmp_path / "a.json", ablated.run_config(common))

    assert find_result(tmp_path, ablated, common, variants) is not None
    assert find_result(tmp_path, control, common, variants) is None


def test_configure_federation_is_required_not_optional() -> None:
    """num-clients is this project's partitioning key; the Simulation Runtime's own
    supernode count is `num_supernodes` and defaults to 2. Nothing in the repo ever set
    it, so every sweep cell would have run 2 clients while recording num-clients=20 --
    and in robustness.yaml a '10% malicious' cell would have had the entire participating
    federation compromised."""
    import inspect


    source = inspect.getsource(run_sweep.main)
    assert "configure_federation(" in source
    assert "Refusing" in source, "a failed federation setup must abort, not warn"


def test_a_variantless_control_still_matches_its_own_result(tmp_path: Path) -> None:
    """The other half of the discrimination, and the one a key-presence check gets wrong.

    Every variant key is declared in pyproject (it must be, or `flwr run` rejects it), so
    it appears in EVERY resolved run_config carrying its default -- not only in the
    variant that overrides it. Rejecting candidates that merely carry the key therefore
    rejects the control's own result: verified on a live sweep, where `clean` was handed
    its own file, reported "no result file" and would have re-run forever.
    """
    common = {"num-clients": 4}
    variants = [{"name": "clean"}, {"name": "nofb", "aco-safety-fallback": False}]
    control = Cell("fedaco", "iid", 0, {}, variant="clean")
    ablated = Cell("fedaco", "iid", 0, {"aco-safety-fallback": False}, variant="nofb")

    # As flwr resolves them: the declared default is present on the control's run too.
    _write_result(tmp_path / "clean.json", {**control.run_config(common), "aco-safety-fallback": True})
    _write_result(tmp_path / "nofb.json", ablated.run_config(common))

    assert find_result(tmp_path, control, common, variants).name == "clean.json"
    assert find_result(tmp_path, ablated, common, variants).name == "nofb.json"
