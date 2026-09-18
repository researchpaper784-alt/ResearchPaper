"""Config lint: every configs/experiment/*.yaml file must parse, every strategy/
partition file: reference must resolve to a real file, and expand_grid must produce
a non-empty, well-formed run list. Catches YAML typos and dangling file references
without needing a live Flower runtime -- exactly the same expansion scripts/
run_sweep.py itself performs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fedswarm.sweep import expand_grid, order_seed_first, resolve_entry_overrides

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_DIR = REPO_ROOT / "configs" / "experiment"

# configs/experiment/centralized*.yaml is a different, older config family
# (scripts/run_experiment.py's OmegaConf-composed layers -- Phase 2, centralized
# training) with no strategies/partitions/seeds grid at all; only the FL sweep
# configs (scripts/run_sweep.py's schema) belong to this lint.
_ALL_EXPERIMENT_FILES = sorted(EXPERIMENT_DIR.glob("*.yaml"))
EXPERIMENT_FILES = [p for p in _ALL_EXPERIMENT_FILES if "strategies" in yaml.safe_load(p.read_text())]


@pytest.fixture(autouse=True, scope="module")
def _guard_nonempty():
    assert EXPERIMENT_FILES, "no FL-sweep configs/experiment/*.yaml files found -- did the schema change?"
    assert len(EXPERIMENT_FILES) < len(_ALL_EXPERIMENT_FILES), (
        "expected at least one non-sweep config (centralized*.yaml) to be filtered out -- "
        "if that file was removed, this assertion (and its comment above) is stale, not a real failure"
    )


@pytest.mark.parametrize("path", EXPERIMENT_FILES, ids=[p.stem for p in EXPERIMENT_FILES])
def test_experiment_config_parses_and_expands(path: Path) -> None:
    with open(path) as f:
        experiment = yaml.safe_load(f)

    for key in ("strategies", "partitions", "seeds"):
        assert key in experiment, f"{path.name} is missing required key {key!r}"
    assert len(experiment["strategies"]) > 0, f"{path.name} has an empty strategies list"
    assert len(experiment["partitions"]) > 0, f"{path.name} has an empty partitions list"
    assert len(experiment["seeds"]) > 0, f"{path.name} has an empty seeds list"

    runs = expand_grid(
        strategies=experiment["strategies"],
        partitions=experiment["partitions"],
        seeds=experiment["seeds"],
        base_overrides=experiment.get("base_overrides", {}),
        group=path.stem,
        base_dir=REPO_ROOT,
    )
    assert len(runs) == len(experiment["strategies"]) * len(experiment["partitions"]) * len(
        experiment["seeds"]
    )

    ordered = order_seed_first(runs)
    assert len(ordered) == len(runs)
    for run in runs:
        assert "seed" in run.overrides
        assert isinstance(run.label, str) and run.label


@pytest.mark.parametrize("path", EXPERIMENT_FILES, ids=[p.stem for p in EXPERIMENT_FILES])
def test_every_file_reference_resolves(path: Path) -> None:
    with open(path) as f:
        experiment = yaml.safe_load(f)

    for entry in experiment["strategies"] + experiment["partitions"]:
        overrides = resolve_entry_overrides(entry, base_dir=REPO_ROOT)
        assert isinstance(overrides, dict)
        if "file" in entry:
            assert (REPO_ROOT / entry["file"]).exists(), f"{entry['file']} referenced by {path.name} does not exist"


def test_every_strategy_config_file_sets_strategy_name() -> None:
    for path in (REPO_ROOT / "configs" / "strategy").glob("*.yaml"):
        with open(path) as f:
            overrides = yaml.safe_load(f)
        assert "strategy-name" in overrides, f"{path.name} never sets strategy-name"


def test_every_data_config_file_sets_regime() -> None:
    for path in (REPO_ROOT / "configs" / "data").glob("*.yaml"):
        with open(path) as f:
            overrides = yaml.safe_load(f)
        assert "regime" in overrides, f"{path.name} never sets regime"
