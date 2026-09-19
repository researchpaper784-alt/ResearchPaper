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
# training) with no strategies/partitions/seeds grid at all.
#
# Two independent sweep-config schemas coexist in this directory (merge of two
# parallel sessions' Phase 6 work, see docs/OPEN_QUESTIONS.md): this repo's own
# `fedswarm.sweep`-driven shape (`strategies`/`partitions` as file-or-inline-override
# entries, run by `scripts/run_sweep_granular.py`) -- what this lint checks -- and a
# second shape (`strategies` as bare name strings, `regimes` instead of
# `partitions`, `common` instead of `base_overrides`) run by `scripts/run_sweep.py`
# over `fedswarm.utils.runner` (`main.yaml`, `ablation_all.yaml`, `robustness.yaml`).
# Selecting on `partitions` (not just `strategies`, which both shapes have) is what
# keeps the second family out of this lint.
_ALL_EXPERIMENT_FILES = sorted(EXPERIMENT_DIR.glob("*.yaml"))
EXPERIMENT_FILES = [
    p for p in _ALL_EXPERIMENT_FILES if "partitions" in yaml.safe_load(p.read_text())
]


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


def test_every_config_key_is_declared_in_pyproject() -> None:
    """The check that was missing, and the reason every granular ablation and three of
    the robustness sweeps would have failed on contact.

    `flwr run` rejects any `--run-config` key absent from `[tool.flwr.app.config]` with a
    bare `[code: 15] Invalid run configuration` that names no key -- **and exits 0 while
    doing it**, so a sweep runner sees a successful subprocess and simply records "no
    result file". Verified live against `fedaco-search-method`.

    This project has now been bitten by that four separate times: the partition keys, the
    FedACO knobs, every baseline hyperparameter, and then the whole `fedaco-*` family
    that the granular configs used while pyproject declared only `aco-*`. The other
    checks in this file parse the configs and resolve their file references, which all
    passed throughout -- nothing compared the keys against what flwr would accept.
    """
    import tomllib

    declared = set(
        tomllib.load((REPO_ROOT / "pyproject.toml").open("rb"))["tool"]["flwr"]["app"]["config"]
    )
    # Sweep-runner bookkeeping, not run config: these never reach `flwr run --run-config`.
    runner_only = {"output-dir", "name", "file", "overrides"}

    offenders: dict[str, list[str]] = {}
    for path in sorted((REPO_ROOT / "configs").rglob("*.yaml")):
        used: set[str] = set()

        def collect(node) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in ("overrides", "base_overrides") and isinstance(value, dict):
                        used.update(value)
                    collect(value)
            elif isinstance(node, list):
                for value in node:
                    collect(value)

        spec = yaml.safe_load(path.read_text())
        collect(spec)
        if isinstance(spec, dict):
            used.update(spec.get("common") or {})
            for section in ("variants", "regimes"):
                for entry in spec.get(section) or []:
                    used.update(k for k in entry if k != "name")
            # configs/strategy/*.yaml and configs/data/*.yaml are flat run_config maps.
            if path.parent.name in ("strategy", "data"):
                used.update(spec)

        missing = sorted(used - declared - runner_only)
        if missing:
            offenders[str(path.relative_to(REPO_ROOT))] = missing

    assert not offenders, "configs use keys `flwr run` will reject:\n" + "\n".join(
        f"  {name}: {', '.join(keys)}" for name, keys in offenders.items()
    )
