"""`scripts/preflight_sweep_configs.py` decides which arms of a sweep actually execute.

Until 2026-09-25 it ran every strategy against `partitions[0]` and no other partition. So a
partition axis carrying a *code path* -- an attack type, a DP sigma, a cold start hiding
nodes -- was executed for its first value only, and the report printed "ok" for the config
as a whole. 96 arms passed under that rule; the same configs have 137 arms under this one.

The sharp case is `robustness_r2_reduced.yaml`: `gaussian` inflates the update norm and
`sign_flip` preserves it and reverses the direction, which are two different paths through
`aco/heuristics.py`. Only the first ran. And writing `robustness_r1_reduced.yaml` with its
clean arm first would have preflighted label-flipping not at all while reporting 4/4 ok.

This is the project's recurring failure shape rather than a new one: a check that runs,
passes, and means nothing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs/experiment"


def _preflight():
    path = REPO_ROOT / "scripts/preflight_sweep_configs.py"
    spec = importlib.util.spec_from_file_location("preflight_sweep_configs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def preflight():
    return _preflight()


def _spec(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text()) or {}


@pytest.mark.parametrize(
    "name",
    ["robustness_r1_reduced.yaml", "robustness_r2_reduced.yaml",
     "ablation_a2_reduced.yaml", "robustness_r2_update_attack.yaml",
     "robustness_r4_dp_noise.yaml", "ablation_a6.yaml"],
)
def test_every_partition_is_executed_at_least_once(preflight, name: str) -> None:
    """A declared partition the preflight never runs is a config the preflight has not
    checked, however many arms it reports."""
    declared = {(p or {}).get("name") for p in _spec(name).get("partitions") or []}
    if not declared:
        pytest.skip(f"{name} declares no partitions")

    arm_labels = [label for label, _ in preflight.arms(CONFIGS / name)]
    # partitions[0] is implicit in the bare strategy labels; the rest are suffixed.
    covered = {lab.split("/", 1)[1] for lab in arm_labels if "/" in lab}
    first = (_spec(name)["partitions"][0] or {}).get("name")
    covered.add(first)

    missing = declared - covered
    assert not missing, f"{name}: partitions never executed by the preflight: {sorted(missing)}"


def test_r2_reduced_executes_both_attack_mechanisms(preflight) -> None:
    """The regression this fix exists for, asserted on the resolved config rather than on
    labels -- a rename of the partition would slip past a label-only check."""
    attacks = {cfg.get("attack") for _, cfg in preflight.arms(CONFIGS / "robustness_r2_reduced.yaml")}
    assert {"gaussian", "sign_flip"} <= attacks, f"only {attacks} reach the runtime"


def test_r1_reduced_executes_every_strategy_under_attack(preflight) -> None:
    """Krum derives its `f` from the cell's own `attack-fraction`; a mis-derivation is
    invisible with no attackers present. So all four strategies must meet the attack, which
    is why `label_flip_30pct` is `partitions[0]` in that config and not `clean`."""
    attacked = {
        label for label, cfg in preflight.arms(CONFIGS / "robustness_r1_reduced.yaml")
        if cfg.get("attack") == "label_flip"
    }
    declared = {s["name"] for s in _spec("robustness_r1_reduced.yaml")["strategies"]}
    assert declared <= attacked, f"never attacked: {sorted(declared - attacked)}"


def test_extra_partition_arms_are_linear_not_multiplicative(preflight) -> None:
    """A6 has 27 strategy arms. Covering partitions by crossing the axes would make it
    27 x P and put the preflight back into GPU-session territory, which is the thing it
    exists to stand in front of."""
    spec = _spec("ablation_a6.yaml")
    n_strategies = len(spec["strategies"]) * max(1, len(spec.get("variants") or []))
    n_parts = len(spec.get("partitions") or [{}])
    got = len(preflight.arms(CONFIGS / "ablation_a6.yaml"))
    assert got == n_strategies + max(0, n_parts - 1), (
        f"A6 produced {got} arms; expected {n_strategies} + {max(0, n_parts - 1)}"
    )


def test_the_probe_strategy_is_the_method_where_there_is_one(preflight) -> None:
    """Extra partitions are paired with FedACO where it is in the list: the attacked and
    noised partitions exercise `aco/heuristics.py`'s r_k and dispersion terms, so pairing
    them with `fedavg` would execute the attack and skip the code that reads it."""
    extra = [
        (label, cfg) for label, cfg in preflight.arms(CONFIGS / "robustness_r2_update_attack.yaml")
        if "/" in label
    ]
    assert extra, "r2 declares 6 partitions; some should be probed"
    for label, cfg in extra:
        assert label.startswith("fedaco/"), f"{label} probes a partition without FedACO"
        assert cfg.get("strategy-name") == "fedaco", f"{label} resolved to {cfg.get('strategy-name')}"


# --------------------------------------------------------------------------------------
# The resume key. Added 2026-09-25 after the resume added the day before reported four
# arms as "already ran" whose configs had changed underneath it.


def test_the_resume_key_separates_arms_that_resolve_differently(preflight) -> None:
    """Keyed on `config_stem + label` alone, reordering a config's partitions keeps every
    strategy label identical -- so the arms come back "already ran" holding results
    computed under the previous partition. Reordering `robustness_r1_reduced.yaml` to put
    the attacked partition first did exactly that: 4/4 ok, label-flipping never executed.
    """
    from fedswarm.utils.results import make_run_id

    arms = preflight.arms(CONFIGS / "robustness_r1_reduced.yaml")
    attacked = next(cfg for label, cfg in arms if label == "fedavg")
    clean = {**attacked, "attack": "none", "attack-fraction": 0.0}

    fp = lambda cfg: make_run_id(cfg, 0, length=8).rsplit("_", 1)[0]  # noqa: E731
    assert fp(attacked) != fp(clean), (
        "the same strategy label under two partitions fingerprints identically, so resume "
        "cannot tell them apart"
    )


def test_every_arm_of_a_config_gets_a_distinct_resume_key(preflight) -> None:
    """Two arms sharing a key means one is handed the other's result -- and the arm that
    never ran is counted as passed. `ablation_a6` is the case that matters: 27 arms, and
    two of its axes deliberately share the same centre value."""
    from fedswarm.utils.results import make_run_id

    for name in ("ablation_a6.yaml", "robustness_r2_reduced.yaml",
                 "robustness_r1_reduced.yaml", "robustness.yaml"):
        keys = {}
        for label, cfg in preflight.arms(CONFIGS / name):
            fp = make_run_id(cfg, 0, length=8).rsplit("_", 1)[0]
            key = f"{label.replace('/', '_')}__{fp}"
            assert key not in keys, (
                f"{name}: arms {keys.get(key)!r} and {label!r} share resume key {key}"
            )
            keys[key] = label


def test_a_config_named_twice_is_walked_once(preflight) -> None:
    """Naming a config explicitly and then also matching it with a glob is the natural way
    to say "these first, then the rest". Walked twice, the second pass is skipped by resume
    -- but skips count toward `passed`, so the summary read "147 arms ran" for 137 distinct
    arms. A pass count that overstates coverage is this script's own failure mode."""
    overlapping = [
        "configs/experiment/robustness_r1_reduced.yaml",
        "configs/experiment/robustness_r2_reduced.yaml",
        "configs/experiment/robustness_r*.yaml",
    ]
    paths = preflight.resolve_config_paths(overlapping)
    assert len(paths) == len(set(paths)), "the same config is walked more than once"
    # And the explicitly named ones still come first, which is the point of naming them.
    assert paths[0].name == "robustness_r1_reduced.yaml"
    assert paths[1].name == "robustness_r2_reduced.yaml"


def test_resolve_config_paths_refuses_a_pattern_matching_nothing(preflight) -> None:
    with pytest.raises(SystemExit):
        preflight.resolve_config_paths(["configs/experiment/no_such_thing_*.yaml"])
    with pytest.raises(SystemExit):
        preflight.resolve_config_paths(["configs/experiment/no_such_thing.yaml"])
