"""The FedACO defaults are written down in three places. They must agree.

Until 2026-09-24 they did not, and nothing could tell:

* `pyproject.toml`'s `[tool.flwr.app.config]` is what every `flwr run` uses -- so it is
  what the Kaggle gate runs, and what any experiment YAML listing strategies as bare
  names (`- fedaco`) gets: `main.yaml`, `ablation_all.yaml`, `robustness.yaml`.
* `configs/strategy/fedaco.yaml` is layered on top by `sweep.resolve_entry_overrides`
  for any YAML using `file:` -- `main_client_scale.yaml`, `robustness_r1..r4`, `smoke.yaml`.
* `docs/IMPLEMENTATION_PLAN.md` §14 is the spec both are supposed to encode.

They disagreed on three keys -- `aco-q0` (0.9 vs 0.70), `aco-rho-round` (0.1 vs 0.30) and
`aco-gamma-dispersion` (1.0 vs 0.50). `aco-gamma-dispersion` is a weight *in the fitness*,
so the objective being optimised differed too, not just the search over it. The paper prints
main.yaml's table and main_client_scale.yaml's table as the same method at different K, and
robustness.yaml next to robustness_r1..r4 -- each of those pairs straddles the split.

This is the shape of failure this repository keeps producing: not a wrong value, but two
sources of truth with no check that they match. A value here is cheap to change deliberately;
what this file forbids is changing one copy and not the other.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _flwr_defaults() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["flwr"]["app"]["config"]


def _strategy_file(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs/strategy" / f"{name}.yaml").read_text()) or {}


# ======================================================================================
# pyproject.toml vs configs/strategy/*.yaml
# ======================================================================================


@pytest.mark.parametrize(
    "strategy",
    sorted(p.stem for p in (ROOT / "configs/strategy").glob("*.yaml")),
)
def test_strategy_files_never_silently_disagree_with_the_flwr_defaults(strategy: str) -> None:
    """A strategy file may only *add* keys, never contradict a default.

    A key it repeats with a different value is not an override -- the two runner paths make
    it a coin flip decided by how an unrelated experiment YAML happens to list its
    strategies. A genuine per-strategy value belongs in the strategy file and nowhere else;
    a genuine global default belongs in pyproject.toml and nowhere else.
    """
    defaults = _flwr_defaults()
    overrides = _strategy_file(strategy)

    conflicts = {
        key: (defaults[key], value)
        for key, value in overrides.items()
        if key in defaults and defaults[key] != value and key != "strategy-name"
    }

    assert not conflicts, (
        f"configs/strategy/{strategy}.yaml repeats pyproject.toml keys with different "
        f"values: {conflicts}. Sweeps listing strategies as bare names get the pyproject "
        f"value and sweeps using `file:` get this one, so the same named method runs two "
        f"ways. Set it in exactly one place."
    )


def test_every_strategy_file_key_is_a_real_run_config_key() -> None:
    """A typo'd key in a strategy file is silently ignored by `flwr run` if it is declared,
    and rejected with a bare `[code: 15]` if it is not. Either way the intended value never
    reaches the strategy."""
    defaults = _flwr_defaults()
    unknown = {}
    for path in sorted((ROOT / "configs/strategy").glob("*.yaml")):
        extra = [k for k in (_strategy_file(path.stem)) if k not in defaults]
        if extra:
            unknown[path.name] = extra

    assert not unknown, f"keys not declared in pyproject.toml: {unknown}"


# ======================================================================================
# ...and both against the plan
# ======================================================================================

# Plan §14's `fedaco:` block, mapped to this repo's flat keys. Only the values the plan
# states as scalars; `levels` is checked separately because it is a list there.
PLAN_SECTION_14 = {
    "alpha_exponent_a": ("aco-pheromone-exp", 1.0),
    "beta_exponent_b": ("aco-heuristic-exp", 2.0),
    "rho_evaporation": ("aco-rho", 0.10),
    "q0_exploitation": ("aco-q0", 0.70),
    "n_ants": ("aco-ants-start", 30),
    "n_iterations": ("aco-iters-start", 10),
    "tau_init": ("aco-tau0", 1.0),
    "tau_min": ("aco-tau-min", 0.01),
    "tau_max": ("aco-tau-max", 10.0),
    "pheromone_persistence": ("aco-persistence", "decayed"),
    "rho_round": ("aco-rho-round", 0.30),
    "fitness_mode": ("aco-fitness-mode", "data_free"),
    "gamma1_alignment": ("aco-gamma-alignment", 1.0),
    "gamma2_dispersion": ("aco-gamma-dispersion", 0.50),
    "gamma3_entropy": ("aco-gamma-entropy", 0.10),
    "weight_sum": ("aco-target-sum", 1.0),
    "safety_fallback": ("aco-safety-fallback", True),
}


def test_the_plan_table_this_test_encodes_still_says_what_it_says() -> None:
    """`PLAN_SECTION_14` is a transcription, so it can go stale the moment the plan is
    edited -- which would turn every assertion below into a check against a remembered
    number. Parse §14 and confirm the transcription still matches it.
    """
    plan = (ROOT / "docs/IMPLEMENTATION_PLAN.md").read_text()
    block = plan.split("## 14. Default hyperparameters")[1].split("```")[1]

    parsed = {}
    for line in block.splitlines():
        match = re.match(r"\s*([a-z0-9_]+):\s*([^#\n]+?)\s*(?:#.*)?$", line)
        if match and not match.group(2).startswith("["):
            parsed[match.group(1)] = match.group(2).strip()

    for plan_name, (flat_key, expected) in PLAN_SECTION_14.items():
        assert plan_name in parsed, f"plan §14 no longer lists {plan_name}"
        raw = parsed[plan_name]
        actual: object
        if isinstance(expected, bool):
            actual = raw == "true"
        elif isinstance(expected, str):
            actual = raw
        else:
            actual = type(expected)(float(raw))
        assert actual == expected, (
            f"plan §14 now says {plan_name}: {raw}, but this test file transcribes "
            f"{expected} for {flat_key}. Update PLAN_SECTION_14 deliberately."
        )


@pytest.mark.parametrize(("plan_name", "flat_key", "expected"),
                         [(n, k, v) for n, (k, v) in PLAN_SECTION_14.items()])
def test_flwr_defaults_match_plan_section_14(plan_name: str, flat_key: str,
                                             expected: object) -> None:
    """A6 exists to tune these, so they will move -- but they move in the plan first and in
    both config files together, never in one copy."""
    actual = _flwr_defaults()[flat_key]
    assert actual == expected, (
        f"pyproject.toml has {flat_key} = {actual!r}, plan §14 says {plan_name} = "
        f"{expected!r}. `flwr run` uses the pyproject value, so this is what the Kaggle "
        f"gate and main.yaml actually execute."
    )


def test_the_levels_are_linear_not_log_spaced() -> None:
    """Plan §14 gives the levels as an explicit evenly-spaced list. An earlier version of
    this repo generated them log-spaced, which put 6 of 11 levels below 0.5 and is recorded
    in docs/EXPERIMENT_LOG.md as a real experimental error. `paper/ALGORITHM.md` still
    described them as log-spaced until 2026-09-24."""
    defaults = _flwr_defaults()
    plan = (ROOT / "docs/IMPLEMENTATION_PLAN.md").read_text()
    block = plan.split("## 14. Default hyperparameters")[1].split("```")[1]
    listed = re.search(r"levels:\s*\[([^\]]+)\]", block)
    assert listed, "plan §14 no longer lists levels explicitly"
    values = [float(v) for v in listed.group(1).split(",")]

    assert defaults["aco-level-spacing"] == "linear"
    assert defaults["aco-num-levels"] == len(values)
    assert defaults["aco-level-low"] == min(values)
    assert defaults["aco-level-high"] == max(values)


# ======================================================================================
# The runner split that made the divergence invisible
# ======================================================================================


def test_fedaco_gets_identical_hyperparameters_by_either_runner_path() -> None:
    """The end-to-end version of the first test: build the run_config exactly as each
    runner does and require them to agree on every `aco-*` key.

    `run_sweep.py` expands bare strategy names and never reads configs/strategy/, so
    main.yaml's FedACO is pyproject's. `run_sweep_granular.py` goes through
    `sweep.resolve_entry_overrides`, so robustness_r1..r4's FedACO is the strategy file's.
    """
    from fedswarm.sweep import resolve_entry_overrides

    defaults = _flwr_defaults()
    by_bare_name = {**defaults, "strategy-name": "fedaco"}
    by_file = {**defaults, **resolve_entry_overrides(
        {"name": "fedaco", "file": "configs/strategy/fedaco.yaml"}, base_dir=ROOT
    )}

    differing = {
        key: (by_bare_name.get(key), by_file.get(key))
        for key in set(by_bare_name) | set(by_file)
        if key.startswith("aco-") and by_bare_name.get(key) != by_file.get(key)
    }

    assert not differing, (
        f"main.yaml and robustness_r1..r4 would run different FedACO: {differing}. "
        "The paper prints their tables as one method."
    )


# ======================================================================================
# A6 names its centre arms after the defaults
# ======================================================================================


def test_a6_arms_labelled_default_actually_are_the_default() -> None:
    """A6 is a one-at-a-time sensitivity sweep, so each axis carries the default as its
    centre arm, named `..._default`. If such an arm sets a value that is not the default,
    the axis has two arms doing the same thing, no true centre, and every sensitivity curve
    on it is read against the wrong reference point.

    This is not hypothetical: before the defaults were reconciled, `pyproject.toml` had
    `aco-q0 = 0.9`, so A6's `q0=0.7_default` arm was not the default and its `q0=0.9` arm
    silently was -- 20 of A6's 270 cells measuring one point twice. A6 has not been run yet,
    so nothing was lost; the check exists so that stays true.
    """
    defaults = _flwr_defaults()
    spec = yaml.safe_load((ROOT / "configs/experiment/ablation_a6.yaml").read_text())

    mismatched = {}
    for entry in spec["strategies"]:
        name = entry["name"] if isinstance(entry, dict) else str(entry)
        if not name.endswith("_default") or not isinstance(entry, dict):
            continue
        for key, value in (entry.get("overrides") or {}).items():
            if defaults.get(key) != value:
                mismatched[f"{name}:{key}"] = (value, defaults.get(key))

    assert not mismatched, (
        f"A6 arms named `_default` that are not the default (arm value, real default): "
        f"{mismatched}"
    )


# ======================================================================================
# Every sweep must write where its table target looks
# ======================================================================================


@pytest.mark.parametrize(
    ("pattern", "expected"),
    # `ablation_a[0-9]*` rather than `ablation_a[0-9]`: the reduced variants added on
    # 2026-09-25 are named `ablation_a2_reduced.yaml`, which the exact-length pattern did
    # not match -- so the newest configs were the ones this check did not cover.
    # `ablation_all.yaml` still falls outside it (it is an aggregate config using `common:`)
    # and sets its own output-dir.
    [("ablation_a[0-9]*.yaml", "results/fl/ablation"),
     ("robustness_r*.yaml", "results/fl/robustness")],
)
def test_every_granular_config_sets_its_output_dir(pattern: str, expected: str) -> None:
    """None of the 15 granular configs set `output-dir` until 2026-09-24, so all of them fell
    through to pyproject's `results/fl` -- while `make tables-ablation` reads
    `results/fl/ablation` and `make tables-robustness` reads `results/fl/robustness`.

    C would have run 1,025 cells and then had both table commands find nothing, or worse,
    `tables-ablation` would have produced a plausible-looking table from `ablation_all` alone.
    The three aggregate configs (main/robustness/ablation_all) already set theirs, which is
    how the gap stayed invisible.
    """
    paths = sorted((ROOT / "configs/experiment").glob(pattern))
    assert paths, f"no configs matched {pattern}"
    for path in paths:
        spec = yaml.safe_load(path.read_text()) or {}
        actual = (spec.get("base_overrides") or {}).get("output-dir")
        assert actual == expected, (
            f"{path.name} writes to {actual!r}; its table target reads {expected!r}"
        )


def test_no_sweep_config_silently_inherits_the_smoke_round_count() -> None:
    """pyproject's `num-rounds = 2` is the smoke default. A sweep config that does not
    override it runs 2 rounds per cell and records `status: completed` -- a whole sweep's
    worth of results that look finished and mean nothing."""
    smoke_rounds = _flwr_defaults()["num-rounds"]
    assert smoke_rounds == 2, "this test's premise moved; re-read it"

    missing = []
    for path in sorted((ROOT / "configs/experiment").glob("*.yaml")):
        spec = yaml.safe_load(path.read_text()) or {}
        if not (spec.get("strategies")):
            continue
        block = {**(spec.get("common") or {}), **(spec.get("base_overrides") or {})}
        if "num-rounds" not in block:
            missing.append(path.name)

    # smoke.yaml is the one config that legitimately wants the smoke round count.
    assert [m for m in missing if m != "smoke.yaml"] == [], (
        f"configs that would run {smoke_rounds} rounds per cell: {missing}"
    )


# --------------------------------------------------------------------------------------
# The reduced main sweep (2026-09-25). `main_reduced.yaml` exists because main.yaml is
# 120-240 GPU-h and the project has ~100. Trimming a sweep is legitimate; trimming it
# below the point where its headline claim is *arithmetically* obtainable is not, and the
# difference is invisible in a result file -- every cell says `completed` either way.


def _sweep_spec(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs/experiment" / name).read_text()) or {}


@pytest.mark.parametrize("name", ["main.yaml", "main_reduced.yaml"])
def test_headline_sweeps_keep_the_eight_seed_significance_floor(name: str) -> None:
    """The signed-rank test's p-value floor is set by the pair count alone: 5 seeds floors
    at 0.0625 and cannot clear alpha=0.05 under any data. Both main sweeps print the
    paper's headline comparison, so both need 8."""
    seeds = _sweep_spec(name)["seeds"]
    assert len(seeds) >= 8, (
        f"{name} has {len(seeds)} seeds; the headline comparison needs 8 to be able to "
        "reach p < 0.05 at all (5 seeds floors at p = 0.0625)"
    )


@pytest.mark.parametrize("name", ["main.yaml", "main_reduced.yaml"])
def test_headline_sweeps_keep_fedavg_and_fedaco(name: str) -> None:
    """FedAvg is the reference every claim is stated relative to, and `add_deltas` has
    nothing to subtract without it -- the delta column silently renders as "—", which is
    also what a cell with no paired seeds renders as. FedACO is the method."""
    strategies = _sweep_spec(name)["strategies"]
    names = {s if isinstance(s, str) else s["name"] for s in strategies}
    assert "fedavg" in names, f"{name} dropped the reference strategy"
    assert "fedaco" in names, f"{name} dropped the method under test"


def test_the_reduced_sweep_is_actually_cheaper_than_the_one_it_replaces() -> None:
    """A "reduced" config that is not smaller is a naming trap: someone runs it believing
    they bought headroom. Compares cells x rounds, which is what a GPU-hour buys."""
    def cost(name: str) -> int:
        spec = _sweep_spec(name)
        cells = len(spec["strategies"]) * len(spec["regimes"]) * len(spec["seeds"])
        return cells * spec["common"]["num-rounds"]

    full, reduced = cost("main.yaml"), cost("main_reduced.yaml")
    assert reduced < full / 2, (
        f"main_reduced is {reduced:,} rounds against main's {full:,} -- not the order-of-"
        "magnitude saving the 9-day budget needs"
    )


def test_the_reduced_sweep_keeps_a_monotone_heterogeneity_ladder() -> None:
    """Plan figure 9 (gain vs measured Jensen-Shannon heterogeneity) and claim C1 both
    need more than one point on the heterogeneity axis. A trim that kept three regimes at
    the same heterogeneity would still be three regimes, and would carry neither."""
    regimes = _sweep_spec("main_reduced.yaml")["regimes"]
    kinds = [r["regime"] for r in regimes]
    assert "iid" in kinds, "no low-heterogeneity control"
    dirichlet_alphas = sorted(r["alpha"] for r in regimes if r["regime"] == "dirichlet")
    assert len(dirichlet_alphas) >= 2, (
        f"only {len(dirichlet_alphas)} dirichlet alpha(s): {dirichlet_alphas} -- figure 9 "
        "needs a spread of measured heterogeneity, not a single point plus iid"
    )


# --------------------------------------------------------------------------------------
# C's 9-day reduced set (2026-09-25). R1/R2/A2 full are 240 cells and 100-200 GPU-h; the
# reduced trio is 110 cells / ~23 GPU-h. The trim introduced one coupling that no result
# file would reveal: `robustness_r2_reduced.yaml` has NO clean arm, because
# `robustness_r1_reduced.yaml` writes one to the same directory. Trim R1's clean arm away
# and R2 loses its baseline -- `add_deltas` leaves every delta None, which renders as the
# same "—" as a cell with no paired seeds.

_REDUCED = ("robustness_r1_reduced.yaml", "robustness_r2_reduced.yaml",
            "ablation_a2_reduced.yaml", "ablation_a1_reduced.yaml")


def _granular(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs/experiment" / name).read_text()) or {}


def _cells(spec: dict) -> int:
    return len(spec["strategies"]) * len(spec["partitions"]) * len(spec["seeds"])


def test_r1_reduced_keeps_the_clean_arm_that_r2_reduced_depends_on() -> None:
    r1 = _granular("robustness_r1_reduced.yaml")
    unattacked = [
        p["name"] for p in r1["partitions"]
        if (p.get("overrides") or {}).get("attack") in (None, "none")
    ]
    assert unattacked, (
        "robustness_r1_reduced has no unattacked arm. robustness_r2_reduced deliberately "
        "ships without one and relies on this; removing it leaves both sweeps with no "
        "0%-attacker baseline in results/fl/robustness and every delta empty."
    )


def test_r2_reduced_does_not_duplicate_r1_reduced_clean_arm() -> None:
    """Two arms resolving to the same config in one output directory is the resume bug
    this project hit in September: a control handed a sibling's result file."""
    r2 = _granular("robustness_r2_reduced.yaml")
    clean = [p["name"] for p in r2["partitions"]
             if (p.get("overrides") or {}).get("attack") in (None, "none")]
    assert not clean, f"r2_reduced adds its own clean arm(s) {clean}; r1_reduced already has one"


def test_the_two_reduced_robustness_configs_differ_only_in_the_attack() -> None:
    """R1's clean arm is only a valid baseline for R2's rows if everything except the
    attack matches. They ran at different local-epochs before the trim."""
    r1 = _granular("robustness_r1_reduced.yaml")["base_overrides"]
    r2 = _granular("robustness_r2_reduced.yaml")["base_overrides"]
    attack_keys = {"attack", "attack-fraction", "attack-scale"}
    differing = {
        k for k in set(r1) | set(r2)
        if k not in attack_keys and r1.get(k) != r2.get(k)
    }
    assert not differing, (
        f"r1_reduced and r2_reduced disagree on {sorted(differing)} -- r1's clean arm is "
        "not a valid baseline for r2's rows"
    )


def test_r2_reduced_keeps_both_update_attack_mechanisms() -> None:
    """`gaussian` inflates the update norm; `sign_flip` preserves it and reverses the
    direction. R2 exists to test the norm-ratio heuristic r_k, so dropping either type
    tests the heuristic against half of what it claims to detect. The trim cut attacker
    *fractions* for exactly this reason."""
    attacks = {(p.get("overrides") or {}).get("attack")
               for p in _granular("robustness_r2_reduced.yaml")["partitions"]}
    assert {"gaussian", "sign_flip"} <= attacks, f"r2_reduced only attacks with {attacks}"


def test_a2_reduced_keeps_all_three_persistence_modes() -> None:
    """A2 asks whether cross-round stigmergy helps at all. `decayed` is pyproject's
    default so main_reduced already runs it -- but into results/fl/main, which
    `make tables-ablation` does not read. Two arms and no middle is not an ablation."""
    modes = {(s.get("overrides") or {})["aco-persistence"]
             for s in _granular("ablation_a2_reduced.yaml")["strategies"]}
    assert modes == {"none", "decayed", "full"}, f"a2_reduced runs only {modes}"


@pytest.mark.parametrize(
    ("reduced", "full"),
    [("robustness_r1_reduced.yaml", "robustness_r1_label_flip.yaml"),
     ("robustness_r2_reduced.yaml", "robustness_r2_update_attack.yaml"),
     ("ablation_a2_reduced.yaml", "ablation_a2.yaml")],
)
def test_each_reduced_config_is_cheaper_than_the_one_it_replaces(
    reduced: str, full: str
) -> None:
    r, f = _granular(reduced), _granular(full)
    r_cost = _cells(r) * r["base_overrides"]["num-rounds"] * r["base_overrides"]["num-clients"]
    f_cost = _cells(f) * f["base_overrides"]["num-rounds"] * f["base_overrides"]["num-clients"]
    assert r_cost < f_cost / 2, (
        f"{reduced} is {r_cost:,} client-rounds against {full}'s {f_cost:,} -- not the "
        "saving the 9-day budget needs"
    )


@pytest.mark.parametrize("name", _REDUCED)
def test_reduced_configs_match_main_reduced_where_the_comparison_depends_on_it(
    name: str,
) -> None:
    """Every reduced config's header claims its rows are comparable to main_reduced's.
    That claim is only true if the local work per round is identical -- before the trim,
    main.yaml ran K=20 at 1 local epoch and R1/R2/A2 ran K=20 at 2, so "FedACO under
    attack" and "FedACO clean" differed in training as well as in the attack."""
    main = yaml.safe_load(
        (ROOT / "configs/experiment/main_reduced.yaml").read_text()
    )["common"]
    block = _granular(name)["base_overrides"]
    for key in ("num-clients", "local-epochs", "num-rounds", "image-size",
                "local-lr", "local-batch-size", "model-name", "model-norm"):
        assert block[key] == main[key], (
            f"{name} sets {key}={block[key]!r}, main_reduced uses {main[key]!r} -- the "
            "comparison its header claims does not hold"
        )


# --------------------------------------------------------------------------------------
# Person B's day 1-2 (2026-09-25). `gate_fitness.yaml` decides which fitness fix to use;
# `ablation_a1_reduced.yaml` decides the paper's framing. Roughly 80% of the project's
# remaining compute sits behind the second, so what these two configs measure is not a
# matter of taste.


def test_a1_reduced_keeps_every_search_method() -> None:
    """A1 asks whether the colony beats equal-budget alternatives. Dropping any control is
    dropping a comparison the whole claim rests on, and `coordinate_grid` -- the control
    that beat ACO by the largest margin on the fixture, +0.0501 to +0.0013 -- is the one it
    would be most convenient to lose."""
    full = {(s.get("overrides") or {})["aco-search-method"]
            for s in _granular("ablation_a1.yaml")["strategies"]}
    reduced = {(s.get("overrides") or {})["aco-search-method"]
               for s in _granular("ablation_a1_reduced.yaml")["strategies"]}
    assert reduced == full, f"a1_reduced dropped {sorted(full - reduced)}"


def test_a1_reduced_keeps_eight_seeds_and_both_partitions() -> None:
    """The gate's design is not the cuttable part -- its per-cell cost is. At 5 seeds the
    signed-rank floor of 0.0625 makes "inconclusive, sample too small" arithmetically
    certain, which is the one outcome this experiment cannot afford to report."""
    reduced = _granular("ablation_a1_reduced.yaml")
    assert len(reduced["seeds"]) >= 8, f"{len(reduced['seeds'])} seeds"
    assert len(reduced["partitions"]) == len(_granular("ablation_a1.yaml")["partitions"])
    assert _cells(reduced) == _cells(_granular("ablation_a1.yaml")) == 80


def test_a1_reduced_is_cheaper_per_cell_not_smaller() -> None:
    """The saving has to come from K and local epochs, since the cell count is fixed."""
    def client_epoch_rounds(spec: dict) -> int:
        b = spec["base_overrides"]
        return _cells(spec) * b["num-rounds"] * b["num-clients"] * b["local-epochs"]

    full, reduced = _granular("ablation_a1.yaml"), _granular("ablation_a1_reduced.yaml")
    assert client_epoch_rounds(reduced) < client_epoch_rounds(full) / 2


def test_the_gate_config_keeps_the_control_its_fixes_are_measured_against() -> None:
    """`default` reproduces the +0.6252 corner margin the fixes are compared to. Without it
    a negative margin cannot be attributed to the change rather than to anything else that
    moved since September. `fedavg` is what health check 4 reported as "NO BASELINE"."""
    strategies = _granular("gate_fitness.yaml")["strategies"]
    names = {s["name"] for s in strategies}
    assert "default" in names, "no unmodified FedACO arm to compare the fixes against"
    assert "fedavg" in names, "health check 4 has no baseline without a FedAvg run"

    default = next(s for s in strategies if s["name"] == "default")
    touched = set(default.get("overrides") or {}) - {"strategy-name"}
    assert not touched, f"the `default` arm is not default -- it sets {sorted(touched)}"


def test_the_gate_config_tests_both_candidate_fixes() -> None:
    """Two fixes exist and neither is confirmed on real deltas: raising `aco-gamma-entropy`
    to its closed-form crossing, and switching the dispersion reference to `aggregate`.
    Running one arm makes the gate a confirmation of a guess rather than a comparison."""
    overrides = [s.get("overrides") or {} for s in _granular("gate_fitness.yaml")["strategies"]]
    assert any("aco-gamma-entropy" in o for o in overrides), "no gamma_entropy arm"
    assert any(o.get("aco-dispersion-reference") == "aggregate" for o in overrides), \
        "no aggregate-dispersion arm"


def test_the_gate_config_stays_comparable_to_the_run_it_re_runs() -> None:
    """15 rounds and K=10, matching B's September gate, so `corner_margin` is comparable to
    the +0.6252 on record rather than to nothing."""
    b = _granular("gate_fitness.yaml")["base_overrides"]
    assert b["num-rounds"] == 15, f"{b['num-rounds']} rounds is not the recorded gate's 15"
    assert b["num-clients"] == 10, f"K={b['num-clients']} is not the recorded gate's 10"
    # `make gate-fitness` reads this directory back with check_fedaco_health.py. Left to
    # pyproject's default the runs would land in results/fl beside every other run and the
    # check would average the gate's arms in with whatever else is there.
    assert b["output-dir"] == "results/fl/gate", f"gate writes to {b['output-dir']!r}"
