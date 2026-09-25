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
    [("ablation_a[0-9].yaml", "results/fl/ablation"),
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
