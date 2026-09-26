"""The gate's verdict decides whether 23 GPU-hours are worth spending.

`gate_fitness.yaml` runs three FedACO arms to settle which fitness fix closes the degenerate
single-client optimum. Reading it wrong is expensive in one specific direction: adopting an arm
whose corner is still open means A1's five search methods all inherit the same useless optimum,
so their ranking says nothing -- and A1 is what the paper's framing rests on.

The rule under test is "negative in EVERY round, not on average". gamma_entropy and the
dispersion shape are set once per run, so an arm that clears the mean round while one round
stays positive is still degenerate there.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = REPO_ROOT / "scripts/apply_gate_fix.py"
    spec = importlib.util.spec_from_file_location("apply_gate_fix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return _module()


def _write(tmp_path: Path, name: str, config: dict, margins: list[float], seed: int = 0) -> None:
    """Write a result file in the layout the ServerApp ACTUALLY writes.

    Built through `fedswarm.utils.results.resolved_config`, the function fl/app.py itself uses.
    This helper used to hand-write `{"seed": ..., "config": <flat settings>}` -- a guess at the
    layout that matched the script's own guess, so every test passed while the script read no
    settings from real files. Going through the production function means a test file cannot
    have a shape production never writes.
    """
    from fedswarm.utils.results import resolved_config

    (tmp_path / name).write_text(json.dumps({
        "run_id": f"{name}_{seed}",
        "config": resolved_config({**config, "seed": seed},
                                  config.get("strategy-name", "fedaco"), seed),
        "rounds": [{"train_corner_margin": m} for m in margins],
    }))


FEDACO = {"strategy-name": "fedaco", "aco-gamma-entropy": 0.1,
          "aco-dispersion-reference": "weighted_mean"}
GAMMA = {**FEDACO, "aco-gamma-entropy": 0.45}
AGG = {**FEDACO, "aco-dispersion-reference": "aggregate"}


def test_an_arm_positive_in_one_round_is_not_adopted(gate, tmp_path) -> None:
    """The expensive mistake. Mean is comfortably negative; one round is not."""
    _write(tmp_path, "default.json", FEDACO, [+0.62] * 5)
    _write(tmp_path, "gamma.json", GAMMA, [-0.4, -0.5, +0.02, -0.45, -0.5])
    chosen, lines = gate.verdict(gate.read_gate(tmp_path))
    assert chosen is None, f"adopted {chosen} despite a positive round"
    text = "\n".join(lines)
    assert "NO arm closed the corner" in text
    assert "OPEN (1/5 rounds)" in text


def test_the_arm_with_the_most_headroom_wins(gate, tmp_path) -> None:
    _write(tmp_path, "default.json", FEDACO, [+0.62] * 5)
    _write(tmp_path, "gamma.json", GAMMA, [-0.01, -0.02, -0.01, -0.03, -0.02])
    _write(tmp_path, "agg.json", AGG, [-0.52, -0.55, -0.49, -0.53, -0.51])
    chosen, lines = gate.verdict(gate.read_gate(tmp_path))
    assert chosen == "dispersion_aggregate", chosen
    assert "also closed the corner: gamma_entropy_0.45" in "\n".join(lines)


def test_a_seed_that_disagrees_blocks_the_arm(gate, tmp_path) -> None:
    """Two seeds exist precisely so a sign that flips between them is caught."""
    _write(tmp_path, "agg_s0.json", AGG, [-0.5] * 5, seed=0)
    _write(tmp_path, "agg_s1.json", AGG, [-0.5, -0.5, +0.1, -0.5, -0.5], seed=1)
    chosen, _ = gate.verdict(gate.read_gate(tmp_path))
    assert chosen is None, "adopted an arm that is degenerate on the second seed"


def test_fedavg_rows_do_not_vote(gate, tmp_path) -> None:
    """FedAvg is in the gate to give health check 4 a baseline, not to be adopted."""
    _write(tmp_path, "fedavg.json", {"strategy-name": "fedavg"}, [])
    _write(tmp_path, "agg.json", AGG, [-0.5] * 5)
    rows = gate.read_gate(tmp_path)
    assert {r["strategy"] for r in rows} == {"fedavg", "fedaco"}
    assert gate.verdict(rows)[0] == "dispersion_aggregate"


def test_an_empty_gate_adopts_nothing(gate, tmp_path) -> None:
    chosen, lines = gate.verdict(gate.read_gate(tmp_path))
    assert chosen is None
    assert "no FedACO result with a corner_margin" in "\n".join(lines)


def test_a_result_predating_corner_margin_is_not_read_as_closed(gate, tmp_path) -> None:
    """A file with no corner_margin at all has zero rounds where the corner won. Counting
    that as CLOSED would adopt an arm on the absence of evidence."""
    from fedswarm.utils.results import resolved_config

    (tmp_path / "old.json").write_text(json.dumps({
        "config": resolved_config(AGG, "fedaco", 0), "rounds": [{"test_macro_f1": 0.5}],
    }))
    chosen, _ = gate.verdict(gate.read_gate(tmp_path))
    assert chosen is None, "adopted an arm whose result file has no corner_margin"


def test_the_patch_targets_pyproject_not_the_strategy_yaml(gate) -> None:
    """`ablation_a1_reduced` and `ablation_a2_reduced` set `strategy-name: fedaco` in
    base_overrides and never read configs/strategy/fedaco.yaml. Patching that file would fix
    the sweeps using `file:` and leave the two that decide the paper on the broken default."""
    import yaml
    for name in ("ablation_a1_reduced.yaml", "ablation_a2_reduced.yaml"):
        spec = yaml.safe_load((REPO_ROOT / "configs/experiment" / name).read_text())
        blob = yaml.safe_dump(spec)
        assert "configs/strategy/fedaco.yaml" not in blob, (
            f"{name} now references the strategy YAML; apply_gate_fix's rationale needs revisiting"
        )

    changes = gate.patch_pyproject("aggregate", dry_run=True)
    assert any("aco-dispersion-reference" in c for c in changes), changes


def test_patching_an_undeclared_key_is_refused(gate, monkeypatch) -> None:
    """`flwr run` rejects any --run-config key not declared in pyproject with a bare
    '[code: 15]', so writing one that is not there would fail at runtime, not here."""
    monkeypatch.setitem(gate.FIXES, "bogus", {"aco-not-a-real-key": 1.0})
    with pytest.raises(SystemExit, match="not declared"):
        gate.patch_pyproject("bogus", dry_run=True)



# --------------------------------------------------------------------------------------
# The first real gate, 2026-09-26, Kaggle T4. The per-run mean margins below are the ones the
# user's run printed. The script labelled all six FedACO runs `default` (it read settings flat
# from a nested layout), pooled them, saw a positive round, and refused to apply any fix. These
# reconstruct that run in the real layout; which arm produced which pair is not recoverable from
# the printout, so the pairing here is illustrative -- the point is that the arms are told apart.


def test_the_first_real_gate_is_read_as_three_arms_not_one(gate, tmp_path) -> None:
    def flat(mean):
        return [mean] * 15

    _write(tmp_path, "d0.json", FEDACO, flat(+0.1629), seed=0)
    _write(tmp_path, "d1.json", FEDACO, flat(+0.1591), seed=1)
    _write(tmp_path, "g0.json", GAMMA, flat(-0.3984), seed=0)
    _write(tmp_path, "g1.json", GAMMA, flat(-0.3597), seed=1)
    _write(tmp_path, "a0.json", AGG, flat(-0.5780), seed=0)
    _write(tmp_path, "a1.json", AGG, flat(-0.5760), seed=1)
    _write(tmp_path, "f0.json", {"strategy-name": "fedavg"}, [], seed=0)

    rows = gate.read_gate(tmp_path)
    labels = {gate.arm_label(r) for r in rows}
    assert labels == {"default", "gamma_entropy_0.45", "dispersion_aggregate", "fedavg"}, labels
    assert {r["seed"] for r in rows} == {0, 1}, "seed not read from config['seed']"

    chosen, lines = gate.verdict(rows)
    assert chosen == "dispersion_aggregate", "\n".join(lines)
    assert "corner OPEN (30/30 rounds)" in "\n".join(lines)  # default, both seeds


def test_a_real_result_file_is_read_with_its_settings(gate, tmp_path) -> None:
    """The exact bug: a file in the production layout must yield its settings, not None."""
    _write(tmp_path, "one.json", AGG, [-0.5] * 15, seed=1)
    (row,) = gate.read_gate(tmp_path)
    assert row["dispersion"] == "aggregate"
    assert row["gamma_entropy"] == 0.1
    assert row["seed"] == 1
    assert row["strategy"] == "fedaco"
