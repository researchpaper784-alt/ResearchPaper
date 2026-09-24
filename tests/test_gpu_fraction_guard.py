"""A sweep that trains on CPU on a GPU box is the most expensive failure available here.

Ray hides accelerators from an actor requested with `num_gpus=0`. From the installed Ray
2.55.1 (`ray/_private/worker.py`):

    override_on_zero = env_bool(RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO_ENV_VAR, True)
    ... warns that *future* versions "will no longer override accelerator visible devices
        env var if num_gpus=0 or num_gpus=None (default)"

-- so the current version does override, and a ClientApp actor with no GPU fraction sees
`torch.cuda.is_available() == False`.

**Why it must refuse rather than warn.** The ServerApp runs in the driver process, not a Ray
actor, so it keeps the GPU. Server-side evaluation, `test_macro_f1` and every logged metric
look normal; the only symptom is wall-clock, checked against a cost projection that a
4-core box legitimately makes optimistic anyway. For `main.yaml` that is 576 cells x 100
rounds of client training at CPU speed -- it does not finish inside any Kaggle quota, and no
result file says why.

`make main` passed no `--gpus-per-client` and the flag defaulted to 0.0, so this was the
shipped behaviour until 2026-09-24.

These tests simulate a GPU, because the box they run on does not have one -- which is
precisely why the bug survived: it cannot reproduce anywhere CI or a developer runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fedswarm.sweep import gpu_fraction_problem  # noqa: E402


@pytest.fixture
def pretend_gpu(monkeypatch: pytest.MonkeyPatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


@pytest.fixture
def pretend_no_gpu(monkeypatch: pytest.MonkeyPatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


# ======================================================================================
# The refusal
# ======================================================================================


def test_unset_fraction_on_a_gpu_box_is_refused(pretend_gpu) -> None:
    problem = gpu_fraction_problem(None, num_clients=20)
    assert problem is not None
    assert "train on CPU" in problem


def test_the_refusal_names_a_fraction_that_fits_every_client(pretend_gpu) -> None:
    """A refusal that does not say what to pass costs a Kaggle session to act on."""
    problem = gpu_fraction_problem(None, num_clients=20)
    assert "--gpus-per-client 0.05" in problem, problem
    # 1/20 lets all twenty ClientApps be resident on one card.
    assert "0.05" == str(round(1.0 / 20, 4))


def test_the_refusal_offers_the_deliberate_cpu_escape(pretend_gpu) -> None:
    assert "--gpus-per-client 0 if you really do want CPU" in gpu_fraction_problem(None, 20)


# ======================================================================================
# What must NOT be refused
# ======================================================================================


def test_an_explicit_zero_is_honoured_even_on_a_gpu_box(pretend_gpu) -> None:
    """Someone who asks for CPU gets CPU. Only the unset case is a mistake, and argparse
    cannot tell them apart unless the default is None -- which is why it is."""
    assert gpu_fraction_problem(0.0, num_clients=20) is None


def test_a_real_fraction_is_fine(pretend_gpu) -> None:
    assert gpu_fraction_problem(0.05, num_clients=20) is None


def test_a_cpu_box_is_never_blocked(pretend_no_gpu) -> None:
    """CI and every developer machine here have no GPU. If this returned a problem, the
    guard would block the only environments that can currently run a sweep at all."""
    assert gpu_fraction_problem(None, num_clients=20) is None


def test_a_missing_torch_does_not_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """No torch means no GPU training either way; the guard must not turn an import problem
    into a refusal."""
    import builtins

    real_import = builtins.__import__

    def explode(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", explode)
    assert gpu_fraction_problem(None, num_clients=20) is None


# ======================================================================================
# Both runners are wired to it
# ======================================================================================


def test_run_sweep_preflight_reports_the_problem(pretend_gpu) -> None:
    from run_sweep import preflight

    warnings = preflight(num_clients=20, skip=False, gpus_per_client=None)
    assert any("train on CPU" in w for w in warnings)
    # It must be fatal, not a NOTE: run_sweep.py treats any warning not starting with
    # "NOTE" as a reason to refuse.
    assert not any(w.startswith("NOTE") for w in warnings if "train on CPU" in w)


def test_run_sweep_preflight_is_quiet_when_the_fraction_is_set(pretend_gpu) -> None:
    from run_sweep import preflight

    warnings = preflight(num_clients=20, skip=False, gpus_per_client=0.05)
    assert not any("train on CPU" in w for w in warnings)


@pytest.mark.parametrize("script", ["run_sweep.py", "run_sweep_granular.py"])
def test_both_runners_default_the_flag_to_none_not_zero(script: str) -> None:
    """The whole guard rests on this: with `default=0.0` argparse reports 0.0 whether or not
    the user passed anything, so the unset case becomes indistinguishable from a deliberate
    CPU run and cannot be refused."""
    source = (ROOT / "scripts" / script).read_text()
    block = source.split('"--gpus-per-client"')[1].split(")")[0]
    assert "default=None" in block, f"{script} must use a None sentinel, got: {block}"


@pytest.mark.parametrize("script", ["run_sweep.py", "run_sweep_granular.py"])
def test_both_runners_resolve_none_before_calling_configure_federation(script: str) -> None:
    """`configure_federation` takes a float and compares it with `>`, so a None reaching it
    would raise mid-sweep rather than at argument parsing."""
    source = (ROOT / "scripts" / script).read_text()
    assert "args.gpus_per_client or 0.0" in source
