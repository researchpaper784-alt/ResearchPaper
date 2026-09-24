"""Phase 8, R4 tests -- fl/privacy.py's Gaussian mechanism.

The property that matters most is the one easiest to get silently wrong, and it is the
same shape as the attacks' one: a privacy mechanism that does not actually perturb
anything would make every method look perfectly noise-robust, and nothing would error.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from fedswarm.fl.privacy import clip_and_noise, dp_from_run_config


def _states(dim: int = 256, step: float = 0.1):
    torch.manual_seed(0)
    base = OrderedDict(w=torch.randn(dim), b=torch.randn(8))
    local = OrderedDict((k, v + step) for k, v in base.items())
    return base, local


def _update_norm(base, out) -> float:
    return float(
        sum((out[k] - base[k]).pow(2).sum() for k in base) ** 0.5
    )


def test_noise_actually_perturbs_the_update() -> None:
    """The inert-mechanism failure. A no-op here would report every method as perfectly
    DP-robust, with no error anywhere."""
    base, local = _states()

    noised = clip_and_noise(base, local, sigma=0.1, clip_norm=1.0,
                            generator=torch.Generator().manual_seed(0))

    assert not torch.allclose(noised["w"], local["w"])


def test_clipping_bounds_the_update_norm() -> None:
    """Noise calibrated to a bound means nothing if what it is added to is unbounded, so
    a mechanism that only added noise would not be one. Checked with noise off, so this
    measures the clip rather than the clip plus a random draw."""
    base = OrderedDict(w=torch.zeros(100))
    huge = OrderedDict(w=torch.full((100,), 10.0))  # L2 norm 100, far over the bound

    clipped = clip_and_noise(base, huge, sigma=1e-12, clip_norm=2.0,
                             generator=torch.Generator().manual_seed(0))

    assert _update_norm(base, clipped) == pytest.approx(2.0, abs=0.01)


def test_an_update_inside_the_bound_is_not_scaled_up() -> None:
    """"Clip" is one-sided. Scaling a small update *up* to the bound would inject
    magnitude the client never produced, which the norm-ratio heuristic reads as drift."""
    base = OrderedDict(w=torch.zeros(100))
    small = OrderedDict(w=torch.full((100,), 0.001))  # L2 norm 0.01, well inside

    out = clip_and_noise(base, small, sigma=1e-12, clip_norm=5.0,
                         generator=torch.Generator().manual_seed(0))

    assert _update_norm(base, out) == pytest.approx(0.01, abs=1e-4)


def test_sigma_zero_is_a_complete_no_op() -> None:
    """R4's control cell. It has to be byte-identical to an ordinary run, or the control
    is not comparable to the main sweep -- clipping included, which is the documented
    judgement call in `clip_and_noise`."""
    base, local = _states()

    assert clip_and_noise(base, local, sigma=0.0, clip_norm=0.5) is local


def test_more_noise_moves_the_update_further() -> None:
    """Monotonicity: if sigma did not control the perturbation size, R4's sigma sweep
    would be four labels on one experiment."""
    base, local = _states()

    def spread(sigma: float) -> float:
        out = clip_and_noise(base, local, sigma, clip_norm=1.0,
                             generator=torch.Generator().manual_seed(0))
        return float((out["w"] - local["w"]).norm())

    assert spread(0.01) < spread(0.05) < spread(0.2)


def test_it_is_reproducible_under_a_seeded_generator() -> None:
    base, local = _states()

    first = clip_and_noise(base, local, 0.1, 1.0, torch.Generator().manual_seed(7))
    second = clip_and_noise(base, local, 0.1, 1.0, torch.Generator().manual_seed(7))

    assert torch.allclose(first["w"], second["w"])


def test_non_float_and_unmatched_tensors_pass_through() -> None:
    """Same rule as `attacks.poison_update`: integer buffers and SCAFFOLD control
    variates are protocol state, not the model. Noising them is a different mechanism."""
    base = OrderedDict(w=torch.randn(4))
    local = OrderedDict(w=torch.randn(4), counter=torch.tensor([3, 4]), extra=torch.randn(2))

    out = clip_and_noise(base, local, 0.5, 1.0, torch.Generator().manual_seed(0))

    assert torch.equal(out["counter"], local["counter"])
    assert torch.equal(out["extra"], local["extra"])
    assert not torch.equal(out["w"], local["w"])


# ======================================================================================
# Config parsing
# ======================================================================================


def test_defaults_are_mechanism_off() -> None:
    assert dp_from_run_config({}) == (0.0, 1.0)


def test_a_negative_sigma_raises_rather_than_running_clean() -> None:
    """A typo'd `dp-noise-sigma=-0.1` that quietly ran without noise would be recorded as
    a privacy result showing perfect robustness -- the same failure shape
    `attacks.attack_from_run_config` guards against."""
    with pytest.raises(ValueError, match="dp-noise-sigma"):
        dp_from_run_config({"dp-noise-sigma": -0.1})
    with pytest.raises(ValueError, match="dp-clip-norm"):
        dp_from_run_config({"dp-clip-norm": 0.0})


def test_the_r4_keys_are_declared_in_pyproject() -> None:
    """R4's config referenced `dp-noise-sigma` for weeks with no implementation and no
    declaration, so every cell would have died on `flwr run`'s bare `[code: 15]`."""
    import tomllib
    from pathlib import Path

    declared = tomllib.load(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb")
    )["tool"]["flwr"]["app"]["config"]
    assert "dp-noise-sigma" in declared
    assert "dp-clip-norm" in declared
