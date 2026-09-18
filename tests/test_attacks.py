"""Phase 8 tests -- fedswarm.fl.attacks.

The property that matters most here is the one that is easiest to get silently wrong: an
attack that does not actually attack. A robustness sweep whose adversaries were inert
would report every defence as perfect, and nothing would look broken.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from fedswarm.fl.attacks import (
    ATTACKS,
    attack_from_run_config,
    is_malicious,
    malicious_ids,
    poison_labels,
    poison_update,
)


def _states(dim: int = 8, step: float = 0.5):
    torch.manual_seed(0)
    base = OrderedDict(w=torch.randn(dim), b=torch.randn(3))
    local = OrderedDict((k, v + step) for k, v in base.items())
    return base, local


# ======================================================================================
# Who is compromised
# ======================================================================================


def test_malicious_set_is_deterministic() -> None:
    """Resampling the attacker set every round measures something quite different from a
    fixed compromised subset, and a seeded-random choice would make the result depend on a
    seed the analysis never sees."""
    assert malicious_ids(20, 0.2) == malicious_ids(20, 0.2) == {0, 1, 2, 3}


def test_a_nonzero_fraction_always_compromises_someone() -> None:
    """ceil, not round. A '20% malicious' run on 4 clients that silently compromised none
    would read as evidence of robustness."""
    assert malicious_ids(4, 0.2) == {0}
    assert malicious_ids(3, 0.01) == {0}


def test_zero_fraction_compromises_nobody() -> None:
    assert malicious_ids(20, 0.0) == set()
    assert not is_malicious(0, 20, 0.0)


def test_fraction_is_capped_at_every_client() -> None:
    assert malicious_ids(5, 2.0) == {0, 1, 2, 3, 4}


# ======================================================================================
# The attacks actually attack
# ======================================================================================


def test_sign_flip_reverses_the_update() -> None:
    base, local = _states()

    poisoned = poison_update(base, local, "sign_flip", scale=1.0)

    for key in base:
        honest_delta = local[key] - base[key]
        assert torch.allclose(poisoned[key] - base[key], -honest_delta)


def test_scaled_keeps_direction_and_inflates_magnitude() -> None:
    """Magnitude-only: an alignment-based heuristic (FedACO's a_k) sees nothing wrong
    here, and only a magnitude one (r_k) does. That asymmetry is the point of having both
    this and sign_flip."""
    base, local = _states()

    poisoned = poison_update(base, local, "scaled", scale=5.0)

    for key in base:
        honest = local[key] - base[key]
        attacked = poisoned[key] - base[key]
        assert torch.allclose(attacked, 5.0 * honest)
        cosine = torch.nn.functional.cosine_similarity(
            honest.flatten(), attacked.flatten(), dim=0
        )
        assert cosine > 0.999  # same direction
        assert attacked.norm() > honest.norm()


def test_gaussian_destroys_alignment_with_the_honest_update() -> None:
    base, local = _states(dim=512)
    generator = torch.Generator().manual_seed(0)

    poisoned = poison_update(base, local, "gaussian", scale=1.0, generator=generator)

    honest = (local["w"] - base["w"]).flatten()
    attacked = (poisoned["w"] - base["w"]).flatten()
    cosine = torch.nn.functional.cosine_similarity(honest, attacked, dim=0)
    assert abs(float(cosine)) < 0.2


def test_none_is_a_genuine_no_op() -> None:
    """A non-robustness run must behave exactly as it did before attacks existed."""
    base, local = _states()

    assert poison_update(base, local, "none", scale=3.0) is local


def test_gaussian_is_reproducible_under_a_seeded_generator() -> None:
    base, local = _states(dim=64)

    first = poison_update(base, local, "gaussian", 1.0, torch.Generator().manual_seed(7))
    second = poison_update(base, local, "gaussian", 1.0, torch.Generator().manual_seed(7))

    assert torch.allclose(first["w"], second["w"])


def test_non_float_and_unmatched_tensors_pass_through_untouched() -> None:
    """SCAFFOLD control variates and integer buffers are protocol state, not the model.
    Corrupting them would be an attack on the protocol -- a different threat model, and
    conflating the two would make the result uninterpretable."""
    base = OrderedDict(w=torch.randn(4))
    local = OrderedDict(w=torch.randn(4), counter=torch.tensor([3, 4]), extra=torch.randn(2))

    poisoned = poison_update(base, local, "sign_flip", scale=1.0)

    assert torch.equal(poisoned["counter"], local["counter"])  # integer buffer
    assert torch.equal(poisoned["extra"], local["extra"])  # no matching base entry
    assert not torch.equal(poisoned["w"], local["w"])  # the model itself is attacked


# ======================================================================================
# Label poisoning
# ======================================================================================


def test_label_flip_is_a_consistent_lie_not_noise() -> None:
    """A cyclic shift means the client trains to convergence on a coherent but wrong
    mapping -- the realistic compromised-data case. Random per-sample labels would mostly
    stop that client learning at all, which resembles a weak client, not an adversary."""
    labels = torch.tensor([0, 1, 2, 3])

    flipped = poison_labels(labels, num_classes=4)

    assert torch.equal(flipped, torch.tensor([1, 2, 3, 0]))
    # Deterministic: the same input always maps the same way.
    assert torch.equal(poison_labels(labels, 4), flipped)
    # And no label is left correct.
    assert not (flipped == labels).any()


# ======================================================================================
# Config parsing
# ======================================================================================


def test_unknown_attack_raises_rather_than_running_clean() -> None:
    """A typo'd `--run-config "attack='signflip'"` that quietly ran a clean sweep would be
    recorded as a robustness result showing perfect robustness."""
    with pytest.raises(ValueError, match="Unknown attack"):
        attack_from_run_config({"attack": "signflip"})


def test_out_of_range_fraction_raises() -> None:
    with pytest.raises(ValueError, match="attack-fraction"):
        attack_from_run_config({"attack": "sign_flip", "attack-fraction": 1.5})


def test_defaults_are_no_attack() -> None:
    assert attack_from_run_config({}) == ("none", 0.0, 1.0)


def test_every_declared_attack_is_reachable_from_run_config() -> None:
    for attack in ATTACKS:
        name, _, _ = attack_from_run_config({"attack": attack})
        assert name == attack


def test_attack_keys_are_declared_in_pyproject() -> None:
    """`flwr run` rejects any --run-config key absent from [tool.flwr.app.config] with a
    bare `[code: 15]`. An undeclared attack key would make every robustness run fail with
    an error naming nothing -- the same bug class that hit the partition keys and every
    baseline hyperparameter."""
    import tomllib
    from pathlib import Path

    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_bytes()
    declared = set(tomllib.loads(pyproject.decode())["tool"]["flwr"]["app"]["config"])

    for key in ("attack", "attack-fraction", "attack-scale"):
        assert key in declared, f"{key} is not declared in pyproject.toml"


def test_label_flip_preserves_per_epoch_augmentation() -> None:
    """Materializing one augmented pass would give a poisoned client identical images
    every epoch while honest clients get fresh draws, confounding the label-flip effect
    with a reduced-augmentation effect that has nothing to do with poisoning."""
    from fedswarm.fl.app import _poisoned_loader

    class _RandomAugmented(torch.utils.data.Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, index):
            # Stands in for RandomResizedCrop et al: a fresh draw on every read.
            return torch.rand(3, 4, 4), index % 4

    loader = torch.utils.data.DataLoader(_RandomAugmented(), batch_size=8)
    poisoned = _poisoned_loader(loader, num_classes=4)

    first = next(iter(poisoned))[0]
    second = next(iter(poisoned))[0]

    assert not torch.equal(first, second), "augmentation was frozen"


def test_label_flip_does_not_mutate_the_shared_source_dataset() -> None:
    """The source reads from a memory-mapped cache shared with every other client, so
    poisoning in place would poison the honest clients too."""
    from fedswarm.fl.app import _poisoned_loader

    images = torch.zeros(4, 3, 2, 2)
    labels = torch.tensor([0, 1, 2, 3])
    base = torch.utils.data.TensorDataset(images, labels)
    loader = torch.utils.data.DataLoader(base, batch_size=4)

    _, poisoned_labels = next(iter(_poisoned_loader(loader, num_classes=4)))

    assert torch.equal(poisoned_labels.sort().values, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(base.tensors[1], torch.tensor([0, 1, 2, 3])), "source was mutated"
