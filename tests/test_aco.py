"""Unit tests for aco/colony.py, aco/pheromone.py, aco/heuristics.py, aco/schedules.py --
pure tensor math, no Flower runtime needed."""

from __future__ import annotations

import math

import pytest
import torch

from fedswarm.aco.colony import ColonyConfig, run_colony
from fedswarm.aco.fitness import DataFreeFitness, DataFreeFitnessConfig
from fedswarm.aco.gram import precompute_gram
from fedswarm.aco.heuristics import HeuristicWeights, desirability_matrix, desirability_scores
from fedswarm.aco.pheromone import Pheromone, PheromoneConfig
from fedswarm.aco.schedules import colony_budget, level_set


def test_level_set_contains_exactly_one_and_is_sorted() -> None:
    levels = level_set(num_levels=11, low=0.0, high=2.5)
    assert levels.numel() == 11
    assert torch.any(levels == 1.0)
    assert torch.equal(levels, torch.sort(levels).values)
    assert float(levels.min()) == 0.0
    assert float(levels.max()) <= 2.5 + 1e-6


def test_colony_budget_decays_from_start_to_end() -> None:
    ants0, iters0 = colony_budget(0, 10, ants_start=30, ants_end=10, iters_start=10, iters_end=4)
    ants_last, iters_last = colony_budget(9, 10, ants_start=30, ants_end=10, iters_start=10, iters_end=4)
    assert ants0 == 30 and iters0 == 10
    assert ants_last == 10 and iters_last == 4


def test_simplex() -> None:
    """test_simplex (plan §4.8 acceptance #5): the colony's returned alpha is
    non-negative and sums to the configured target within 1e-6."""
    torch.manual_seed(0)
    num_clients, dim = 6, 200
    deltas = torch.randn(num_clients, dim)
    gram = precompute_gram(deltas)
    fitness = DataFreeFitness(gram, DataFreeFitnessConfig())
    levels = level_set(num_levels=7)
    tau0 = torch.full((num_clients, levels.numel()), 1.0)
    eta = torch.ones(num_clients, levels.numel())
    base_weights = torch.full((num_clients,), 1.0 / num_clients)

    for target_sum in (1.0, 0.8):
        result = run_colony(
            tau0, eta, levels, base_weights, fitness.evaluate,
            num_ants=10, num_iterations=5, config=ColonyConfig(), target_sum=target_sum,
        )
        assert torch.all(result.alpha_best >= 0.0)
        assert abs(float(result.alpha_best.sum()) - target_sum) < 1e-6


def test_pheromone_persists() -> None:
    """test_pheromone_persists (plan §4.8 acceptance #4): pheromone state at round t+1
    is a decayed function of round t and is keyed by client id under a changing
    participation set."""
    config = PheromoneConfig(tau0=1.0, rho_round=0.2, persistence="decayed")
    pheromone = Pheromone(num_levels=5, config=config)

    tau_round1 = pheromone.begin_round(["a", "b"])
    assert torch.all(tau_round1 == 1.0)  # fresh rows, all at tau0

    boosted = tau_round1.clone()
    boosted[:, 2] = 5.0  # simulate a colony run depositing pheromone on level index 2
    pheromone.end_round(["a", "b"], boosted)

    # Client "c" is new; "b" drops out this round; "a" returns.
    tau_round2 = pheromone.begin_round(["a", "c"])
    row_a = tau_round2[0]
    row_c = tau_round2[1]

    expected_a = (1.0 - config.rho_round) * boosted[0] + config.rho_round * config.tau0
    assert torch.allclose(row_a, expected_a, atol=1e-6)
    assert torch.all(row_c == config.tau0)  # never-seen client starts fresh at tau0

    # "b" is absent, not forgotten: its row still exists internally and has decayed
    # toward tau0 rather than being reset outright (persistence="decayed", not "none").
    assert "b" in pheromone._rows
    assert not torch.equal(pheromone._rows["b"], boosted[1])
    assert not torch.all(pheromone._rows["b"] == config.tau0)


def test_pheromone_none_persistence_resets_every_round() -> None:
    config = PheromoneConfig(tau0=1.0, persistence="none")
    pheromone = Pheromone(num_levels=4, config=config)
    tau = pheromone.begin_round(["x"])
    boosted = tau.clone()
    boosted[:, 0] = 9.0
    pheromone.end_round(["x"], boosted)

    tau_next = pheromone.begin_round(["x"])
    assert torch.all(tau_next == config.tau0)


def test_desirability_prefers_well_aligned_high_data_client() -> None:
    torch.manual_seed(2)
    dim = 100
    consensus_direction = torch.randn(dim)
    aligned = consensus_direction + 0.01 * torch.randn(dim)
    misaligned = -consensus_direction + 0.01 * torch.randn(dim)
    deltas = torch.stack([aligned, misaligned, aligned * 0.9], dim=0)

    gram = precompute_gram(deltas, trim_fraction=0.0)
    num_examples = torch.tensor([100.0, 100.0, 100.0])
    d_k = desirability_scores(gram, num_examples, weights=HeuristicWeights())

    assert d_k[0] > d_k[1]  # aligned client scores higher desirability than the negated one

    levels = level_set(num_levels=5)
    eta = desirability_matrix(d_k, levels)
    # The misaligned client's peak desirability should sit at a lower multiplier level
    # than the well-aligned client's.
    assert levels[eta[0].argmax()] >= levels[eta[1].argmax()]


def test_fitness_is_scale_invariant_under_normalization() -> None:
    """Regression guard for the dispersion-scale bug (2026-09-17, docs/EXPERIMENT_LOG.md).

    F mixes a cosine in [-1, 1] with a raw sum of squared distances. Normalizing
    dispersion by trace(G)/K makes every term dimensionless, so scaling all client
    updates by a constant -- which is what raising the local learning rate or the number
    of local epochs does -- must leave F exactly unchanged. Without the normalization
    the dispersion term scales as c^2 and swamps the other two.
    """
    torch.manual_seed(0)
    deltas = torch.randn(8, 500)
    alpha = torch.rand(8)
    alpha = alpha / alpha.sum()

    normalized = DataFreeFitnessConfig(normalize_dispersion=True)
    raw = DataFreeFitnessConfig(normalize_dispersion=False)

    for scale in (0.1, 10.0, 100.0):
        f_base = DataFreeFitness(precompute_gram(deltas), normalized).evaluate(alpha)
        f_scaled = DataFreeFitness(precompute_gram(deltas * scale), normalized).evaluate(alpha)
        assert abs(f_scaled - f_base) < 1e-4, (scale, f_base, f_scaled)

    # ...and the un-normalized form is exactly what it is not: kept as an explicit
    # contrast so the test documents the bug, not just the fix.
    f_raw_base = DataFreeFitness(precompute_gram(deltas), raw).evaluate(alpha)
    f_raw_scaled = DataFreeFitness(precompute_gram(deltas * 10.0), raw).evaluate(alpha)
    assert abs(f_raw_scaled - f_raw_base) > 1.0


def test_pheromone_still_carries_signal_when_updates_are_large() -> None:
    """The operational consequence of the bug above: `colony.py` floors deposits at
    `max(F, 0)`, so a fitness driven negative for every candidate deposits nothing at
    all, tau stays uniform across levels, and `tau^a * eta^b` collapses to `eta^b` --
    deterministic heuristic-greedy weighting wearing an ACO costume. Measured on real
    SimpleCNN deltas this happened for every config with lr >= 0.05 or more than one
    local epoch. Assert tau actually develops within-row structure at a delta scale
    that used to kill it.
    """
    torch.manual_seed(0)
    num_clients = 8
    # x100 reproduces the delta magnitude measured at lr=0.1 / 5 local epochs.
    deltas = torch.randn(num_clients, 400) * 100.0
    gram = precompute_gram(deltas)
    fitness = DataFreeFitness(gram, DataFreeFitnessConfig())
    levels = level_set(num_levels=11)
    d_k = desirability_scores(gram, torch.full((num_clients,), 100.0))
    eta = desirability_matrix(d_k, levels)
    pheromone = Pheromone(11, PheromoneConfig())
    tau0 = pheromone.begin_round([str(i) for i in range(num_clients)])

    result = run_colony(
        tau0, eta, levels, torch.full((num_clients,), 1.0 / num_clients), fitness.evaluate,
        num_ants=20, num_iterations=10, config=ColonyConfig(),
        generator=torch.Generator().manual_seed(0),
    )

    row_spread = (result.tau_final.max(dim=1).values - result.tau_final.min(dim=1).values).mean()
    assert float(row_spread) > 1e-3, (
        f"pheromone is uniform across levels (spread={float(row_spread):.3e}) -- the "
        "colony has degenerated to heuristic-only selection"
    )


# ======================================================================================
# The degenerate optimum (2026-09-18)
# ======================================================================================


def _consensus_deltas(num_clients: int, noise: float, seed: int = 0, dim: int = 512):
    """A shared direction plus per-client isotropic noise; `noise` is the heterogeneity."""
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(dim, generator=generator)
    direction /= direction.norm()
    return direction + noise * torch.randn(num_clients, dim, generator=generator) / math.sqrt(dim)


def test_dispersion_is_exactly_zero_at_every_vertex() -> None:
    """The root of the whole problem, and it needs no data to state: dispersion is a
    weighted variance about the weighted mean, and at `alpha = e_j` the mean *is*
    `delta_j`. The term does not become small, it vanishes -- so the fitness pays nothing
    at all for throwing every client but one away."""
    from fedswarm.aco.gram import weighted_dispersion

    gram = precompute_gram(_consensus_deltas(6, noise=1.0))

    for j in range(6):
        assert float(weighted_dispersion(torch.eye(6)[j], gram.gram)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("num_clients,noise", [(2, 1.0), (4, 2.0), (20, 0.5)])
def test_vertex_fitness_has_a_closed_form(num_clients: int, noise: float) -> None:
    """`F(e_j) = gamma_1 * cos(delta_j, robust_mean) - gamma_3 * log K`, exactly, for any
    Gram matrix. `corner_margin` relies on this to be exact and O(K^2) instead of a search
    over the simplex, so it is checked against the implementation with non-default gammas
    -- the defaults would hide a term that had been dropped."""
    from fedswarm.aco.fitness import DataFreeFitness, DataFreeFitnessConfig

    gram = precompute_gram(_consensus_deltas(num_clients, noise))
    config = DataFreeFitnessConfig(gamma_alignment=1.3, gamma_dispersion=0.7, gamma_entropy=0.2)
    fitness = DataFreeFitness(gram, config)
    norms = torch.diagonal(gram.gram).sqrt()
    cosines = gram.g_rob / (norms * math.sqrt(gram.rob_norm_sq))

    for j in range(num_clients):
        predicted = config.gamma_alignment * float(cosines[j]) - config.gamma_entropy * math.log(
            num_clients
        )
        assert fitness.evaluate(torch.eye(num_clients)[j]) == pytest.approx(predicted, abs=1e-5)


def test_corner_margin_agrees_with_evaluating_both_points() -> None:
    """The closed form is an optimization, not a different quantity."""
    from fedswarm.aco.fitness import DataFreeFitness, DataFreeFitnessConfig, corner_margin

    num_clients = 5
    gram = precompute_gram(_consensus_deltas(num_clients, noise=1.5))
    base = torch.full((num_clients,), 1.0 / num_clients)
    config = DataFreeFitnessConfig(gamma_entropy=0.1)
    fitness = DataFreeFitness(gram, config)

    direct = max(
        fitness.evaluate(torch.eye(num_clients)[j]) for j in range(num_clients)
    ) - fitness.evaluate(base)

    assert corner_margin(gram, base, config) == pytest.approx(direct, abs=1e-5)


def test_a_small_heterogeneous_federation_prefers_one_client() -> None:
    """The finding, pinned. At K=2 with the project's default `gamma_entropy=0.1`, the
    fitness ranks a single-client answer above the FedAvg point -- so a colony that
    searches well returns a useless aggregation, while `fallback_used=0` and a
    `best_fitness` well above `fedavg_fitness` both report success.

    This is the regime the project's only FedACO run was in, by way of the
    `num_supernodes=2` bug, and its alpha of [0.99, 0.01] is that optimum rather than a
    search pathology."""
    from fedswarm.aco.fitness import corner_margin

    gram = precompute_gram(_consensus_deltas(2, noise=1.5))

    assert corner_margin(gram, torch.full((2,), 0.5)) > 0


def test_the_guard_strengthens_with_K_and_with_gamma_entropy() -> None:
    """Both stated remedies, checked rather than asserted. The penalty at a vertex is
    `gamma_3 * log K`, so either term buys the same thing -- which is why a sweep at K=20
    can be safe while the K=4 smoke test that is supposed to validate it is not."""
    from fedswarm.aco.fitness import DataFreeFitnessConfig, corner_margin

    small = precompute_gram(_consensus_deltas(2, noise=1.5))
    large = precompute_gram(_consensus_deltas(20, noise=1.5))

    assert corner_margin(large, torch.full((20,), 1 / 20)) < corner_margin(
        small, torch.full((2,), 0.5)
    )
    assert corner_margin(small, torch.full((2,), 0.5), DataFreeFitnessConfig(gamma_entropy=1.0)) < 0


# ======================================================================================
# The concentration penalty's shape
# ======================================================================================


def test_both_penalty_shapes_are_zero_at_uniform_and_maximal_at_a_vertex() -> None:
    """The invariant that makes them comparable at all: each charges nothing for the
    FedAvg point and its full price for a single-client answer."""
    from fedswarm.aco.fitness import concentration_penalty

    for shape, expected_max in (("entropy", math.log(8)), ("gini", 1 - 1 / 8)):
        assert concentration_penalty(torch.full((8,), 1 / 8), shape) == pytest.approx(0.0, abs=1e-6)
        assert concentration_penalty(torch.eye(8)[0], shape) == pytest.approx(expected_max, abs=1e-6)


def test_gini_stays_on_dispersions_scale_as_K_grows_and_entropy_does_not() -> None:
    """Why the shape matters rather than the constant. Dispersion is bounded and sits near
    1 at every K once normalized, so the penalty holding it in check has to be bounded too.
    `log K` is not: by K=50 it charges four times what it charged at K=2 for the same
    degenerate answer, so one `gamma_entropy` cannot be right at both ends."""
    from fedswarm.aco.fitness import concentration_penalty

    entropy = [concentration_penalty(torch.eye(k)[0], "entropy") for k in (2, 50)]
    gini = [concentration_penalty(torch.eye(k)[0], "gini") for k in (2, 50)]

    assert entropy[1] / entropy[0] > 4
    assert gini[1] / gini[0] < 2
    assert all(0 < g < 1 for g in gini)


def test_corner_margin_follows_the_shape() -> None:
    """`corner_margin` computes the vertex side in closed form, so it has to know which
    penalty is in force -- reading `log K` under a "gini" run would report a margin the
    run does not have."""
    from fedswarm.aco.fitness import DataFreeFitness, DataFreeFitnessConfig, corner_margin

    gram = precompute_gram(_consensus_deltas(6, noise=1.5))
    base = torch.full((6,), 1 / 6)

    for shape in ("entropy", "gini"):
        config = DataFreeFitnessConfig(concentration_penalty=shape, gamma_entropy=0.25)
        fitness = DataFreeFitness(gram, config)
        direct = max(fitness.evaluate(torch.eye(6)[j]) for j in range(6)) - fitness.evaluate(base)
        assert corner_margin(gram, base, config) == pytest.approx(direct, abs=1e-5)


def test_an_unknown_penalty_shape_raises_rather_than_defaulting() -> None:
    """Silently falling back to "entropy" would turn a mistyped ablation into a
    valid-looking run of the control -- the same failure mode `aco-persistence` guards
    against."""
    from fedswarm.aco.fitness import concentration_penalty

    with pytest.raises(ValueError, match="Unknown concentration_penalty"):
        concentration_penalty(torch.full((4,), 0.25), "simpson")


def test_the_default_is_still_the_method_as_proposed() -> None:
    """"gini" exists so Phase 7 can measure the choice on real data. Making it the default
    on synthetic evidence would be changing the method under the paper's own description
    of it."""
    from fedswarm.aco.fitness import DataFreeFitnessConfig

    assert DataFreeFitnessConfig().concentration_penalty == "entropy"


# ======================================================================================
# The level set, against the plan's own hyperparameter table
# ======================================================================================


def test_the_default_level_set_is_exactly_the_one_the_plan_specifies() -> None:
    """docs/IMPLEMENTATION_PLAN.md §14 writes the level set out literally. This function
    was log-spaced until 2026-09-19 -- a deviation nobody could catch, because the plan
    was not in the repository. Pinned here so the code and the paper describe the same
    search space."""
    expected = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]

    assert [round(float(x), 6) for x in level_set()] == expected


def test_log_spacing_is_still_reachable_and_materially_different() -> None:
    """Kept rather than deleted: it is what every result before 2026-09-19 used, and A6 --
    the hyperparameter-sensitivity ablation that owns L -- is where the choice belongs.

    The difference is not cosmetic. Log spacing lets one ant construct alpha ratios of
    1000:1, putting a near-vertex weighting like [0.99, 0.01] one greedy draw away; linear
    caps it at 10:1. That changes how reachable the degenerate single-client optimum is,
    so a conclusion about the fitness drawn under one spacing does not transfer to the
    other."""
    linear = [float(x) for x in level_set(spacing="linear") if x > 0]
    log = [float(x) for x in level_set(spacing="log") if x > 0]

    assert max(linear) / min(linear) == pytest.approx(10.0, rel=1e-3)
    assert max(log) / min(log) > 100
    # And log pushes most of the grid below the FedAvg point.
    assert sum(1 for x in level_set(spacing="log") if x < 1.0) > sum(
        1 for x in level_set(spacing="linear") if x < 1.0
    )


@pytest.mark.parametrize("spacing", ["linear", "log"])
def test_fedavg_stays_reachable_under_either_spacing(spacing: str) -> None:
    """lambda=1 is the level that makes an ant's choice recover the FedAvg weight at every
    station. Losing it would remove FedAvg from the search space entirely, which
    `test_fedavg_recoverable` (plan §4.8 acceptance #2) depends on."""
    levels = level_set(spacing=spacing)

    assert torch.any(levels == 1.0)
    assert torch.equal(levels, torch.sort(levels).values)


def test_an_unknown_spacing_raises_rather_than_silently_picking_one() -> None:
    with pytest.raises(ValueError, match="Unknown level spacing"):
        level_set(spacing="geometric")


def test_the_spacing_reaches_the_strategy_from_run_config() -> None:
    """A6 varies this through `--run-config`, so an unwired key would run the default
    under the ablation's label -- the same silent-substitution failure A1 had."""
    from fedswarm.strategies.factory import strategy_from_run_config

    strategy = strategy_from_run_config(
        {"strategy-name": "fedaco", "aco-level-spacing": "log"}
    )

    assert strategy.aco_config.level_spacing == "log"
    assert float(strategy.levels[1]) < 0.01  # the log grid's second level, not 0.25


# ======================================================================================
# required_gamma_entropy -- the number the first GPU run's health check could not name
# ======================================================================================


@pytest.mark.parametrize("num_clients", [2, 4, 10, 20])
@pytest.mark.parametrize("noise", [0.5, 1.5, 3.0])
def test_the_returned_gamma_actually_zeroes_the_margin(num_clients: int, noise: float) -> None:
    """The round trip, and the only guarantee worth having: feed the answer back in and the
    corner stops winning.

    Written this way on purpose rather than asserting the algebra. `required_gamma_entropy`
    derives its closed form from the *claim* that the margin is linear in gamma_entropy --
    true for today's fitness, and exactly the kind of assumption that rots when someone adds
    a term. Checking the returned value against the real `corner_margin` catches that; a
    test of the formula against itself would not.
    """
    from fedswarm.aco.fitness import DataFreeFitnessConfig, corner_margin, required_gamma_entropy

    gram = precompute_gram(_consensus_deltas(num_clients, noise=noise))
    base = torch.full((num_clients,), 1.0 / num_clients)
    config = DataFreeFitnessConfig(gamma_entropy=0.1)

    needed = required_gamma_entropy(gram, base, config)
    at_crossing = corner_margin(gram, base, DataFreeFitnessConfig(gamma_entropy=needed))

    assert at_crossing == pytest.approx(0.0, abs=1e-5), (
        f"K={num_clients} noise={noise}: gamma_entropy={needed} left a margin of "
        f"{at_crossing}, so the reported value does not fix what it claims to"
    )


def test_headroom_puts_the_margin_strictly_negative() -> None:
    """Landing exactly on the crossing leaves the corner *tied* with FedAvg, not beaten --
    and a tie is not a safe place to run 576 cells from, because any drift in the deltas
    puts it back on the wrong side."""
    from fedswarm.aco.fitness import DataFreeFitnessConfig, corner_margin, required_gamma_entropy

    num_clients = 10
    gram = precompute_gram(_consensus_deltas(num_clients, noise=1.5))
    base = torch.full((num_clients,), 1.0 / num_clients)
    config = DataFreeFitnessConfig(gamma_entropy=0.1)

    with_headroom = required_gamma_entropy(gram, base, config, headroom=0.2)
    margin = corner_margin(gram, base, DataFreeFitnessConfig(gamma_entropy=with_headroom))

    assert margin < 0.0
    assert with_headroom > required_gamma_entropy(gram, base, config)


def test_a_non_uniform_base_needs_more_than_the_uniform_estimate() -> None:
    """Why the closed form subtracts `P(base_weights)` rather than using `log K` alone.

    FedACO's reference point is the num-examples-weighted average, not the uniform one, and
    under dirichlet(0.3) those are far apart. A non-uniform base is *already* concentrated,
    so the penalty charges it too -- which shrinks the gap the penalty can open between the
    corner and the reference, and raises the gamma_entropy needed to close it. Reading
    `log K` as the denominator therefore under-reports the requirement, which is the
    direction that matters: it would say the corner is fixed when it is not.
    """
    from fedswarm.aco.fitness import DataFreeFitnessConfig, required_gamma_entropy

    num_clients = 10
    gram = precompute_gram(_consensus_deltas(num_clients, noise=1.5))
    config = DataFreeFitnessConfig(gamma_entropy=0.1)

    uniform = torch.full((num_clients,), 1.0 / num_clients)
    skewed = torch.tensor([0.55, 0.20, 0.10, 0.05, 0.04, 0.03, 0.01, 0.01, 0.005, 0.005])

    assert required_gamma_entropy(gram, skewed, config) > required_gamma_entropy(
        gram, uniform, config
    )


def test_a_degenerate_base_is_reported_as_impossible_not_as_a_big_number() -> None:
    """If one client holds every example, the reference point *is* the corner. The penalty
    charges both identically and no gamma_entropy separates them. Returning `inf` says that;
    returning 1e9 would look like a value someone could set."""
    from fedswarm.aco.fitness import DataFreeFitnessConfig, required_gamma_entropy

    num_clients = 4
    gram = precompute_gram(_consensus_deltas(num_clients, noise=1.5))
    vertex = torch.tensor([1.0, 0.0, 0.0, 0.0])

    assert required_gamma_entropy(gram, vertex, DataFreeFitnessConfig()) == float("inf")


# ======================================================================================
# dispersion_reference -- removing the vertex degeneracy at its root
# ======================================================================================


@pytest.mark.parametrize("num_clients", [4, 10, 20])
def test_a_fixed_reference_charges_a_vertex_what_a_moving_one_does_not(num_clients: int) -> None:
    """The root of the whole problem, stated as the difference between the two modes.

    `weighted_dispersion` is a weighted variance about `Delta(alpha)`, and at `alpha = e_j`
    the mean *is* `delta_j` -- so the term is exactly zero and a single-client answer pays
    nothing. About a fixed reference the same vertex costs `||delta_j - Delta_base||^2`, which
    is large. This is not "small vs smaller": one is zero.
    """
    from fedswarm.aco.gram import dispersion_about_reference, weighted_dispersion

    gram = precompute_gram(_consensus_deltas(num_clients, noise=1.5))
    base = torch.full((num_clients,), 1.0 / num_clients)
    vertex = torch.eye(num_clients)[0]

    assert float(weighted_dispersion(vertex, gram.gram)) == pytest.approx(0.0, abs=1e-9)
    assert float(dispersion_about_reference(vertex, base, gram.gram)) > 0.1


@pytest.mark.parametrize("num_clients", [10, 20])
@pytest.mark.parametrize("noise", [1.5, 3.0])
def test_the_fixed_reference_kills_the_corner_at_the_default_penalty(
    num_clients: int, noise: float
) -> None:
    """Both halves at once, at `gamma_entropy=0.1`, which is what makes this the fix rather
    than another trade.

    Raising `gamma_entropy` to 0.6 also made the margin negative -- and pushed the whole
    landscape down with it: the MAX-MIN rule deposits `rho * Q * max(F, 0)`, so 7 of 15 real
    rounds deposited nothing, tau never moved, and FedACO finished 0.21 macro-F1 behind plain
    FedAvg. A fix has to leave the fitness positive where the colony actually searches.
    """
    from fedswarm.aco.fitness import DataFreeFitness, DataFreeFitnessConfig, corner_margin

    gram = precompute_gram(_consensus_deltas(num_clients, noise=noise), trim_fraction=0.2)
    base = torch.full((num_clients,), 1.0 / num_clients)
    cfg = DataFreeFitnessConfig(gamma_entropy=0.1, dispersion_reference="base")

    assert corner_margin(gram, base, cfg) < 0.0, "the corner still wins"
    # The FedAvg point -- where the colony starts and what it is judged against -- must stay
    # in positive territory, or the deposit rule silently switches the mechanism off.
    assert DataFreeFitness(gram, cfg, reference=base).evaluate(base) > 0.0


def test_base_mode_refuses_to_run_without_a_reference() -> None:
    """A silently-uniform default would remove the vertex degeneracy too, so every diagnostic
    would look repaired while the term measured spread about a point the aggregation never
    uses. That is the failure mode this project keeps finding; refusing is cheaper."""
    from fedswarm.aco.fitness import DataFreeFitness, DataFreeFitnessConfig

    gram = precompute_gram(_consensus_deltas(4, noise=1.5))

    with pytest.raises(ValueError, match="needs `reference`"):
        DataFreeFitness(gram, DataFreeFitnessConfig(dispersion_reference="base"))


def test_corner_margin_under_base_mode_agrees_with_evaluating_the_vertices() -> None:
    """`corner_margin`'s closed form assumes dispersion vanishes at a vertex, which is false
    in this mode -- so it switches to direct evaluation. Same quantity, checked."""
    from fedswarm.aco.fitness import DataFreeFitness, DataFreeFitnessConfig, corner_margin

    num_clients = 6
    gram = precompute_gram(_consensus_deltas(num_clients, noise=1.5))
    base = torch.full((num_clients,), 1.0 / num_clients)
    cfg = DataFreeFitnessConfig(gamma_entropy=0.1, dispersion_reference="base")
    fitness = DataFreeFitness(gram, cfg, reference=base)

    direct = max(
        fitness.evaluate(torch.eye(num_clients)[j]) for j in range(num_clients)
    ) - fitness.evaluate(base)

    assert corner_margin(gram, base, cfg) == pytest.approx(direct, abs=1e-6)


def test_the_default_dispersion_reference_is_still_the_method_as_proposed() -> None:
    """Every result produced before 2026-09-20 used the moving mean. Changing the default
    would silently make them incomparable with anything produced after -- and "base" is
    supported by synthetic deltas so far, which is exactly the evidence that mislead this
    project on the level set."""
    from fedswarm.aco.fitness import DataFreeFitnessConfig

    assert DataFreeFitnessConfig().dispersion_reference == "weighted_mean"
