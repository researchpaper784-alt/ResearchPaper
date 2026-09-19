"""Integration tests for strategies/fedaco.py's FedACO(Strategy) -- exercises the real
Message/ArrayRecord/MetricRecord plumbing end-to-end (constructing replies by hand, the
same pattern tests/test_fl_app.py uses for the ClientApp/ServerApp handlers), not just
the pure-tensor aco/ modules underneath it.
"""

from __future__ import annotations

import time
from collections import OrderedDict

import pytest
import torch
from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict

from fedswarm.aco.colony import ColonyConfig
from fedswarm.aco.fitness import DataFreeFitnessConfig
from fedswarm.aco.heuristics import HeuristicWeights
from fedswarm.strategies.factory import strategy_from_run_config
from fedswarm.strategies.fedaco import FedACO, FedACOConfig
from fedswarm.fl.app import local_train
from fedswarm.models.simple_cnn import SimpleCNN
from torch.utils.data import DataLoader, TensorDataset


def _state(dim_a: int = 20, dim_b: int = 5, seed: int = 0) -> "OrderedDict[str, torch.Tensor]":
    g = torch.Generator().manual_seed(seed)
    return OrderedDict(w=torch.randn(dim_a, generator=g), b=torch.randn(dim_b, generator=g))


def _add(
    state: "OrderedDict[str, torch.Tensor]", delta: "OrderedDict[str, torch.Tensor]"
) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict((k, state[k] + delta[k]) for k in state)


def _make_reply(
    client_id: int,
    state: "OrderedDict[str, torch.Tensor]",
    num_examples: float,
    loss_before: float = 1.0,
    loss_after: float = 0.5,
) -> Message:
    content = RecordDict(
        {
            "arrays": ArrayRecord(state),
            "metrics": MetricRecord(
                {
                    "client_id": client_id,
                    "num-examples": num_examples,
                    "train_loss_before": loss_before,
                    "train_loss_after": loss_after,
                }
            ),
        }
    )
    return Message(content=content, dst_node_id=client_id, message_type="train")


def _new_strategy(**config_kwargs) -> FedACO:
    return FedACO(min_train_nodes=2, min_evaluate_nodes=2, aco_config=FedACOConfig(**config_kwargs))


def test_fedavg_recoverable(monkeypatch: pytest.MonkeyPatch) -> None:
    """test_fedavg_recoverable (plan §4.8 acceptance #2): with pheromone forced
    uniform (guaranteed on a strategy's first round -- fresh Pheromone rows all start
    at tau0), q0=1, and eta forced to peak at lambda=1, FedACO's output equals
    FedAvg's output to within float tolerance."""
    import fedswarm.strategies.fedaco as fedaco_module

    one_index = int(torch.argmin(torch.abs(fedaco_module.level_set(11, 0.0, 2.5) - 1.0)))

    def _one_hot_eta(d_k: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        eta = torch.zeros(d_k.numel(), levels.numel())
        eta[:, one_index] = 1.0
        return eta

    monkeypatch.setattr(fedaco_module, "desirability_matrix", _one_hot_eta)

    global_state = _state(seed=0)
    client_states = [
        _add(global_state, _state(seed=s)) for s in (1, 2, 3)
    ]  # arbitrary distinct "post-training" states
    num_examples = [100.0, 50.0, 200.0]
    replies = [
        _make_reply(i, s, n) for i, (s, n) in enumerate(zip(client_states, num_examples))
    ]

    strategy = _new_strategy(colony=ColonyConfig(q0=1.0), num_rounds=1)
    strategy._current_arrays = ArrayRecord(global_state)
    arrays_out, metrics_out = strategy.aggregate_train(1, list(replies))
    assert arrays_out is not None and metrics_out is not None

    from flwr.serverapp.strategy import FedAvg

    fedavg_arrays, _ = FedAvg().aggregate_train(1, list(replies))

    got = arrays_out.to_torch_state_dict()
    want = fedavg_arrays.to_torch_state_dict()
    for key in want:
        assert torch.allclose(got[key], want[key], atol=1e-4), key


def test_planted_bad_client() -> None:
    """test_planted_bad_client (plan §4.8 acceptance #3): a client whose update is a
    negated, amplified copy of the consensus direction ends up with a weight materially
    below its FedAvg (data-size-only) share."""
    torch.manual_seed(7)
    dim = 300
    consensus = torch.randn(dim) * 5.0

    global_state = OrderedDict(w=torch.zeros(dim))
    good_deltas = [consensus + 0.1 * torch.randn(dim) for _ in range(3)]
    bad_delta = -2.0 * consensus + 0.1 * torch.randn(dim)
    all_deltas = good_deltas + [bad_delta]

    num_examples = [100.0, 100.0, 100.0, 100.0]  # equal FedAvg shares (0.25 each)
    replies = [
        _make_reply(i, OrderedDict(w=global_state["w"] + d), n)
        for i, (d, n) in enumerate(zip(all_deltas, num_examples))
    ]
    bad_index = 3

    strategy = _new_strategy(
        num_rounds=1,
        ants_start=40,
        ants_end=40,
        iters_start=15,
        iters_end=15,
        colony=ColonyConfig(q0=0.9, rho=0.15),
        fitness=DataFreeFitnessConfig(gamma_alignment=1.0, gamma_dispersion=1.0, gamma_entropy=0.1),
        heuristics=HeuristicWeights(),
    )
    strategy._current_arrays = ArrayRecord(global_state)
    _, metrics_out = strategy.aggregate_train(1, list(replies))
    assert metrics_out is not None

    alpha = metrics_out["alpha"]
    fedavg_share = 1.0 / len(num_examples)
    assert alpha[bad_index] < fedavg_share * 0.5, alpha


def test_overhead() -> None:
    """test_overhead (plan §4.8 acceptance #6): ACO wall-clock per round stays below a
    configured fraction (e.g. 5%) of a full round's wall-clock in the smoke config --
    approximated here as ACO time vs. the wall-clock of the local training that would
    happen on the client side that same round, since aggregate_train alone (server-side)
    has no visibility into client-side compute."""
    torch.manual_seed(0)
    model = SimpleCNN(num_classes=4, norm="groupnorm")
    # A real client-sized batch count, not one toy batch -- with only 1-2 batches, wall
    # clock is dominated by Python/dispatch overhead rather than real compute, which
    # makes the 5% bar meaningless in either direction. 16 batches approximates one
    # real local epoch over a small-but-real client partition.
    images = torch.randn(512, 3, 32, 32)
    labels = torch.randint(0, 4, (512,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=32)

    train_start = time.perf_counter()
    local_train(model, loader, torch.device("cpu"), epochs=2, lr=0.01)
    one_client_train_time_s = time.perf_counter() - train_start

    global_state = OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())
    num_clients = 4
    replies = []
    for i in range(num_clients):
        client_model = SimpleCNN(num_classes=4, norm="groupnorm")
        client_model.load_state_dict(global_state)
        local_train(client_model, loader, torch.device("cpu"), epochs=2, lr=0.01)
        replies.append(_make_reply(i, client_model.state_dict(), num_examples=16.0))

    strategy = _new_strategy(num_rounds=1, ants_start=30, ants_end=30, iters_start=10, iters_end=10)
    strategy._current_arrays = ArrayRecord(global_state)
    _, metrics_out = strategy.aggregate_train(1, list(replies))
    assert metrics_out is not None

    aco_time_s = metrics_out["aco_time_ms"] / 1000.0
    round_time_s = num_clients * one_client_train_time_s  # sequential-equivalent, a conservative floor
    # The plan's own 5% bar (§4.8). This previously asserted 15% "to absorb timing
    # noise" without the margin ever having been measured; it is actually ~1.8% in this
    # setup, and 0.26% at the real primary-config scale (K=20, SimpleCNN@112, d=390,404:
    # 145ms of ACO against a 56.8s sequential round, of which 60ms is the one-off Gram
    # precompute -- docs/EXPERIMENT_LOG.md, 2026-09-17). A ~2.8x margin here is enough
    # for wall-clock noise in a shared test environment without weakening the criterion.
    assert aco_time_s < 0.05 * round_time_s, (aco_time_s, round_time_s)


def test_colony_is_reproducible_and_independent_of_global_rng() -> None:
    """The colony's exploration draws (`aco/colony.py::_select_levels`) used to come from
    global torch RNG, because `FedACO.aggregate_train` never passed `run_colony` the
    `generator` it accepts. A seeded run is now a function of (aco_config.seed,
    server_round) alone -- asserted by deliberately churning global RNG between two
    otherwise-identical runs, which under the old behavior changed the result.
    """
    global_state = _state()
    replies = [
        _make_reply(i, _add(global_state, _state(seed=100 + i)), num_examples=float(50 + 10 * i))
        for i in range(4)
    ]

    def run_once(seed: int, server_round: int = 1) -> list[float]:
        strategy = _new_strategy(num_rounds=4, seed=seed, colony=ColonyConfig(q0=0.5))
        strategy._current_arrays = ArrayRecord(global_state)
        _, metrics = strategy.aggregate_train(server_round, list(replies))
        assert metrics is not None
        return list(metrics["alpha"])

    first = run_once(7)
    torch.manual_seed(999)
    torch.randn(5000)  # churn global RNG; the old implementation would drift here
    second = run_once(7)
    assert first == second

    # The seed is load-bearing, not decorative: a different seed explores differently.
    assert run_once(8) != first
    # ...and so is the round index, so a resumed run replays the same trajectory rather
    # than repeating round 1's draws on every round.
    assert run_once(7, server_round=2) != first


def test_run_config_reaches_every_fedaco_knob() -> None:
    """Phase 6's sweep and Phase 7's ablations vary FedACO through `--run-config`; until
    2026-09-17 the factory forwarded only `num-rounds`, so `aco-persistence="none"` (the
    A1 ablation) and the `aco-target-sum` shrinkage sweep had no way in at all."""
    strategy = strategy_from_run_config(
        {
            "strategy-name": "fedaco",
            "num-rounds": 9,
            "seed": 5,
            "aco-persistence": "none",
            "aco-target-sum": 0.75,
            "aco-num-levels": 7,
            "aco-ants-start": 12,
            "aco-q0": 0.5,
            "aco-gamma-dispersion": 0.25,
            "aco-beta-drift": 3.0,
            "aco-safety-fallback": False,
            "aco-concentration-penalty": "gini",
        }
    )
    assert isinstance(strategy, FedACO)
    cfg = strategy.aco_config
    assert cfg.num_rounds == 9 and cfg.seed == 5
    assert cfg.pheromone.persistence == "none"
    assert cfg.target_sum == 0.75 and cfg.num_levels == 7 and cfg.ants_start == 12
    assert cfg.colony.q0 == 0.5
    assert cfg.fitness.gamma_dispersion == 0.25
    assert cfg.heuristics.beta_drift == 3.0
    assert cfg.safety_fallback is False
    # Unspecified knobs keep their documented defaults rather than being zeroed.
    assert cfg.iters_start == FedACOConfig.iters_start
    assert cfg.fitness.normalize_dispersion is True
    assert cfg.fitness_mode == "data_free"
    assert cfg.fitness.concentration_penalty == "gini"


def test_a_mistyped_penalty_shape_is_rejected_at_construction() -> None:
    """Not at first use. `concentration_penalty` runs once per ant per iteration, so an
    unknown value would surface from inside the colony mid-round -- where `flwr run`
    reports it as "Exit Code: 700" while the outer process still exits 0, and a sweep
    counts the cell as simply having produced no result file."""
    with pytest.raises(ValueError, match="aco-concentration-penalty"):
        strategy_from_run_config(
            {"strategy-name": "fedaco", "aco-concentration-penalty": "simpson"}
        )


def test_server_val_fitness_mode_requires_its_dependencies() -> None:
    with pytest.raises(ValueError, match="server_val"):
        FedACO(min_train_nodes=2, aco_config=FedACOConfig(fitness_mode="server_val"))
    with pytest.raises(ValueError, match="client_probe"):
        FedACO(min_train_nodes=2, aco_config=FedACOConfig(fitness_mode="client_probe"))


def test_resumed_colony_uses_the_true_round_not_the_restarted_one() -> None:
    """`Strategy.start()` renumbers its rounds from 1 on every invocation -- the same
    reason build_evaluate_fn needs a round_offset. Without the offset, a run resumed after
    round 50 re-seeds the colony with round 1's exploration draws and restarts the ant
    budget decay from its start-of-run value, so a resumed cell is not the same experiment
    as an uninterrupted one."""
    global_state = _state()
    replies = [
        _make_reply(i, _add(global_state, _state(seed=200 + i)), num_examples=float(40 + i))
        for i in range(3)
    ]

    def alpha_for(server_round: int, offset: int) -> list[float]:
        strategy = _new_strategy(num_rounds=10, seed=3, colony=ColonyConfig(q0=0.5))
        strategy.round_offset = offset
        strategy._current_arrays = ArrayRecord(global_state)
        _, metrics = strategy.aggregate_train(server_round, list(replies))
        assert metrics is not None
        return list(metrics["alpha"])

    # Round 4 reached directly, and round 4 reached as round 1 of a run resumed after 3.
    assert alpha_for(4, 0) == alpha_for(1, 3)
    # And the offset genuinely changes the draw, rather than being ignored.
    assert alpha_for(1, 0) != alpha_for(1, 3)


def test_corner_margin_is_recorded_every_round() -> None:
    """A degenerate fitness optimum is invisible in every other logged field: the colony
    finds the corner, `fallback_used` stays 0 (both sides of that comparison use the same
    fitness), `best_fitness` sits comfortably above `fedavg_fitness`, and alpha looks
    confident. `corner_margin` is the only field that can tell "the colony searched well"
    from "the colony searched well for a useless answer", so a run without it cannot be
    diagnosed after the fact."""
    global_state = _state()
    replies = [
        _make_reply(i, _add(global_state, _state(seed=100 + i)), num_examples=float(50 + 10 * i))
        for i in range(4)
    ]
    strategy = _new_strategy(num_rounds=4)
    strategy._current_arrays = ArrayRecord(global_state)

    _, metrics = strategy.aggregate_train(1, list(replies))

    assert metrics is not None
    assert "corner_margin" in metrics
    assert isinstance(metrics["corner_margin"], float)


# ======================================================================================
# Phase 7, A1 -- the equal-budget search controls
# ======================================================================================


def _a1_replies(global_state):
    return [
        _make_reply(i, _add(global_state, _state(seed=100 + i)), num_examples=float(50 + 10 * i))
        for i in range(4)
    ]


@pytest.mark.parametrize("method", ["aco", "random", "coordinate_grid", "pso", "ga"])
def test_every_search_method_actually_runs_a_round(method: str) -> None:
    """`aco/controls.py` shipped with 14 passing unit tests and no caller: nothing in the
    strategy ever invoked it, so A1 -- the plan's "make-or-break experiment" -- would have
    produced five identical colonies under five different labels. Unit tests on the
    controls cannot catch that; only calling them through `aggregate_train` can."""
    global_state = _state()
    strategy = _new_strategy(num_rounds=4, search_method=method)
    strategy._current_arrays = ArrayRecord(global_state)

    _, metrics = strategy.aggregate_train(1, _a1_replies(global_state))

    assert metrics is not None
    alpha = torch.tensor(list(metrics["alpha"]))
    assert alpha.numel() == 4
    assert float(alpha.sum()) == pytest.approx(1.0, abs=1e-5)


def test_controls_search_differently_from_the_colony() -> None:
    """The substitution A1 exists to detect, in reverse: if a control returned the same
    alpha as the colony, the ablation would report a tie no matter how the method
    performed. Uses a seed where the fallback does not fire, so what is compared is the
    search output rather than the FedAvg point every method falls back to."""
    global_state = _state()
    replies = _a1_replies(global_state)

    alphas = {}
    for method in ("aco", "random", "pso", "ga"):
        strategy = _new_strategy(num_rounds=4, search_method=method, safety_fallback=False)
        strategy._current_arrays = ArrayRecord(global_state)
        _, metrics = strategy.aggregate_train(1, list(replies))
        alphas[method] = tuple(round(float(a), 6) for a in metrics["alpha"])

    assert len(set(alphas.values())) > 1, f"every method returned the same alpha: {alphas}"


@pytest.mark.parametrize("method", ["aco", "random", "coordinate_grid", "pso", "ga"])
def test_no_method_outspends_the_shared_evaluation_budget(method: str) -> None:
    """A1's whole claim is "at equal budget". A control that quietly evaluated twice as
    often would win on effort rather than on search quality, and nothing in the accuracy
    columns would show it. `BudgetedFitness` caps every method at the colony's own
    `ants x iterations`, and the usage is logged so the analysis can check rather than
    assume."""
    global_state = _state()
    strategy = _new_strategy(num_rounds=4, search_method=method)
    strategy._current_arrays = ArrayRecord(global_state)

    _, metrics = strategy.aggregate_train(1, _a1_replies(global_state))

    assert metrics["evaluations_used"] <= metrics["evaluation_budget"]
    assert metrics["evaluation_budget"] > 0
    # A method that spent almost nothing is not running a real search.
    assert metrics["evaluations_used"] > 0.25 * metrics["evaluation_budget"]


def test_a_control_run_reports_no_colony_metrics() -> None:
    """A control has no pheromone. Logging `pheromone_entropy: 0` for one would read to
    every downstream check as a collapsed colony rather than as "not a colony" -- and the
    health check's question 1 would report INERT for a run that never had a colony."""
    global_state = _state()

    colony = _new_strategy(num_rounds=4, search_method="aco")
    colony._current_arrays = ArrayRecord(global_state)
    _, colony_metrics = colony.aggregate_train(1, _a1_replies(global_state))

    control = _new_strategy(num_rounds=4, search_method="random")
    control._current_arrays = ArrayRecord(global_state)
    _, control_metrics = control.aggregate_train(1, _a1_replies(global_state))

    for key in ("pheromone_entropy", "realized_ants", "realized_iterations"):
        assert key in colony_metrics, key
        assert key not in control_metrics, key
    # But the method-agnostic diagnostics stay, or A1 cells become unanalysable.
    for key in ("best_fitness", "fedavg_fitness", "corner_margin", "evaluations_used"):
        assert key in control_metrics, key


def test_a1_config_key_is_declared_in_pyproject() -> None:
    """`flwr run` rejects any undeclared --run-config key with a bare "[code: 15]" naming
    nothing -- and exits 0 while doing it, so a sweep records only "no result file".
    Verified live: every cell of ablation_a1.yaml failed exactly that way."""
    import tomllib
    from pathlib import Path

    declared = tomllib.load(open(Path(__file__).resolve().parents[1] / "pyproject.toml", "rb"))
    assert "aco-search-method" in declared["tool"]["flwr"]["app"]["config"]


# ======================================================================================
# server_round reaches the client -- found in the first real GPU run's metrics
# ======================================================================================


def test_the_wrapper_sets_the_round_number_before_delegating() -> None:
    """The ServerApp builds `train_config` once, before round 1, and never put a round
    number in it. `train_handler` reads `config.get("server_round", -1)`, so every train
    reply in the first real GPU run recorded `server_round = -1`.

    That is a correctness bug, not a cosmetic one, because of R4: the Gaussian mechanism
    seeds its noise with `seed*104729 + partition*1000003 + server_round`. With that term
    pinned at 0, every round drew the SAME noise -- a fixed per-client perturbation the
    model trains around, not DP noise. R4 would have reported the method as far more
    noise-robust than it is, and nothing in the result file would have shown why.
    """
    from flwr.app import ConfigRecord

    from fedswarm.strategies.factory import _with_server_round

    seen: list[int] = []

    class _Stub:
        def configure_train(self, server_round, arrays, config, grid):
            # Reads it back out, so this asserts the assignment happens BEFORE delegation
            # -- a wrapper that set it afterwards would leave the client reading -1.
            seen.append(int(config["server_round"]))
            return []

    stub = _Stub()
    config = ConfigRecord({"local-epochs": 1})
    _with_server_round(stub).configure_train(7, None, config, None)

    assert seen == [7]
    assert config["server_round"] == 7


@pytest.mark.parametrize(
    "strategy_name",
    ["fedavg", "fedprox", "fedadam", "fedyogi", "median", "fedtrimmedavg", "krum",
     "scaffold", "fednova", "loss-based", "fedaco"],
)
def test_every_strategy_the_factory_builds_goes_through_the_wrapper(strategy_name: str) -> None:
    """Parametrized over the BUILT-INS on purpose. Five strategies in this repo override
    `configure_train` and could have been fixed individually; `fedavg`, `fedprox`,
    `fedadam`, `fedyogi`, `median`, `fedtrimmedavg` and `krum` are Flower built-ins
    with no subclass here, and they are every baseline the paper compares against -- so a
    per-subclass fix would have left the comparison arm broken and the method arm correct.

    Checks the wrapper is in place rather than calling it: `configure_train` on a real
    strategy reaches Flower's `sample_nodes`, which blocks waiting for SuperNodes that no
    unit test has. The wrapper's own behaviour is covered by the test above.
    """
    from fedswarm.strategies.factory import strategy_from_run_config

    strategy = strategy_from_run_config({"strategy-name": strategy_name, "num-clients": 2})

    assert getattr(strategy.configure_train, "__qualname__", "").startswith(
        "_with_server_round"
    ), (
        f"{strategy_name} is returned unwrapped, so it never tells the client the round "
        "number and R4's DP noise would be identical in every round"
    )
