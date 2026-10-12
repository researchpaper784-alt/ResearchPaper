"""Phase 8, R6 -- clients joining partway through a run.

The thing worth testing hardest here is not the filtering, it is the refusal. Flower's
`sample_nodes` is `while len(grid.get_node_ids()) < min_available_nodes: sleep(1)` with no
give-up, so a cold start that hides one node too many does not fail -- it logs a line a second
forever. On Kaggle that consumes the whole nine-hour session and writes nothing, which is the
most expensive failure mode this project has available.
"""

from __future__ import annotations

import pytest

from fedswarm.fl.cold_start import (
    EarlyJoinersOnlyGrid,
    cold_start_from_run_config,
    late_node_ids,
    visible_before_join,
    with_cold_start,
)
from fedswarm.strategies.factory import strategy_from_run_config


class FakeGrid:
    """Node ids deliberately not 0..n-1: Flower's are opaque integers assigned by the
    runtime, and anything here that assumed they were partition ids would pass against a
    0-based fake and be wrong in a real run."""

    def __init__(self, n: int, first: int = 100) -> None:
        self.ids = list(range(first, first + n))
        self.messages_created = 0

    def get_node_ids(self):
        return list(self.ids)

    def create_message(self, *args, **kwargs):
        self.messages_created += 1
        return "message"


# ======================================================================================
# Who joins late
# ======================================================================================


@pytest.mark.parametrize(
    ("fraction", "expected"), [(0.0, 0), (0.05, 1), (0.1, 2), (0.25, 5), (0.5, 10)]
)
def test_the_late_set_is_ceil_and_never_silently_empty(fraction: float, expected: int) -> None:
    """`ceil`, so any nonzero fraction holds back at least one client. A "25% join late" run
    that silently held back none would be recorded as cold start costing nothing -- the same
    failure shape `attacks.malicious_ids` uses `ceil` to avoid."""
    assert len(late_node_ids(range(100, 120), fraction)) == expected


def test_the_late_set_is_deterministic_across_calls() -> None:
    """Sampled would mean the same config held back different clients in different rounds of
    one run, so a client could 'join' and then vanish again."""
    ids = list(range(100, 120))
    assert late_node_ids(ids, 0.25) == late_node_ids(reversed(ids), 0.25)


def test_late_joiners_and_attackers_are_disjoint_sets() -> None:
    """`malicious_ids` takes the LOWEST ids and this takes the highest, so a robustness cell
    combining an attack with a cold start delays different clients than it compromises.
    Overlapping them would quietly test one condition twice and report it as two."""
    from fedswarm.fl.attacks import malicious_ids

    num_clients = 20
    attackers = malicious_ids(num_clients, 0.25)
    late = late_node_ids(range(num_clients), 0.25)

    assert attackers and late
    assert not (attackers & late)


# ======================================================================================
# The filtered Grid
# ======================================================================================


def test_the_grid_hides_the_late_joiners_and_passes_everything_else_through() -> None:
    grid = FakeGrid(20)
    proxy = EarlyJoinersOnlyGrid(grid, late_node_ids(grid.get_node_ids(), 0.25))

    assert len(proxy.get_node_ids()) == 15
    assert set(proxy.get_node_ids()) < set(grid.get_node_ids())
    # Messaging is untouched -- only visibility changes. A proxy that swallowed
    # `create_message` would break every strategy silently.
    assert proxy.create_message() == "message"
    assert grid.messages_created == 1


def test_the_clients_are_hidden_before_the_join_round_and_present_after() -> None:
    """The whole point: they exist from round 1, the aggregation just cannot see them. So the
    pheromone meets their ids for the first time at the join round, with every other id
    twenty rounds of deposit ahead."""
    seen = {}

    class _Stub:
        min_available_nodes = 10
        min_train_nodes = 10
        min_evaluate_nodes = 10
        fraction_train = 1.0

        def configure_train(self, server_round, arrays, config, grid):
            seen[server_round] = len(grid.get_node_ids())
            return []

        def configure_evaluate(self, server_round, arrays, config, grid):
            return []

    strategy = with_cold_start(_Stub(), join_round=21, fraction=0.25, num_clients=20)
    for rnd in (1, 20, 21, 100):
        strategy.configure_train(rnd, None, None, FakeGrid(20))

    assert seen[1] == 15
    assert seen[20] == 15, "round 20 is still before a join_round of 21"
    assert seen[21] == 20, "the late joiners must be visible from the join round itself"
    assert seen[100] == 20


def test_evaluation_is_filtered_too() -> None:
    """Leaving evaluate unfiltered would report a late joiner's local-val metrics on a model
    its data never touched, for twenty rounds -- and R6's question is precisely what happens
    when a client first appears."""
    seen = []

    class _Stub:
        min_available_nodes = 10
        min_train_nodes = 10
        min_evaluate_nodes = 10
        fraction_train = 1.0

        def configure_train(self, server_round, arrays, config, grid):
            return []

        def configure_evaluate(self, server_round, arrays, config, grid):
            seen.append(len(grid.get_node_ids()))
            return []

    strategy = with_cold_start(_Stub(), join_round=21, fraction=0.25, num_clients=20)
    strategy.configure_evaluate(5, None, None, FakeGrid(20))
    strategy.configure_evaluate(50, None, None, FakeGrid(20))

    assert seen == [15, 20]


# ======================================================================================
# The refusal -- the expensive failure this module exists to prevent
# ======================================================================================


def test_a_cold_start_that_would_hang_the_run_is_refused_at_construction() -> None:
    """`sample_nodes` waits for nodes that are never coming, one log line a second, forever.
    It does not raise, so nothing downstream can catch it and a Kaggle session dies quietly
    having produced nothing. The only place to stop it is before the run starts."""
    strategy = strategy_from_run_config(
        {"strategy-name": "fedavg", "num-clients": 20,
         "min-train-nodes": 20, "min-available-nodes": 20}
    )

    with pytest.raises(ValueError, match="never gives up"):
        with_cold_start(strategy, join_round=21, fraction=0.25, num_clients=20)


def test_a_workable_cold_start_is_accepted() -> None:
    strategy = strategy_from_run_config(
        {"strategy-name": "fedavg", "num-clients": 20,
         "min-train-nodes": 10, "min-available-nodes": 10}
    )
    wrapped = with_cold_start(strategy, join_round=21, fraction=0.25, num_clients=20)

    assert wrapped.configure_train.__qualname__.startswith("with_cold_start")


def test_the_shipped_r6_config_does_not_hang() -> None:
    """The guard is only useful if the config we ask someone to run actually passes it. This
    builds every strategy R6 declares, exactly as the runner will."""
    import yaml
    from pathlib import Path

    spec = yaml.safe_load(
        (Path(__file__).resolve().parents[1]
         / "configs/experiment/robustness_r6_cold_start.yaml").read_text()
    )
    for entry in spec["strategies"]:
        run_config = {**spec["base_overrides"], **entry.get("overrides", {})}
        strategy_from_run_config(run_config)  # raises if the cold start cannot be satisfied


# ======================================================================================
# Off by default
# ======================================================================================


def test_every_other_sweep_is_untouched() -> None:
    """`cold-start-round=0` must return the strategy itself, not a wrapper. 989 of C's 990
    cells set nothing here."""
    strategy = strategy_from_run_config({"strategy-name": "fedavg", "num-clients": 20})

    assert not strategy.configure_train.__qualname__.startswith("with_cold_start")


def test_a_fraction_of_one_is_refused_rather_than_clamped() -> None:
    """Holding back every client leaves nothing to aggregate. Clamping would run a sweep that
    reports cold start as free, which is the shape of error
    `attacks.attack_from_run_config` raises for."""
    with pytest.raises(ValueError, match="nothing to aggregate"):
        cold_start_from_run_config({"cold-start-round": 21, "cold-start-fraction": 1.0})


def test_visible_before_join_matches_the_late_set() -> None:
    assert visible_before_join(20, 0.25) == 15
    assert visible_before_join(20, 0.0) == 20
