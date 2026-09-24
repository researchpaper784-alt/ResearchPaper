"""Phase 8, R6 -- clients that join the federation partway through a run.

R6 is the one test in the plan's robustness table (§8) that had no config file at all, and
docs/OPEN_QUESTIONS.md recorded why: which SuperNodes exist is Flower's Simulation Runtime's
business (Ray actor lifecycle), and nothing `run_config` can express says "partition 15 must
not participate before round 20".

That is true, and it is also not the only way in. A strategy never asks the runtime for nodes
directly -- it asks the `Grid` it is handed. So the runtime can start every SuperNode at round
1 and a **filtered Grid** can keep a deterministic subset out of the strategy's view until the
join round. The clients exist; the aggregation simply does not see them, which is exactly the
condition R6 is about: does the pheromone initialise sensibly for a client id it meets for the
first time at round 20, when every other id has twenty rounds of deposit behind it?

`Pheromone.begin_round` already initialises a fresh row for any unseen client
(`test_pheromone_persists`), so the mechanism is built. What was missing was a way to make it
happen on purpose.

⚠️ **The hazard here is a hang, not an error, and it is why this module validates rather than
trusts.** Flower's own `sample_nodes` is

    while len(all_nodes := list(grid.get_node_ids())) < min_available_nodes:
        sleep(1)

an unbounded wait. Hide enough nodes that fewer remain than `max(min_available_nodes,
sample_size)` and the run does not fail -- it logs one line a second forever, which on a
Kaggle session silently burns the entire nine-hour budget and produces nothing. So the
visible count is checked against the strategy's own minimums up front, and a configuration
that cannot satisfy them raises at construction.
"""

from __future__ import annotations

import math
from typing import Iterable

from flwr.serverapp.strategy import Strategy


def late_node_ids(node_ids: Iterable[int], fraction: float) -> set[int]:
    """The deterministic subset of nodes that joins late: the highest ids.

    Deterministic rather than sampled, for the same reason `attacks.malicious_ids` is: every
    run at the same settings holds back the same clients, and a result file identifies them
    without needing a seed the analysis does not have. `ceil` so any nonzero fraction holds
    back at least one -- a "25% join late" run that silently held back none would read as
    evidence that cold start costs nothing.

    Taking the *highest* ids rather than the lowest keeps this disjoint from
    `malicious_ids`, which takes the lowest. A robustness cell combining an attack with a
    cold start then compromises early joiners and delays different clients, instead of
    quietly testing one thing twice.
    """
    if fraction <= 0.0:
        return set()
    ordered = sorted(node_ids)
    if not ordered:
        return set()
    count = min(len(ordered), math.ceil(fraction * len(ordered)))
    return set(ordered[len(ordered) - count :])


class EarlyJoinersOnlyGrid:
    """A `Grid` that hides the late joiners, delegating everything else untouched.

    A proxy rather than a subclass: `Grid` is abstract and the concrete class the runtime
    passes in is an implementation detail of `flwr.serverapp`, so wrapping the instance is
    the only way to do this without depending on which one it is.
    """

    def __init__(self, grid, hidden: set[int]) -> None:
        self._grid = grid
        self._hidden = hidden

    def get_node_ids(self):
        return [n for n in self._grid.get_node_ids() if n not in self._hidden]

    def __getattr__(self, name):
        # Everything else -- create_message, push_messages, send_and_receive, run, set_run --
        # goes straight through. Only visibility is changed, not messaging.
        return getattr(self._grid, name)


def visible_before_join(num_clients: int, fraction: float) -> int:
    """How many clients the strategy can see before the join round."""
    return num_clients - len(late_node_ids(range(num_clients), fraction))


def with_cold_start(
    strategy: Strategy, join_round: int, fraction: float, num_clients: int
) -> Strategy:
    """Hide a deterministic subset of clients from `strategy` until `join_round`.

    `join_round` is the first round at which the late joiners participate, so `join_round=21`
    means "twenty rounds without them". 0 or less disables the wrapper entirely and returns
    the strategy untouched, which is what every non-R6 sweep needs.

    Raises rather than hanging when the early-joiner count cannot satisfy the strategy's own
    minimums -- see the module docstring: Flower's sampler waits forever rather than failing,
    so the only place this can be caught is before the run starts.
    """
    if join_round <= 0 or fraction <= 0.0:
        return strategy

    visible = visible_before_join(num_clients, fraction)
    needed = max(
        int(getattr(strategy, "min_available_nodes", 0)),
        int(getattr(strategy, "min_train_nodes", 0)),
        int(getattr(strategy, "min_evaluate_nodes", 0)),
        math.ceil(visible * float(getattr(strategy, "fraction_train", 1.0))),
    )
    if visible < needed:
        raise ValueError(
            f"cold start holds back {num_clients - visible} of {num_clients} clients until "
            f"round {join_round}, leaving {visible} visible, but this strategy needs at "
            f"least {needed} (min_available_nodes / min_train_nodes / min_evaluate_nodes). "
            f"Flower's sample_nodes waits for them in a `while ...: sleep(1)` loop that "
            f"never gives up, so this would hang the run rather than fail it -- on a Kaggle "
            f"session, silently, for the whole budget. Lower cold-start-fraction or lower "
            f"the min-*-nodes for this sweep."
        )

    original_train = strategy.configure_train
    original_evaluate = strategy.configure_evaluate

    def _filtered(grid, server_round: int):
        if server_round >= join_round:
            return grid
        return EarlyJoinersOnlyGrid(grid, late_node_ids(grid.get_node_ids(), fraction))

    def configure_train(server_round, arrays, config, grid):  # type: ignore[no-untyped-def]
        return original_train(server_round, arrays, config, _filtered(grid, server_round))

    def configure_evaluate(server_round, arrays, config, grid):  # type: ignore[no-untyped-def]
        # Evaluation is filtered too. Leaving it unfiltered would evaluate on clients the
        # aggregation never trained with, so a late joiner's local-val split would be
        # reporting on a model it had no part in -- and R6's whole question is what happens
        # when a client first appears.
        return original_evaluate(server_round, arrays, config, _filtered(grid, server_round))

    strategy.configure_train = configure_train  # type: ignore[method-assign]
    strategy.configure_evaluate = configure_evaluate  # type: ignore[method-assign]
    return strategy


def cold_start_from_run_config(run_config: dict) -> tuple[int, float]:
    """`(join_round, fraction)`, validated. `join_round <= 0` means the wrapper is off.

    Raises on an out-of-range fraction rather than clamping, the same rule
    `attacks.attack_from_run_config` follows: a typo'd `cold-start-fraction=1.5` that quietly
    held back every client would be recorded as a cold-start result, and one that held back
    none would be recorded as cold start costing nothing.
    """
    join_round = int(run_config.get("cold-start-round", 0))
    fraction = float(run_config.get("cold-start-fraction", 0.0))
    if not 0.0 <= fraction < 1.0:
        raise ValueError(
            f"cold-start-fraction must be in [0, 1), got {fraction} -- holding back every "
            "client leaves nothing to aggregate"
        )
    return join_round, fraction
