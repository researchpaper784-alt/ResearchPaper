"""CI's smoke job asserted `status == "completed"` and nothing else -- which is precisely the
assertion the GPU device bug satisfied. Every case here is a bug this repository shipped, so
these are regression tests for the *checker*, not hypotheticals.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_smoke_result import check_complete, check_reproduces  # noqa: E402


def _healthy() -> dict:
    return {
        "status": "completed",
        "rounds": [
            {"round": 1, "test_macro_f1": 0.2, "train_server_round": 1.0,
             "client_eval_accuracy": 0.25, "client_eval_glioma_tp": 0.0},
            {"round": 2, "test_macro_f1": 0.3, "train_server_round": 2.0,
             "client_eval_accuracy": 0.50, "client_eval_glioma_tp": 1.0},
        ],
        "final": {"final_test_macro_f1": 0.3, "final_val_macro_f1": 0.3,
                  "num_rounds_completed": 2},
        # Populated because `write_result` accepted this field from the day the Phase 6
        # schema was written and no caller ever passed it, so it was null in every result
        # file ever produced -- which removed plan §9.2's figure 9 for want of an x-axis.
        "partition_stats": {"js_divergence": 0.32, "size_gini": 0.17,
                            "client_sizes": [113, 72, 121, 54]},
    }


def test_a_healthy_result_passes() -> None:
    """A checker that fires on everything gets disabled, so this matters as much as the rest."""
    assert check_complete(_healthy()) == []


def test_the_device_bug_is_caught() -> None:
    """On Kaggle every client's evaluation failed -- model on CPU, batches on GPU -- Flower
    printed `Aggregated ClientApp-side Evaluate Metrics: {}`, and the run wrote a result
    marked `completed` that looked identical to a healthy one. Three GPU sessions went past."""
    payload = _healthy()
    for entry in payload["rounds"]:
        for key in [k for k in entry if k.startswith("client_eval_")]:
            entry.pop(key)

    failures = check_complete(payload)

    assert failures and "client-side evaluation" in failures[0]


def test_a_round_number_that_never_arrived_is_caught() -> None:
    """The ServerApp never put the round number in the train config, so every reply recorded
    -1 -- and R4's DP noise, seeded partly from it, drew the same values every round."""
    payload = _healthy()
    payload["rounds"][-1]["train_server_round"] = -1.0

    failures = check_complete(payload)

    assert failures and "train_server_round is -1" in failures[0]


def test_a_run_that_aggregated_nothing_is_caught() -> None:
    payload = _healthy()
    payload["rounds"] = []

    assert "aggregated nothing" in check_complete(payload)[0]


@pytest.mark.parametrize("missing", ["final_test_macro_f1", "num_rounds_completed"])
def test_an_empty_final_block_is_caught(missing: str) -> None:
    payload = _healthy()
    payload["final"].pop(missing)

    assert check_complete(payload)


def test_two_runs_that_disagree_at_the_same_seed_are_caught() -> None:
    """The exact numbers two Kaggle runs of an identical FedAvg config returned."""
    first = _healthy()
    second = copy.deepcopy(first)
    second["final"]["final_test_macro_f1"] = 0.3286
    first["final"]["final_test_macro_f1"] = 0.1363

    failures = check_reproduces(first, second)

    assert failures and "not being seeded deterministically" in failures[0]


def test_identical_runs_agree() -> None:
    payload = _healthy()

    assert check_reproduces(payload, copy.deepcopy(payload)) == []


def test_the_comparison_is_exact_by_default() -> None:
    """Not a tolerance band. Both runs execute on the same machine, same build, same seed, so
    there is no float drift to absorb and any difference is a real determinism bug. A
    tolerance would hide exactly the failure this exists for -- the shipped reproducibility
    band was 0.15 wide while the variance it was meant to absorb was 0.19."""
    first = _healthy()
    second = copy.deepcopy(first)
    second["final"]["final_test_macro_f1"] += 1e-9

    assert check_reproduces(first, second)


def test_ci_actually_invokes_this_checker() -> None:
    """The recurring defect in this repo is a capability nothing calls. A checker CI does not
    run is worth less than no checker, because it reads as coverage."""
    workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml").read_text()

    assert "scripts/check_smoke_result.py" in workflow
    assert "--compare-against" in workflow, "the determinism half must run too"


def test_a_null_partition_stats_is_caught() -> None:
    """`write_result` has taken a `partition_stats` argument since the Phase 6 schema was
    written and nothing ever passed it. Every result file recorded `null` for the measured
    heterogeneity of its own partition, and plan §9.2's figure 9 -- gain against measured
    Jensen-Shannon divergence -- had no x-axis as a result. Nothing failed; the figure was
    simply never wired, which is how a missing field costs a figure."""
    payload = _healthy()
    payload["partition_stats"] = None
    failures = check_complete(payload)
    assert any("partition_stats" in f for f in failures)


def test_a_partition_stats_error_is_reported_rather_than_accepted() -> None:
    """The collector is wrapped so a diagnostics failure cannot lose a completed run, which
    means it can return an error marker instead. Treating that as present would restore the
    silence this check exists to break."""
    payload = _healthy()
    payload["partition_stats"] = {"error": "FileNotFoundError: manifest.csv"}
    assert any("could not be computed" in f for f in check_complete(payload))


def test_partition_stats_without_js_divergence_is_caught() -> None:
    """`alpha` in the config is the dirichlet parameter; js_divergence is the skew the draw
    actually produced. Only the second one can be the x-axis of figure 9."""
    payload = _healthy()
    payload["partition_stats"] = {"size_gini": 0.17}
    assert any("js_divergence" in f for f in check_complete(payload))
