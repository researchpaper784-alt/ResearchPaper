"""Phase 10 -- fedswarm.repro, against synthetic result payloads matching the real
write_result() schema."""

from __future__ import annotations

import json
from pathlib import Path

from fedswarm.repro import check_metric, load_reference, verify_result

REFERENCE = {
    "metrics": {"final_test_macro_f1": {"reference": 0.45, "tolerance_abs": 0.15, "min_acceptable": 0.10}},
    "num_rounds_completed": {"reference": 2, "exact_match_required": True},
}


def test_check_metric_passes_within_tolerance() -> None:
    passed, _ = check_metric(0.40, REFERENCE["metrics"]["final_test_macro_f1"])
    assert passed


def test_check_metric_fails_outside_tolerance() -> None:
    passed, _ = check_metric(0.05, REFERENCE["metrics"]["final_test_macro_f1"])
    assert not passed


def test_check_metric_fails_below_floor_even_if_within_raw_tolerance() -> None:
    # 0.45 - 0.15 = 0.30 is the raw band's lower edge, but min_acceptable=0.10 -- test
    # a value that's within the raw +/- band arithmetic but demonstrates the floor is
    # a real, separate check (here the floor is looser than the band, so use a spec
    # where the floor actually binds).
    spec = {"reference": 0.45, "tolerance_abs": 1.0, "min_acceptable": 0.20}
    passed, _ = check_metric(0.15, spec)  # within +/-1.0 of 0.45, but below the 0.20 floor
    assert not passed


def test_verify_result_passes_a_matching_completed_run() -> None:
    result = {"final": {"final_test_macro_f1": 0.44, "num_rounds_completed": 2}}
    outcome = verify_result(result, REFERENCE)
    assert outcome["passed"] is True
    assert all(c["passed"] for c in outcome["checks"])


def test_verify_result_fails_when_metric_out_of_band() -> None:
    result = {"final": {"final_test_macro_f1": 0.0, "num_rounds_completed": 2}}
    outcome = verify_result(result, REFERENCE)
    assert outcome["passed"] is False


def test_verify_result_fails_when_a_referenced_metric_is_missing() -> None:
    result = {"final": {"num_rounds_completed": 2}}  # no final_test_macro_f1 at all
    outcome = verify_result(result, REFERENCE)
    assert outcome["passed"] is False
    missing = [c for c in outcome["checks"] if c["metric"] == "final_test_macro_f1"]
    assert missing[0]["detail"] == "missing from result"


def test_verify_result_fails_on_wrong_round_count_when_exact_match_required() -> None:
    result = {"final": {"final_test_macro_f1": 0.44, "num_rounds_completed": 1}}
    outcome = verify_result(result, REFERENCE)
    assert outcome["passed"] is False


def test_load_reference_reads_the_real_repro_reference_json() -> None:
    """The actual committed docs/repro_reference.json must itself be loadable and
    have the shape verify_result expects -- a real regression guard, not a synthetic
    fixture check."""
    repo_root = Path(__file__).resolve().parent.parent
    reference = load_reference(repo_root / "docs" / "repro_reference.json")
    assert "final_test_macro_f1" in reference["metrics"]
    assert "reference" in reference["metrics"]["final_test_macro_f1"]


def test_the_real_reference_source_run_still_passes_its_own_recorded_band() -> None:
    """Sanity check: the actual run that seeded docs/repro_reference.json must pass
    verify_result against that same reference (it would be a real bug if the file
    that recorded a real result's own tolerance band rejected that exact result)."""
    repo_root = Path(__file__).resolve().parent.parent
    reference = load_reference(repo_root / "docs" / "repro_reference.json")
    source_path = repo_root / "results" / "fl" / f"{reference['reference_source_run_id']}.json"
    if not source_path.exists():
        return  # gitignored, real result -- only present locally if this session's own artifact is there
    result = json.loads(source_path.read_text())
    outcome = verify_result(result, reference)
    assert outcome["passed"] is True
