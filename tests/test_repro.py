"""Phase 10 -- fedswarm.repro, against synthetic result payloads matching the real
write_result() schema."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from fedswarm.repro import (
    DETERMINISM_FIX_COMMIT,
    check_metric,
    load_reference,
    verify_result,
)

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
    fixture check.

    `allow_stale=True` because the committed band is currently stale by design: it was seeded
    before client-side seeding existed. This test checks its SHAPE, which is still the shape a
    regenerated band will have. Verifying a result against it is what `load_reference` refuses
    by default, and that refusal has its own test below."""
    repo_root = Path(__file__).resolve().parent.parent
    reference = load_reference(repo_root / "docs" / "repro_reference.json", allow_stale=True)
    assert "final_test_macro_f1" in reference["metrics"]
    assert "reference" in reference["metrics"]["final_test_macro_f1"]


def test_the_real_reference_source_run_still_passes_its_own_recorded_band() -> None:
    """Sanity check: the actual run that seeded docs/repro_reference.json must pass
    verify_result against that same reference (it would be a real bug if the file
    that recorded a real result's own tolerance band rejected that exact result)."""
    repo_root = Path(__file__).resolve().parent.parent
    reference = load_reference(repo_root / "docs" / "repro_reference.json", allow_stale=True)
    source_path = repo_root / "results" / "fl" / f"{reference['reference_source_run_id']}.json"
    if not source_path.exists():
        return  # gitignored, real result -- only present locally if this session's own artifact is there
    result = json.loads(source_path.read_text())
    outcome = verify_result(result, reference)
    assert outcome["passed"] is True


# ======================================================================================
# A band that cannot mean anything must refuse, not compare
# ======================================================================================


def test_the_committed_band_is_refused_by_default() -> None:
    """The band this repo shipped was seeded at `9b01a462...`, before `seed_everything`
    reached the ClientApp -- so every client's train loader shuffled from OS entropy and its
    reference value is one draw from an unrecorded distribution. Its `tolerance_abs: 0.15` was
    described as "wide enough to absorb float-determinism drift"; two Kaggle runs of an
    identical FedAvg config later returned 0.1363 and 0.3286, a 0.19 swing -- wider than the
    band itself.

    So the check was comparing a non-reproducible quantity against a tolerance narrower than
    its own variance, and passing or failing at random. Post-fix it is worse: a deterministic
    value has no reason to land near an old random draw. Either way
    "verify_repro: PASSED" never meant what Phase 10's acceptance criterion says it means, and
    a reproducibility claim in a paper resting on it is not recoverable.
    """
    from fedswarm.repro import StaleReferenceError

    repo_root = Path(__file__).resolve().parent.parent
    with pytest.raises(StaleReferenceError, match="seeded before client-side seeding"):
        load_reference(repo_root / "docs" / "repro_reference.json")


def test_a_band_stamped_with_the_fix_commit_loads() -> None:
    from fedswarm.repro import DETERMINISM_FIX_COMMIT

    band = {"seeded_after_commit": DETERMINISM_FIX_COMMIT, "metrics": {}}
    path = Path(tempfile.mkdtemp()) / "ref.json"
    path.write_text(json.dumps(band))

    assert load_reference(path)["seeded_after_commit"] == DETERMINISM_FIX_COMMIT


def test_a_regenerated_band_accepts_the_run_it_came_from() -> None:
    """The round trip: seed from a result, then verify that result against it. If this failed,
    `--seed-reference` would be writing bands nothing can satisfy."""
    from fedswarm.repro import reference_from_result

    result = {
        "status": "completed",
        "final": {"final_test_macro_f1": 0.4712, "num_rounds_completed": 2},
        "config": {"run_config": {"strategy-name": "fedavg", "num-clients": 2,
                                  "num-rounds": 2, "_run_id": "abc_0"}},
        "provenance": {"git_sha": "deadbeef", "gpu": {"available": True}},
    }

    band = reference_from_result(result)

    assert band["seeded_after_commit"] == DETERMINISM_FIX_COMMIT
    assert verify_result(result, band)["passed"] is True


def test_the_regenerated_tolerance_is_tighter_than_the_stale_one() -> None:
    """0.05, not the old 0.15. The old value was sized to absorb variance the code no longer
    has -- an unseeded shuffle -- and a band three times wider than the effect it measures
    cannot catch a broken pipeline, which is the one job it has."""
    from fedswarm.repro import reference_from_result

    repo_root = Path(__file__).resolve().parent.parent
    stale = load_reference(repo_root / "docs" / "repro_reference.json", allow_stale=True)
    fresh = reference_from_result(
        {"final": {"final_test_macro_f1": 0.5, "num_rounds_completed": 2}, "config": {}}
    )

    assert (
        fresh["metrics"]["final_test_macro_f1"]["tolerance_abs"]
        < stale["metrics"]["final_test_macro_f1"]["tolerance_abs"]
    )


def test_seeding_a_band_from_an_incomplete_run_is_refused() -> None:
    """A band seeded from a crashed run would enshrine whatever half-finished number it left,
    and every later run would be checked against it."""
    from fedswarm.repro import reference_from_result

    with pytest.raises(ValueError, match="incomplete run"):
        reference_from_result({"final": {"num_rounds_completed": 1}, "config": {}})
