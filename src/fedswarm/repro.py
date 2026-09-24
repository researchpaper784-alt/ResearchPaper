"""Phase 10 -- reproducibility verification (plan §10): checks a completed run's
final metrics against a recorded reference tolerance band
(`docs/repro_reference.json`). Pure Python, no Flower dependency -- fully testable;
`scripts/verify_repro.py` is the thin CLI that actually invokes `flwr run` first
(Colab/Kaggle only) and then calls `verify_result` on whatever it produced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# The commit that made FL runs reproducible at all: before it, `seed_everything` ran only in
# `server_app.main()`, and in the Simulation Runtime a ClientApp is a separate Ray actor
# process -- so every client's torch RNG started from OS entropy and the train loader shuffled
# from it. A reference band seeded before this describes one draw from an unrecorded
# distribution, not a reproducible number.
DETERMINISM_FIX_COMMIT = "2b7c4796652566eb4452988df9e70bc257eb8e1b"


class StaleReferenceError(RuntimeError):
    """The reference band predates the fix that made runs reproducible."""


def load_reference(path: str | Path, allow_stale: bool = False) -> dict[str, Any]:
    """Load the band, refusing one that cannot mean anything.

    This check exists because the band this repo shipped was seeded from a run at
    `9b01a462...`, before client-side seeding existed, with `tolerance_abs: 0.15` described
    as "wide enough to absorb float-determinism drift". The drift was not float-sized: two
    Kaggle runs of an identical FedAvg config, same seed, returned 0.1363 and 0.3286 -- a
    0.19 swing, **wider than the band itself**. So the check was comparing a
    non-reproducible quantity against a tolerance narrower than its own variance, and passed
    or failed at random.

    Post-fix the runs are reproducible, which makes it worse rather than better: a
    deterministic value has no reason to land near an old random draw, so the check now fails
    for the right runs. Either way "verify_repro: PASSED" would not have meant what Phase 10's
    acceptance criterion claims it means.

    Refusing is the conservative choice: a band that has to be regenerated is a five-minute
    job for whoever has a GPU session open (`--seed-reference`), and a reproducibility claim
    in a paper that rests on a meaningless check is not recoverable.
    """
    with open(path) as f:
        reference = json.load(f)

    seeded_after = reference.get("seeded_after_commit")
    if not allow_stale and seeded_after != DETERMINISM_FIX_COMMIT:
        raise StaleReferenceError(
            f"{path} has seeded_after_commit={seeded_after!r}, not "
            f"{DETERMINISM_FIX_COMMIT[:12]!r}. It was seeded before client-side seeding "
            "existed, so its reference value is one draw from an unrecorded distribution and "
            "its tolerance is narrower than the run-to-run variance it was meant to absorb "
            "(0.15 against a measured 0.19 swing). Regenerate it from a completed post-fix "
            "run:\n"
            "    python scripts/verify_repro.py --seed-reference\n"
            "Pass allow_stale=True only to inspect the old band, never to verify against it."
        )
    return reference


def reference_from_result(
    result: dict[str, Any],
    tolerance_abs: float = 0.05,
    min_acceptable: float = 0.1,
) -> dict[str, Any]:
    """A fresh band from a completed, post-fix run.

    `tolerance_abs` defaults to 0.05 rather than the old 0.15. The old value was chosen to
    absorb variance the code no longer has: with the clients seeded, the same config and seed
    reproduces, so the band only needs to cover genuine cross-platform float drift (different
    torch build, CPU vs GPU kernels) rather than an unseeded shuffle. A band much wider than
    the thing it measures cannot catch a broken pipeline, which is the one job it has.
    """
    final = result.get("final", {})
    if "final_test_macro_f1" not in final:
        raise ValueError(
            "result has no final_test_macro_f1 -- cannot seed a band from an incomplete run"
        )
    config = result.get("config", {})
    return {
        "_comment": (
            "Reference tolerance band for scripts/verify_repro.py, seeded from a completed "
            "run made AFTER client-side seeding existed, so the reference value is "
            "reproducible rather than one draw from an unrecorded distribution. Tolerance "
            "covers cross-platform float drift only. Regenerate with "
            "`python scripts/verify_repro.py --seed-reference`."
        ),
        "seeded_after_commit": DETERMINISM_FIX_COMMIT,
        "reference_source_run_id": config.get("run_config", {}).get("_run_id")
        or result.get("run_id"),
        "provenance": {
            "git_sha": result.get("provenance", {}).get("git_sha"),
            "gpu_available": result.get("provenance", {}).get("gpu", {}).get("available"),
        },
        "run_config": {
            k: config.get("run_config", {}).get(k)
            for k in ("strategy-name", "num-clients", "num-rounds", "regime", "model-name",
                      "model-norm", "image-size", "seed")
            if config.get("run_config", {}).get(k) is not None
        },
        "metrics": {
            "final_test_macro_f1": {
                "reference": final["final_test_macro_f1"],
                "tolerance_abs": tolerance_abs,
                "min_acceptable": min_acceptable,
            }
        },
        "num_rounds_completed": {
            "reference": final.get("num_rounds_completed"),
            "exact_match_required": True,
        },
    }


def check_metric(actual: float, spec: dict[str, Any]) -> tuple[bool, str]:
    reference = spec["reference"]
    tolerance = spec["tolerance_abs"]
    min_acceptable = spec.get("min_acceptable")
    within_band = abs(actual - reference) <= tolerance
    above_floor = min_acceptable is None or actual >= min_acceptable
    detail = f"actual={actual:.4f}, reference={reference:.4f} +/- {tolerance}"
    if min_acceptable is not None:
        detail += f", floor={min_acceptable}"
    return (within_band and above_floor), detail


def verify_result(result: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Checks `result` (a real `results/fl/<run_id>.json` payload) against
    `reference` (`docs/repro_reference.json`'s schema). A metric named in
    `reference["metrics"]` but absent from `result["final"]` fails explicitly
    (`"missing from result"`), rather than being skipped silently -- a missing
    metric on a "completed" run is itself a real reproducibility problem, not a
    non-event."""
    checks: list[dict[str, Any]] = []
    overall_passed = True

    final = result.get("final", {})
    for metric_name, spec in reference.get("metrics", {}).items():
        actual = final.get(metric_name)
        if actual is None:
            checks.append({"metric": metric_name, "passed": False, "detail": "missing from result"})
            overall_passed = False
            continue
        passed, detail = check_metric(actual, spec)
        checks.append({"metric": metric_name, "passed": passed, "detail": detail})
        overall_passed = overall_passed and passed

    rounds_spec = reference.get("num_rounds_completed")
    if rounds_spec:
        actual_rounds = final.get("num_rounds_completed")
        expected_rounds = rounds_spec["reference"]
        passed = (
            actual_rounds == expected_rounds if rounds_spec.get("exact_match_required") else True
        )
        checks.append(
            {
                "metric": "num_rounds_completed",
                "passed": passed,
                "detail": f"actual={actual_rounds}, expected={expected_rounds}",
            }
        )
        overall_passed = overall_passed and passed

    return {"passed": overall_passed, "checks": checks}
