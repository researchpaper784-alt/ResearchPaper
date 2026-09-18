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


def load_reference(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


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
