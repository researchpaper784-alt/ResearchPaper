"""Phase 10 -- reproducibility verification CLI (plan §10: "runs the smoke config
and checks final metrics fall within a recorded tolerance band"). Checking logic
lives in `fedswarm.repro` (pure Python, `tests/test_repro.py`, including a sanity
check against the real run that seeded `docs/repro_reference.json`); this script
runs `flwr run .` itself (needs the `simulation` extra -- Colab/Kaggle only,
`docs/FLOWER_API_NOTES.md`) using the pyproject.toml smoke defaults (no overrides),
then loads whatever result that produced and checks it.

Run:
  .venv/bin/python scripts/verify_repro.py
  .venv/bin/python scripts/verify_repro.py --skip-run  # just check an already-produced result
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedswarm.repro import (  # noqa: E402
    StaleReferenceError,
    load_reference,
    reference_from_result,
    verify_result,
)
from fedswarm.sweep import is_completed, predict_run_id, pyproject_flat_defaults  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/fl")
    parser.add_argument("--reference", default=str(REPO_ROOT / "docs" / "repro_reference.json"))
    parser.add_argument("--skip-run", action="store_true", help="Don't invoke flwr run; check an existing result")
    parser.add_argument(
        "--seed-reference",
        action="store_true",
        help=(
            "write a new tolerance band from the completed run instead of checking against "
            "the old one. Needed once after the client-side seeding fix: the shipped band was "
            "seeded from a run whose clients were unseeded, so its reference value was one "
            "draw from an unrecorded distribution"
        ),
    )
    parser.add_argument("--tolerance-abs", type=float, default=0.05)
    args = parser.parse_args()

    defaults = pyproject_flat_defaults(REPO_ROOT / "pyproject.toml")
    run_id = predict_run_id(defaults, {})  # the smoke config IS the pyproject defaults, no overrides

    if not args.skip_run:
        print("Running: flwr run . --stream")
        subprocess.run(["flwr", "run", ".", "--stream"], check=False)

    if not is_completed(run_id, args.output_dir):
        print(f"FAIL: no completed result found for run_id {run_id} under {args.output_dir}")
        sys.exit(1)

    result = json.loads((Path(args.output_dir) / f"{run_id}.json").read_text())

    if args.seed_reference:
        # Writing the band is its own mode rather than a fallback on a stale one: silently
        # re-seeding whenever the check fails would turn every regression into a new reference
        # and the check into a no-op.
        fresh = reference_from_result(result, tolerance_abs=args.tolerance_abs)
        Path(args.reference).write_text(json.dumps(fresh, indent=1) + "\n")
        print(f"Wrote a fresh reference band to {args.reference}")
        print(f"  reference final_test_macro_f1 = "
              f"{fresh['metrics']['final_test_macro_f1']['reference']:.4f} "
              f"+/- {args.tolerance_abs}")
        print(f"  from run_id {fresh['reference_source_run_id']}, "
              f"git_sha {fresh['provenance']['git_sha']}")
        print("Commit it, then re-run this script without --seed-reference to verify.")
        return

    try:
        reference = load_reference(args.reference)
    except StaleReferenceError as exc:
        # Not a crash to work around: the band cannot mean anything, and a reproducibility
        # claim resting on a meaningless check is worse than an absent one.
        print(f"verify_repro: REFUSED\n\n{exc}")
        sys.exit(2)

    outcome = verify_result(result, reference)

    for check in outcome["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {check['metric']}: {check['detail']}")

    if outcome["passed"]:
        print("verify_repro: PASSED")
    else:
        print("verify_repro: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
