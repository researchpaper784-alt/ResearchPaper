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

from fedswarm.repro import load_reference, verify_result  # noqa: E402
from fedswarm.sweep import is_completed, predict_run_id, pyproject_flat_defaults  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/fl")
    parser.add_argument("--reference", default=str(REPO_ROOT / "docs" / "repro_reference.json"))
    parser.add_argument("--skip-run", action="store_true", help="Don't invoke flwr run; check an existing result")
    args = parser.parse_args()

    reference = load_reference(args.reference)
    defaults = pyproject_flat_defaults(REPO_ROOT / "pyproject.toml")
    run_id = predict_run_id(defaults, {})  # the smoke config IS the pyproject defaults, no overrides

    if not args.skip_run:
        print("Running: flwr run . --stream")
        subprocess.run(["flwr", "run", ".", "--stream"], check=False)

    if not is_completed(run_id, args.output_dir):
        print(f"FAIL: no completed result found for run_id {run_id} under {args.output_dir}")
        sys.exit(1)

    result = json.loads((Path(args.output_dir) / f"{run_id}.json").read_text())
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
