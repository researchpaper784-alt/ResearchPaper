"""Check that a smoke result is COMPLETE, not merely finished -- and optionally that it
reproduces.

CI's smoke job asserted `payload["status"] == "completed"` and nothing else. That is exactly
the assertion the GPU bug satisfied: on Kaggle every client's evaluation failed (the model was
on CPU while the batches were on the GPU), Flower printed
`Aggregated ClientApp-side Evaluate Metrics: {}`, and the run still wrote a result marked
`completed`. It took three GPU sessions to notice, because nothing that looked at a result
file could tell.

Every check here is a bug this repository actually shipped:

* **client-side evaluation present** -- the device-placement bug above.
* **`train_server_round` counts** -- the ServerApp never put the round number in the train
  config, so every reply recorded `-1`, and R4's DP noise (seeded partly from it) drew the
  same values every round.
* **per-round metrics exist at all** -- a run can finish having aggregated nothing.
* **reproducibility** (`--compare-against`) -- `seed_everything` ran in `server_app.main()`
  only, and a ClientApp is a separate Ray actor process, so every client's train loader
  shuffled from OS entropy. Two runs of the identical config and seed returned 0.1363 and
  0.3286. No unit test catches that: it needs two real processes, which CI has and a
  developer's `pytest` run does not.

Run:
  python scripts/check_smoke_result.py --results-dir <dir>
  python scripts/check_smoke_result.py --results-dir <dir> --compare-against <first.json>
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

# Fields whose presence proves the corresponding subsystem actually produced something.
# A prefix rather than an exact name: the per-class confusion counts are named after the
# classes, which the smoke dataset shares with the real one but a future dataset may not.
CLIENT_EVAL_PREFIX = "client_eval_"


def _load_one(results_dir: str) -> tuple[Path, dict]:
    paths = sorted(glob.glob(f"{results_dir}/*.json"))
    if not paths:
        raise SystemExit(f"FAIL: no result file under {results_dir}")
    if len(paths) > 1:
        raise SystemExit(
            f"FAIL: {len(paths)} result files under {results_dir}, expected exactly one -- "
            "a stale result from an earlier run would make every check below ambiguous"
        )
    return Path(paths[0]), json.loads(Path(paths[0]).read_text())


def check_complete(payload: dict) -> list[str]:
    """Returns a list of failures; empty means the result is complete."""
    failures: list[str] = []

    if payload.get("status") != "completed":
        failures.append(f"status is {payload.get('status')!r}, not 'completed'")

    rounds = payload.get("rounds") or []
    if not rounds:
        failures.append("no per-round records -- the run finished having aggregated nothing")
        return failures

    last = rounds[-1]

    client_eval = [k for k in last if k.startswith(CLIENT_EVAL_PREFIX)]
    if not client_eval:
        failures.append(
            "no client-side evaluation metrics in the last round. Either every client's "
            "evaluate_handler failed (the CPU/GPU device-placement bug) or "
            "merge_evaluate_metrics stopped being called -- both produce a result that says "
            "'completed' and contains no evidence of client evaluation"
        )

    server_round = last.get("train_server_round")
    if server_round is None:
        failures.append("no train_server_round -- the strategy is not reporting its round")
    elif int(server_round) < 1:
        failures.append(
            f"train_server_round is {server_round}, so the ServerApp is not putting the round "
            "number in the train config. Anything seeded from it (R4's DP noise) draws "
            "identically every round"
        )

    final = payload.get("final") or {}
    if final.get("final_test_macro_f1") is None:
        failures.append("no final_test_macro_f1 -- nothing was evaluated at the end")
    if not final.get("num_rounds_completed"):
        failures.append("num_rounds_completed is missing or zero")

    return failures


def check_reproduces(first: dict, second: dict, tolerance: float = 0.0) -> list[str]:
    """Bit-for-bit by default.

    Not a tolerance band: both runs execute on the same machine, same build, same seed, so
    there is no float drift to absorb and any difference at all is a real determinism bug.
    A tolerance here would hide exactly the failure this check exists for -- the shipped
    reproducibility band was 0.15 wide while the variance it was meant to absorb was 0.19.
    """
    failures: list[str] = []
    for key in ("final_test_macro_f1", "final_val_macro_f1", "num_rounds_completed"):
        a, b = (first.get("final") or {}).get(key), (second.get("final") or {}).get(key)
        if a is None or b is None:
            failures.append(f"{key} missing from one of the runs ({a!r} vs {b!r})")
        elif abs(float(a) - float(b)) > tolerance:
            failures.append(
                f"{key} differs between two runs of the identical config and seed: "
                f"{a} vs {b}. The clients are not being seeded deterministically"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument(
        "--compare-against",
        default=None,
        help="a previous run's result JSON; the two must agree exactly",
    )
    args = parser.parse_args()

    path, payload = _load_one(args.results_dir)
    print(f"Checking {path}")

    failures = check_complete(payload)
    if args.compare_against:
        first = json.loads(Path(args.compare_against).read_text())
        failures += check_reproduces(first, payload)
        print(f"  compared against {args.compare_against}")

    if failures:
        print("\nFAIL:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    final = payload["final"]
    print(f"  status completed, {len(payload['rounds'])} rounds")
    print(f"  final_test_macro_f1 = {final['final_test_macro_f1']:.4f}")
    print("  client-side evaluation present, train_server_round counts")
    if args.compare_against:
        print("  and it reproduces exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
