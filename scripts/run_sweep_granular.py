"""Phase 6, Step 6.2 -- sweep runner CLI for the *granular* per-cell experiment
configs (`smoke.yaml`, `main_client_scale.yaml`, `overhead.yaml`,
`ablation_a1..a9.yaml`, `robustness_r1..r5.yaml`). These configs cover ground the
combined `ablation_all.yaml`/`robustness.yaml` configs don't (A2, A4-A8 individually;
R1-R5 individually) -- see docs/OPEN_QUESTIONS.md. `scripts/run_sweep.py` (a
different, independently-developed runner over `fedswarm.utils.runner`) drives
`main.yaml`/`ablation_all.yaml`/`robustness.yaml` instead; the two runners exist
side by side because they read two different YAML shapes, not by accident.

All real logic (grid expansion, run_id prediction, resume-skip, locking, ordering)
lives in `fedswarm.sweep` and is unit-tested there without needing a live Flower
runtime; this script only parses args, loads the experiment YAML, and actually shells
out to `flwr run` -- which needs the `simulation` extra (Colab/Kaggle only,
`docs/FLOWER_API_NOTES.md`).

Experiment YAML shape:

    base_overrides: {<flat run_config keys>: <value>, ...}
    strategies:
      - name: fedavg
        overrides: {strategy-name: fedavg}
    partitions:
      - name: iid
        overrides: {regime: iid}
    seeds: [0, 1, 2, 3, 4]

Run:
  .venv/bin/python scripts/run_sweep_granular.py --config configs/experiment/ablation_a2.yaml
  .venv/bin/python scripts/run_sweep_granular.py --config configs/experiment/ablation_a2.yaml --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedswarm.sweep import (  # noqa: E402
    gpu_fraction_problem,
    granular_runs,
    run_sweep,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment YAML (see module docstring)")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned run list; run nothing")
    parser.add_argument("--output-dir", default="results/fl")
    parser.add_argument("--manifest-path", default="results/manifest.jsonl")
    parser.add_argument("--lock-dir", default="results/fl/_locks")
    parser.add_argument("--lock-timeout-s", type=float, default=3600.0)
    parser.add_argument("--cpus-per-client", type=int, default=1)
    parser.add_argument(
        "--gpus-per-client",
        type=float,
        # None, not 0.0 -- see `sweep.gpu_fraction_problem`: argparse cannot otherwise tell
        # "left alone" from "asked for CPU", and only the first is refused on a GPU box.
        default=None,
        help=(
            "fraction of a GPU per ClientApp (0.2 = five share one card). Required on a GPU "
            "box -- unset, Ray hides the GPU from every ClientApp and the sweep trains on "
            "CPU while looking healthy. Pass 0 to ask for CPU deliberately"
        ),
    )
    parser.add_argument(
        "--pyproject-path", default=str(REPO_ROOT / "pyproject.toml"), help="For predicting run_ids"
    )
    args = parser.parse_args()

    # One expansion, shared with the notebooks via fedswarm.sweep.granular_runs, so "which
    # cells belong to this sweep" cannot mean different things in the two places.
    runs = granular_runs(args.config, REPO_ROOT)

    # Before anything starts: a sweep that trains on CPU on a GPU box looks healthy and
    # never finishes (`sweep.gpu_fraction_problem`). A dry run is exempt -- it runs nothing.
    if not args.dry_run:
        largest_k = max(
            (int(r.overrides.get("num-clients", 2)) for r in runs), default=2
        )
        problem = gpu_fraction_problem(args.gpus_per_client, largest_k)
        if problem:
            print(problem)
            print("\nRefusing to start. Nothing has been run.")
            return 1

    if args.dry_run:
        print(f"{len(runs)} runs planned:")

    entries = run_sweep(
        runs,
        pyproject_path=args.pyproject_path,
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        lock_dir=args.lock_dir,
        lock_timeout_s=args.lock_timeout_s,
        dry_run=args.dry_run,
        cpus_per_client=args.cpus_per_client,
        gpus_per_client=args.gpus_per_client or 0.0,
    )

    for entry, run in zip(entries, runs):
        print(f"  [{entry['status']:>17}] {run.label} ({entry['run_id']})")

    completed = sum(1 for e in entries if e["status"] in ("completed", "skipped_completed"))
    print(f"\n{completed}/{len(entries)} runs completed (this call + already-done).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
