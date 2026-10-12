"""Plan §9.3's acceptance criterion, as a command.

    "`make figures tables` regenerates every artifact from `results/` with no manual steps.
     Deleting `paper/figures/` and `paper/tables/` and re-running restores them
     byte-for-byte identical."

Nothing checked it until 2026-09-24, and the first run of the full chain found two real
faults that no unit test could see, because both needed the whole pipeline pointed at real
result files:

* `_variant_of` compared each knob against a **hardcoded** default. Reconciling
  `aco-gamma-dispersion` to plan §14's 0.50 made every run -- FedAvg included, since
  pyproject supplies the key to every resolved config -- read as a `g2=0.5` variant. Both
  `convergence` and `comparison`, the paper's two headline figures, filter to
  `variant == "default"`, so both silently skipped with a full set of valid results present,
  and every delta column came out empty.
* `aggregate_results.py` accepted `--partitions` and `--baselines` names that matched
  nothing, printed "(no overlapping-seed comparisons found)", wrote both CSVs and exited 0.
  Holm-Bonferroni corrects by family size, so a typo there inflates the significance of
  every comparison that survives.

Byte-for-byte is the right bar, not a tolerance: both runs read the same JSON with the same
code on the same machine, so any difference is non-determinism in the artifact pipeline
itself -- a dict iteration order, an unsorted glob, an embedded timestamp -- and each of those
makes "did this number change?" unanswerable across a re-run.

Run:
  python scripts/verify_phase9_repro.py --results-dir results/fl --expected-seeds 5 \
      --baselines fedavg --partitions dirichlet_0.3 iid
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run(cmd: list[str]) -> tuple[int, str]:
    done = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return done.returncode, (done.stdout + done.stderr)


def generate(out: Path, args: argparse.Namespace) -> list[str]:
    """Produce every Phase 9 artifact into `out`. Returns failure messages."""
    out.mkdir(parents=True, exist_ok=True)
    failures = []

    steps = [
        ("tables", [PYTHON, "scripts/make_tables.py",
                    "--results-dir", args.results_dir, "--out", str(out / "tables"),
                    "--expected-seeds", str(args.expected_seeds)]),
        ("figures", [PYTHON, "scripts/make_figures.py",
                     "--results-dir", args.results_dir, "--out", str(out / "figures"),
                     "--expected-seeds", str(args.expected_seeds)]),
        ("aggregate", [PYTHON, "scripts/aggregate_results.py",
                       "--results-dir", args.results_dir,
                       "--baselines", *args.baselines,
                       "--partitions", *args.partitions,
                       "--out-summary", str(out / "summary.csv"),
                       "--out-comparison", str(out / "comparison.csv")]),
    ]
    for name, cmd in steps:
        code, output = _run(cmd)
        if code != 0:
            failures.append(f"{name} exited {code}:\n{output[-800:]}")
    return failures


def compare(first: Path, second: Path) -> tuple[list[str], int]:
    """Every file under `first` must exist under `second` with identical bytes."""
    differing = []
    checked = 0
    for path in sorted(p for p in first.rglob("*") if p.is_file()):
        relative = path.relative_to(first)
        other = second / relative
        checked += 1
        if not other.exists():
            differing.append(f"{relative}: missing from the second run")
        elif not filecmp.cmp(path, other, shallow=False):
            differing.append(
                f"{relative}: differs ({path.stat().st_size} vs {other.stat().st_size} bytes)"
            )
    for path in sorted(p for p in second.rglob("*") if p.is_file()):
        relative = path.relative_to(second)
        if not (first / relative).exists():
            differing.append(f"{relative}: appeared only in the second run")
    return differing, checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/fl")
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--baselines", nargs="+", default=["fedavg"])
    parser.add_argument("--partitions", nargs="+", required=True)
    parser.add_argument("--keep", action="store_true", help="leave both output trees on disk")
    args = parser.parse_args()

    results = Path(args.results_dir)
    if not results.is_absolute():
        results = REPO_ROOT / results
    if not sorted(results.glob("*.json")):
        print(f"No result files under {results} -- run a sweep first.")
        return 2

    workdir = Path(tempfile.mkdtemp(prefix="phase9-repro-"))
    try:
        print(f"Generating every Phase 9 artifact twice from {results}\n")
        for label in ("first", "second"):
            failures = generate(workdir / label, args)
            if failures:
                print(f"FAIL: the {label} pass did not complete:")
                for failure in failures:
                    print(f"  - {failure}")
                return 1
            print(f"  {label} pass complete")

        differing, checked = compare(workdir / "first", workdir / "second")
        print(f"\n{checked} artifacts compared byte-for-byte.")
        if differing:
            print("\nFAIL -- these are not reproducible:")
            for item in differing:
                print(f"  - {item}")
            print(
                "\nBoth passes read the same JSON with the same code on one machine, so a "
                "difference is non-determinism in the artifact pipeline itself: an unsorted "
                "glob, dict iteration order, or an embedded timestamp."
            )
            return 1
        print("PASS -- every artifact is identical across two independent passes.")
        return 0
    finally:
        if args.keep:
            print(f"\nOutputs kept in {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
