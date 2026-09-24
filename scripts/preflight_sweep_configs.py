"""Execute one tiny cell of every arm of a sweep config, to prove the config runs at all.

`strategy_from_run_config` catching a bad config is necessary and not sufficient: it builds the
strategy, not the round. A config can construct fine and then die at round 1 -- an attack that
needs more malicious clients than exist, a cold start that hides too many nodes, a fitness mode
missing its val loader, a `min-train-nodes` above the supernode count (which does not raise at
all: Flower's `sample_nodes` waits in a `while ...: sleep(1)` loop that never gives up, so the
run consumes the whole Kaggle session and writes nothing).

This runs every declared arm at 2 rounds, K=6, image-size 32 against the synthetic fixture, so
the strategy, attack, cold-start and aggregation code paths all execute with real Flower
messages. It is minutes of CPU against C's 214-427 GPU-h, and a failure here is a failure that
would otherwise be found on Kaggle.

What it does NOT check: that the numbers mean anything. Rounds, K and image size are all cut,
so this is "does the code path run", not "is the result valid".

Run:
  python scripts/preflight_sweep_configs.py --fixture /path/to/het --configs configs/experiment/robustness_r*.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from fedswarm.sweep import build_run_config_arg, resolve_entry_overrides  # noqa: E402

# Cut to the bone, but not below what the conditions need. K=6 keeps every ceil-based subset
# (malicious_ids, late_node_ids) at one client or more; min-train-nodes has to stay at or below
# the count a cold start leaves visible, which is why it is 3 rather than 6.
SHRINK = {
    "num-rounds": 2,
    "num-clients": 6,
    "min-client-size": 3,
    "min-train-nodes": 3,
    "min-evaluate-nodes": 3,
    "min-available-nodes": 3,
    "image-size": 32,
    "local-batch-size": 8,
    "local-epochs": 1,
    "num-classes": 4,
    "model-name": "simple_cnn",
    "model-pretrained": False,
}


def arms(path: Path) -> list[tuple[str, dict]]:
    spec = yaml.safe_load(path.read_text()) or {}
    base = {**(spec.get("common") or {}), **(spec.get("base_overrides") or {})}
    out = []
    parts = spec.get("partitions") or spec.get("regimes") or [{"name": "-"}]
    for entry in spec.get("strategies") or []:
        if isinstance(entry, str):
            name, overrides = entry, {"strategy-name": entry}
        else:
            name, overrides = entry["name"], resolve_entry_overrides(entry, base_dir=REPO_ROOT)
        # One partition and one variant per arm: the axes multiply, and the point is to touch
        # every strategy/variant code path once, not to re-run the grid.
        part = parts[0]
        part_overrides = (
            resolve_entry_overrides(part, base_dir=REPO_ROOT)
            if isinstance(part, dict) and ("file" in part or "overrides" in part)
            else {k: v for k, v in (part or {}).items() if k != "name"}
        )
        for variant in (spec.get("variants") or [{}]):
            variant_overrides = {k: v for k, v in variant.items() if k != "name"}
            label = name + (f"/{variant['name']}" if variant.get("name") else "")
            out.append((label, {**base, **part_overrides, **overrides, **variant_overrides}))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, help="dir holding manifest.csv and cache/")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--out", default=None, help="scratch dir for results (default: temp)")
    parser.add_argument("--timeout", type=int, default=420, help="seconds per arm")
    args = parser.parse_args()

    fixture = Path(args.fixture)
    out = Path(args.out) if args.out else Path("/tmp/preflight-sweeps")
    out.mkdir(parents=True, exist_ok=True)

    # The federation has to be set before any run: num_supernodes defaults to 2 and no
    # run-config key can carry it, so a K=6 cell would wait forever for nodes 3-6.
    subprocess.run(
        ["flwr", "federation", "simulation-config", "--num-supernodes",
         str(SHRINK["num-clients"]), "--client-resources-num-cpus", "1"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )

    paths = []
    for pattern in args.configs:
        paths.extend(sorted(REPO_ROOT.glob(pattern)) if "*" in pattern else [Path(pattern)])

    failures, passed = [], 0
    for path in paths:
        print(f"\n=== {path.name} ===")
        for label, overrides in arms(path):
            cell = out / f"{path.stem}__{label.replace('/', '_')}"
            run_config = {
                **overrides,
                **SHRINK,
                "seed": 0,
                "cache-dir": str(fixture / "cache"),
                "manifest-path": str(fixture / "manifest.csv"),
                "partition-cache-dir": str(fixture / "parts"),
                "output-dir": str(cell),
                "checkpoint-dir": str(cell / "ckpt"),
            }
            started = time.time()
            done = subprocess.run(
                ["flwr", "run", ".", "--stream", "--run-config",
                 build_run_config_arg(run_config)],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=args.timeout,
            )
            elapsed = time.time() - started
            wrote = bool(list(cell.glob("*.json")))
            if wrote:
                passed += 1
                print(f"  ok      {label:34} {elapsed:5.1f}s")
            else:
                # `flwr run` exits 0 even when the simulation dies, so the result file is the
                # only trustworthy signal -- the same reason check_smoke_result.py exists.
                tail = "\n".join((done.stdout + done.stderr).strip().splitlines()[-6:])
                failures.append((path.name, label, tail))
                print(f"  FAILED  {label:34} {elapsed:5.1f}s (exit {done.returncode})")

    print(f"\n{passed} arms ran, {len(failures)} failed.")
    for name, label, tail in failures:
        print(f"\n--- {name} :: {label} ---\n{tail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
