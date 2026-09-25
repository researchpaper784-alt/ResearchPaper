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
import shutil
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

    def part_overrides_of(part: object) -> dict:
        return (
            resolve_entry_overrides(part, base_dir=REPO_ROOT)
            if isinstance(part, dict) and ("file" in part or "overrides" in part)
            else {k: v for k, v in (part or {}).items() if k != "name"}
        )

    strategies = spec.get("strategies") or []

    def resolved(entry: object) -> tuple[str, dict]:
        if isinstance(entry, str):
            return entry, {"strategy-name": entry}
        return entry["name"], resolve_entry_overrides(entry, base_dir=REPO_ROOT)

    for entry in strategies:
        name, overrides = resolved(entry)
        # Every strategy against the FIRST partition: the axes multiply, and the point is to
        # touch every strategy/variant code path once, not to re-run the grid.
        part_overrides = part_overrides_of(parts[0])
        for variant in (spec.get("variants") or [{}]):
            variant_overrides = {k: v for k, v in variant.items() if k != "name"}
            label = name + (f"/{variant['name']}" if variant.get("name") else "")
            out.append((label, {**base, **part_overrides, **overrides, **variant_overrides}))

    # Then every REMAINING partition once, against one strategy. Until 2026-09-25 only
    # `parts[0]` ever ran, so a partition axis carrying a code path -- an attack type, a DP
    # sigma, a cold start hiding nodes -- was executed for its first value only, and the
    # report said "ok" for the whole config. `robustness_r2_reduced` is the sharp case:
    # `gaussian` inflates the update norm and `sign_flip` preserves it and reverses the
    # direction, two different paths through aco/heuristics.py, and only the first ran.
    #
    # Linear in partitions rather than multiplicative, so A6's 27 strategies stay 27 arms
    # plus its partitions. Paired with the method under test where there is one, because
    # FedACO's r_k / dispersion heuristics are the code the attacked partitions exercise --
    # pairing them with `fedavg` would execute the attack and skip the thing reading it.
    if len(parts) > 1 and strategies:
        names = [resolved(e)[0] for e in strategies]
        probe = next((e for e, n in zip(strategies, names) if "fedaco" in n), strategies[0])
        probe_name, probe_overrides = resolved(probe)
        for part in parts[1:]:
            label = f"{probe_name}/{(part or {}).get('name', '-')}"
            out.append((label, {**base, **part_overrides_of(part), **probe_overrides}))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, help="dir holding manifest.csv and cache/")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--out", default=None, help="scratch dir for results (default: temp)")
    parser.add_argument("--timeout", type=int, default=420, help="seconds per arm")
    parser.add_argument("--force", action="store_true",
                        help="re-run arms that already have a result")
    args = parser.parse_args()

    fixture = Path(args.fixture)
    out = Path(args.out) if args.out else Path("/tmp/preflight-sweeps")
    out.mkdir(parents=True, exist_ok=True)

    paths = []
    for pattern in args.configs:
        # Any glob metacharacter, not just `*`. Checking for `*` alone silently treated
        # `configs/experiment/ablation_a[0-9].yaml` as a literal filename and died on
        # FileNotFoundError after 36 arms had already passed -- and a character class is the
        # natural way to name A1-A9, so this is the pattern a caller reaches for first.
        if any(ch in pattern for ch in "*?["):
            matched = sorted(REPO_ROOT.glob(pattern))
            if not matched:
                raise SystemExit(f"no config matched {pattern!r}")
            paths.extend(matched)
        else:
            path = Path(pattern)
            if not path.is_absolute():
                path = REPO_ROOT / path
            if not path.exists():
                raise SystemExit(f"no such config: {path}")
            paths.append(path)

    if shutil.which("flwr") is None:
        raise SystemExit(
            "`flwr` is not on PATH. Activate the venv first (export PATH=\"$PWD/.venv/bin:"
            "$PATH\"), or this dies with a bare FileNotFoundError from subprocess."
        )

    # The federation has to be set before any run: num_supernodes defaults to 2 and no
    # run-config key can carry it, so a K=6 cell would wait forever for nodes 3-6.
    subprocess.run(
        ["flwr", "federation", "simulation-config", "--num-supernodes",
         str(SHRINK["num-clients"]), "--client-resources-num-cpus", "1"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )

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
            # Resume, for the same reason the sweep runners have it: this walks ~96 arms at
            # about 40s each, and a container that reaps the process partway through should
            # not mean starting over. An arm that already wrote a result is an arm that ran.
            if not args.force and list(cell.glob("*.json")):
                passed += 1
                print(f"  skip    {label:34} (already ran)")
                continue

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
