"""Phase 2.1 — summarize centralized results across seeds.

Prints mean +/- std test macro-F1/accuracy per (model, image_size, norm), which is the
performance ceiling every FL method in later phases is measured against.

The BatchNorm arm run alongside GroupNorm here is a *centralized-only* control -- it
shows the accuracy cost of GroupNorm with federation held out. It is not the plan's A9
ablation (BatchNorm-vs-GroupNorm *aggregation* behavior under non-IID federated data),
which needs actual federated training and belongs to Phase 4/7, not this script.

Run: python scripts/summarize_centralized.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="results/centralized")
    args = parser.parse_args()

    groups: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for path in sorted(Path(args.dir).glob("*.json")):
        result = json.loads(path.read_text())
        key = (
            result["config"]["model"]["name"],
            result["config"]["data"]["image_size"],
            result["config"]["model"]["norm"],
        )
        groups[key].append(result)

    if not groups:
        print(f"No results found in {args.dir}")
        return 1

    print(
        f"{'model':<12}{'size':>5}  {'norm':<11}{'n':>3}  "
        f"{'test_macro_f1':<20}{'test_accuracy':<20}{'best_epoch':<12}wall_clock_s"
    )
    for (model, size, norm), results in sorted(groups.items()):
        f1s = np.array([r["final"]["test_macro_f1"] for r in results])
        accs = np.array([r["final"]["test_accuracy"] for r in results])
        epochs = [r["final"]["best_epoch"] for r in results]
        wall = np.array([r["final"]["wall_clock_s"] for r in results])
        seeds = sorted(r["config"]["seed"] for r in results)

        print(
            f"{model:<12}{size:>5}  {norm:<11}{len(results):>3}  "
            f"{f1s.mean():.4f} +/- {f1s.std():.4f}    "
            f"{accs.mean():.4f} +/- {accs.std():.4f}    "
            f"{np.mean(epochs):<12.1f}{wall.mean():.1f}"
        )
        print(f"  seeds={seeds}  per-seed f1={[round(f,4) for f in f1s]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
