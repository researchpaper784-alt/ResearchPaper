"""Apply the fitness fix that `gate_fitness.yaml` chose, and report the gate's evidence.

`configs/experiment/gate_fitness.yaml` runs three FedACO arms -- the unmodified default, a
raised `aco-gamma-entropy`, and `aco-dispersion-reference: aggregate` -- to settle, on real
data, which one closes the degenerate single-client optimum recorded in
docs/OPEN_QUESTIONS.md (best vertex beat the FedAvg point in 15/15 rounds, mean margin
+0.6252).

Reading the gate and then acting on it are separate jobs and this script does both explicitly,
because the acting part has a trap. A1, A2, main_reduced, R1 and R2 do NOT all read
`configs/strategy/fedaco.yaml`: `ablation_a2_reduced` and `ablation_a1_reduced` set
`strategy-name: fedaco` in `base_overrides` and never reference that file. Patching it would
fix the sweeps that use `file:` and silently leave the two that decide the paper running the
broken default. So the single lever that every config inherits is pyproject's
`[tool.flwr.app.config]`, and that is what this writes.

Changing a default is deliberate, not incidental: docs/EXPERIMENT_LOG.md records that raising
`aco-gamma-entropy` repairs the corner but also charges *legitimate* concentration, which is
what down-weighting a straggler or an adversary requires. So this prints the before/after and
refuses to guess -- `--fix` must be named, and `--from-results` shows what the gate measured.

    python scripts/apply_gate_fix.py --from-results results/fl/gate
    python scripts/apply_gate_fix.py --fix aggregate
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fedswarm.utils.results import result_settings  # noqa: E402

# key -> value, matching gate_fitness.yaml's arms.
FIXES = {
    "none": {},
    "gamma_entropy": {"aco-gamma-entropy": 0.45},
    "aggregate": {"aco-dispersion-reference": "aggregate"},
}


def round_margins(result: dict) -> list[float]:
    """Per-round `corner_margin`, looking under both the `train_` prefix and the bare key.

    Mirrors `check_fedaco_health._round_metric`: FedACO's diagnostics travel in
    aggregate_train's MetricRecord, which different result-writing paths have folded in
    under different names.
    """
    out = []
    for entry in result.get("rounds", []):
        for candidate in ("train_corner_margin", "corner_margin"):
            value = entry.get(candidate)
            if value is not None:
                out.append(float(value))
                break
    return out


def read_gate(results_dir: Path) -> list[dict]:
    """One row per result file: which arm, and what its corner margin did."""
    rows = []
    for path in sorted(results_dir.rglob("*.json")):
        try:
            result = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        config = result.get("config") or {}
        # Settings are NESTED under config["run_config"] (fedswarm.utils.results.
        # resolved_config). This read them flat until the first real Kaggle gate, where every
        # arm therefore came back with no settings and was labelled `default`.
        settings = result_settings(result)
        if not settings:
            continue
        margins = round_margins(result)
        rows.append({
            "file": path.name,
            "strategy": config.get("strategy") or settings.get("strategy-name") or "?",
            "gamma_entropy": settings.get("aco-gamma-entropy"),
            "dispersion": settings.get("aco-dispersion-reference"),
            "seed": config.get("seed", settings.get("seed")),
            "rounds": len(margins),
            "mean_margin": sum(margins) / len(margins) if margins else None,
            "max_margin": max(margins) if margins else None,
            "corner_wins": sum(1 for m in margins if m > 0),
        })
    return rows


def arm_label(row: dict) -> str:
    if row["strategy"] != "fedaco":
        return row["strategy"]
    if row["dispersion"] == "aggregate":
        return "dispersion_aggregate"
    if row["gamma_entropy"] not in (None, 0.1):
        return f"gamma_entropy_{row['gamma_entropy']}"
    return "default"


def verdict(rows: list[dict]) -> tuple[str | None, list[str]]:
    """Which arm to adopt, and the lines explaining why.

    The rule is the one the health check uses and the one the open question demands: the
    margin must be negative in EVERY round, not on average. gamma_entropy and the dispersion
    shape are set once per run, so a value that clears the mean round leaves the worst rounds
    degenerate.
    """
    by_arm: dict[str, list[dict]] = {}
    for row in rows:
        if row["strategy"] == "fedaco" and row["rounds"]:
            by_arm.setdefault(arm_label(row), []).append(row)

    lines, clean = [], []
    for arm in sorted(by_arm):
        group = by_arm[arm]
        wins = sum(r["corner_wins"] for r in group)
        total = sum(r["rounds"] for r in group)
        worst = max(r["max_margin"] for r in group)
        mean = sum(r["mean_margin"] * r["rounds"] for r in group) / total
        status = "CLOSED" if wins == 0 else f"OPEN ({wins}/{total} rounds)"
        lines.append(
            f"  {arm:24} mean {mean:+.4f}  worst round {worst:+.4f}  corner {status}"
        )
        if wins == 0:
            clean.append((worst, arm))

    if not by_arm:
        lines.append("  no FedACO result with a corner_margin found")
        return None, lines

    if not clean:
        lines.append("")
        lines.append("  NO arm closed the corner. Do not run A1 or A2 on this -- every search")
        lines.append("  method inherits the same degenerate optimum and their ranking would")
        lines.append("  say nothing. Raise aco-gamma-entropy past 0.45 (the health check")
        lines.append("  prints the closed-form requirement) and re-run the gate.")
        return None, lines

    # Most negative worst-round margin = most headroom.
    clean.sort()
    best = clean[0][1]
    lines.append("")
    lines.append(f"  ADOPT: {best}  (largest margin below zero across every round)")
    if len(clean) > 1:
        others = ", ".join(a for _, a in clean[1:])
        lines.append(f"  also closed the corner: {others}")
    return best, lines


def patch_pyproject(fix: str, dry_run: bool = False) -> list[str]:
    """Set the fix's keys in `[tool.flwr.app.config]`, in place, preserving comments."""
    if fix not in FIXES:
        raise SystemExit(f"unknown --fix {fix!r}; choose from {sorted(FIXES)}")
    path = REPO_ROOT / "pyproject.toml"
    text = path.read_text()
    changes = []
    for key, value in FIXES[fix].items():
        literal = f'"{value}"' if isinstance(value, str) else repr(value)
        pattern = re.compile(rf'^(\s*){re.escape(key)}\s*=\s*(\S+)', re.MULTILINE)
        match = pattern.search(text)
        if match is None:
            raise SystemExit(
                f"{key!r} is not declared in pyproject's [tool.flwr.app.config]. `flwr run` "
                "rejects any --run-config key not declared there, so adding it here without "
                "declaring it would fail at runtime with a bare '[code: 15]'."
            )
        if match.group(2) == literal:
            changes.append(f"{key}: already {literal}")
            continue
        changes.append(f"{key}: {match.group(2)} -> {literal}")
        text = text[:match.start()] + f"{match.group(1)}{key} = {literal}" + text[match.end():]

    if not FIXES[fix]:
        changes.append("no keys to change (fix='none' keeps the documented defaults)")
    if not dry_run and changes:
        path.write_text(text)
    return changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-results", type=Path, default=None,
                        help="gate results dir to read and summarise (e.g. results/fl/gate)")
    parser.add_argument("--fix", choices=sorted(FIXES), default=None,
                        help="which fix to write into pyproject's config table")
    parser.add_argument("--apply-verdict", action="store_true",
                        help="with --from-results, write whichever arm the gate chose")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    chosen = None
    if args.from_results is not None:
        if not args.from_results.exists():
            raise SystemExit(f"no such directory: {args.from_results}")
        rows = read_gate(args.from_results)
        print(f"gate results in {args.from_results} ({len(rows)} file(s))\n")
        for row in sorted(rows, key=lambda r: (arm_label(r), r["seed"] or 0)):
            margin = f"{row['mean_margin']:+.4f}" if row["mean_margin"] is not None else "   n/a"
            print(f"  {arm_label(row):24} seed {row['seed']}  rounds {row['rounds']:3}  "
                  f"mean margin {margin}  corner won {row['corner_wins']}")
        print()
        chosen, lines = verdict(rows)
        print("\n".join(lines))
        print()

    fix = args.fix
    if args.apply_verdict:
        if chosen is None:
            raise SystemExit(
                "--apply-verdict but the gate chose nothing. Nothing written; see above."
            )
        fix = {"default": "none", "dispersion_aggregate": "aggregate"}.get(chosen)
        if fix is None and chosen.startswith("gamma_entropy"):
            fix = "gamma_entropy"
        if fix is None:
            raise SystemExit(f"cannot map gate verdict {chosen!r} to a --fix")
        print(f"applying the gate's verdict: {chosen} -> --fix {fix}")

    if fix is None:
        print("no --fix and no --apply-verdict, so pyproject.toml was not touched.")
        return 0

    for line in patch_pyproject(fix, dry_run=args.dry_run):
        print(("[dry-run] " if args.dry_run else "") + line)

    print()
    print("Every sweep inherits pyproject's [tool.flwr.app.config], so this reaches A1, A2,")
    print("main_reduced, R1 and R2 -- including the two that never read")
    print("configs/strategy/fedaco.yaml. Record the change in docs/EXPERIMENT_LOG.md: it makes")
    print("results incomparable with runs made before it, which is fine now (none exist) and")
    print("will not be later.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
