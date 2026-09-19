"""Phase 9 — turn a directory of result JSONs into the tables the paper reports.

Aggregates over seeds within each (strategy, regime, variant) cell and emits Markdown and
CSV. The arithmetic is mean and standard deviation; the work is in refusing to present a
number more confidently than it deserves.

Four things this does that a straightforward groupby would not, each of which is a way a
results table can mislead without containing a single wrong number:

**Incomplete cells are shown, not dropped.** A cell with 2 of 5 seeds appears with n=2 and
a marker. Silently averaging whatever happened to finish is how a table ends up with one
strategy's mean over 5 seeds next to another's over 2, presented identically.

**`n` is always a column.** Not a footnote, not implied by the presence of a std.

**FedACO carries its own health columns.** `fallback_used` rate and `pheromone_entropy`
sit next to its score, because a headline macro-F1 says nothing about whether the colony
was searching or whether the safety fallback quietly turned FedACO into FedAvg for most of
the run. A table that reports only the score cannot be read honestly.

**The delta column is paired by seed where it can be.** Comparing mean-to-mean throws away
the fact that both methods saw the same seeds; a paired mean difference is both tighter
and more honest, and the code says which one it used.

Run:
    python scripts/make_tables.py --results-dir results/fl/main
    python scripts/make_tables.py --results-dir results/fl/ablation --out paper/tables
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from fedswarm.analysis import (
    cohens_d,
    holm_bonferroni,
    min_achievable_p,
    seeds_needed_for,
    wilcoxon_paired_test,
)
from fedswarm.tables import main_results_table
from fedswarm.utils.runner import load_result_files

REPO_ROOT = Path(__file__).resolve().parents[1]

# The reported metric. Val exists in every result file too and is what hyperparameter
# selection reads (scripts/run_hparam_search.py); it is shown alongside so a reader can
# see whether the two agree, but test is what the paper quotes.
PRIMARY = "final_test_macro_f1"
SECONDARY = "best_val_macro_f1"


def load_results(results_dir: Path) -> list[dict]:
    """Every completed result under `results_dir`. Incomplete and half-written runs are
    dropped here rather than filtered downstream: a table is the one artefact a reader
    takes at face value, so a crashed run must not contribute a number to it."""
    out = []
    for path, result in load_result_files(results_dir, recursive=True):
        if result.get("status") != "completed" or "final" not in result:
            continue
        result["_path"] = str(path)
        out.append(result)
    return out


def cell_key(result: dict) -> tuple[str, str, str]:
    config = result.get("config", {})
    run_config = config.get("run_config", {})
    strategy = str(config.get("strategy", run_config.get("strategy-name", "unknown")))
    regime = str(run_config.get("regime", "unknown"))
    # Dirichlet's alpha is part of the regime's identity: dirichlet@0.1 and dirichlet@1.0
    # are different experimental conditions and must never be pooled into one row.
    if regime == "dirichlet" and run_config.get("alpha") is not None:
        regime = f"dirichlet_{run_config['alpha']}"
    return strategy, regime, _variant_of(run_config)


def _variant_of(run_config: dict) -> str:
    """Reconstruct the ablation variant from the run config.

    The sweep runner knows the variant name, but a result file records only the resolved
    config, so the variant has to be recovered from the knobs that differ from default.
    Anything unrecognised falls back to "default" rather than being guessed at.
    """
    marks = []
    if str(run_config.get("aco-persistence", "decayed")) != "decayed":
        marks.append(f"persistence={run_config['aco-persistence']}")
    if str(run_config.get("aco-fitness-mode", "data_free")) != "data_free":
        marks.append(f"fitness={run_config['aco-fitness-mode']}")
    if str(run_config.get("model-norm", "groupnorm")) != "groupnorm":
        marks.append(f"norm={run_config['model-norm']}")
    if float(run_config.get("aco-target-sum", 1.0)) != 1.0:
        marks.append(f"s={run_config['aco-target-sum']}")
    if float(run_config.get("aco-gamma-dispersion", 1.0)) != 1.0:
        marks.append(f"g2={run_config['aco-gamma-dispersion']}")
    if run_config.get("aco-safety-fallback") is False:
        marks.append("no-fallback")
    # Phase 8. Without these every robustness variant collapses to "default": the 9
    # variants x 5 seeds per strategy would key into the same per-seed dict, whichever
    # file was read last would win, and the published table would average a clean control
    # together with a 30%-sign-flip run and report it as one number with n=5.
    attack = str(run_config.get("attack", "none"))
    if attack != "none":
        marks.append(f"attack={attack}@{run_config.get('attack-fraction', 0)}")
        if float(run_config.get("attack-scale", 1.0)) != 1.0:
            marks.append(f"x{run_config['attack-scale']}")
    if float(run_config.get("fraction-train", 1.0)) != 1.0:
        marks.append(f"participation={run_config['fraction-train']}")
    return ",".join(marks) if marks else "default"


def _seed_of(result: dict) -> int:
    config = result.get("config", {})
    return int(config.get("seed", result.get("seed", 0)))


def _round_mean(result: dict, key: str) -> float | None:
    values = []
    for entry in result.get("rounds", []):
        for candidate in (f"train_{key}", key):
            if entry.get(candidate) is not None:
                values.append(float(entry[candidate]))
                break
    return statistics.fmean(values) if values else None


def summarize(results: list[dict], expected_seeds: int | None) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for result in results:
        grouped[cell_key(result)].append(result)

    rows = []
    for (strategy, regime, variant), runs in sorted(grouped.items()):
        scores = {_seed_of(r): r["final"].get(PRIMARY) for r in runs}
        scores = {s: v for s, v in scores.items() if v is not None}
        val = [r["final"].get(SECONDARY) for r in runs if r["final"].get(SECONDARY) is not None]
        if not scores:
            continue

        fallbacks = [_round_mean(r, "fallback_used") for r in runs]
        fallbacks = [f for f in fallbacks if f is not None]
        entropies = [_round_mean(r, "pheromone_entropy") for r in runs]
        entropies = [e for e in entropies if e is not None]

        rows.append(
            {
                "strategy": strategy,
                "regime": regime,
                "variant": variant,
                "n": len(scores),
                "expected_n": expected_seeds,
                # Marked rather than hidden: a cell averaged over 2 of 5 seeds is not
                # comparable to one averaged over 5, and presenting them identically is
                # the most common way a results table lies without a wrong number in it.
                "incomplete": expected_seeds is not None and len(scores) < expected_seeds,
                "seeds": sorted(scores),
                "by_seed": scores,
                "mean": statistics.fmean(scores.values()),
                "std": statistics.stdev(scores.values()) if len(scores) > 1 else 0.0,
                "val_mean": statistics.fmean(val) if val else None,
                "fallback_rate": statistics.fmean(fallbacks) if fallbacks else None,
                "pheromone_entropy": statistics.fmean(entropies) if entropies else None,
            }
        )
    return rows


def add_deltas(rows: list[dict], baseline: str) -> None:
    """Delta against the baseline strategy in the same regime, paired by seed when both
    ran the same ones.

    Mean-to-mean throws away the pairing that makes the comparison tight: both methods
    saw the same seeds, so the per-seed difference has far less variance than the
    difference of the means. `delta_paired` records which was used, so a reader is never
    left guessing whether a small margin survived pairing."""
    by_regime_seed = {
        (r["regime"], s): v
        for r in rows
        if r["strategy"] == baseline and r["variant"] == "default"
        for s, v in r["by_seed"].items()
    }
    for row in rows:
        if row["strategy"] == baseline and row["variant"] == "default":
            row["delta"] = 0.0
            row["delta_paired"] = True
            continue
        shared = [s for s in row["by_seed"] if (row["regime"], s) in by_regime_seed]
        if shared:
            row["delta"] = statistics.fmean(
                row["by_seed"][s] - by_regime_seed[(row["regime"], s)] for s in shared
            )
            row["delta_paired"] = True
            row["delta_n"] = len(shared)
        else:
            row["delta"] = None
            row["delta_paired"] = False


def add_significance(rows: list[dict], baseline: str, alternative: str = "two-sided") -> None:
    """Paired Wilcoxon + Cohen's d per row, Holm-Bonferroni corrected across the whole
    family of comparisons at once.

    Mean +/- std cannot support "FedACO beats FedAvg". At 5 seeds the per-seed spread is
    comparable to the margins these methods differ by, so a table without a test invites
    the reader to eyeball two overlapping error bars and conclude whatever they came in
    believing. The plan (§9.1) calls for a signed-rank test specifically because n=5 does
    not justify assuming normality.

    Holm is applied once across every comparison in the table rather than per regime:
    correcting within a regime and then reporting six regimes under-corrects by exactly
    the factor the correction exists to supply.

    `alternative="greater"` is available and is *not* the default. A one-sided test halves
    the achievable p and is legitimate when the hypothesis is directional -- which this
    one is -- but choosing it after seeing the two-sided result is not, so it has to be
    set deliberately and is recorded in the table when it is.
    """
    paired = {
        (r["regime"], s): v
        for r in rows
        if r["strategy"] == baseline and r["variant"] == "default"
        for s, v in r["by_seed"].items()
    }
    tested = []
    for row in rows:
        row["p_value"] = None
        row["p_value_holm"] = None
        row["cohens_d"] = None
        if row["strategy"] == baseline and row["variant"] == "default":
            continue
        shared = sorted(s for s in row["by_seed"] if (row["regime"], s) in paired)
        if len(shared) < 2:
            continue
        a = np.array([row["by_seed"][s] for s in shared], dtype=float)
        b = np.array([paired[(row["regime"], s)] for s in shared], dtype=float)
        test = wilcoxon_paired_test(a, b) if alternative == "two-sided" else _wilcoxon(a, b, alternative)
        row["p_value"] = test["p_value"]
        row["cohens_d"] = cohens_d(a, b)
        row["n_pairs"] = len(shared)
        tested.append(row)

    if tested:
        adjusted = holm_bonferroni([r["p_value"] for r in tested])
        for row, value in zip(tested, adjusted):
            row["p_value_holm"] = value


def _wilcoxon(a, b, alternative: str) -> dict:
    """`wilcoxon_paired_test` with a non-default alternative, sharing its nan handling."""
    from scipy import stats

    if np.all(a - b == 0):
        return {"p_value": float("nan")}
    try:
        return {"p_value": float(stats.wilcoxon(a, b, alternative=alternative).pvalue)}
    except ValueError:
        return {"p_value": float("nan")}


def power_note(rows: list[dict], alpha: float, alternative: str) -> str | None:
    """A warning when the table's p-values cannot clear `alpha` at any data.

    This is the check that makes the statistics honest rather than decorative. The
    signed-rank p-value has a floor set by the pair count alone, so a table can report
    "not significant" for every row while the data is as favourable as data can be. Saying
    so in the table beats leaving it for whoever writes the discussion section.
    """
    tested = [r for r in rows if r.get("p_value") is not None and r.get("n_pairs")]
    if not tested:
        return None
    family = len(tested)
    n_pairs = min(r["n_pairs"] for r in tested)
    floor = min_achievable_p(n_pairs, alternative)
    if floor * family <= alpha:
        return None

    needed = seeds_needed_for(alpha, family, alternative)
    target = f"{needed} seeds" if needed else "more than 40 seeds"
    return (
        f"\n⚠️ **These p-values cannot reach {alpha} at this seed count.** The signed-rank "
        f"test's smallest possible {alternative} p-value at n={n_pairs} pairs is "
        f"{floor:.4f}; across {family} comparison{'s' if family != 1 else ''} "
        f"Holm-Bonferroni multiplies the smallest "
        f"by {family}, so the best achievable adjusted p is "
        f"{min(floor * family, 1.0):.4f} — reached when *every* seed favours the method. "
        f"Every 'not significant' in this table is therefore a statement about the sample "
        f"size, not about the data. Clearing {alpha} over this family needs {target}, or a "
        f"smaller family of claims. Do not read these rows as evidence of no effect.\n"
    )


def to_markdown(rows: list[dict], baseline: str) -> str:
    lines = []
    regimes = sorted({r["regime"] for r in rows})
    for regime in regimes:
        subset = [r for r in rows if r["regime"] == regime]
        lines.append(f"\n### {regime}\n")
        lines.append(
            "| strategy | variant | n | macro-F1 | val | Δ vs " + baseline
            + " | p (Holm) | d | fallback | τ entropy |"
        )
        lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|")
        for row in sorted(subset, key=lambda r: (-r["mean"],)):
            mark = " ⚠️" if row["incomplete"] else ""
            score = f"{row['mean']:.4f} ± {row['std']:.4f}"
            val = f"{row['val_mean']:.4f}" if row["val_mean"] is not None else "—"
            if row["delta"] is None:
                delta = "—"
            else:
                delta = f"{row['delta']:+.4f}" + ("" if row["delta_paired"] else " (unpaired)")
            fallback = f"{row['fallback_rate']:.0%}" if row["fallback_rate"] is not None else "—"
            entropy = (
                f"{row['pheromone_entropy']:.3f}" if row["pheromone_entropy"] is not None else "—"
            )
            # The raw p is deliberately not shown next to the adjusted one: two p-values
            # in a row invites quoting whichever is smaller. The CSV carries both.
            if row.get("p_value_holm") is None or row["p_value_holm"] != row["p_value_holm"]:
                p_holm = "—"
            else:
                p_holm = f"{row['p_value_holm']:.3f}"
            effect = (
                f"{row['cohens_d']:+.2f}"
                if row.get("cohens_d") is not None and row["cohens_d"] == row["cohens_d"]
                else "—"
            )
            lines.append(
                f"| {row['strategy']} | {row['variant']} | {row['n']}{mark} | {score} | "
                f"{val} | {delta} | {p_holm} | {effect} | {fallback} | {entropy} |"
            )

    incomplete = [r for r in rows if r["incomplete"]]
    if incomplete:
        lines.append(
            f"\n⚠️ **{len(incomplete)} cell(s) have fewer seeds than expected** and are not "
            "comparable to complete cells. Listed with their actual n above; re-run the "
            "sweep to fill them before quoting anything from those rows.\n"
        )
    if any(r["strategy"] == "fedaco" for r in rows):
        lines.append(
            "\n**Reading the FedACO rows.** `fallback` is the fraction of rounds where the "
            "colony lost to the FedAvg point and FedACO silently *was* FedAvg — a high "
            "rate means the score largely belongs to FedAvg. `τ entropy` near its ceiling "
            "(log of the level count, 2.398 at the default 11) means pheromone was flat, "
            "so the colony was not searching and the cross-round stigmergy the method "
            "claims was not happening. Neither is visible in the macro-F1 column.\n"
        )
    return "\n".join(lines)


def to_latex(rows: list[dict], baseline: str, path: Path, name: str) -> None:
    """Plan §9.3's deliverable: booktabs, best-in-bold, significance markers.

    Built from the *same* `rows` the markdown and CSV come from, and rendered by
    `fedswarm.tables.main_results_table` rather than a second LaTeX emitter -- the plan's
    instruction is "never hand-type a number into the paper", and two formatters drifting
    apart is the same failure one step later. `tables.py` takes the
    `fedswarm.analysis`-shaped frame, so this adapts rather than reimplements.

    Rows are labelled by `variant` when a sweep has more than one (the ablation tables,
    where every row is FedACO) and by `strategy` otherwise (the main table). Labelling an
    ablation by strategy would print twelve identical "fedaco" rows.
    """
    import pandas as pd

    variants = {r["variant"] for r in rows}
    label_key = "variant" if len(variants) > 1 else "strategy"

    summary = pd.DataFrame(
        [
            {
                "strategy": r[label_key],
                "partition": r["regime"],
                "mean": r["mean"],
                "std": r["std"],
                "n": r["n"],
            }
            for r in rows
        ]
    )
    # `main_results_table` reads markers off (partition, baseline) pairs, where `baseline`
    # is the row being marked -- so the frame is keyed the same way the rows are labelled.
    comparison = pd.DataFrame(
        [
            {
                "partition": r["regime"],
                "baseline": r[label_key],
                "p_value_holm": r["p_value_holm"],
            }
            for r in rows
            if r.get("p_value_holm") is not None
        ]
    )

    # `main_results_table`'s `method=` parameter is "the row that gets no marker". Its own
    # convention marks the baselines ("fedaco vs. this baseline"); ours marks the method
    # ("this row differs significantly from fedavg"), which is what the caption states and
    # what `rows` carries -- each row's p-value is its own comparison against the baseline.
    # So the row to leave unmarked is the baseline's. Passing `baseline` here looks
    # inverted and is not; do not "fix" it without changing the caption to match.
    method = baseline if label_key == "strategy" else "default"
    tex = main_results_table(
        summary,
        comparison if not comparison.empty else None,
        method=method,
        caption=(
            f"{name}: test macro-F1 (mean $\\pm$ std over seeds). "
            f"Significance vs.\\ {_escape(baseline)}, Wilcoxon signed-rank with "
            "Holm--Bonferroni correction across the table "
            "($^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$)."
        ),
        label=f"tab:{name}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex + "\n")


def _escape(text: str) -> str:
    return str(text).replace("_", r"\_")


def to_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "strategy", "regime", "variant", "n", "expected_n", "incomplete",
        "mean", "std", "val_mean", "delta", "delta_paired",
        "p_value", "p_value_holm", "cohens_d", "n_pairs",
        "fallback_rate", "pheromone_entropy", "seeds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "seeds": " ".join(str(s) for s in row["seeds"])})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/fl/main")
    parser.add_argument("--out", default="paper/tables")
    parser.add_argument("--baseline", default="fedavg")
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="significance level for the power warning"
    )
    parser.add_argument(
        "--alternative",
        default="two-sided",
        choices=["two-sided", "greater"],
        help=(
            "signed-rank alternative. 'greater' halves the achievable p and is legitimate "
            "for a directional hypothesis -- but only if chosen before seeing the result"
        ),
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=8,
        help="seeds each cell should have; cells with fewer are marked incomplete",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = REPO_ROOT / results_dir

    results = load_results(results_dir)
    if not results:
        print(f"No completed result files under {results_dir}.")
        print("Run the sweep first: make main  (or make main-plan to see what it would cost)")
        return 2

    rows = summarize(results, args.expected_seeds)
    add_deltas(rows, args.baseline)
    add_significance(rows, args.baseline, args.alternative)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    markdown = to_markdown(rows, args.baseline)
    warning = power_note(rows, args.alpha, args.alternative)
    if warning:
        markdown += warning
    header = (
        f"# Results — {results_dir.name}\n\n"
        f"{len(results)} completed runs across {len(rows)} cells. "
        f"Metric: `{PRIMARY}`, mean ± std over seeds. `p (Holm)` is a paired Wilcoxon "
        f"signed-rank test against {args.baseline}, Holm-Bonferroni corrected across every "
        f"comparison in this table; `d` is Cohen's d"
        + (f" ({args.alternative})" if args.alternative != "two-sided" else "")
        + ".\n"
    )
    (out_dir / f"{results_dir.name}.md").write_text(header + markdown + "\n")
    to_csv(rows, out_dir / f"{results_dir.name}.csv")
    to_latex(rows, args.baseline, out_dir / f"{results_dir.name}.tex", results_dir.name)

    print(header + markdown)
    print(f"\nWrote {out_dir / f'{results_dir.name}.md'}, .csv and .tex")

    incomplete = sum(1 for r in rows if r["incomplete"])
    if incomplete:
        print(f"\n{incomplete} incomplete cell(s) -- see the warning in the table.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
