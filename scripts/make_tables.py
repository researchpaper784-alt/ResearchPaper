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
import functools
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


@functools.lru_cache(maxsize=1)
def _defaults() -> dict:
    """pyproject's `[tool.flwr.app.config]`, read once.

    `_variant_of` used to hardcode the default it compared each knob against, and on
    2026-09-24 that broke: reconciling `aco-gamma-dispersion` to plan §14's 0.50 (it had
    drifted to 1.0) made **every** FedACO run read as a `g2=0.5` variant. Since `add_deltas`
    only computes a delta for rows whose variant is "default", every comparison column in
    every table would have come out empty -- a fifth copy of a default, failing the same way
    the other four did. Read the real thing instead.
    """
    import tomllib

    path = REPO_ROOT / "pyproject.toml"
    return tomllib.loads(path.read_text())["tool"]["flwr"]["app"]["config"]


# Keys that are not experimental conditions. Everything else in the resolved config is, and
# `_variant_of` marks any of it that differs from the default -- see that function for why the
# previous hand-listed approach could not work.
_NOT_A_CONDITION = frozenset({
    # Paths and bookkeeping.
    "cache-dir", "checkpoint-dir", "manifest-path", "output-dir", "partition-cache-dir",
    # Already part of the grouping key, so marking them would duplicate the row label.
    "strategy-name", "regime", "alpha", "seed", "partition-seed",
    # Reporting/plumbing rather than condition.
    "deterministic", "num-classes", "num-rounds",
})

# Short names for the marks, so a label stays readable. Anything unlisted is marked by its
# key with the `aco-`/`fedaco-` prefix stripped.
_MARK_NAMES = {
    "aco-persistence": "persistence", "aco-fitness-mode": "fitness", "model-norm": "norm",
    "aco-target-sum": "s", "aco-gamma-dispersion": "g2", "aco-gamma-entropy": "g3",
    "aco-gamma-alignment": "g1", "aco-q0": "q0", "aco-rho": "rho",
    "aco-pheromone-exp": "a", "aco-heuristic-exp": "b", "aco-num-levels": "levels",
    "aco-search-method": "search", "aco-desirability-scaling": "scaling",
    "aco-dispersion-reference": "ref", "fraction-train": "participation",
    "num-clients": "K", "aco-ants-start": "ants", "aco-iters-start": "iters",
}


def _mark_name(key: str) -> str:
    if key in _MARK_NAMES:
        return _MARK_NAMES[key]
    for prefix in ("aco-", "fedaco-", "model-"):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _variant_of(run_config: dict) -> str:
    """Reconstruct the experimental condition from the resolved config.

    The sweep runner knows the arm's name, but a result file records only the resolved
    config, so the condition has to be recovered from whatever differs from the defaults.

    **Why this compares everything rather than a hand-picked list.** It used to check about
    eight knobs by name, and on 2026-09-24 a check of every declared sweep arm found that
    **42 distinct arms collapsed onto the single key `(fedaco, dirichlet_0.3, default)`** --
    including all five of `ablation_a1`'s search methods, which *is* the project's go/no-go
    gate, plus most of A6's 27 arms, all of A5's, both of A8's, R5's four client counts and
    R6's cold-start arms. `make_tables` groups by (strategy, regime, variant), so each
    collision averages unrelated arms into one row: A1 would have reported ACO, random
    search, coordinate grid, PSO and GA as a single number.

    The log already recorded this failure for the robustness sweeps -- "whichever file was
    read last would win, and the published table would average a clean control together with
    a 30%-sign-flip run and report it as one number with n=5" -- and the fix then was to add
    the two keys those sweeps varied. Adding keys one at a time is what left the ablations
    broken, because every new ablation axis needs another entry here and nothing fails when
    one is missing. Comparing the whole config cannot go stale that way.

    `_NOT_A_CONDITION` holds the exclusions: paths, the keys already in the grouping label,
    and plumbing. Defaults come from `_defaults()`, never from literals -- a default that
    moves would otherwise relabel every existing result (which is exactly what reconciling
    `aco-gamma-dispersion` to plan §14's 0.50 did).
    """
    defaults = _defaults()
    marks = []

    for key in sorted(run_config):
        if key in _NOT_A_CONDITION or key not in defaults:
            continue
        actual, default = run_config[key], defaults[key]
        if isinstance(default, bool) or isinstance(actual, bool):
            differs = bool(actual) != bool(default)
        elif isinstance(default, (int, float)) and isinstance(actual, (int, float)):
            differs = float(actual) != float(default)
        else:
            differs = str(actual) != str(default)
        if not differs:
            continue
        if key == "aco-safety-fallback" and actual is False:
            marks.append("no-fallback")
        else:
            marks.append(f"{_mark_name(key)}={actual}")

    # The attack is one condition spread over several keys; collapsing it keeps the label
    # readable and keeps "10% gaussian" distinct from "10% sign-flip".
    attack = str(run_config.get("attack", defaults.get("attack", "none")))
    if attack != str(defaults.get("attack", "none")):
        marks = [m for m in marks if not m.startswith(("attack=", "attack-fraction=",
                                                       "attack-scale=", "num-malicious-nodes="))]
        mark = f"attack={attack}@{run_config.get('attack-fraction', 0)}"
        scale = run_config.get("attack-scale")
        if scale is not None and float(scale) != float(defaults.get("attack-scale", 1.0)):
            mark += f"x{scale}"
        marks.insert(0, mark)

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
    # Baseline values keyed by (regime, variant, seed), plus the same map restricted to the
    # default variant as a fallback.
    #
    # Matching the variant matters because several sweeps set a tracked knob in
    # `base_overrides`, so **every** row in them carries that mark and none is "default":
    # `main_client_scale.yaml` sets fraction-train 0.3, and robustness_r1/r2/r3 set an
    # attack. Keying the baseline on variant == "default" alone found nothing in those
    # sweeps, so every delta came out None -- the robustness tables lost the exact column
    # they exist for, silently, because an empty delta renders as the same "—" as a cell
    # with no paired seeds.
    #
    # Within-variant first is also the comparison those sweeps intend: FedACO against FedAvg
    # *at the same attack level*, not against an unattacked control. Falling back to the
    # default variant preserves the ablation behaviour, where the baseline only ever exists
    # at default and each variant is meant to be read against it.
    by_variant_seed: dict[tuple[str, str, object], float] = {
        (r["regime"], r["variant"], s): v
        for r in rows
        if r["strategy"] == baseline
        for s, v in r["by_seed"].items()
    }
    by_default_seed = {
        (regime, seed): value
        for (regime, variant, seed), value in by_variant_seed.items()
        if variant == "default"
    }

    for row in rows:
        if row["strategy"] == baseline:
            # A baseline row is its own reference, whatever variant it sits at.
            row["delta"] = 0.0
            row["delta_paired"] = True
            continue

        same_variant = [
            s for s in row["by_seed"] if (row["regime"], row["variant"], s) in by_variant_seed
        ]
        if same_variant:
            reference = {s: by_variant_seed[(row["regime"], row["variant"], s)] for s in same_variant}
            shared = same_variant
        else:
            shared = [s for s in row["by_seed"] if (row["regime"], s) in by_default_seed]
            reference = {s: by_default_seed[(row["regime"], s)] for s in shared}

        if shared:
            row["delta"] = statistics.fmean(row["by_seed"][s] - reference[s] for s in shared)
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
    floor_info = _power_floor(rows, alternative)
    if floor_info is None:
        return None
    family, n_pairs, floor = floor_info
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


def _power_floor(rows: list[dict], alternative: str) -> tuple[int, int, float] | None:
    """`(family_size, n_pairs, floor)` for the tested rows, or None if nothing was tested.

    Factored out so `power_note` and `_latex_power_note` cannot disagree about whether a
    table is underpowered -- two formatters computing the same thing separately is how the
    markdown ended up warning about something the LaTeX advertised.
    """
    tested = [r for r in rows if r.get("p_value") is not None and r.get("n_pairs")]
    if not tested:
        return None
    family = len(tested)
    n_pairs = min(r["n_pairs"] for r in tested)
    return family, n_pairs, min_achievable_p(n_pairs, alternative)


def _latex_power_note(rows: list[dict], alpha: float, alternative: str) -> str | None:
    """The same warning as `power_note`, as a caption sentence.

    Separate from `power_note` because that one is markdown -- emoji, `**bold**`, newlines --
    and none of it survives in a LaTeX caption. The substance has to, though: without it a
    caption prints "$^{*}p<0.05$" for a threshold the signed-rank test cannot cross at this
    pair count, and a reader of the paper has no way to know. The markdown table says so in a
    paragraph the LaTeX never carried.
    """
    floor_info = _power_floor(rows, alternative)
    if floor_info is None:
        return None
    family, n_pairs, floor = floor_info
    if floor * family <= alpha:
        return None
    needed = seeds_needed_for(alpha, family, alternative)
    target = f"{needed} seeds" if needed else "more than 40 seeds"
    return (
        f" No significance markers are shown: at $n={n_pairs}$ paired seeds the signed-rank "
        f"test's smallest attainable {alternative} $p$ is {floor:.4f}, and across "
        f"{family} comparison{'s' if family != 1 else ''} the Holm--Bonferroni adjusted "
        f"floor is {min(floor * family, 1.0):.4f} -- so $p<{alpha}$ is unreachable at any "
        f"data, and every non-significant result here reflects the sample size rather than "
        f"the effect. Clearing {alpha} over this family requires {target}."
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


def to_latex(
    rows: list[dict], baseline: str, path: Path, name: str, power_warning: str | None = None
) -> None:
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
                # Carried into the LaTeX so an incomplete cell is marked THERE too. It was
                # dropped here, which is how `0.510 $\pm$ 0.010` from 3 of 8 seeds reached the
                # paper looking identical to a complete cell while the markdown beside it
                # showed `3 ⚠️` and two paragraphs of warning.
                "expected_n": r.get("expected_n"),
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
            "Holm--Bonferroni correction across the table"
            # The legend is conditional. Printing "$^{*}p<0.05$" when the signed-rank test
            # cannot reach 0.05 at this seed count advertises a threshold no data can cross,
            # and a reader of the paper has no way to know that -- the markdown table says so
            # in a paragraph the LaTeX never carried.
            + (
                "."
                if power_warning
                else " ($^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$)."
            )
        ),
        label=f"tab:{name}",
        power_warning=power_warning,
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
    to_latex(
        rows,
        args.baseline,
        out_dir / f"{results_dir.name}.tex",
        results_dir.name,
        # The same note the markdown carries. It belongs in the LaTeX caption too: a table
        # copied into a draft takes its caption with it and nothing else.
        power_warning=_latex_power_note(rows, args.alpha, args.alternative),
    )

    print(header + markdown)
    print(f"\nWrote {out_dir / f'{results_dir.name}.md'}, .csv and .tex")

    incomplete = sum(1 for r in rows if r["incomplete"])
    if incomplete:
        print(f"\n{incomplete} incomplete cell(s) -- see the warning in the table.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
