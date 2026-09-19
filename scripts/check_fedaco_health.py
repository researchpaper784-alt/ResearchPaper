"""Does FedACO's mechanism actually do anything on real data?

Every federated run so far has been on a synthetic pixel cache, which verifies plumbing
and nothing else. This script reads the result files from the first real-data run and
answers the three questions that decide whether Phase 6 is worth building as designed.
They are not accuracy questions -- a method can post a respectable macro-F1 while its
stated mechanism sits inert.

**1. Is the colony searching at all?**
    `pheromone_entropy` is the mean per-client entropy of tau across the L multiplier
    levels. Its maximum is log(L) = log(11) = 2.398, reached exactly when tau is uniform.
    Uniform tau means `tau^a * eta^b` reduces to `eta^b`: no colony search, no cross-round
    stigmergy, just deterministic heuristic-greedy weighting. The "Adaptive"/"swarm" claim
    in the paper's title is then decorative, which `aco/pheromone.py`'s own docstring
    names as the failure to watch for.

    Judged against what the run's own budget allows, not against a fixed gap. tau starts
    *at* the ceiling and walks away from it at a rate set by rho, the deposit magnitude
    and the iteration count, so an absolute threshold applied to a short run measures the
    run length. `fedswarm.aco.diagnostics.best_case_gaps` computes the fastest the rule
    permits -- one level winning at every iteration -- and this check reports the fraction
    of that a run realized. The first version of this check did not, and returned INERT on
    a four-round run whose best possible score was itself barely above the threshold
    (docs/EXPERIMENT_LOG.md, 2026-09-17).

**2. Has the deposit floor engaged?**
    `colony.py` deposits `rho * Q * max(F, 0)`. If F is negative for every candidate in a
    round, every deposit is zero and tau can only decay toward uniform -- question 1's
    failure, by a specific cause. Normalizing dispersion by trace(G)/K removed the
    *scale*-driven path into this (docs/EXPERIMENT_LOG.md, 2026-09-17), but F can still
    go negative on its own terms when client updates have no shared direction and
    alignment sits near zero. That residual is recorded as open in docs/OPEN_QUESTIONS.md
    and this is how it gets settled.

**3. Is the fitness optimum degenerate?**
    Dispersion is `sum_k alpha_k ||delta_k - Delta(alpha)||^2`, a weighted variance about
    the weighted mean -- and at `alpha = e_j` the mean *is* `delta_j`, so it is exactly
    zero at every single-client vertex. The only thing standing against that is the
    concentration penalty `gamma_3 * log K`, which weakens as the federation shrinks. When
    it loses, the fitness ranks "discard every client but one" above the aggregation the
    method exists to improve on, and a *correctly working* colony optimizes toward it. In
    the logs that is indistinguishable from success: `fallback_used=0`, `best_fitness`
    comfortably above `fedavg_fitness`, a confident alpha. `corner_margin` is the exact
    O(K^2) test, logged per round.

**4. Does FedACO actually beat FedAvg, and is the fallback carrying it?**
    `fallback_used=1` means the colony lost to the FedAvg point that round and FedACO
    silently *was* FedAvg. A high fallback rate with a good final score means the score
    belongs to FedAvg. Note the fallback compares both sides under the same fitness, so
    it cannot detect a degenerate fitness -- which is exactly why questions 1 and 2 are
    asked separately rather than inferred from it.

Also reports measured per-round wall-clock, which docs/EXPERIMENT_LOG.md's compute-budget
table explicitly says to substitute for its estimates ("Do not cite these numbers until
measured").

Run:
    python scripts/check_fedaco_health.py --results-dir results/fl
    python scripts/check_fedaco_health.py --results-dir results/fl --num-levels 11
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from fedswarm.aco.diagnostics import MEASURED_FITNESS, best_case_gaps
from fedswarm.utils.runner import load_result_files

REPO_ROOT = Path(__file__).resolve().parents[1]

# How close to log(L) counts as "uniform". Entropy is a log-scale quantity and the top of
# its range is flat, so a small absolute gap is a large behavioural one: at L=11, 2% below
# maximum still leaves tau nearly flat across levels. This threshold is a judgement call,
# not a figure from the plan -- stated here rather than buried, and adjustable via --flat.
DEFAULT_FLAT_FRACTION = 0.02

# tau has "not moved" when it has covered less than this fraction of the distance its own
# budget allows. Unlike the absolute gap above, this is scale-free: it means the same
# thing in round 2 of a smoke test as in round 90 of the main sweep. 0.15 is a judgement
# call, stated here rather than buried -- below it, selection is eta-driven in any
# practical sense.
INERT_REALIZED_FRACTION = 0.15

# Above this required fraction, SEARCHING is effectively unreachable: passing would take a
# colony concentrating tau nearly as fast as one that picks the same level every
# iteration, i.e. one that has stopped exploring. Such a run gets UNDERPOWERED rather than
# a verdict its length cannot support.
UNDERPOWERED_REQUIRED_FRACTION = 0.6


def load_results(results_dir: Path, strategy: str | None = None) -> list[dict]:
    """Every result under `results_dir` that has per-round records, optionally one
    strategy's. A run still in progress or killed mid-write has no `rounds` and is
    skipped; this check reads round-by-round behaviour, so it has nothing to say about a
    result that has none."""
    out = []
    for path, result in load_result_files(results_dir, recursive=True):
        if "rounds" not in result or "config" not in result:
            continue
        if strategy and str(result["config"].get("strategy", "")).lower() != strategy:
            continue
        result["_path"] = str(path)
        out.append(result)
    return out


def _round_metric(result: dict, key: str) -> list[float]:
    """Per-round values of an aggregate_train metric.

    FedACO's metrics travel in the MetricRecord returned by `aggregate_train`, which is
    not the same place as the evaluation metrics in `rounds`. Different result-writing
    paths have put them in different spots, so look in both rather than assume one.
    """
    values = []
    for entry in result.get("rounds", []):
        # `merge_train_metrics` folds aggregate_train's MetricRecord in under a `train_`
        # prefix, to keep it clear of the test_/val_ evaluation metrics. Bare keys are
        # accepted too so this still reads files written by other paths.
        for candidate in (f"train_{key}", key):
            if entry.get(candidate) is not None:
                values.append(float(entry[candidate]))
                break
    return values


def check_colony_searching(result: dict, num_levels: int, flat_fraction: float) -> dict:
    entropies = _round_metric(result, "pheromone_entropy")
    ceiling = math.log(num_levels)
    if not entropies:
        return {
            "question": "Is the colony searching?",
            "verdict": "NO DATA",
            "detail": (
                "no `pheromone_entropy` in the result file. Either this is not a FedACO "
                "run, or it predates the Phase 4 close-out that started logging it."
            ),
        }

    gaps = [(ceiling - e) / ceiling for e in entropies]
    mean_gap = statistics.fmean(gaps)
    best = _best_case(result, len(gaps), num_levels)
    best_mean = statistics.fmean(best) if best else 0.0

    # `realized` is what the colony did as a fraction of what the budget allowed;
    # `required` is what it would have had to do to clear the absolute threshold.
    realized = mean_gap / best_mean if best_mean else 0.0
    required = flat_fraction / best_mean if best_mean else float("inf")
    budget_note = (
        f" Best case at this budget averages {best_mean:.2%}, so the colony realized "
        f"{realized:.0%} of the concentration available to it and would have needed "
        f"{required:.0%} to clear the {flat_fraction:.0%} threshold."
    )

    if realized < INERT_REALIZED_FRACTION:
        verdict = "INERT"
        detail = (
            f"tau covered {realized:.0%} of the distance from uniform its own budget "
            f"allows ({mean_gap:.2%} against a best case of {best_mean:.2%}). Pheromone "
            "is carrying essentially no signal, so selection is driven by eta alone -- "
            "the colony is not searching and cross-round stigmergy is not happening."
        )
    elif required > UNDERPOWERED_REQUIRED_FRACTION:
        verdict = "UNDERPOWERED"
        detail = (
            f"tau is moving ({realized:.0%} of the best case, {mean_gap:.2%} below "
            f"uniform) but this run is too short to call it. Reporting SEARCHING would "
            f"take {required:.0%} of a best case that picks the same level every "
            "iteration -- a colony that has stopped exploring. Lengthen the run, or raise "
            "the deposit (`aco-q-deposit`) or the iteration budget (`aco-iters-start`), "
            "before reading a verdict here."
        )
    elif gaps[-1] < gaps[0]:
        verdict = "DEGRADING"
        detail = (
            f"tau started structured ({gaps[0]:.2%} below uniform) and ended flatter "
            f"({gaps[-1]:.2%}). Pheromone is washing out as rounds progress." + budget_note
        )
    else:
        verdict = "SEARCHING"
        detail = (
            f"tau is {mean_gap:.2%} below uniform on average, {gaps[-1]:.2%} by the final "
            "round." + budget_note
        )
    return {
        "question": "Is the colony searching?",
        "verdict": verdict,
        "detail": detail,
        "entropy_ceiling": ceiling,
        "entropy_per_round": entropies,
        "mean_gap_fraction": mean_gap,
        "best_case_gap_fraction": best_mean,
        "realized_fraction": realized,
        "required_fraction": required,
    }


def _best_case(result: dict, rounds: int, num_levels: int) -> list[float]:
    """The fastest tau could have concentrated under this run's own colony settings.

    Read from the result's resolved run_config rather than from defaults: a run with a
    smaller `aco-iters-start` or a lower fitness has a lower ceiling, and comparing it
    against the plan's default budget would understate how well it did.
    """
    run_config = result.get("config", {}).get("run_config", {})
    fitnesses = _round_metric(result, "best_fitness")
    return best_case_gaps(
        rounds,
        iters_start=int(run_config.get("aco-iters-start", 10)),
        iters_end=int(run_config.get("aco-iters-end", 4)),
        fitness=statistics.fmean(fitnesses) if fitnesses else MEASURED_FITNESS,
        num_levels=num_levels,
        num_clients=int(run_config.get("num-clients", 4)),
        rho=float(run_config.get("aco-rho", 0.1)),
        tau0=float(run_config.get("aco-tau0", 1.0)),
        rho_round=float(run_config.get("aco-rho-round", 0.1)),
        q_deposit=float(run_config.get("aco-q-deposit", 1.0)),
        global_best_every=int(run_config.get("aco-global-best-every", 5)),
    )


def check_deposit_floor(result: dict) -> dict:
    fitness = _round_metric(result, "best_fitness")
    scales = _round_metric(result, "delta_mean_sq_norm")
    if not fitness:
        return {
            "question": "Has the deposit floor engaged?",
            "verdict": "NO DATA",
            "detail": "no `best_fitness` in the result file.",
        }

    negative = [i + 1 for i, f in enumerate(fitness) if f <= 0.0]
    scale_note = (
        f" delta_mean_sq_norm ranged {min(scales):.4f}–{max(scales):.4f}." if scales else ""
    )
    if not negative:
        verdict = "CLEAR"
        detail = (
            f"best_fitness stayed positive every round "
            f"({min(fitness):.4f}–{max(fitness):.4f}), so every round deposited."
            + scale_note
        )
    elif len(negative) == len(fitness):
        verdict = "ENGAGED"
        detail = (
            f"best_fitness was <= 0 in all {len(fitness)} rounds "
            f"({min(fitness):.4f}–{max(fitness):.4f}): every deposit was zeroed by the "
            "max(F, 0) floor, which fully explains an INERT verdict above." + scale_note
        )
    else:
        verdict = "PARTIAL"
        detail = (
            f"best_fitness was <= 0 in {len(negative)} of {len(fitness)} rounds "
            f"(rounds {negative[:8]}{'...' if len(negative) > 8 else ''}); those rounds "
            "deposited nothing." + scale_note
        )
    return {
        "question": "Has the deposit floor engaged?",
        "verdict": verdict,
        "detail": detail,
        "best_fitness_per_round": fitness,
        "negative_rounds": negative,
    }


def check_degenerate_optimum(result: dict) -> dict:
    """Does the fitness rank a single-client answer above the FedAvg point?

    `corner_margin` is exact, not sampled: F at a vertex has a closed form
    (`gamma_1 * cos(delta_j, robust_mean) - gamma_3 * log K`) because dispersion vanishes
    there. Positive in most rounds means the colony's target is degenerate, whatever the
    colony itself is doing.
    """
    margins = _round_metric(result, "corner_margin")
    alpha_max = _round_metric(result, "alpha_max")
    num_clients = result.get("config", {}).get("run_config", {}).get("num-clients", "?")

    # The value that would have fixed it, taken from the run itself. The max across rounds,
    # not the mean: gamma_entropy is set once for the whole run, so a value that clears the
    # average round still leaves the worst rounds degenerate. Finite values only -- `inf`
    # means a round whose base weights were a single client, which no real partition
    # produces and which would otherwise swallow the max.
    required = [g for g in _round_metric(result, "required_gamma_entropy") if g != float("inf")]
    current = result.get("config", {}).get("run_config", {}).get("aco-gamma-entropy")
    if required:
        worst = max(required)
        fix = (
            f" **Set `aco-gamma-entropy` to at least {worst:.3f}** (currently "
            f"{current if current is not None else 'default 0.1'}); that is the largest "
            f"zero-crossing over {len(required)} rounds, computed in closed form from each "
            f"round's own Gram matrix. Add headroom -- landing on the crossing leaves the "
            f"corner tied, not beaten -- and re-run this gate to confirm, because the value "
            f"moves with K and the partition."
        )
    else:
        fix = (
            " This result has no `required_gamma_entropy` (it predates 2026-09-19), so the "
            "value needed is not recorded. Re-run one round to get it; it is a closed form "
            "over the Gram matrix already computed, and costs nothing."
        )

    if not margins:
        return {
            "question": "Is the fitness optimum degenerate?",
            "verdict": "NO DATA",
            "detail": (
                "no `corner_margin` in the result file -- it predates 2026-09-18. Re-run "
                "to get it; it costs O(K^2) once per round."
            ),
        }

    positive = sum(1 for m in margins if m > 0)
    rate = positive / len(margins)
    mean_margin = statistics.fmean(margins)
    concentration = f" Observed alpha_max averaged {statistics.fmean(alpha_max):.3f}." if alpha_max else ""

    if rate >= 0.5:
        verdict = "DEGENERATE"
        detail = (
            f"the best single-client vertex outscored the FedAvg point in {positive}/"
            f"{len(margins)} rounds (mean margin {mean_margin:+.4f}) at K={num_clients}. "
            "The fitness is asking for 'use one client, discard the rest', so a colony "
            "that finds it is working correctly toward a useless answer -- and nothing "
            "else in this report can tell the two apart." + fix + concentration
        )
    elif mean_margin > -0.02:
        verdict = "MARGINAL"
        detail = (
            f"the corner lost, but only just (mean margin {mean_margin:+.4f} at "
            f"K={num_clients}, positive in {positive}/{len(margins)} rounds). A different "
            "partition or a noisier round could flip it, so this is not a comfortable "
            "pass." + fix + concentration
        )
    else:
        verdict = "CLEAR"
        detail = (
            f"the FedAvg point outscores the best single-client vertex by {-mean_margin:.4f} "
            f"on average at K={num_clients}; the concentration penalty is doing its job."
            + concentration
        )
    return {
        "question": "Is the fitness optimum degenerate?",
        "verdict": verdict,
        "detail": detail,
        "mean_corner_margin": mean_margin,
        "degenerate_round_fraction": rate,
        "required_gamma_entropy": max(required) if required else None,
    }


def check_beats_fedavg(fedaco: dict, fedavg: dict | None) -> dict:
    fallbacks = _round_metric(fedaco, "fallback_used")
    rate = statistics.fmean(fallbacks) if fallbacks else None
    aco_score = fedaco.get("final", {}).get("final_test_macro_f1")

    if fedavg is None:
        detail = (
            "no FedAvg run found to compare against -- run one with the same seed and "
            "partition. If you believe you did run one, the listing under `strategies "
            "present` above says what is actually on disk: the first GPU gate reported "
            "this while a FedAvg run was visibly starting in the same cell, and the log "
            "alone could not distinguish 'still running' from 'wrote a file this check "
            "cannot match'."
        )
        verdict = "NO BASELINE"
        margin = None
    else:
        avg_score = fedavg.get("final", {}).get("final_test_macro_f1")
        if aco_score is None or avg_score is None:
            return {
                "question": "Does FedACO beat FedAvg?",
                "verdict": "NO DATA",
                "detail": "one of the runs has no final_test_macro_f1.",
            }
        margin = aco_score - avg_score
        verdict = "AHEAD" if margin > 0 else "BEHIND"
        detail = f"FedACO {aco_score:.4f} vs FedAvg {avg_score:.4f} (margin {margin:+.4f})."

    if rate is not None:
        detail += f" Fallback fired in {rate:.0%} of rounds."
        if rate >= 0.5:
            detail += (
                " At that rate FedACO *was* FedAvg for most of the run, so any margin"
                " above is largely FedAvg's own number, not the colony's."
            )
    return {
        "question": "Does FedACO beat FedAvg?",
        "verdict": verdict,
        "detail": detail,
        "fallback_rate": rate,
        "margin": margin,
    }


def wall_clock_note(result: dict) -> str:
    final = result.get("final", {})
    total = final.get("wall_clock_s")
    rounds = final.get("num_rounds_completed")
    if not total or not rounds:
        return "no wall-clock recorded."
    per_round = total / rounds
    run_config = result.get("config", {}).get("run_config", {})
    clients = run_config.get("num-clients", "?")
    epochs = run_config.get("local-epochs", "?")
    return (
        f"{per_round:.1f}s per round ({total:.0f}s over {rounds} rounds, K={clients}, "
        f"{epochs} local epoch(s)). A 100-round run at this shape would take "
        f"{per_round * 100 / 3600:.2f} GPU-hours. docs/EXPERIMENT_LOG.md's compute table "
        "is estimates only and says to replace it with exactly this number."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/fl")
    parser.add_argument("--num-levels", type=int, default=11, help="aco-num-levels used in the run")
    parser.add_argument(
        "--flat",
        type=float,
        default=DEFAULT_FLAT_FRACTION,
        help="fraction below max entropy that counts as uniform tau (default 0.02)",
    )
    parser.add_argument("--out", default=None, help="optional path to write the report JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "exit nonzero unless the colony is SEARCHING and the fitness optimum is not "
            "DEGENERATE -- for gating a later, expensive step on this check passing"
        ),
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = REPO_ROOT / results_dir

    fedaco_runs = load_results(results_dir, "fedaco")
    fedavg_runs = load_results(results_dir, "fedavg")
    if not fedaco_runs:
        print(f"No FedACO result files under {results_dir}.")
        print("Run one first -- see `make validate-fedaco` or notebooks/colab_fl_smoke.ipynb.")
        return 2

    fedaco = max(fedaco_runs, key=lambda r: r.get("final", {}).get("num_rounds_completed", 0))
    fedavg = (
        max(fedavg_runs, key=lambda r: r.get("final", {}).get("num_rounds_completed", 0))
        if fedavg_runs
        else None
    )

    print(f"FedACO health check\n  run: {fedaco['_path']}")
    # Named explicitly because check 4's "NO BASELINE" is ambiguous on its own -- it cannot
    # tell a FedAvg run that has not finished from one whose result file this check failed to
    # match. Reading the directory once here settles it in the output instead of in the next
    # GPU session.
    everything = load_results(results_dir)
    present = sorted({str(r["config"].get("strategy", "?")) for r in everything})
    print(f"  strategies present in {results_dir}: {', '.join(present) or 'none'}"
          f" ({len(everything)} result file(s) with per-round records)")
    if fedavg:
        print(f"  baseline: {fedavg['_path']}")
    print(f"  timing: {wall_clock_note(fedaco)}\n")

    checks = [
        check_colony_searching(fedaco, args.num_levels, args.flat),
        check_deposit_floor(fedaco),
        check_degenerate_optimum(fedaco),
        check_beats_fedavg(fedaco, fedavg),
    ]
    for i, check in enumerate(checks, 1):
        print(f"{i}. {check['question']}")
        print(f"   {check['verdict']}: {check['detail']}\n")

    verdict = checks[0]["verdict"]
    searching = verdict == "SEARCHING"
    if checks[2]["verdict"] == "DEGENERATE":
        print(
            "Before anything else: the fitness optimum is degenerate at this K. A colony\n"
            "that searches well will find 'use one client and discard the rest', and every\n"
            "other signal in this report -- fallback rate, best vs FedAvg fitness, a\n"
            "confident alpha -- will read as success while it does. Fix that first; the\n"
            "other three questions are not meaningful until it is negative. Check 3 above "
            "names the gamma_entropy that closes it.\n"
        )
    if verdict == "UNDERPOWERED":
        print(
            "This run cannot answer question 1 -- it is too short for tau to have left its\n"
            "uniform initialization by a margin the threshold can see. That is not evidence\n"
            "either way about the mechanism. Re-run longer, or with a larger deposit or\n"
            "iteration budget, before concluding anything; `scripts/analyze_pheromone_"
            "dynamics.py`\nprints what each budget makes reachable."
        )
    elif not searching:
        print(
            "The colony is not searching. Before Phase 6 spends GPU-hours, this needs\n"
            "resolving -- a sweep built on an inert mechanism produces results that say\n"
            "nothing about the method's contribution. Start with check 2: if the deposit\n"
            "floor is ENGAGED, the fix is the deposit rule (a rank-based or\n"
            "baseline-centred deposit), not the fitness weights."
        )
    else:
        print("The mechanism is live on real data. Phase 6 can be built against it.")

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"run": fedaco["_path"], "checks": checks}, indent=2))
        print(f"\nWrote {out_path}")

    # DEGENERATE fails --strict as well, and that is the point of having the flag. A1 costs
    # 80 cells to answer "does the colony beat random search at equal budget", and its four
    # controls optimize the SAME fitness -- so on a degenerate landscape they all inherit the
    # same useless optimum and a tie says nothing about ACO. The plan's week-5 go/no-go is
    # only interpretable once check 3 is clear, so a caller spending that budget should be
    # able to make it conditional on this exit code rather than on someone reading the text.
    degenerate = checks[2]["verdict"] == "DEGENERATE"
    return 1 if (args.strict and (not searching or degenerate)) else 0


if __name__ == "__main__":
    sys.exit(main())
