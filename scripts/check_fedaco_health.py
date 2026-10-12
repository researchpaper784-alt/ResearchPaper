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

# A run smaller than this cannot certify a 100-round K=20 sweep. Deliberately well below the
# real sweep shape -- the point is to exclude smoke and fixture runs, not to demand a rerun of
# the thing being gated.
MIN_ROUNDS_TO_CERTIFY = 15
MIN_CLIENTS_TO_CERTIFY = 10

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

    # A non-positive best case is not a colony verdict, it is the absence of one, and
    # reporting it as INERT is worse than reporting nothing. `best_case_gaps` scales the
    # deposit by the run's mean `best_fitness`, and raising `aco-gamma-entropy` to 0.6 to
    # close the corner drove that mean negative -- so the "best case" came out at -0.000002
    # and `mean_gap / best_mean` printed **-315130%** of the budget, which then fell below
    # the INERT threshold and was announced as "pheromone is carrying essentially no
    # signal". The colony had not been measured at all; the denominator had changed sign.
    #
    # The real content of a non-positive mean fitness belongs to check 2: the MAX-MIN rule
    # deposits `rho * Q * max(F, 0)`, so a negative fitness deposits nothing by design. That
    # is a fitness problem, and check 1 cannot see it.
    if best_mean <= 0:
        return {
            "question": "Is the colony searching?",
            "verdict": "NO DATA",
            "detail": (
                f"cannot be answered: this run's mean `best_fitness` is non-positive, so the "
                f"best-case concentration the deposit rule permits is {best_mean:.2%} -- zero "
                "or less. The MAX-MIN deposit is `rho * Q * max(F, 0)`, so a colony given a "
                "negative fitness deposits nothing however well it searches, and tau cannot "
                "move. Read check 2, not this one: the question is why the fitness is "
                "negative, not whether the colony is trying."
            ),
            "entropy_ceiling": ceiling,
            "entropy_per_round": entropies,
            "mean_gap_fraction": mean_gap,
            "best_case_gap_fraction": best_mean,
        }

    # `realized` is what the colony did as a fraction of what the budget allowed;
    # `required` is what it would have had to do to clear the absolute threshold.
    realized = mean_gap / best_mean
    required = flat_fraction / best_mean
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

    # F at the REFERENCE point, which decides whether the fitness is a usable objective at
    # all -- and which was logged as `fedavg_fitness` from the start while nothing read it.
    # The first gate run's numbers: mean -0.0005, negative in 7 of 15 rounds, at the DEFAULT
    # gamma_entropy. The aggregation FedACO exists to improve on scores zero, so a colony
    # beating it by +0.16 is beating noise -- and every attempt to fix the corner by raising
    # the penalty pushes this further below zero until nothing deposits at all. Reported here
    # because "the objective does not rank the reference point above zero" is a different and
    # prior failure to "some rounds did not deposit"; the second is a symptom of the first.
    reference = _round_metric(result, "fedavg_fitness")
    reference_note = ""
    if reference:
        ref_mean = statistics.fmean(reference)
        ref_negative = sum(1 for f in reference if f <= 0.0)
        reference_note = (
            f" F at the FedAvg point itself averaged {ref_mean:+.4f} and was <= 0 in "
            f"{ref_negative}/{len(reference)} rounds."
        )
        if ref_negative > len(reference) / 3:
            reference_note += (
                " **This is the prior problem.** The objective does not rank the aggregation"
                " this method exists to improve on above zero, so the colony's margin over it"
                " is a margin over noise -- and raising `aco-gamma-entropy` to close the"
                " corner pushes it further down until nothing deposits. Use"
                " `aco-dispersion-reference=aggregate`, which measures"
                " ||Delta(alpha) - Delta_rob||^2 -- how far the AGGREGATE lands from the"
                " robust consensus. Averaging cancels per-client noise, so an interior point"
                " is genuinely closer than any single client rather than merely less"
                " penalized. Not `base`: that one is exactly LINEAR in alpha, so it has no"
                " interior optimum and a vertex on the nearest client is cheaper than any"
                " mix."
            )
    if not negative:
        verdict = "CLEAR"
        detail = (
            f"best_fitness stayed positive every round "
            f"({min(fitness):.4f}–{max(fitness):.4f}), so every round deposited."
            + scale_note + reference_note
        )
    elif len(negative) == len(fitness):
        verdict = "ENGAGED"
        detail = (
            f"best_fitness was <= 0 in all {len(fitness)} rounds "
            f"({min(fitness):.4f}–{max(fitness):.4f}): every deposit was zeroed by the "
            "max(F, 0) floor, which fully explains an INERT verdict above." + scale_note + reference_note
        )
    else:
        verdict = "PARTIAL"
        detail = (
            f"best_fitness was <= 0 in {len(negative)} of {len(fitness)} rounds "
            f"(rounds {negative[:8]}{'...' if len(negative) > 8 else ''}); those rounds "
            "deposited nothing." + scale_note + reference_note
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
    parser.add_argument(
        "--require-sound",
        action="store_true",
        help=(
            "exit nonzero unless the mechanism is affirmatively SOUND: check 1 SEARCHING, "
            "check 3 CLEAR, and the run long enough and large enough to stand in for a "
            "100-round K=20 sweep. Stricter than --strict, which only blocks on a fatal "
            "verdict. Use this before spending GPU hours on ablations of colony internals"
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

    def _most_relevant(runs: list[dict]) -> dict:
        """Longest run, then most recent. The tiebreak is not cosmetic.

        Changing `aco-gamma-entropy` changes the config hash, so a re-run at a new value
        writes a NEW file beside the old one -- both completed, both 15 rounds. `max` returns
        the first maximal element, so without a tiebreak this check could read the previous
        attempt and report the old verdict against the new setting. That is the exact shape
        of mistake that costs a GPU session: you change the value, re-run, and the report
        tells you nothing changed.
        """
        return max(
            runs,
            key=lambda r: (
                r.get("final", {}).get("num_rounds_completed", 0),
                Path(r["_path"]).stat().st_mtime,
            ),
        )

    fedaco = _most_relevant(fedaco_runs)
    fedavg = _most_relevant(fedavg_runs) if fedavg_runs else None

    print(f"FedACO health check\n  run: {fedaco['_path']}")
    # Named explicitly because check 4's "NO BASELINE" is ambiguous on its own -- it cannot
    # tell a FedAvg run that has not finished from one whose result file this check failed to
    # match. Reading the directory once here settles it in the output instead of in the next
    # GPU session.
    everything = load_results(results_dir)
    present = sorted({str(r["config"].get("strategy", "?")) for r in everything})
    print(f"  strategies present in {results_dir}: {', '.join(present) or 'none'}"
          f" ({len(everything)} result file(s) with per-round records)")
    # Printed because this is the knob being tuned between attempts, and a report that does
    # not name the value it was produced at cannot be told apart from a stale one.
    reported_run_config = fedaco.get("config", {}).get("run_config", {})
    reported_gamma = reported_run_config.get("aco-gamma-entropy")
    # `aco-dispersion-reference` belongs here for the same reason gamma does, and its absence
    # cost a round trip: a run was made to test it and the report named neither the setting
    # nor anything that identified which of the three shapes had actually been used.
    reported_ref = reported_run_config.get("aco-dispersion-reference")
    print(f"  aco-gamma-entropy of the run above: {reported_gamma if reported_gamma is not None else 'unset (default 0.1)'}")
    print(f"  aco-dispersion-reference: {reported_ref if reported_ref is not None else 'unset (default weighted_mean)'}")
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

    # What --strict blocks on, and what it deliberately does not.
    #
    # DEGENERATE blocks. A1 costs 80 cells to ask "does the colony beat random search at
    # equal budget", and its four controls optimize the SAME fitness -- so on a degenerate
    # landscape they inherit the same useless optimum and a tie says nothing about ACO.
    #
    # UNDERPOWERED does NOT block, and getting this wrong would be worse than not having the
    # flag. It does not mean the colony failed; it means THIS RUN CANNOT TELL, because the
    # gate is 15 rounds by design and tau has not had time to leave uniform by a margin the
    # threshold can see. Blocking on it would leave a caller who has just fixed the corner
    # still refused, with the only remedy being to lengthen a gate whose whole purpose is to
    # be short -- and A1 itself runs at 100 rounds, where the question is answerable.
    # Inconclusive is not failure, and a gate that cannot say so is a gate nobody can pass.
    #
    # INERT and DEGRADING do block: those are real negative verdicts about the mechanism.
    inconclusive = {"UNDERPOWERED", "NO DATA"}
    colony_verdict = checks[0]["verdict"]
    colony_blocks = not searching and colony_verdict not in inconclusive
    degenerate = checks[2]["verdict"] == "DEGENERATE"
    if args.strict and colony_verdict in inconclusive and not degenerate:
        print(
            f"\n--strict: check 1 is {colony_verdict}, which is inconclusive rather than a\n"
            "failure -- this gate is too short to answer it, and A1 runs at 100 rounds where\n"
            "it is answerable. Not treating it as a blocker. Check 3 is what --strict gates on."
        )
    if args.strict and (colony_blocks or degenerate):
        return 1

    # `--require-sound` answers a DIFFERENT question from `--strict`, and conflating them was a
    # real fault. `--strict` asks "is anything here fatal?", which is what person B's 15-round
    # gate needs: UNDERPOWERED must not block it, or a caller who has just fixed the corner is
    # still refused with no remedy but to lengthen a gate whose whole purpose is to be short.
    #
    # `--require-sound` asks "is the mechanism sound enough to ablate?", which is what person
    # C's notebook needs before spending 515 cells and 107-215 GPU-hours on A2/A4-A8: those
    # sweeps measure colony internals, so on a colony that is not searching every arm collapses
    # toward the same behaviour and the ablation measures noise rather than the term it names.
    #
    # Reading `--strict`'s exit 0 as "sound" is how a 6-round K=6 toy run on a synthetic fixture
    # certifies the mechanism: checks 2 and 3 come back CLEAR because a small K makes the
    # entropy penalty relatively stronger, check 1 is inconclusive and does not block, and the
    # caller unlocks the sweeps. "Not proven degenerate" is not "sound".
    if args.require_sound:
        blockers = []
        if colony_verdict != "SEARCHING":
            blockers.append(
                f"check 1 is {colony_verdict}, not SEARCHING -- ablating the colony's own "
                "machinery needs evidence the colony searches at all"
            )
        if checks[2]["verdict"] != "CLEAR":
            blockers.append(f"check 3 is {checks[2]['verdict']}, not CLEAR")

        # A run far smaller than the sweeps it is certifying cannot stand in for them.
        rounds = len([r for r in fedaco.get("rounds", []) if r.get("train_best_fitness") is not None])
        clients = int((fedaco.get("config") or {}).get("run_config", {}).get("num-clients", 0))
        if rounds < MIN_ROUNDS_TO_CERTIFY:
            blockers.append(
                f"the run judged is {rounds} rounds; at least {MIN_ROUNDS_TO_CERTIFY} are "
                "needed before it stands in for a 100-round sweep"
            )
        if clients and clients < MIN_CLIENTS_TO_CERTIFY:
            blockers.append(
                f"the run judged is K={clients}; the sweeps being gated run K=20, and a small "
                "K makes the concentration penalty relatively stronger, so check 3 passing "
                "here says little about them"
            )

        if blockers:
            print("\n--require-sound: REFUSING to certify the mechanism.")
            for blocker in blockers:
                print(f"  - {blocker}")
            print(
                "\nThis is not a verdict that the method is broken; it is a refusal to call it "
                "sound\non this evidence. Run the gate at full length on the real dataset first."
            )
            return 1
        print("\n--require-sound: mechanism certified -- check 1 SEARCHING, check 3 CLEAR, "
              f"on a {rounds}-round K={clients} run.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
