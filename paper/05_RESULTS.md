# 5. Results — skeleton, blocked on the sweeps

**Nothing in this file may be hand-typed.** Plan §9.3: every figure and table is generated from
`results/`, and `make verify-phase9` checks that regeneration is byte-identical. This file holds
the prose scaffold and the command that fills each slot.

## 5.1 Main results — `make main-reduced` then `make tables figures`

Table 1: 6 strategies × 3 regimes, mean ± std over 8 seeds, best in bold, Holm-corrected
significance markers against FedAvg. Figures 1–2: convergence curves with 95% CI bands (one
panel per regime) and the final macro-F1 bar chart.

*To write:* whether any gain exists, and whether it grows with measured heterogeneity (that is
Figure 9, not a claim to assert from the table). **If the IID column shows a wide spread between
strategies, stop** — `scripts/check_iid_band.py` gates on this, because with almost no
heterogeneity to exploit every method should land in a narrow band, and a large spread means
something other than the method is driving the result.

## 5.2 The equal-budget comparison — `make a1-reduced`

The experiment the premise rests on. 5 search methods × 2 regimes × 8 seeds.

*To write:* the ranking, and the anchoring mechanism from §1.3(2) as its explanation. Report
the fallback rate and `corner_margin` beside the scores — a method that "ties" while its safety
fallback fires in 44% of rounds is not tying, it is declining to act.

**Read `gate_fitness`'s verdict before interpreting anything here.** If the fitness corner is
still open, all five methods optimize the same degenerate objective and their ranking says
nothing about search.

## 5.3 Robustness — `make r1-reduced r2-reduced` then `make tables-robustness`

Table 4 and Figure 6: label-flipping at 30%, Gaussian and sign-flip update attacks at 30%,
against FedAvg / Krum / Trimmed-Mean, with the clean arm as the 0%-attacker reference.

*To write:* whether update inspection buys Byzantine resilience as a side effect. Plan §8's
acceptance criterion is explicit that **any configuration where the method loses stays in the
table** and is stated in §6. Do not quietly drop a losing arm.

## 5.4 Efficiency — `make overhead` then `make figures`

*Already measured, needs writing up:* `aco_time_ms + gram_time_ms` averaged **198 ms against
7,500 ms per round at K=10 — 2.6% of wall-clock.** The microbenchmark
(`scripts/bench_aggregation_overhead.py`) extends this across K ∈ {5…800}.

*To write honestly:* the constant is small where measured; the $O(K^2)$ *shape* does not appear
over the measured range. Report the measurement, not a fitted curve. Plan figure 5 asks for the
$K^2$ overlay — show it, and say it does not fit.

## 5.5 Interpretability — `make figures`

Figure 3 (α heatmap, clients × rounds, one representative run, with true client quality
annotated) and Figure 9 (gain vs measured Jensen–Shannon heterogeneity, scatter with fitted
trend). Figure 3 is the one reviewers remember: it either shows the method doing something
legible or it shows it doing nothing legible, and both are reportable.

Figure 8 (partition diagnostics) comes from `make figures-data` and needs no federated run.

## Not in this paper

**Figure 7 (sensitivity heatmap).** A6 varies one hyperparameter at a time, so its cells form a
cross, not a grid. `make_figures.py` refuses to emit it rather than render a mostly-empty
heatmap that reads as a grid. Stated as a gap in §6.3.
