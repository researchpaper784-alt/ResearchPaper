# 5. Results

**Nothing in this file may be hand-typed as a metric value.** Plan §9.3: every figure and table
is generated from `results/`, and `make verify-phase9` checks that regeneration is byte-identical.
Prose below refers to `paper/tables/*.csv`/`.md` and `paper/figures/*.png` for exact numbers
rather than repeating them to four decimals; read the tables alongside this section.

All planned sweeps have run (2026-10-11): gate, A1, A2, main, R1, R2 — 366/366 cells. What
follows is a first pass at the prose, grounded in the real tables now in this repo, not a
skeleton. Two items below are flagged **not settled** rather than written as findings, because
writing them as findings would overstate what the data supports.

## 5.1 Main results

Table 1 (`paper/tables/main.csv`/`.md`): 7 strategies (FedAvg, FedProx, Trimmed-Mean, Krum,
FedLAW, SCAFFOLD, FedACO) × 3 regimes (`dirichlet_0.1`, `dirichlet_0.3`, `iid`) × 8 seeds,
168/168 cells complete. `p (Holm)` and Cohen's `d` are populated now (a second effect of the
Figures-1/2 fix below — `docs/OPEN_QUESTIONS.md` has the detail); read them against the
table's own footer, which states the 8-seed/18-comparison adjusted-p floor directly — these
have not been independently re-read or interpreted further in this draft.

**The IID band check fails, and the result points at one baseline, not at the method under
test.** `scripts/check_iid_band.py` — written specifically to catch this before it is
over-interpreted — reports the spread of strategy means under `iid` at roughly 14x the expected
band (and well above twice the seed noise). Reading the raw result JSONs rather than only the
aggregate: **SCAFFOLD has collapsed to predicting a single class in the large majority of its
runs**, across all three regimes, not recovering by the final round. Its row in Table 1 is close
to the score of a trivial one-class classifier and should not be read as a measurement of
SCAFFOLD's relative merit on this task.

**Root-caused and fixed, 2026-10-11 (`docs/OPEN_QUESTIONS.md`, "FIXED, confirmed over the full
100-round budget") — not yet re-run on real data.** A genuine numerical blow-up in the SCAFFOLD
control-variate update (not an environment artifact): under this sweep's real settings
(`min-client-size=5`, `local-batch-size=32`, skewed partitioning), several clients legitimately
see one local batch per round, which the update rule's `/(tau·lr)` term then turns into a ~100x
amplified contribution to the shared control variate — an undamped positive feedback loop that
reaches NaN by round ~16 of 100. Reproduced directly (`scripts/debug_scaffold_divergence.py`,
no Ray, no Kaggle needed) and fixed with three bounds, none of them alone sufficient: clipping a
client's contribution to its own actual update magnitude, an EMA + hard norm clip on the server
side's accumulator, and clipping the correction term itself where it is injected into local
training (a single bounded-but-nonzero correction, applied every round under SGD momentum, was
enough to still diverge even with the first two in place). Confirmed stable over the real
100-round budget: `‖w‖` grows from 67.7 to 73.7 — ordinary training, not divergence.

**This table's SCAFFOLD row is still the old, broken one** — the fix has not yet been re-run on
Kaggle (~3.7 GPU-h: SCAFFOLD is 24 of `main_reduced.yaml`'s 168 cells, not the whole sweep). Until that real re-run
lands, the action from before still holds: **exclude the SCAFFOLD row or mark it explicitly as
non-convergent** wherever this table is read. Krum and FedLAW also sit below the
FedAvg/FedProx/Trimmed-Mean/FedACO cluster even under `iid` and are worth the same scrutiny,
though neither shows SCAFFOLD's single-class signature.

**Reading the cluster that does converge.** Under `dirichlet_0.1` (the most heterogeneous
regime run), FedProx leads, FedAvg and Trimmed-Mean follow closely, and FedACO is near the
bottom of the converging group — its fallback rate (the fraction of rounds the colony's
construction defaulted to the FedAvg point) is 0% here, so the low score is not explained by the
colony silently reverting to FedAvg; it is actually searching and landing somewhere worse. Under
`dirichlet_0.3`, FedACO's fallback rate rises to 8%, and under `iid`, to 1%, while its score sits
within the converging cluster at `iid` and is the second-worst (after the two flagged baselines)
at `dirichlet_0.3`. **There is no regime in this table where FedACO is the best strategy**, and
no clean trend of "gain grows with heterogeneity" — if anything the opposite, since its
relative position is worst at `dirichlet_0.3`, not at the most heterogeneous setting. Figure 9
(`gain_vs_heterogeneity.png`, generated) is the sharper way to look at this than the table.

Figures 1–2 (`convergence.png`, `comparison.png`) **are now generated** — the
`figure_convergence`/`figure_comparison` bug (`docs/OPEN_QUESTIONS.md`, "Figures 1 and 2 ...
skip every real sweep") is fixed as of 2026-10-11. They confirm the reading above directly: the
SCAFFOLD baseline (§5.1, item 1 in `HANDOVER.md`) is visibly flat near the bottom of every
panel, not recovering across 100 rounds, and `comparison.png`'s error bars make FedACO's lack
of a win in any regime immediately legible rather than something to infer from the table.

## 5.2 The equal-budget comparison (A1) — real data, not the synthetic screen

5 search methods (ACO, PSO, GA, coordinate-grid, random) × 2 regimes (`iid`, `dirichlet_0.3`) ×
8 seeds, 80/80 cells. This is the experiment the method's original premise rests on, and real
data changes the shape of the synthetic-screen result rather than reversing it.

At `dirichlet_0.3`, **every one of the five search methods scores below the FedAvg point** —
including plain random search, which is the single worst of the five here. ACO is not uniquely
bad (PSO lands within a hundredth of it) but is also not rescued by real data: it remains in the
worse half, clearly behind GA and coordinate-grid, which come within roughly three- and
four-hundredths of the FedAvg point respectively — a real cost, but the smallest of the five. At
`iid`, all five cluster within two-hundredths of each other and of the FedAvg point, which is
the narrow-band behavior the pipeline expects there and not where this comparison has anything
to say.

**This is a different negative result than the synthetic screens gave**, which found every
*other* method beating the FedAvg point and ACO alone failing to — a result specifically about
ACO. Here, the whole family of equal-budget, equal-fitness search underperforms FedAvg under
real heterogeneity, GA and coordinate-grid included, just by a smaller margin than ACO and
random. Full numbers, the cross-check against Table 1's own FedACO/`dirichlet_0.3` row (which
agree exactly, since it is the same cell under two sweeps' bookkeeping), and the reading of what
this does and does not say about the method are in `paper/06_LIMITATIONS.md` §6.1.

**Not yet done:** this is a first read of `ablation.csv`, not run through
`scripts/aggregate_results.py`'s Wilcoxon/Holm significance machinery against this specific
5-arm family. The ranking is clear; whether any gap here clears significance at 8 seeds is not
yet checked.

## 5.2b Cross-round persistence (A2) — the one structural claim left, and it holds

This is the ablation `HANDOVER.md` calls the most important in the project: FedAWA, Adp-FL-PSO
and FedPSO (the published prior work closest to this method, `paper/REFERENCES.md`) are all
stateless across rounds, so cross-round pheromone persistence is the one mechanism this work has
that they do not. A2 tests it directly: `full` persistence vs `none` vs the method's own default
(`decayed`), 5 seeds, both heterogeneous regimes it ran in (`dirichlet_0.1`, `dirichlet_0.3`).

**Full persistence beats both alternatives in both regimes.** At `dirichlet_0.1` the gap is
large — full persistence scores roughly double the no-persistence arm, and both comfortably
beat the method's own `decayed` default, which is the *worst* of the three there. At
`dirichlet_0.3` the ordering is full > default > none, a smaller but consistent gap in the same
direction. **This is the one place in the real-data results where the ablated mechanism clearly
helps**, and it is the part of the method with no analogue in the closest published prior work —
unlike §5.1/§5.2's results, this does not need hedging about what it does and does not show.

Two things worth stating precisely rather than overclaiming: (i) the project's own default
(`decayed` persistence) is not the best arm in this ablation — `full` is — which is itself
worth a sentence in the paper and a candidate default-config change; (ii) only 5 seeds ran here
(`ablation_a2_reduced.yaml`'s own budget cut), so this finding is a clear effect size, not yet a
significance claim at the standard this project otherwise holds itself to (§6.5's 8-seed floor).

## 5.3 Robustness

Table 4 (`paper/tables/robustness.md`): label-flipping at 30% (R1), Gaussian and sign-flip
update attacks at 30% (R2), against FedAvg / Krum / Trimmed-Mean / FedACO, with the `clean` arm
(R1) as the 0%-attacker reference. 80/80 cells across both sweeps.

**Regenerated from the real raw JSONs, 2026-10-11** (`results/fl/robustness`, 80/80 files,
pulled from Kaggle) — the table below matches the earlier reconstruction from the Kaggle
session's printed log exactly (same numbers to four decimals), which is a good independent
cross-check that the reconstruction was accurate. `.csv`/`.tex` and `paper/figures/robustness.png`
(Figure 6) are now generated through the normal pipeline; `make verify-phase9` still needs a
run to confirm byte-identical regeneration, but the data itself is no longer provisional.

Reading the `dirichlet_0.3` table as printed: under the clean arm and every attack arm, the
ranking is consistently **FedAvg ≳ Trimmed-Mean > FedACO ≈ Krum**, with Krum clearly worst under
every condition including clean — a Byzantine-robust aggregator that is also the weakest
performer with no attacker present. Sign-flip is the hardest attack for every strategy (lowest
scores across the board for all four), consistent with it being a norm-preserving,
inner-product-manipulating attack that the literature already identifies as a known-hard case
for Krum and coordinate-wise methods (Xie et al., *Fall of Empires*, 2019 —
`paper/REFERENCES.md`). **FedACO does not buy Byzantine resilience as a side effect** in this
data: it tracks its own clean-arm score downward under attack by roughly the same margin as
FedAvg does, rather than degrading less. This matches §5.2's reading — the method is not
distinguishing itself from the simplest baseline, here under adversarial conditions rather than
under the equal-budget search comparison.

Per §6.5, plan §8's acceptance criterion holds: every arm is reported, including the ones FedACO
loses, rather than only the favorable ones.

## 5.4 Efficiency

Already measured and does not depend on the GPU sweeps above: `aco_time_ms + gram_time_ms`
averaged roughly 198 ms against 7,500 ms per round at K=10 on the one real-hardware measurement
this project has — about 2.6% of wall-clock. The microbenchmark
(`scripts/bench_aggregation_overhead.py`) extends this across K from 5 to 200 on CPU, anchored
to that one real GPU measurement (`docs/EXPERIMENT_LOG.md`, "claim C3 holds, but not for the
reason it claims"): overhead *share* falls from roughly 3% at K=5 to under 1% at K=200, i.e. it
improves with K rather than degrading.

**Report honestly:** the constant is small where measured, and the share *falls* with K over the
measured range — the opposite direction from what an uncompensated $O(K^2)$ cost would show
(fixed per-round cost grows faster than K² only if the proportionality constant is large enough
to dominate at small K, which it is not here). `overhead.yaml`'s GPU sweep (K up to 200 on the
real dataset) was never run and is not needed for this claim — the CPU proxy is anchored and
argued as trustworthy in the log entry above. Plan figure 5's $K^2$ overlay should show the
measured curve and say plainly that the data does not fit a dominant-quadratic shape over this
range, not fit one anyway.

## 5.5 Interpretability

Figure 3-equivalent (`alpha_heatmap.png`) and Figure 4-equivalent (`alpha_entropy.png`) are
generated from `results/fl/main`. Figure 9 (`gain_vs_heterogeneity.png`) is generated and is the
figure to read alongside §5.1's "no clean trend" claim — it should be looked at directly rather
than summarized further here, since whether it shows a legible pattern or not is exactly the
kind of judgment call a reader should make from the picture.

Figure 8 (partition diagnostics, 9 files under `paper/figures/partition_*.pdf`, generated via
`make figures-data`) needs no federated run. Measured JS divergence confirms the regimes are
what they claim to be: `iid` at 0.007 (negligible), `dirichlet_0.1` at 0.541 (most
heterogeneous), `dirichlet_0.3` at 0.435, consistent with the ordering the main table and A1 use
throughout.

## Not in this paper

**Figure 7 (sensitivity heatmap).** A6 varies one hyperparameter at a time, so its cells form a
cross, not a grid. `make_figures.py` refuses to emit it rather than render a mostly-empty
heatmap that reads as a grid. Stated as a gap in §6.3.
