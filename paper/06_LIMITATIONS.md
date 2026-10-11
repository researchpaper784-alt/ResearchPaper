# 6. Limitations

Stated as design constraints rather than as concessions. Plan §1.4 fixed most of these before
implementation; §6.1 and §6.5 are what the experiments added.

## 6.1 The central claim did not survive its own control

Our equal-budget comparison (§5.2) is the experiment the method's premise rests on, and it does
not support the premise. Three screens on a heterogeneous synthetic fixture found the colony
gaining less than uniform random search at identical evaluation budget and identical fitness.
**A1 on real data (80/80 cells, 8 seeds, `configs/experiment/ablation_a1_reduced.yaml`) has now
run, and changes the shape of the finding rather than reversing it:**

| regime | fedavg point | random | ga | coordinate_grid | pso | **aco** |
|---|---:|---:|---:|---:|---:|---:|
| iid | 0.7880 | +0.0053 | +0.0033 | +0.0043 | +0.0036 | **+0.0035** |
| dirichlet_0.3 | 0.6654 | -0.1163 | -0.0331 | -0.0397 | -0.1009 | **-0.1015** |

(macro-F1 gain over the FedAvg point, mean of 8 seeds; `dirichlet_0.3`'s aco row cross-checks
exactly against `paper/tables/main.csv`'s own fedaco/dirichlet_0.3 cell, -0.1015, which is the
same run under two different sweeps' bookkeeping.)

**What changed from the synthetic screens:** there, every non-aco method beat the FedAvg point
and aco alone did not — a result about aco specifically. Here, under real heterogeneity, **no
method beats the FedAvg point** — random search included, at -0.1163 the single worst score in
the table. The ranking among methods is preserved in neither direction: aco is not uniquely bad
(pso lands within 0.0006 of it), but it is also not rescued by real data — it remains in the
worse half, essentially tied with pso and clearly behind ga/coordinate_grid, both of which
confine the real cost of the full search to roughly a third of aco's and random's.

At `iid` all five cluster within 0.0020 of each other and the delta column shows nothing larger
than two-hundredths of a macro-F1 point — the narrow IID band the pipeline is built to expect
(§5.1's gate), and not where this comparison has anything to say.

**Reading this honestly:** on real, heterogeneous data, the equal-budget premise fails for the
whole family of search-based weighting, not only for ACO — which is a different and arguably
more interesting negative result than the synthetic screens alone suggested, but it does not
rescue ACO, which still underperforms the two cheaper, simpler search methods (ga,
coordinate_grid) under the one condition (dirichlet_0.3) where the comparison is informative.

We report this as the result rather than working around it, for two reasons. The mechanism is
identified and is a property of the *desirability heuristic's dynamic range*, not of ant colony
optimization — so it is informative about search-based aggregation weighting in general. And
the diagnostic that found it (§5.2) is reusable by anyone proposing a search over $\alpha$.

*(Numbers above are a first read of `paper/tables/ablation.csv` right after the sweep finished —
worth a second pass with the actual significance test, `scripts/aggregate_results.py`'s
Wilcoxon/Holm machinery, before this goes in the paper as a claim rather than a table.)*

Readers should take the paper's positive contributions as (i) the degeneracy diagnostic and its
closed-form remedy, (ii) the equal-budget methodology, and (iii) the de-duplicated split — not
as a claim that ACO improves federated aggregation.

## 6.2 Incompatible with plain secure aggregation

Like FedLAW, Krum, Trimmed-Mean and every other update-inspecting method, this approach
**requires per-client updates at the server**. Plain secure aggregation delivers only the sum,
so the two cannot be combined as-is. We do not hide this behind a "future work" sentence:
§5.3 runs a DP-noise experiment (R4) to show what survives when updates are perturbed, and
clustered or partial SecAgg — where the server sees group-level sums — is the concrete
compatibility path, not a hand-wave.

## 6.3 Hyperparameters

A metaheuristic brings its own hyperparameters, and a method that is knife-edge in them is not
usable. The planned sensitivity study (A6, 200 cells) **was cut for compute** and is not in this
paper. We therefore cannot claim the method is insensitive to its hyperparameters, and we do not.
What we do report is narrower and verifiable: one axis, `aco-gamma-entropy`, is measured at three
values because it is the only term standing against the degeneracy of §5.2, and its trade-off is
real — a penalty large enough to close the corner also charges *legitimate* concentration, which
is exactly what down-weighting a straggler or an adversary requires.

Plan figure 7 (a sensitivity heatmap over the two most important hyperparameters) has no sweep
behind it: A6 varies one knob at a time, so its cells form a cross rather than a grid.
`scripts/make_figures.py` **refuses to emit that figure** from one-at-a-time data rather than
producing a mostly-empty heatmap that looks like a grid. This is a gap, and it is a stated one.

## 6.4 Single dataset, and a rebalanced variant of it

Results are on one dataset, so they generalize weakly. Two specifics beyond the usual caveat:

- The archive we obtained is a **rebalanced variant**, and the balance is an artifact of
  duplicating the no-tumour class (67.9% redundancy against 18–26% for the tumour classes;
  §4.1). Our numbers are on the de-duplicated distribution and are **not directly comparable**
  to results on the canonical imbalanced release.
- The data layer is dataset-agnostic, so a second dataset is a config change rather than a port.
  We did not run one.

## 6.5 Compute scope, stated rather than implied

The experiments here are a documented reduction of a larger planned grid, cut to fit roughly 70
GPU-hours:

| dimension | run | planned | what the cut costs |
|---|---|---|---|
| clients K | 10 | 20 | the one K with a measured per-round cost; R5 varies it |
| partition regimes | 3 | 6 | pathological, quantity-skew, source-shift dropped — **quantity-skew is the costly one**, it attacks $n_k$ weighting directly |
| strategies | 6 | 12 | FedAdam, FedYogi, median, FedNova, loss-based, SCAFFOLD dropped |
| attacker fractions | 1 (30%) | 3 (10/20/30%) | figure 6 is a bar comparison, not a degradation curve |
| ablations | A1, A2 | A1–A9 | sensitivity (A6), fitness-term ablations (A4/A5/A7/A8) absent |
| robustness | R1, R2 | R1–R6 | stragglers, DP noise, client scaling, cold start absent |

Each cut is recorded at the point it is made, in the config header that makes it — see
`configs/experiment/main_reduced.yaml`, `robustness_r1_reduced.yaml`,
`robustness_r2_reduced.yaml`, `ablation_a2_reduced.yaml`, `ablation_a1_reduced.yaml`. The full
grid is still in the repository and still runnable.

**Seeds were not cut.** Eight is the floor at which the Wilcoxon signed-rank test can return
p < 0.05 at all (§4.6), and the significance family is held to the three comparisons 8 seeds
supports rather than stretched over all fifteen.

## 6.6 Rounds

100 rounds at K=10 on one dataset. Convergence curves are reported so a reader can see whether
the comparison is being made at a plateau or mid-descent; where a strategy has not plateaued we
say so rather than reporting its final-round number as converged.
