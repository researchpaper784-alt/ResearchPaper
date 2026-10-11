# Results — ablation

105 completed runs across 15 cells. Metric: `final_test_macro_f1`, mean ± std over seeds. `p (Holm)` is a paired Wilcoxon signed-rank test against fedavg, Holm-Bonferroni corrected across every comparison in this table; `d` is Cohen's d.

### dirichlet_0.1

| strategy | variant | n | macro-F1 | val | Δ vs fedavg | p (Holm) | d | fallback | τ entropy |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| fedaco | persistence=full,K=10 | 5 | 0.4380 ± 0.1214 | 0.5690 | — | — | — | 0% | 1.248 |
| fedaco | persistence=none,K=10 | 5 | 0.2092 ± 0.0526 | 0.4078 | — | — | — | 0% | 2.398 |
| fedaco | K=10 | 5 | 0.1684 ± 0.0646 | 0.3820 | — | — | — | 0% | 2.397 |

### dirichlet_0.3

| strategy | variant | n | macro-F1 | val | Δ vs fedavg | p (Holm) | d | fallback | τ entropy |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| fedaco | search=ga,K=10 | 8 | 0.6323 ± 0.1092 | 0.7189 | — | — | — | 0% | — |
| fedaco | search=coordinate_grid,K=10 | 8 | 0.6257 ± 0.0961 | 0.7310 | — | — | — | 0% | — |
| fedaco | persistence=full,K=10 | 5 | 0.6054 ± 0.0694 | 0.7257 | — | — | — | 0% | 1.026 |
| fedaco | search=pso,K=10 | 8 | 0.5645 ± 0.0994 | 0.7262 | — | — | — | 0% | — |
| fedaco | K=10 | 8 | 0.5639 ± 0.1182 | 0.6662 | — | — | — | 8% | 2.394 |
| fedaco | persistence=none,K=10 | 5 | 0.5543 ± 0.1371 | 0.6838 | — | — | — | 10% | 2.398 |
| fedaco | search=random,K=10 | 8 | 0.5491 ± 0.1449 | 0.6972 | — | — | — | 1% | — |

### iid

| strategy | variant | n | macro-F1 | val | Δ vs fedavg | p (Holm) | d | fallback | τ entropy |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| fedaco | search=random,K=10 | 8 | 0.7933 ± 0.0208 | 0.8176 | — | — | — | 72% | — |
| fedaco | search=coordinate_grid,K=10 | 8 | 0.7923 ± 0.0213 | 0.8179 | — | — | — | 0% | — |
| fedaco | search=pso,K=10 | 8 | 0.7916 ± 0.0219 | 0.8175 | — | — | — | 12% | — |
| fedaco | K=10 | 8 | 0.7916 ± 0.0214 | 0.8155 | — | — | — | 1% | 2.384 |
| fedaco | search=ga,K=10 | 8 | 0.7913 ± 0.0219 | 0.8169 | — | — | — | 26% | — |

**Reading the FedACO rows.** `fallback` is the fraction of rounds where the colony lost to the FedAvg point and FedACO silently *was* FedAvg — a high rate means the score largely belongs to FedAvg. `τ entropy` near its ceiling (log of the level count, 2.398 at the default 11) means pheromone was flat, so the colony was not searching and the cross-round stigmergy the method claims was not happening. Neither is visible in the macro-F1 column.

