# Results — main

168 completed runs across 21 cells. Metric: `final_test_macro_f1`, mean ± std over seeds. `p (Holm)` is a paired Wilcoxon signed-rank test against fedavg, Holm-Bonferroni corrected across every comparison in this table; `d` is Cohen's d.

### dirichlet_0.1

| strategy | variant | n | macro-F1 | val | Δ vs fedavg | p (Holm) | d | fallback | τ entropy |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| fedprox | K=10 | 8 | 0.3754 ± 0.0715 | 0.4297 | +0.0499 | — | — | — | — |
| fedavg | K=10 | 8 | 0.3255 ± 0.0944 | 0.4176 | +0.0000 | — | — | — | — |
| trimmed-mean | K=10 | 8 | 0.3172 ± 0.0581 | 0.5007 | -0.0083 | — | — | — | — |
| krum | K=10 | 8 | 0.1557 ± 0.0749 | 0.2217 | -0.1698 | — | — | — | — |
| fedaco | K=10 | 8 | 0.1553 ± 0.0539 | 0.3951 | -0.1702 | — | — | 0% | 2.397 |
| fedlaw | K=10 | 8 | 0.1253 ± 0.0192 | 0.1377 | -0.2003 | — | — | — | — |
| scaffold | K=10 | 8 | 0.1082 ± 0.0532 | 0.1097 | -0.2173 | — | — | — | — |

### dirichlet_0.3

| strategy | variant | n | macro-F1 | val | Δ vs fedavg | p (Holm) | d | fallback | τ entropy |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| fedprox | K=10 | 8 | 0.6664 ± 0.0612 | 0.7271 | +0.0009 | — | — | — | — |
| fedavg | K=10 | 8 | 0.6654 ± 0.0632 | 0.7265 | +0.0000 | — | — | — | — |
| trimmed-mean | K=10 | 8 | 0.6613 ± 0.0343 | 0.7155 | -0.0041 | — | — | — | — |
| fedaco | K=10 | 8 | 0.5639 ± 0.1182 | 0.6662 | -0.1015 | — | — | 8% | 2.394 |
| fedlaw | K=10 | 8 | 0.2757 ± 0.0222 | 0.2986 | -0.3898 | — | — | — | — |
| krum | K=10 | 8 | 0.2131 ± 0.1342 | 0.3522 | -0.4524 | — | — | — | — |
| scaffold | K=10 | 8 | 0.1082 ± 0.0532 | 0.1097 | -0.5573 | — | — | — | — |

### iid

| strategy | variant | n | macro-F1 | val | Δ vs fedavg | p (Holm) | d | fallback | τ entropy |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| trimmed-mean | K=10 | 8 | 0.7926 ± 0.0179 | 0.8166 | +0.0046 | — | — | — | — |
| fedaco | K=10 | 8 | 0.7916 ± 0.0214 | 0.8155 | +0.0035 | — | — | 1% | 2.384 |
| fedavg | K=10 | 8 | 0.7880 ± 0.0193 | 0.8165 | +0.0000 | — | — | — | — |
| fedprox | K=10 | 8 | 0.7879 ± 0.0192 | 0.8168 | -0.0001 | — | — | — | — |
| krum | K=10 | 8 | 0.7202 ± 0.0817 | 0.7897 | -0.0679 | — | — | — | — |
| fedlaw | K=10 | 8 | 0.6038 ± 0.0589 | 0.7399 | -0.1843 | — | — | — | — |
| scaffold | K=10 | 8 | 0.1082 ± 0.0532 | 0.1097 | -0.6798 | — | — | — | — |

**Reading the FedACO rows.** `fallback` is the fraction of rounds where the colony lost to the FedAvg point and FedACO silently *was* FedAvg — a high rate means the score largely belongs to FedAvg. `τ entropy` near its ceiling (log of the level count, 2.398 at the default 11) means pheromone was flat, so the colony was not searching and the cross-round stigmergy the method claims was not happening. Neither is visible in the macro-F1 column.

