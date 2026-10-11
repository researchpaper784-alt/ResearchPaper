# Results — main

168 completed runs across 21 cells. Metric: `final_test_macro_f1`, mean ± std over seeds. `p (Holm)` is a paired Wilcoxon signed-rank test against fedavg, Holm-Bonferroni corrected across every comparison in this table; `d` is Cohen's d.

### dirichlet_0.1

| strategy | variant | n | macro-F1 | val | Δ vs fedavg | p (Holm) | d | fallback | τ entropy |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| fedprox | default | 8 | 0.3754 ± 0.0715 | 0.4297 | +0.0499 | 1.000 | +0.60 | — | — |
| fedavg | default | 8 | 0.3255 ± 0.0944 | 0.4176 | +0.0000 | — | — | — | — |
| trimmed-mean | default | 8 | 0.3172 ± 0.0581 | 0.5007 | -0.0083 | 1.000 | -0.11 | — | — |
| krum | default | 8 | 0.1557 ± 0.0749 | 0.2217 | -0.1698 | 0.234 | -1.99 | — | — |
| fedaco | default | 8 | 0.1553 ± 0.0539 | 0.3951 | -0.1702 | 0.141 | -2.21 | 0% | 2.397 |
| fedlaw | default | 8 | 0.1253 ± 0.0192 | 0.1377 | -0.2003 | 0.141 | -2.94 | — | — |
| scaffold | default | 8 | 0.1082 ± 0.0532 | 0.1097 | -0.2173 | 0.141 | -2.84 | — | — |

### dirichlet_0.3

| strategy | variant | n | macro-F1 | val | Δ vs fedavg | p (Holm) | d | fallback | τ entropy |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| fedprox | default | 8 | 0.6664 ± 0.0612 | 0.7271 | +0.0009 | 1.000 | +0.02 | — | — |
| fedavg | default | 8 | 0.6654 ± 0.0632 | 0.7265 | +0.0000 | — | — | — | — |
| trimmed-mean | default | 8 | 0.6613 ± 0.0343 | 0.7155 | -0.0041 | 1.000 | -0.08 | — | — |
| fedaco | default | 8 | 0.5639 ± 0.1182 | 0.6662 | -0.1015 | 0.766 | -1.07 | 8% | 2.394 |
| fedlaw | default | 8 | 0.2757 ± 0.0222 | 0.2986 | -0.3898 | 0.141 | -8.23 | — | — |
| krum | default | 8 | 0.2131 ± 0.1342 | 0.3522 | -0.4524 | 0.141 | -4.31 | — | — |
| scaffold | default | 8 | 0.1082 ± 0.0532 | 0.1097 | -0.5573 | 0.141 | -9.54 | — | — |

### iid

| strategy | variant | n | macro-F1 | val | Δ vs fedavg | p (Holm) | d | fallback | τ entropy |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| trimmed-mean | default | 8 | 0.7926 ± 0.0179 | 0.8166 | +0.0046 | 0.625 | +0.25 | — | — |
| fedaco | default | 8 | 0.7916 ± 0.0214 | 0.8155 | +0.0035 | 0.891 | +0.17 | 1% | 2.384 |
| fedavg | default | 8 | 0.7880 ± 0.0193 | 0.8165 | +0.0000 | — | — | — | — |
| fedprox | default | 8 | 0.7879 ± 0.0192 | 0.8168 | -0.0001 | 1.000 | -0.01 | — | — |
| krum | default | 8 | 0.7202 ± 0.0817 | 0.7897 | -0.0679 | 0.492 | -1.14 | — | — |
| fedlaw | default | 8 | 0.6038 ± 0.0589 | 0.7399 | -0.1843 | 0.141 | -4.20 | — | — |
| scaffold | default | 8 | 0.1082 ± 0.0532 | 0.1097 | -0.6798 | 0.141 | -17.00 | — | — |

**Reading the FedACO rows.** `fallback` is the fraction of rounds where the colony lost to the FedAvg point and FedACO silently *was* FedAvg — a high rate means the score largely belongs to FedAvg. `τ entropy` near its ceiling (log of the level count, 2.398 at the default 11) means pheromone was flat, so the colony was not searching and the cross-round stigmergy the method claims was not happening. Neither is visible in the macro-F1 column.

⚠️ **These p-values cannot reach 0.05 at this seed count.** The signed-rank test's smallest possible two-sided p-value at n=8 pairs is 0.0078; across 18 comparisons Holm-Bonferroni multiplies the smallest by 18, so the best achievable adjusted p is 0.1406 — reached when *every* seed favours the method. Every 'not significant' in this table is therefore a statement about the sample size, not about the data. Clearing 0.05 over this family needs 10 seeds, or a smaller family of claims. Do not read these rows as evidence of no effect.

