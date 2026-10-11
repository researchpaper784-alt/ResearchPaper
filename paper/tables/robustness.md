# Results — robustness

80 completed runs across 16 cells. Metric: `final_test_macro_f1`, mean ± std over seeds. `p (Holm)` is a paired Wilcoxon signed-rank test against fedavg, Holm-Bonferroni corrected across every comparison in this table; `d` is Cohen's d.

### dirichlet_0.3

| strategy | variant | n | macro-F1 | val | Δ vs fedavg | p (Holm) | d | fallback | τ entropy |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| fedavg | attack=gaussian@0.3x0.5 | 5 | 0.7349 ± 0.0119 | 0.7396 | +0.0000 | 0.938 | +1.73 | — | — |
| fedaco | attack=gaussian@0.3x0.5,g3=0.1 | 5 | 0.7115 ± 0.0071 | 0.7224 | +0.0555 | 0.938 | +1.23 | 0% | 2.397 |
| trimmed-mean | attack=gaussian@0.3x0.5 | 5 | 0.6741 ± 0.0098 | 0.7031 | -0.0608 | 1.000 | +0.40 | — | — |
| fedavg | default | 5 | 0.6559 ± 0.0636 | 0.7314 | +0.0000 | — | — | — | — |
| trimmed-mean | default | 5 | 0.6533 ± 0.0426 | 0.7158 | -0.0026 | 1.000 | -0.05 | — | — |
| trimmed-mean | attack=label_flip@0.3 | 5 | 0.5855 ± 0.0404 | 0.5910 | +0.1333 | 0.938 | -1.32 | — | — |
| fedaco | g3=0.1 | 5 | 0.5645 ± 0.0773 | 0.7077 | -0.0915 | 0.938 | -1.29 | 0% | 2.393 |
| fedaco | attack=label_flip@0.3,g3=0.1 | 5 | 0.5172 ± 0.1075 | 0.6923 | -0.1387 | 0.938 | -1.57 | 0% | 2.393 |
| fedavg | attack=sign_flip@0.3x0.5 | 5 | 0.5121 ± 0.0343 | 0.5757 | +0.0000 | 0.938 | -2.81 | — | — |
| fedavg | attack=label_flip@0.3 | 5 | 0.4522 ± 0.0918 | 0.6177 | +0.0000 | 0.938 | -2.58 | — | — |
| fedaco | attack=sign_flip@0.3x0.5,g3=0.1 | 5 | 0.4139 ± 0.0467 | 0.4860 | -0.2421 | 0.938 | -4.34 | 0% | 2.397 |
| trimmed-mean | attack=sign_flip@0.3x0.5 | 5 | 0.2631 ± 0.0587 | 0.2839 | -0.2490 | 0.938 | -6.42 | — | — |
| krum | attack=gaussian@0.3x0.5 | 5 | 0.2159 ± 0.0279 | 0.2404 | -0.5190 | 0.938 | -8.96 | — | — |
| krum | default | 5 | 0.2038 ± 0.1143 | 0.3543 | -0.4522 | 0.938 | -4.89 | — | — |
| krum | attack=label_flip@0.3 | 5 | 0.1786 ± 0.0849 | 0.4038 | -0.2736 | 0.938 | -6.36 | — | — |
| krum | attack=sign_flip@0.3x0.5 | 5 | 0.1112 ± 0.0041 | 0.2144 | -0.4010 | 0.938 | -12.09 | — | — |

**Reading the FedACO rows.** `fallback` is the fraction of rounds where the colony lost to the FedAvg point and FedACO silently *was* FedAvg — a high rate means the score largely belongs to FedAvg. `τ entropy` near its ceiling (log of the level count, 2.398 at the default 11) means pheromone was flat, so the colony was not searching and the cross-round stigmergy the method claims was not happening. Neither is visible in the macro-F1 column.

⚠️ **These p-values cannot reach 0.05 at this seed count.** The signed-rank test's smallest possible two-sided p-value at n=5 pairs is 0.0625; across 15 comparisons Holm-Bonferroni multiplies the smallest by 15, so the best achievable adjusted p is 0.9375 — reached when *every* seed favours the method. Every 'not significant' in this table is therefore a statement about the sample size, not about the data. Clearing 0.05 over this family needs 10 seeds, or a smaller family of claims. Do not read these rows as evidence of no effect.

