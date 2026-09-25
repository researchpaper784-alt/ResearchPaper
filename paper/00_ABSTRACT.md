# Abstract — write this last

Blocked on §5 having numbers. Drafting it now means hand-typing a result, which plan §9.3
forbids for exactly the reason that it then diverges from the table.

**Structure to fill, once `make tables` has run:**

1. One sentence on the setting: non-IID federated brain-tumour MRI across simulated sites.
2. One sentence on what was tried: an ACO metaheuristic over the aggregation weight vector,
   with pheromone persisting across rounds.
3. **The finding.** Framing depends on A1 — see the two options in `01_INTRODUCTION.md`. If the
   negative framing holds, lead with the degenerate optimum (measured: best single-client vertex
   beat the reference point in 15/15 rounds) and the equal-budget result, not with the method.
4. The reusable contributions: the `corner_margin` diagnostic, its closed-form penalty
   threshold, the equal-budget protocol, the de-duplicated pseudo-patient split.
5. One sentence of scope: one dataset, K=10, 3 regimes, 6 baselines, 8 seeds.

**Numbers to pull in, each from a generated artifact rather than retyped:**

| claim | source |
|---|---|
| headline macro-F1 and delta vs FedAvg | `paper/tables/main_*.tex` |
| p-value and effect size | same table (Holm-corrected, 3-comparison family) |
| equal-budget ranking | `paper/tables/ablation_*.tex` |
| overhead percentage | overhead table / Figure 5 |
| centralized ceiling | `docs/EXPERIMENT_LOG.md` 2026-09-16 — already measured |
| leakage and redundancy | `data/processed/leakage_report.json` — already measured |

**Do not claim** significance against any baseline other than FedAvg: 8 seeds supports a
three-comparison family (§4.6). `make tables` prints the requirement for whatever family the
table actually contains.
