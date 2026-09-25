# paper/ — what is written, what is blocked, and what produces each artifact

Plan §13 maps eight paper sections to the artifacts that produce them. This directory holds
the sections whose content does not depend on federated results, so they can be written while
the sweeps run. **Never hand-type a number into any of these files** — plan §9.3 is explicit
that every figure and table is generated from `results/`.

| § | File | State | Produced by |
|---|---|---|---|
| — | `00_ABSTRACT.md` | skeleton | last, once §5 has numbers |
| 1 | `01_INTRODUCTION.md` | **written** | plan §1.1–1.2 |
| 2 | `02_RELATED_WORK.md` | **written**, citations marked `[CITE]` | plan §1.2, §0.3 |
| 3 | `ALGORITHM.md` | **written** (pre-existing) | plan §4.1–4.7 |
| 4 | `04_EXPERIMENTAL_SETUP.md` | **written, with measured numbers** | Phase 1 reports, `main_reduced.yaml` |
| 5 | `05_RESULTS.md` | skeleton + command map | `make figures tables` |
| 6 | `06_LIMITATIONS.md` | **written** | plan §1.4 + today's findings |
| 7 | `07_REPRODUCIBILITY.md` | **written** | Phase 10 |

## The citations are the one thing that cannot be finished here

`[CITE: ...]` marks a claim that needs a real reference. The implementation plan names
FedAAW, FedLAW, FedNolowe and DaWa but carries no bibliographic details, and this container
has no literature access. **Every `[CITE]` must be resolved by a human against the actual
paper before submission** — inventing an author, year or venue is the single fastest way to
have a submission desk-rejected. Grep for them:

```bash
grep -rn "\[CITE" paper/
```

## Figures and tables → the command that makes them

| artifact | command |
|---|---|
| Table 1 (main results) | `make tables` → `paper/tables/main_*.tex` |
| Tables 2–3 (ablations) | `make tables-ablation` |
| Table 4 (robustness) | `make tables-robustness` |
| Figures 1–2 (convergence, final bars) | `make figures` |
| Figure 3 (α heatmap) | `make figures` |
| Figure 4 (pheromone / α entropy) | `make figures` |
| Figure 5 (overhead vs K) | `make overhead` then `make figures` |
| Figure 6 (robustness curves) | `make figures` |
| Figure 8 (partition diagnostics) | `make figures-data` |
| Figure 9 (gain vs measured heterogeneity) | `make figures` |
| Figure 7 (sensitivity heatmap) | **no sweep exists** — A6 is one-at-a-time, so its cells form a cross, not a grid. Needs a small factorial config written after A6 names the two knobs that matter. Dropped from the 9-day set. |

`make verify-phase9` regenerates every artifact twice and compares byte-for-byte, which is
plan §9.3's acceptance criterion.
