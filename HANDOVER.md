# Handover — FedSwarm, as of 2026-09-25

You are picking this up cold. Read this file and nothing else first; it tells you where the
ground truth for everything is.

Branch: **`claude/happy-hamilton-c5jjil`**. 724 tests pass, `ruff check .` clean, working tree
clean, in sync with origin.

---

## 1. What the project is, in four sentences

FedSwarm applies an ant-colony metaheuristic to the **aggregation weight vector** in federated
learning — instead of FedAvg's fixed $n_k$-proportional weights, a colony searches for weights
each round, with pheromone persisting across rounds as a memory of client reliability. The task
is 4-class brain tumour MRI classification across simulated hospital sites. The code is
finished and tested. **The science is barely started, and the central claim is currently
failing.**

## 2. The single most important thing to know

**The paper's original claim does not hold, and this is already settled well enough to plan
around.**

Claim C2 was "the gain comes from ACO specifically, not from optimizing the weights somehow".
Three independent local screens measured, at identical fitness and identical evaluation budget:

| search method | gain over the FedAvg point |
|---|---|
| coordinate_grid | **+0.0501** |
| pso | +0.0484 |
| ga | +0.0312 |
| random | +0.0304 |
| **aco** | **+0.0013**, → +0.0066 after a config fix |

ACO beat every control on **0 of 3 seeds**. The plan's stated failure condition was a *tie*;
this is a loss. The mechanism is understood and is not a bug: the per-client desirability signal
spans ~0.04 against a level spacing of 0.25, so every client's argmax lands on the same level; a
uniform level assignment renormalises to $n_k$-proportional weights **exactly**; so the greedy
branch of the construction rule reproduces the FedAvg point ~70% of the time. The colony is
anchored to the thing it is supposed to beat.

**And the novelty claim was never true either.** A literature search on 2026-09-25 found
published work in the exact slot:

- **Adp-FL-PSO** (Srinivas et al., NMITCON 2025) — PSO at the server computing optimal
  aggregation weights.
- **FedPSO** (Park et al., Sensors 2021, 119 citations) — PSO replacing FedAvg's aggregation.
- **FedAWA** (Shi et al., **CVPR 2025**, 49 citations) — adaptive weights from client update
  vectors, no proxy data, aligning with the global direction. Same signal as our alignment term.

Note that the published swarm work chose **PSO**, and our own measurements put PSO far ahead of
ACO. Those agree. Full metadata in `paper/REFERENCES.md` — **nothing there was written from
memory, and you must not fill the remaining `[CITE]` markers from memory either.**

### What is still genuinely ours

1. The first **equal-budget, equal-fitness** comparison across ACO / PSO / GA / coordinate grid
   / random for aggregation-weight search. Each existing paper validates one method against
   FedAvg; none answers *which* search matters.
2. **`corner_margin`** — a diagnostic that detects a degenerate objective from quantities a
   round already computes — plus a closed-form threshold for the penalty weight that removes it.
   This applies to the published PSO methods too, because the degeneracy is a property of the
   objective, not of ACO.
3. **Cross-round stigmergy.** FedAWA, Adp-FL-PSO and FedPSO are all stateless between rounds.
   This is the only structural novelty left, and it rests entirely on ablation **A2**.

**Therefore A2 is the most important experiment in the project.** It is 30 cells, ~6 GPU-hours,
and it was written as a routine ablation before anyone realised it carries the contribution.

## 3. Three decisions already taken — each reversible, each yours to overturn

| decision | what was chosen | where, and how to reverse |
|---|---|---|
| **Framing** | Framing A: the degenerate optimum and the equal-budget loss are the findings | `docs/OPEN_QUESTIONS.md` "The reframe decision". The alternative is written out in full at the end of `paper/01_INTRODUCTION.md` — if A1 reverses the result, delete §1.3 and paste that block. Nothing else in the paper changes. |
| **Dataset variant** | Option (a): keep this archive, document it precisely | `docs/OPEN_QUESTIONS.md` "Dataset variant … RESOLVED". Reversing means fetching the canonical release, which needs Kaggle credentials and costs the compute budget twice. |
| **Scope** | 342 cells / ~70 GPU-h instead of 351–701 | every `*_reduced.yaml` header names its own cuts. The full-size configs are untouched and still runnable. |

These were taken by an agent because work was blocked on them, **not** because they belong to
an agent. If you disagree with any, the reversal path is written down.

## 4. What to run, in order

Set `GPUS` on a GPU box. This is not optional — see §6.

```bash
make setup                      # uv venv, editable install
make test                       # 724 tests, ~50s

# Person B — the critical path. Nothing else is worth starting before this resolves.
make gate-fitness GPUS=0.1      # 8 cells, ~15 min. Decides WHICH fitness fix to use.
make a1-reduced   GPUS=0.1      # 80 cells, ~17 GPU-h. The go/no-go on the framing.

# Person C — 254 cells, ~53 GPU-h. r1 MUST precede r2 (see §6).
make main-reduced GPUS=0.1      # 144 cells, ~30 GPU-h
make r1-reduced   GPUS=0.1      # 40 cells, ~8 GPU-h
make r2-reduced   GPUS=0.1      # 40 cells, ~8 GPU-h
make a2-reduced   GPUS=0.1      # 30 cells, ~6 GPU-h  <-- carries the contribution

# Person A — after any results exist
make tables tables-ablation tables-robustness figures figures-data
make verify-phase9              # regenerates everything twice, byte-compares
```

`make b-all` and `make c-all` chain each person's set. Every sweep resumes per-cell, so a killed
Kaggle session costs only the cell in flight.

**Before spending GPU hours on anything you have edited:**

```bash
python scripts/synthetic_heterogeneous_dataset.py --out-dir /tmp/fx --num-images 600 --image-size 32
make preflight-configs FIXTURE=/tmp/fx CONFIGS='configs/experiment/*.yaml'
```

155 arms, ~100 minutes of CPU. It proves every code path executes. It proves nothing about
whether numbers mean anything.

## 5. What is left

| # | work | GPU-h | state |
|---|---|---|---|
| 1 | `gate-fitness` → `a1-reduced` | 17 | ready, never run |
| 2 | `main-reduced`, `r1`, `r2`, **`a2`** | 53 | ready, never run |
| 3 | Phase 9: statistics, figures, tables | 0 | built and byte-verified, never run on real results |
| 4 | **The paper** — §5 results prose + abstract | 0 | blocked on 1–2 |
| 5 | **11 open `[CITE]` markers** | 0 | needs literature access |
| 6 | Add **Adp-FL-PSO as a baseline** | ~8 | not implemented; it is the direct competitor |
| 7 | Zenodo DOI, `v1.0` tag | 0 | tag pushes 403 here (§6) |

`grep -rn "\[CITE" paper/` lists the open citations. Never invent an author, year or venue.

**Item 4 is the real risk, not the GPU rows.** 70 GPU-hours is a `make` target. Eight pages with
two related-work sections is a person's undivided attention for days, and as of now nobody is
assigned to it. Sections 1, 2, 4, 6 and 7 are already written (`paper/`); §5 and the abstract
need the numbers.

## 6. Landmines — each of these has already cost someone a day

**Flower / Ray**
- **`flwr run` exits 0 when the simulation dies.** Judge a cell by whether a result JSON
  appeared, never by exit status. Every runner here already does.
- **A wrong supernode count hangs forever, silently.** Flower's `sample_nodes` is
  `while len(nodes) < min_available: sleep(1)` with no give-up. `num_supernodes` defaults to 2
  and **no `--run-config` key can set it** — it is a federation setting. A cold-start config
  hiding one node too many eats a whole Kaggle session and writes nothing.
- **Ray hides the GPU from any actor requested with `num_gpus=0`.** With `GPUS` unset, every
  client trains on **CPU** while the server keeps the card, and every logged metric looks
  normal. The only symptom is wall-clock. `1/num_clients` is the right value. The sweep runner
  now refuses to start rather than do this.
- `flwr run` executes an *installed* copy of the app, not your working tree.

**Statistics**
- **8 seeds is a floor, not a preference.** The Wilcoxon signed-rank p-value has a floor set by
  the pair count alone: at 5 seeds the smallest attainable two-sided p is **0.0625**, so a
  5-seed study cannot report significance at α=0.05 *under any data*. Measured: Cohen's d = 7.91
  reported as p = 0.25.
- The supported family is **3 comparisons** (method vs FedAvg in each of 3 regimes). Claiming
  significance against all baselines in all regimes is a 15-comparison family that 8 seeds does
  not clear. `make tables` prints the requirement for whatever family your table contains —
  believe it over any comment.

**Config coupling**
- **`r1-reduced` must run before `r2-reduced`.** R2 ships with no unattacked arm by design and
  uses the `clean` arm R1 writes to the same `results/fl/robustness`. Trim R1's clean arm and
  every R2 delta becomes `None`, which renders as the same "—" as a cell with no paired seeds.
  Pinned by tests.

**This container specifically**
- `data/raw` holds a dataset card and **no images**. `www.kaggle.com` is blocked by the egress
  proxy (curl returns 000) and there are no credentials. You cannot train here.
- **Tag pushes return HTTP 403** while branch pushes succeed, so `v0.5-pipeline` and
  `v0.6-pipeline-verified` exist locally but not on the remote. Commit SHAs in the log are the
  durable references. Phase 10's release step will hit this.

**The rule that matters most** (from `CLAUDE.md`): *never fabricate a number, dataset property,
or API signature — if it can't be verified, stop and record it in `docs/OPEN_QUESTIONS.md`.*

## 7. The failure mode this repository keeps producing

**A check that runs, passes, and means nothing.** Thirteen instances found and fixed. Three were
found today, and two of those were introduced the day before by the same agent that found them:

- the preflight ran only the first partition of every config and reported the rest as ok — so
  "96/96 arms execute" was 96 of 137, and `sign_flip`, R4's higher DP sigmas and R3's larger
  straggler fractions had **never executed**;
- its resume keyed on the arm's *label* and not its config, so editing a config left four arms
  reporting `already ran` from results computed under the old settings;
- its pass count counted skips, printing "147 arms ran" for 137.

Earlier ones: a `partition_stats` parameter no caller passed (one figure had no x-axis); nine
plot functions imported by nothing (five paper figures missing); 42 sweep arms collapsing onto
one cell key (A1's five search methods would have reported as one number); 15 configs with no
`output-dir` (1,025 cells run, both table commands find nothing).

**When you add a check, ask what it would look like if the thing it checks were broken.** If the
answer is "the same", the check is not a check. This is the most useful habit to carry forward.

## 8. Where ground truth lives

| question | file |
|---|---|
| Flower API signatures | `docs/FLOWER_API_NOTES.md` — **authoritative**, not recollection |
| What happened, chronologically, including failures | `docs/EXPERIMENT_LOG.md` (~2,900 lines) |
| Open questions and decisions taken | `docs/OPEN_QUESTIONS.md` |
| The spec | `docs/IMPLEMENTATION_PLAN.md` (§14 = default hyperparameters) |
| Conventions: uv, commit format, determinism | `CLAUDE.md` |
| Per-section paper status and artifact map | `paper/README.md` |
| References, resolved and open | `paper/REFERENCES.md` |
| What each sweep cut and why | the header of each `configs/experiment/*_reduced.yaml` |

Read `docs/EXPERIMENT_LOG.md` from the bottom up. The last four entries cover today and are the
ones that change what you should do.
