# Handover — FedSwarm, as of 2026-10-11

You are picking this up cold. Read this file and nothing else first; it tells you where the
ground truth for everything is.

Branch: **`claude/happy-hamilton-c5jjil`**. 910/914 tests pass locally (the 4 failures are
`test_config_defaults.py` false alarms from applying the gate fix to `pyproject.toml` to
regenerate tables/figures — revert `pyproject.toml` and they pass; see §5 item 3), working tree
has new results/tables/figures (not yet committed — see §5 item 0).

---

## 1. What the project is, in four sentences

FedSwarm applies an ant-colony metaheuristic to the **aggregation weight vector** in federated
learning — instead of FedAvg's fixed $n_k$-proportional weights, a colony searches for weights
each round, with pheromone persisting across rounds as a memory of client reliability. The task
is 4-class brain tumour MRI classification across simulated hospital sites. **All ~75 GPU-hours
of planned experiments are now done** (gate, A1, A2, main table, R1, R2 — see §1a). The code is
finished and tested. **The central claim is still failing, and A1's real-data run (§2) makes
the failure sharper, not better.**

## 1a. GPU work — complete, as of 2026-10-11

| sweep | cells | status |
|---|---|---|
| `gate-fitness` | 8 | done |
| `a1-reduced` | 80 | done — pulled into `results/fl/ablation`, `paper/tables/ablation.{md,csv,tex}` regenerated locally |
| `a2-reduced` | 25 (of a nominal 30; 5 cells coincide with A1's) | done — same directory as A1, combined 105/105 |
| `main-reduced` | 168 | done — pulled into `results/fl/main`, `paper/tables/main.{md,csv,tex}` regenerated locally |
| `r1-reduced` | 40 | done |
| `r2-reduced` | 40 | done — combined 80/80 in `results/fl/robustness`. **Raw per-cell JSONs not yet in this clone**, only the printed `robustness.md` table recovered from the Kaggle notebook's log output (see §5 item 1) |

Nothing left to run on Kaggle. What is left is local: writing, citations, and pulling the
robustness JSONs in properly (§5).

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

**A1 on real data has now run (80/80 cells, 8 seeds) and the shape of the result changed:**

| regime | fedavg point | random | ga | coordinate_grid | pso | **aco** |
|---|---:|---:|---:|---:|---:|---:|
| iid | 0.7880 | +0.0053 | +0.0033 | +0.0043 | +0.0036 | **+0.0035** |
| dirichlet_0.3 | 0.6654 | -0.1163 | -0.0331 | -0.0397 | -0.1009 | **-0.1015** |

On the synthetic fixture every non-ACO method *beat* the FedAvg point and ACO alone did not —
a result about ACO specifically. On real data under heterogeneity **no method beats the FedAvg
point**, random included (worst of the five, at -0.1163). ACO is not uniquely bad here (pso is
within 0.0006 of it) but it is not rescued either — it stays in the worse half, well behind
ga/coordinate_grid. At iid all five are within 0.0020 of each other, as expected (§5.1's gate).
Full numbers and the cross-check against `paper/tables/main.csv` are in `paper/06_LIMITATIONS.md`
§6.1 — **treat that as a first read, not yet re-run through the Wilcoxon/Holm significance
machinery** (`scripts/aggregate_results.py`).

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

## 3. Three decisions — all confirmed by the project owner on 2026-09-25

| decision | chosen | if the evidence changes |
|---|---|---|
| **Framing** | **A** — the degenerate optimum and the equal-budget loss are the findings | The alternative framing is written out in full at the end of `paper/01_INTRODUCTION.md`. If A1 on real data reverses the result, delete §1.3 and paste that block. Nothing else in the paper changes — §4, §6 and §7 are framing-independent by construction. |
| **Dataset variant** | **(a)** — keep this archive, document it precisely, restate the macro-F1 rationale against the de-duplicated distribution | `docs/OPEN_QUESTIONS.md` "Dataset variant … RESOLVED". Reversing means fetching the canonical release: Kaggle credentials, and the compute budget spent twice. |
| **Scope** | **366 cells / ~75 GPU-h** instead of 351–701 | Every `*_reduced.yaml` header names its own cuts. The full-size configs are untouched and still runnable if the deadline moves. `main_reduced` carries 7 strategies (added `scaffold` back, 2026-09-27), not 6. |

These are settled. Write the paper as though they are true, not hedged. What can still change is
the *evidence* (A1, A2), not the decisions.

## 4. What is left is local now — no GPU box needed for any of it

```bash
make setup                      # uv venv, editable install
make test                       # 914 tests, ~2 min (4 fail iff pyproject.toml has the gate fix applied -- see §5.3)

# Already done, this clone (regenerate if results/ changes):
make tables tables-ablation                     # main.csv/md/tex, ablation.csv/md/tex -- regenerated 2026-10-11
make figures-ablation figures-data              # alpha_heatmap, alpha_entropy, gain_vs_heterogeneity, 9 partition figures
# tables-robustness and figures (main's convergence/comparison) do NOT work yet -- see §5.1 and §5.2
```

Every sweep this project planned has run on Kaggle. **Do not start a new GPU session for
anything in this file** unless one of items 4-5 below turns up a config you don't trust — re-run
only that cell, the sweep runners resume per-cell.

## 5. What is left

| # | work | GPU-h | state |
|---|---|---|---|
| 0 | **Commit `results/`, `paper/tables/`, `paper/figures/`** | 0 | pulled from Kaggle into this clone 2026-10-11, not yet committed. `results/*` is gitignored by design (CLAUDE.md) -- only the tables/figures belong in git. |
| 1 | **Get the real `results/fl/robustness/*.json`** (80 files) **off Kaggle** | 0 | what's in this clone now is `paper/tables/robustness.md` reconstructed by hand from the R2 notebook's printed log, not regenerated from raw JSONs. No `.csv`/`.tex`, no robustness figures, `make verify-phase9` cannot cover it. Same zip-and-Google-Drive path that worked for main/ablation works here -- the notebook's Output tab, zipped. |
| 2 | **Fix `figure_convergence`/`figure_comparison`** (Figures 1-2, the headline ones) | 0 | both silently skip every real sweep -- `_variant_of` marks any result whose `num-clients` differs from `pyproject.toml`'s bare `2` as non-"default", and no real config in this project uses `2`. Root-caused and written up in `docs/OPEN_QUESTIONS.md` ("Figures 1 and 2 ... skip every real sweep"), **not fixed** -- the naive fix (exempt `num-clients`) silently breaks R5's client-scaling study the same quiet way the 2026-09-24 variant bug did. Needs a design decision, not a patch. |
| 3 | `pyproject.toml`'s gate fix is session-local by design (CLAUDE.md) | 0 | regenerating tables/figures locally means running `scripts/apply_gate_fix.py --from-results results/fl/gate --apply-verdict` first (sets `aco-gamma-entropy` 0.1→0.45) and `git checkout -- pyproject.toml` after, or `test_config_defaults.py` fails (4 tests) because the committed default is meant to stay 0.1. |
| 4 | ~~§5 results prose + abstract~~ | 0 | **done 2026-10-11** -- both §5 and `paper/00_ABSTRACT.md` have real drafts now; §1.3's framing updated to match A1's real-data result (confirms the negative framing, does not reverse it) |
| 5 | ~~23 open `[CITE]` markers~~ | 0 | **done 2026-10-11** -- all resolved against a live Consensus/Semantic Scholar search, metadata in `paper/REFERENCES.md`, `test_every_citation_placeholder_is_findable` removed per its own instruction. Still worth a human pass before submission -- automated search, not peer judgment. |
| 6 | Add **Adp-FL-PSO as a baseline** | ~8 | not implemented; it is the direct competitor. Lowest priority of what's left -- the paper's finding does not depend on it. |
| 7 | Zenodo DOI, `v1.0` tag | 0 | tag pushes 403 here (§6) |
| 8 | **Fix the SCAFFOLD baseline** | 0 (debug, not GPU) | found while writing §5.1: 24/24 of its `results/fl/main` runs are frozen bit-identical from round 0 to round 100. Mechanism now confirmed mechanically (not just inferred): `Scaffold.aggregate_train` is returning `(None, None)` every round, which Flower's own `Strategy.start()` handles by silently keeping the previous round's arrays forever -- so every client reply is erroring, every round. Control-variate math checked line-by-line against the SCAFFOLD paper, no bug found there; client-side `train_handler` unit tests pass on a hand-built `Message`. Local repro blocked by an unrelated container bug (`pyarrow`/`numpy` ABI mismatch in `flwr run`'s per-invocation isolated env -- kills every strategy locally, confirmed on plain `fedavg` too, not SCAFFOLD-specific). **Owner is pulling the real Kaggle log for the `kaggle_day3d_main.ipynb` session (2026-10-08, scaffold cells ~08:31-10:17 UTC) to get the actual per-client exception** -- `docs/OPEN_QUESTIONS.md`'s "mechanism confirmed; exact trigger still open" entry has the full chain. |
| 9 | Pull `results/`, `paper/tables/`, `paper/figures/` from this clone | 0 | **done 2026-10-11**, committed in `05d3cda` and the commit that added this revision |

Never invent an author, year or venue, even with item 5 done -- a future `[CITE]` added by hand
follows the same rule. **Item 8 (SCAFFOLD) is now the sharpest open risk** — it sits inside
Table 1, the paper's first table, and the current write-up (§5.1) works around it by caveat
rather than by fix. Item 1 (robustness raw JSONs) is the other real gap: §5.3 is written from a
reconstructed table, not the normal pipeline.

### §5's honest status, past the table in HANDOVER's own bookkeeping

`paper/05_RESULTS.md` now has real prose for every subsection, not a skeleton. Two things in it
are marked **not settled** rather than asserted as findings, and should stay that way until
resolved: the SCAFFOLD row (§5.1, item 8 above) and the robustness table's provenance (§5.3,
item 1 above). Both A1 (§5.2) and A2 (§5.2b) read as genuine, if preliminary, findings — A2 in
particular: **full cross-round persistence beats the method's own default and beats no
persistence, in both regimes it was tested in.** That is the one place in the real data where
the mechanism this project is actually named for does something the closest published prior
work cannot. §1.2 of `HANDOVER.md` above still calls A2 "the most important experiment in the
project" — that call looks right in hindsight, not just in expectation.

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
