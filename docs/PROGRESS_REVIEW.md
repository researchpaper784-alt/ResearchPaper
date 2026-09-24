# Progress review — 2026-09-17

Where the project stands. Phases 0–3 in brief; Phases 4 and 5 in detail, since that is
what this review was originally written about; Phases 6, 7 and 9 appended at the end,
built after it.

Eight commits from `637ec02` through `d517633`: Phase 4, Phase 5, this review, the health
check, the sweep runner, the ablation config and table builder, and the figures.
31 files, +5328 / −49 lines. Tests went from 175 to 251, all passing. Ruff clean.

---

## Where each phase stands

| Phase | State |
|---|---|
| 0 — scaffolding | Done |
| 1 — data pipeline | Done |
| 2 — centralized ceiling | Done, real numbers |
| 3 — FL harness | Done (3 fixes this session) |
| 4 — FedACO | **Done** |
| 5 — baselines | **Done** |
| 6 — main sweep | **Built, never run** |
| 7 — ablations | **Built, never run** |
| 8 — robustness | Not started |
| 9 — analysis | **Built, never run on real results** |

The line is no longer between phases — it is between *code* and *evidence*. Every phase
except 8 now has working, tested infrastructure, and the project has **zero federated
results**. Every FL run so far has been against a synthetic pixel cache (the real Phase-1
manifest, fabricated images) because the raw JPEGs need Kaggle credentials the build
environment does not have. The plumbing is proven end to end; not one number is.

---

## Phases 0–3 (brief)

Not changed in substance: the data pipeline (dedup, the leakage audit that found 34%
train/test contamination, splits rebuilt at pseudo-patient level, six partition regimes),
the centralized ceiling (SimpleCNN@112: 0.9300 macro-F1 with BatchNorm, 0.9203 with
GroupNorm; ResNet-18: 0.9701 over 3 of 5 seeds), and the Flower harness in `fl/app.py`.

Three Phase 3 fixes were needed for later work:

1. **`nan` in result files.** Newer scikit-learn returns `nan` from `roc_auc_score`
   instead of raising, so the `except ValueError` never fired. `json.dumps` writes `nan`
   as a bare `NaN` token, which is not valid JSON. This would have hit exactly the
   label-skewed partitions the project is about, so Phase 6's result files would have
   been unreadable. Fixed.
2. **Val metrics added to result files.** Details under Phase 5.
3. **Test fixtures had no val split.** The real manifest is three-way; the fixtures were
   two-way, so `global_val_loader` was silently empty.

---

## Phase 4 — FedACO

### The dispersion bug

The fitness function adds three terms. Alignment is a cosine, so it sits in [-1, 1]. The
entropy penalty sits in [0, log K]. Dispersion was a raw sum of squared distances, so its
size depended on how big the client updates were — which depends on learning rate and
local epochs, not on whether the weights are any good.

My first guess was that dispersion would always dominate. I measured before changing
anything, and that guess was wrong: at the default config (`lr=0.01`, 1 local epoch) the
three terms are fine. A small CNN at that learning rate produces update norms below 1, so
squaring them makes them smaller, not bigger.

The problem shows up as soon as you leave that config:

| lr | epochs | dispersion | F at FedAvg | fraction F>0 | pheromone spread |
|---|---|---|---|---|---|
| 0.01 | 1 | 0.40 | +0.56 | 0.80 | 4.2e-01 |
| 0.01 | 2 | 1.38 | −0.43 | 0.16 | 7.3e-03 |
| 0.01 | 5 | 2.63 | −1.65 | 0.07 | **0** |
| 0.05 | 1 | 13.4 | −12.4 | 0.00 | **0** |
| 0.1 | 5 | 266.7 | −265.8 | 0.00 | **0** |

Dispersion swings 660x. The colony deposits `rho * Q * max(F, 0)`, so once F is negative
for every candidate, every deposit is zero. Pheromone stays flat, `tau^a * eta^b` becomes
just `eta^b`, and FedACO stops being an ant colony — it becomes a fixed heuristic rule.
That happened in 6 of 9 configs. The default is one local epoch away from it, and 5 local
epochs is a normal FedAvg setting.

**This would not have shown up in a run.** The safety fallback compares the colony's best
F against FedAvg's F using the same broken fitness. The colony "won" in all 9 configs,
including the 6 dead ones. A broken run reports `fallback_used=0` and looks healthy.

**Fix:** divide dispersion by `trace(G)/K`. That is a constant within a round, so it
cannot change which candidate ranks above which — it only fixes the balance against
alignment. Dispersion now stays in [0.763, 0.933] across the same grid, and pheromone
develops real structure in all 9 configs.

I also tried a second fix (deposit `max(F - F_fedavg, 0)`). It made no consistent
difference on real deltas, so I dropped it rather than add a knob the data did not
support.

**One thing I did not decide.** Normalizing makes dispersion about the same size as
alignment, so with `gamma_2 = 1.0` they now carry equal weight. That changes behaviour at
the default config. I left `gamma_2` at 1.0 because that is the value `ALGORITHM.md`
specifies, and quietly retuning a published hyperparameter is not a bug fix. It is now
sweepable as `aco-gamma-dispersion`.

Two tests guard this. The scale-invariance test fails without the fix (fitness
−3,080,542, pheromone spread exactly 0).

### Config knobs

The factory passed exactly one setting to FedACO (`num_rounds`). Everything else used
dataclass defaults. So:

- The Phase 7 persistence ablation could not run. `persistence="none"` was implemented
  and unit-tested, but there was no way to reach it.
- The `target_sum` sweep that `OPEN_QUESTIONS.md` promises to Phase 7 could not run.
- Phase 6's sweep had nothing to vary.

Exposing them in code was not enough. `flwr run` rejects any `--run-config` key that is
not **declared** in `pyproject.toml`, with an error that names nothing (`[code: 15]`).

That also revealed an older problem: `pyproject.toml` documented the command
`--run-config "regime='dirichlet' alpha=0.3"`, but `alpha` was never declared. **That
command has never worked**, which means no non-IID federated run was reachable from the
command line — the main point of the project.

### Other Phase 4 items

- **Determinism.** `run_colony` accepted a random generator that was never passed, so the
  colony used global torch RNG. It is now seeded from `(seed, server_round)`. The test
  scrambles global RNG between two runs to prove they no longer affect each other.
- **Fitness mode** is now selectable (`data_free` or `server_val`). `client_probe` is
  rejected at construction rather than accepted and then failing.
- **Overhead.** The test asserted 15%, a number nobody had measured. The real figure is
  ~1.8% there, and **145 ms at K=20** on the real model — 0.26% of a 56.8s round, of
  which 60 ms is the one-off Gram precompute. Tightened to the plan's actual 5% bar.
- **Live run.** FedACO ran through a real `flwr run`: exit 0, 3/3 rounds, 213s. It beat
  the FedAvg point every round with no fallback.

---

## Phase 5 — baselines

### Result files had no validation metric

The plan says to tune each baseline on the validation split. But `build_evaluate_fn` only
ever evaluated the test split, so result files carried `test_*` and nothing else. Any
search built on those files would have had to rank configurations by test score — tuning
every baseline against the number the paper reports.

`global_val_loader` already existed. It was added during the FedLAW work precisely
because using test for weight selection "would have been a real methodology bug". Nothing
in the result path used it.

Now every round records `val_loss`, `val_accuracy`, `val_macro_f1` and `val_auc`, and
`final` carries `best_val_macro_f1` (the best round, not the last, so a config that peaks
and then drifts is not punished) plus `best_val_round`. Cost is one extra pass over 735
images per round.

### No baseline knob could be swept

`fedprox-mu`, `fedopt-*`, `num-malicious-nodes`, `trim-beta`, `lossweight-temperature`,
`scaffold-server-lr`, `fedlaw-steps`, `fedlaw-lr` — none were declared in
`pyproject.toml`. The comment above them said they "only need overriding via
--run-config when actually sweeping them", which is the one thing that does not work.
**All 51 search trials would have failed.**

Same root cause as the `alpha` bug. My Phase 4 fix covered the keys I was touching and
left these.

There was a trap in fixing it. `fedopt-eta` was read by both FedAdam and FedYogi, which
have different published defaults (0.1 and 0.01, from the FedOpt paper). Declaring one
shared key would have given both the same value and silently replaced FedYogi's default.
They are now separate keys.

### `flwr run` does not wait

A plain `flwr run` submits the run and returns exit 0 straight away. The first end-to-end
search checked for each result file immediately after launching the trial, found nothing,
and reported every trial as failed.

The knock-on effect is worse. Those launched-and-forgotten runs keep going. Two abandoned
2-trial searches left four simulations running across 10 Ray sessions at load 11.8 on a
4-core box. That starved the next trial from ~70s to over 12 minutes per round. Nothing
reported it — the searches had already "finished". Clearing it meant killing
`flower-superlink` to reap the leftover processes.

`flwr run` also returns exit 0 when the simulation itself crashes. So the exit code tells
you nothing either way. The harness now passes `--stream` and decides success by whether
a result file appeared.

**This matters for Phase 6.** A sweep runner that shells out without `--stream` will
launch everything at once, report total failure, and leave the machine thrashing.

### Two bugs in my own code, caught by the tests

- `find_existing` matched trials on the swept keys only. A search at seed 1 would have
  reused seed 0's results, making the second seed a copy of the first and destroying the
  seed variance every error bar needs.
- `diagnose` matched the bare word "error" line by line, so it blamed a Ray
  `FutureWarning` for failed trials.

### What was delivered

- `scripts/run_hparam_search.py` — per-strategy grids, shared budget, one `flwr run` per
  trial, resumable. Ranks on validation only. Reports `budget_shortfall` as a real field,
  so a strategy that got fewer trials than FedACO is visible rather than hidden.
- `scripts/check_iid_band.py` — the IID gate. Returns PASS, FAIL or INCONCLUSIVE, and
  exits nonzero on FAIL.
- `make hparam-search`, `make hparam-search-plan`, `make iid-band`.

Both were checked end to end. The search ran two real trials and picked
`fedprox-mu=0.01` on validation (0.1169 vs 0.1132). Validation and test differed on the
same run (0.1132 vs 0.1119), confirming they are separate splits. The gate was checked
three ways: PASS on a tight cluster, FAIL on a 0.086 FedACO lead, INCONCLUSIVE when the
spread sits inside seed noise.

---

## What is proven, and what is not

**Proven:** 194 tests pass. FedACO and FedProx both run in the real Flower runtime.
Validation metrics reach result files. Baseline knobs reach strategies. The search selects
on validation and waits for each run. The IID gate returns the right verdicts.

**Not proven:** any scientific number. Every federated run here used a **synthetic pixel
cache** — the real Phase 1 manifest with fabricated images — because the raw JPEGs need
Kaggle credentials this environment does not have. The mechanism works. The numbers mean
nothing. The real hyperparameter search is GPU-hours in the training environment and has
not been run.

---

## Still open

- The deposit floor can still hit zero because of the *sign* of F, not its scale, if
  client updates have no shared direction. On real deltas F stayed positive; on random
  noise `pheromone_entropy` sat at its maximum. Check at K=20.
- `fallback_used` cannot detect a broken fitness by design. Read `pheromone_entropy`
  instead.
- In the live 3-round run, `pheromone_entropy` was only ~1% below its maximum, so
  pheromone was contributing little. Three rounds at K=2 proves nothing — check at scale.
- `gamma_2` left at 1.0. Your call whether to retune.
- The IID band default of 0.05 is **my choice, not the plan's**. Flagged in the script
  and the log.
- Phase 6 CPU budget: Flower gives each client 2 cores by default, so K=20 asks for 40.
- Milestone tags now exist (`v0.3-fedaco`, `v0.4-baselines`, fetched 2026-09-17), closing
  an item that was open earlier in the session. Note they point at commits on `main`, not
  at the Phase 4/5 work on this branch, so there is no tag for either of those yet.
- The implementation plan is not in the repo, so every "per plan §4.5" reference cannot
  be checked from the clone.

---

## The pattern worth remembering

Three separate blockers this session were the same bug: a `--run-config` key documented in
a comment but never declared, failing with an error that names nothing. It hit the
partition keys, the FedACO knobs, and every baseline knob.

Anything Phases 6–8 want to sweep must be declared first. There is now a test enforcing
this for the search grids, but not for sweep configs that do not exist yet.

---

## Phases 6, 7 and 9 — added after the original review

- **`scripts/run_sweep.py`** (Phase 6). 576 cells, resumable via `results/manifest.jsonl`.
  Four of its design choices are failure-driven rather than stylistic: `--stream` always
  (a bare `flwr run` returns before the run starts); exit codes ignored entirely (it lies
  in both directions); preflight refuses to start on CPU oversubscription; result matching
  includes seed and regime, because matching on less silently reuses one seed's result for
  another and destroys the variance every error bar depends on.
- **`configs/experiment/ablation_all.yaml`** (Phase 7). 150 cells. Only the ablations this
  repo actually names — A1, A3, A9 — plus the knobs `OPEN_QUESTIONS.md` defers here. A2
  and A4–A8 are deliberately absent: the implementation plan is not in the repository, and
  labelling a guessed ablation with the plan's number would be worse than the gap.
- **`scripts/make_tables.py`** (Phase 9). Incomplete cells are marked with their real `n`,
  never dropped. Deltas are paired by seed and say so. FedACO rows carry `fallback` and
  `τ entropy` beside the score, because a macro-F1 column cannot show that the colony
  never searched.
- **`scripts/make_figures.py`** (Phase 9). Emphasis encoding — FedACO and FedAvg take the
  two validated categorical slots, the other ten baselines fold into one muted field,
  because a palette carries ~8 hues before adjacent ones blur. `colony_health.png` is the
  only figure that can show the mechanism did not run.

Also fixed along the way: the strategy's own diagnostics (`pheromone_entropy`,
`best_fitness`, `fallback_used`, `delta_mean_sq_norm`) were never written to the result
file at all — they went to Flower's console and nowhere a checker could read them, despite
Phase 4 having added `delta_mean_sq_norm` specifically to diagnose the degenerate regime.

## Next

**One real-data run, before the sweep.** `make validate-fedaco` then `make health`,
in an environment with the dataset. It is roughly twenty minutes against the sweep's
several hundred projected hours, and it settles three things at once: whether FedACO's
mechanism actually runs, what a round really costs (the compute table is still estimates
and says in bold not to cite them), and whether client CPU sizing holds at the target K.

The one measurement available said **INERT** — τ 0.92% below uniform while the deposit
floor was clear. That verdict has since been withdrawn: τ starts *at* the entropy ceiling,
and at that run's budget the fastest any colony could concentrate it averages 3.14%, so
the 2% threshold demanded 64% of a theoretical maximum. τ was in fact moving away from
uniform every round. The check now judges against the run's own reachable ceiling and
reports **UNDERPOWERED** there (docs/EXPERIMENT_LOG.md, 2026-09-17).

So the mechanism question is genuinely *unanswered*, not answered badly — which is exactly
why the real run matters: the sweep will faithfully produce 576 cells whether or not the
colony is searching. `make validate-fedaco` at 15 rounds is powered to answer it (22% of
the best case); a shorter run would not be, and `make pheromone-budget` says so before the
compute is spent.
