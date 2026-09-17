# Open questions / unverifiable claims log

Per the implementation plan's ground rule: if a number, dataset property, or API signature
cannot be verified, it is recorded here rather than guessed.

## Naming collision (§0.3 of the plan) — RESOLVED 2026-09-14

The working title "FedACO" collides with an unrelated 2025 PRCV/Springer paper, "FedACo:
Adaptive Collaboration with Fine-Grained Aggregation for Personalized Federated Learning"
(https://link.springer.com/chapter/10.1007/978-981-95-4987-0_1). Same acronym (case
difference only), different expansion, different method.

**Decision (author, 2026-09-14): renamed the project to "FedSwarm."** Alternate names that
were considered and rejected: ACO-Agg, PherAgg, FedPher. If the author wants to reconsider,
alternatives are still on the table — this is a cheap rename now (before any paper text
exists) and an expensive one later.

## Compute environment

Local machine (this Mac) is **Intel macOS** (`macosx_26_0_x86_64` platform tag observed
during `uv pip install`, 2026-09-14). `flwr[simulation]`'s `ray==2.55.1` dependency ships
wheels only for `manylinux2014_{x86_64,aarch64}`, `macosx_12_0_arm64` (Apple Silicon), and
`win_amd64` — **no Intel-macOS wheel exists**. Verified by direct `uv` resolver failure, not
assumed.

Consequence: `flwr[simulation]` cannot be installed on this machine. Local work (Phase 0
scaffolding, unit tests, non-simulation Flower API introspection) uses plain `flwr==1.36.0`.
Actual FL simulation runs (Phase 3 onward) must happen in the author's Colab/Kaggle
environment (both Linux), where the `simulation` extra installs cleanly. This is a hard
blocker for locally running anything from Phase 3 (`ClientApp`/`ServerApp`/`start()`)
onward — flagged here rather than silently worked around.

Confirmed directly (not just via the resolver): downloaded the official
`@flwrlabs/quickstart-pytorch` reference app and ran `flwr run . --stream` on this
machine — it fails with `Unable to launch 'flower-superlink' for local simulation:
[Errno 2] No such file or directory: 'flower-superlink'`, since that binary ships with the
`simulation` extra. So the plan's §0.2 acceptance criterion ("run the unmodified Flower
quickstart ≥2 rounds locally") cannot be satisfied on this machine at all — full detail in
`docs/FLOWER_API_NOTES.md`. Everything else that criterion was meant to de-risk (the real
`Strategy`/`FedAvg`/`ArrayRecord` API surface) was verified by reading the installed
package and the reference app's source directly.

A secondary, unrelated finding from the same install attempts: `torch` has no Intel-macOS
wheel past `2.2.x`, and `torch==2.2.2` needs `numpy<2` (verified `torch.from_numpy` crashes
under numpy 2.1.3, works under 1.26.4) — pinned in `pyproject.toml`. Re-evaluate this pin
in the Linux training environment, where it doesn't apply.

## Kaggle dataset access

Author confirmed a Kaggle account exists and the dataset can be fetched from there, but
`kaggle.json` API credentials are not yet confirmed configured on the machine that will run
`fedswarm.data.download`. Phase 1 must verify `~/.kaggle/kaggle.json` (or `KAGGLE_USERNAME`
/ `KAGGLE_KEY` env vars) exist before relying on the Kaggle-API download path, and fall back
to the documented manual-zip path (`data/raw/`) otherwise.

## `source_shift` partitioning — RESOLVED 2026-09-14, with a documented coverage gap

The plan's `source_shift` regime partitions clients "by original source component
(Figshare / SARTAJ / Br35H)" — the most clinically honest setting, since real
cross-hospital heterogeneity is feature/acquisition shift, not just label imbalance. The
merged Kaggle dataset carries no such label natively.

**Resolved via hash-matching against all three upstream sources**, all now downloaded,
verified, and matched (`src/fedswarm/data/source_provenance.py`; full numbers and the
MATLAB-transpose bug hit along the way in `docs/EXPERIMENT_LOG.md`):

| Source | Matched images |
|---|---|
| Figshare | 1,531 |
| SARTAJ | 1,262 |
| Br35H | 1,222 |
| **Total** | **4,015 / 7,200 (55.8%)**, median distance 0 |

**`source_shift` is implemented in `partition.py`** (`_source_shift()`): clients grouped
by source, group size proportional to that source's available data, each source with any
data guaranteed ≥1 client.

**Known, permanent limitation — state this in the paper:** at the pseudo-patient level
(the actual partitioning unit), coverage is **46.9% of the training split** (1,563/3,330
placed, K=20). Units with no recovered provenance are excluded from this regime
specifically, not forced into a bucket, which would blur the cross-site signal the regime
exists to isolate. **Any `source_shift` result must be reported as computed over this
~47%-coverage subset**, not the full training set — say so explicitly wherever the regime
is used (results tables, figures, methods section). This is an honest answer to "how do
you know the source label," not a hidden assumption, but it does mean `source_shift`'s
client pool is meaningfully smaller than the other four regimes'.

Confirmed dataset slugs (previously unverified guesses, checked via search and live
Kaggle API before downloading):
[SARTAJ](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri),
[Br35H](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection),
[Figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427).

## ⚠️ Dataset variant is class-balanced, not the canonical release — DECISION NEEDED

The archive obtained on 2026-09-14 is **perfectly class-balanced**: exactly 1400 images per
class in `Training/` and 400 per class in `Testing/` (7,200 total). The widely-cited
Nickparvar release is class-*imbalanced* (~1321/1339/1595/1457 train, ~7,023 total). So
this is a rebalanced variant, not the canonical dataset the plan names.

Why it matters beyond pedantry:

1. **The plan's primary-metric justification assumes imbalance.** §6.3 argues macro-F1 over
   accuracy because "the dataset is class-imbalanced and the clinical cost of a missed
   tumor is asymmetric." On a perfectly balanced dataset, macro-F1 and accuracy nearly
   coincide and that argument evaporates. The metric choice is still defensible (per-class
   clinical cost is still asymmetric) but the stated reason has to change.
2. **Comparability to published results is weakened.** Numbers from this variant cannot be
   directly compared to papers using the canonical release.
3. **Unknown rebalancing mechanism.** If balance was achieved by oversampling/augmenting
   minority classes, that is itself a duplication source. The Phase 1.2 audit found 34%
   redundancy overall, but it has not been checked whether redundancy is *concentrated* in
   particular classes, which would be the signature of augmentation-based balancing.

Options: (a) keep this variant, document it precisely, and restate the macro-F1 rationale;
(b) additionally fetch the canonical release and use it as primary; (c) use both and show
results hold on each (strongest, cheapest as a robustness note since the pipeline is
dataset-agnostic).

**Not a blocker for building Phases 1.3–1.4** — the pipeline is identical either way. Decide
before the main sweep (Phase 6) commits GPU-weeks to one variant.

**Next diagnostic to run:** per-class redundancy rate (images per pseudo-patient, broken
down by class). If one class is markedly more redundant, that is evidence of
augmentation-based balancing and pushes toward option (b) or (c).

### Diagnostic result (2026-09-14) — the balance is manufactured by duplicating `notumor`

Per-class redundancy (images vs. unique pseudo-patients at t=5):

| Class | Images | Unique pseudo-patients | Redundancy |
|---|---|---|---|
| glioma | 1,800 | 1,476 | 18.0% |
| meningioma | 1,800 | 1,403 | 22.1% |
| pituitary | 1,800 | 1,326 | 26.3% |
| **notumor** | **1,800** | **577** | **67.9%** |

`notumor` is 2.6–3.8× more redundant than every tumour class, in both Training (60.9%) and
Testing (44.8%). The uniform 1400/400 counts are therefore an artifact of **padding the
`notumor` class with duplicates**, not a property of the underlying data.

Three consequences, one of them good:

1. **The apparent class balance is fake.** In unique-patient terms the dataset is strongly
   imbalanced — and imbalanced *against* `notumor* (577 vs ~1,300–1,500 per tumour class),
   which is the opposite direction from the canonical release where `notumor` is the
   largest class.
2. **Good news: the plan's macro-F1 justification survives.** After de-duplication the
   dataset is genuinely imbalanced, so §6.3's argument for macro-F1 over accuracy holds
   again — just for a different reason than the plan states. Update the wording, keep the
   metric.
3. **Any result computed on the raw variant is doubly compromised** — 28% cross-split
   leakage *and* a majority-duplicate `notumor` class. Our pseudo-patient collapse removes
   both by construction, which is precisely why Phase 1.2 exists.

This raises the value of option (b)/(c) (also fetching the canonical release): it would
show whether this duplication is an artifact of *this* variant or inherited from upstream.
Cheap to check later; not a blocker now, since de-duplication neutralises it either way.

## Phase 3 -- the FL harness is unverified end-to-end pending the first real Colab run

Built `fl/app.py` (originally four modules -- task/client_app/server_app/checkpoint,
consolidated 2026-09-16 into one file at the author's request for a single file that's
easy to hand to Colab; see the note in `docs/FLOWER_API_NOTES.md`) against the verified
`flwr==1.36.0` API (`docs/FLOWER_API_NOTES.md`), and confirmed as much as possible
locally:

- Every handler (`train_handler`/`evaluate_handler`/`build_evaluate_fn`'s closure) is
  directly unit-tested by calling it with hand-built `Message`/`Context`/`ArrayRecord`
  objects (confirmed: `ClientApp`'s and `ServerApp.main()`'s decorators register and
  return the function *unmodified*, by reading their source, not assumed).
- `flwr build` (packaging only, no `simulation` extra needed) validates the
  `[tool.flwr]` schema and resolves both component import paths successfully.

**What none of that can verify**: `strategy.start()`'s actual round-by-round behavior
against a real `Grid` -- whether `node_config["partition-id"]` is populated the way
this code assumes across however many SuperNodes the simulation spins up, whether
`min_train_nodes=2`/`min_available_nodes=2` behaves as expected at `num-clients=2`,
whether the checkpoint/resume round-remapping in `fl/app.py`'s `main()` (`round_offset`)
is actually exercised correctly by a real interrupted-and-resumed run. This machine
cannot install `flwr[simulation]` (no `ray` wheel for Intel macOS -- platform blocker,
`docs/FLOWER_API_NOTES.md`). **The first real `flwr run .` on Colab is Phase 3's actual
acceptance test**, not anything built so far -- treat the code as reviewed-and-tested-
as-far-as-locally-possible, not as verified. Runs on CPU with no code changes (see
`fl/app.py`'s module docstring) if GPU quota is the blocker rather than correctness.

## Phase 3 -- FedProx's `mu` is read by the client but nothing sends it yet

`fl/app.py`'s `local_train` and `train_handler` already support a FedProx proximal
term (`mu > 0`, read from the per-round `ConfigRecord`), per the plan's Step 3.1 design
goal ("one client serves all strategies"). `pyproject.toml`'s `[tool.flwr.app.config]`
default is `mu = 0.0` (off), and no strategy on the server side ever sets it to
anything else yet -- `fl/app.py`'s `main()` only drives the built-in `FedAvg`. A real
`FedProx(Strategy)` (Phase 5) still needs to be written and wired to set `mu` in the
train `ConfigRecord` it builds; SCAFFOLD's control variates are a
separate, not-yet-designed extension (they need the client to receive *and return*
extra per-client state beyond the model weights, which changes the `RecordDict` shape
client and server exchange -- deferred to Phase 5, not stubbed now).

## Phase 3 -- `flwr run` executes an installed copy of the app, not the repo clone

Discovered from a real Colab run, not anticipated in advance: `flwr run .` does not
execute `fl/app.py` from `/content/ResearchPaper` at all. Its own log says so --
`Successfully installed fedswarm to /root/.flwr/apps/fedswarm.fedswarm.0.0.1.<hash>`
-- the Simulation Runtime packages the app and re-installs it into an isolated
location (plus a separate `uv sync`-built environment for dependencies), then runs
client/server code from *there*. That installed copy is just the Python package
(`[tool.hatch.build.targets.wheel] packages = ["src/fedswarm"]`) -- no `data/`, no
`results/`, both gitignored and outside the wheel's scope regardless.

Consequence, confirmed by the first real run's symptoms matching exactly: every
relative `run_config` path (`cache-dir`, `manifest-path`, `output-dir`, ...) was
silently resolving against whatever that isolated process's cwd happened to be, not
the repo. The run took ~12 minutes (plausibly rebuilding the entire image cache from
scratch in the wrong location) and produced no `results/fl/*.json` anywhere under the
actual clone.

**Fixed** with the same pattern this file already used for the *raw dataset* root
(`FEDSWARM_DATA_ROOT`): a `FEDSWARM_REPO_ROOT` env var, read by a new `_repo_path()`
helper, anchoring every relative run_config path to the real clone regardless of
where the installed copy's code executes from. Unset, behavior is unchanged (falls
back to cwd, which is the repo root for every other entry point). `notebooks/
colab_fl_smoke.ipynb` now sets it before calling `flwr run`.

**Not yet verified**: whether `os.environ["FEDSWARM_REPO_ROOT"]`, set in the Colab
kernel process before the `!flwr run` shell-out, actually propagates through
Flower's own process tree (SuperLink → the `uv sync`-built runtime-env's Python
interpreter → the installed app). Standard Unix env-var inheritance says it should
(nothing suggests Flower explicitly filters arbitrary env vars when spawning child
interpreters), but this machine cannot run the simulation extra to check, and this is
exactly the kind of assumption that already broke once already this session (the
per-run install itself). The next real `flwr run .` on Colab is the actual test.

## Phase 3 -- `flwr run .` fails on Colab with "No heartbeat received from the task"

**2026-09-17, after the torch-pin and FEDSWARM_REPO_ROOT fixes above, both confirmed
working on their own terms** (dependency install succeeds; the isolated app copy is a
known, understood quantity) -- every real Colab attempt still ends the same way:
`flwr ls --format json` reports `"status": "finished:failed"`, `"status-details": "No
heartbeat received from the task"`. No Python traceback anywhere (not in `--stream`,
not in `flwr log <run_id> --show`), and no Ray session artifacts exist anywhere on
disk (`find` for `ray_*`/`session_*` under any path came back empty) -- meaning Ray
itself may never be reaching a normal running state, not crashing after starting.

Ruled out, in order, each with real evidence, not guesses:
- Shared memory: `/dev/shm` had 5.7G free before any change -- not the constraint.
- flwr CLI version: identical failure at both 1.36.0 (pyproject.toml's declared
  target) and 1.37.0 (latest) -- not a version mismatch (upgrading to 1.37.0 briefly
  produced a hang instead of a clean failure, but reverting to 1.36.0 reproduced the
  exact same "No heartbeat" failure, so the mismatch wasn't the underlying cause,
  just a confound on top of it).
- Our own dependency pins and path resolution: both fixed and independently
  verified (`flwr build` succeeds; `_repo_path` unit-tested).

**What actually explains it, found via the real GitHub issue tracker (`flwrlabs/
flower`, not `adap/flower` -- the org renamed at some point), not a search-engine
summary**: [PR #7391 "Add dedicated task heartbeat RPCs"](https://github.com/flwrlabs/flower/pull/7391)
is still **open, unmerged**, against a task/runtime system that was very recently and
heavily refactored (`ServerAppIo`/`ClientAppIo` merged into a new Runtime API,
gRPC→HTTP switch landing in 1.37.0 itself -- see that version's changelog). A related
deadlock fix ([#7895](https://github.com/flwrlabs/flower/pull/7895)) already shipped;
this specific heartbeat gap has not. This is circumstantial, not a confirmed root
cause -- there is no GitHub issue with this exact error string to point to directly --
but it is real, verifiable evidence (an open PR, on the right subsystem, in a
framework whose task lifecycle is in active flux) that this class of failure is a
known, current, unresolved area of Flower's own infrastructure, not something in this
repository.

**Correction to an earlier claim in this debugging session**: a web search initially
suggested a "unified heartbeat mechanism" had *already* been fixed in a recent release
and specific PR numbers were cited. Checked directly against the real changelogs
(`framework/docs/source/changelog/v1.36.0.md`, `v1.37.0.md`, fetched from the repo,
not summarized) -- neither mentions any such fix, and the cited PR numbers don't even
fall in this repo's real range. That claim did not hold up and should not be repeated.

**Status: blocked on upstream Flower, not on anything in this repo.** Options, not
yet decided: (a) retry once a Flower release ships past this point (re-check
`flwrlabs/flower`'s changelog for a heartbeat-related fix before trying again --
don't just bump to "latest" blindly, verify against the real changelog file first,
the way this entry itself was written); (b) try Kaggle instead of Colab, a different
container environment that might not hit whatever specifically triggers this on
Colab (blocked itself, separately, on missing phone verification -- see the Kaggle
dataset access entry above); (c) proceed with Phase 4 (`aco/` -- pure tensor math, no
Flower dependency, fully buildable and testable on this Mac) while this sits open,
since it doesn't block that work at all.

**2026-09-17 follow-up -- two more documented workarounds tried, both ruled out with
real evidence, not assumption:**

- `FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1` (env var, set before `flwr run .`,
  confirmed real in the Flower docs -- `how-to-install-app-dependencies-at-runtime.html`,
  verified against raw HTML, not a search summary): **no effect**. The per-run
  isolated `uv sync` environment (`Created env for run in: /root/.flwr/runtime-envs/...`,
  full ~190-package install list) still runs in full on every attempt with this var
  set. Caveat that turned out correct: that doc describes a long-running
  `SuperLink`/`SuperNode` deployment; `flwr run`'s ad-hoc local `SuperLink` (used for
  `Simulation`) apparently does not read the same flag.
- `--federation-config="client-resources-num-cpus=1"` (per-run override, confirmed
  real syntax in `how-to-run-simulations.html`): also **no effect**. Motivating
  theory was CPU starvation -- the docs state the *default* Simulation Runtime
  assigns 2 CPU cores per `ClientApp` for 2 simulated SuperNodes (4 total needed to
  run both at once), and this Colab instance's `nproc` reports only 2 -- a plausible
  deadlock. Downsizing to 1 CPU/client did not change the outcome at all: same
  `"No heartbeat received from the task"`, same `clientapp-seconds: 0.0`.

**The clincher, from `flwr ls --format json` across all 8 attempts so far**:
`compute-time.clientapp-seconds` is exactly `0.0` on *every single run*, including one
that ran for 752s (12.5 minutes, run `1134741074203875862`) before failing. 12 minutes
is far longer than any plausible CPU-queueing delay on a 2-core box -- if this were
resource contention, the ClientApp actor would have gotten scheduled well before then.
Zero client-side compute time, regardless of how long the run is left running, points
at the ClientApp task never being dispatched at the protocol level at all -- consistent
with the open PR #7391 heartbeat-RPC gap, not with any resource-sizing knob this repo
or a `flwr run` flag controls. CPU-resource starvation is now a ruled-out theory, not
an open one.

**Decision, 2026-09-17**: pivoting to Phase 4 (`aco/`, `strategies/fedaco.py`) while
this sits open -- it has zero Flower dependency and is fully buildable/testable on
this Mac, unlike Phase 3's FL harness. Revisit this entry once a Flower release lands
past PR #7391, or if Kaggle is tried as an alternative to Colab.

**UPDATE, 2026-09-17 (later the same day) -- Phase 3's acceptance test has actually
passed, on a teammate's machine.** `results/fl/e35285b33a_0.json` (gitignored, kept
locally, delivered via `fl_results.zip`) is a real, verified result: `"status":
"completed"`, `num_rounds_completed: 2`, `run_id` format matches
`utils/results.py::make_run_id` exactly (10-char config hash + `_seed`), and
`provenance.git_sha = 9b01a4621a256e2c170501fc7608197411165590` is confirmed (via
`git cat-file -t`) to be this repo's own real commit -- the `phase-4` tip, i.e. this
teammate ran genuine Phase-4-era code, not something forked or hand-edited. Round-by-
round macro-F1 (0.054 -> 0.455 -> 0.447) and per-class recall look like real training
(round 0's collapse to predicting one class is the expected random-init signature),
not fabricated numbers.

**The one variable that changed from every failed attempt in this file**:
`provenance.gpu = {"available": false, "backend": "cpu"}` and `packages.torch =
"2.10.0+cpu"` -- a **CPU-only** runtime, not GPU. Every failed attempt logged above
in this same entry was on a GPU-visible Colab T4 instance.

**Decision, 2026-09-17**: not chasing the exact root cause further (whether it's
really GPU-actor scheduling in Ray, or something else CPU-runtime happens to avoid)
-- CPU runtime is now the adopted working recipe for this notebook
(`notebooks/colab_fl_smoke.ipynb` updated to say so explicitly: Runtime > Change
runtime type > CPU, before running). The PR #7391 theory above is kept as
background, not retracted, but isn't being investigated further either; this repo's
Flower FL work doesn't need a GPU (smoke config is tiny), so there's no cost to
just always using CPU runtime for `flwr run` and moving on.

**Status update**: no longer "blocked" -- Phase 3's acceptance test has passed once,
for real, on CPU. Phases 4 and 5 (built on top of Phase 3's FL harness, `fl/app.py`)
are still "not yet run inside a live `flwr run`" until one of the newer strategies is
actually driven through a real run the same way -- straightforward now that CPU
runtime is the known-working recipe.

## Phase 4 -- two deliberate scope reductions in FedACO, not oversights

Both flagged in `paper/ALGORITHM.md` alongside the code; recorded here too per this
file's convention (`docs/OPEN_QUESTIONS.md` is where anything not fully built/verified
gets tracked, not just things that failed).

1. **Global shrinkage $s$ (plan §4.1)** is a fixed `FedACOConfig.target_sum`, not an
   extra per-ant search dimension the colony explores alongside the K per-client
   levels. The plan explicitly frames it as something to "make a config flag and
   ablate" -- sweeping fixed values of $s$ across separate runs is Phase 7 work;
   *searching* $s$ per-round (adding an (K+1)-th decision to the construction graph)
   is not implemented and would need its own design pass if the ablation later shows
   it matters.
2. **`ClientProbeFitness` (plan §4.4's `client_probe` mode)** implements the
   aggregation side only -- turning already-collected, one-round-delayed
   per-candidate client-reported losses into a scalar fitness
   (`aco/fitness.py::ClientProbeFitness.evaluate`). The broadcast-a-candidate-menu /
   collect-next-round round-trip itself is not wired into `FedACO.configure_train` or
   `aggregate_train` -- it's a strategy-level, FL-runtime concern, and the plan itself
   says this mode only needs to run "in the A3 ablation" (Phase 7) at a heavily
   reduced budget, not in the default round loop. `DataFreeFitness` (the method as
   proposed) and `ServerValFitness` (the upper-bound reference) are both fully wired
   and tested now; `client_probe` is the one mode still needing strategy-layer work,
   deferred to whenever Phase 7 actually exercises it.

## Phase 5 -- baselines: what's built, what's a documented interpretation, what's a real deviation

**Built and unit-tested against hand-built Messages/Contexts (`tests/test_baselines.py`,
`tests/test_fl_app.py`), same pattern as Phase 3/4 -- none of this has run inside a
live `flwr run` yet, same reason as Phase 3/4 (the Colab heartbeat blocker above).**

- **Built-ins wired via `strategies/factory.py`**: FedAvg, FedProx, FedAdam, FedYogi,
  Krum/MultiKrum, FedTrimmedAvg (the plan's "Trimmed-Mean"), FedMedian -- all
  `flwr.serverapp.strategy` real constructors, verified against the installed
  package's own source, not assumed.
- **Custom, well-specified**: FedNova (Wang et al. 2020) using the client's own
  already-reported `num_batches` as tau_i; SCAFFOLD (Karimireddy et al. 2020) --
  the one baseline needing a real client-side protocol change, using `Context.state`
  (confirmed real and durable per-node, `flwr/supernode/start_client_internal.py`)
  for the persistent local control variate. **Real deviation from the paper**:
  SCAFFOLD's model-update aggregation here is data-size-weighted (`weighted_by_key`,
  matching every other baseline for a fair comparison), not the paper's uniform
  `1/|S_t|` average. Flag this if SCAFFOLD's numbers are ever compared directly
  against the original paper's reported results.
- **`FedLAW` (Li et al., ICML 2023)**: reproduces the *idea* (weights learned by
  gradient descent against a server val set, via `torch.func.functional_call` for a
  differentiable forward pass) -- not the paper's exact learnable-global-scaling-factor
  formulation, and no page/table from that paper was checked against this
  implementation's numbers.
- **`LossBasedWeighting` ("FedNolowe-style" per the plan's own wording)**: no
  canonical paper named "FedNolowe" was found to verify a formula or numbers against
  -- per CLAUDE.md, this is recorded rather than silently assumed. What's implemented
  (`alpha_i ~ softmax(-loss_i / T)`) is a documented, reasonable interpretation of
  "weight clients by their own local loss," not a reproduction of a specific paper.
- **`global_val_loader`** (`fl/app.py`) was added alongside these -- `FedLAW` and
  `ServerValFitness`-style fitness need a server-held set to choose weights against
  that is genuinely distinct from the **test** split the final metric is measured on
  (`data/splits.py`'s three-way train/val/test), not a reuse of `global_test_loader`.
  Using the test set there would have been a real methodology bug (weight selection
  contaminating the reported metric), not just an engineering shortcut.

**Real bug caught before it shipped**: the installed `FedProx` strategy sends its
proximal coefficient under `config["proximal-mu"]`, not `config["mu"]` -- this
project's own client only ever read `"mu"` (a key nothing in Flower's own FedProx
ever sets), so the two would have silently never connected: every FedProx run would
have quietly behaved like plain FedAvg. Fixed in `train_handler`
(`config.get("proximal-mu", config.get("mu", ...))`), verified by reading FedProx's
real source (`flwr/serverapp/strategy/fedprox.py`), not by guessing the key name.

**Not yet attempted**: honest hyperparameter search per baseline (plan §5's "give
every baseline an honest hyperparameter search on the val split with a budget matched
to FedACO's") and the IID-narrow-band acceptance check -- both need a live FL run to
execute, same blocker as everything else in Phase 3 onward.

## Phase 4 -- closed items, and the two residuals that remain open

**Closed 2026-09-17** (see docs/EXPERIMENT_LOG.md for measurements):

- The dispersion term's scale was silently disabling the colony for any config beyond
  `lr=0.01, local-epochs=1`. Fixed by dividing by `trace(G)/K`; guarded by two new tests.
- No FedACO knob was reachable from `run_config` (the factory forwarded only
  `num-rounds`), which made the Phase 7 A1 persistence ablation and the `target_sum`
  shrinkage sweep literally unrunnable. All knobs are now declared in
  `[tool.flwr.app.config]` as `aco-*` keys.
- The colony's RNG rode on global torch RNG; it now takes a `torch.Generator` seeded
  from `(seed, server_round)`.
- `fitness_mode` is now selectable between `data_free` and `server_val`.
- `test_overhead` was asserting an unmeasured 15%; tightened to the plan's 5% with the
  real K=20 number recorded.

### Residual 1 -- the deposit floor can still zero out, just not from scale

`colony.py` deposits `rho * Q * max(F, 0)`. Normalization removed the *scale-driven* path
into that floor, but F can still be negative on its own terms: with dispersion now
normalized to ~1, a round whose client updates have no consensus direction (alignment
near 0) yields F ~ -1 for every candidate, and the same degeneracy follows -- uniform
tau, `tau^a * eta^b` collapsing to `eta^b`. This was observed directly while measuring
overhead with isotropic-noise deltas: `pheromone_entropy` came back at exactly
log(11) = 2.3979, its maximum.

Whether this matters in practice is an open empirical question, not a known bug: on real
training deltas (which do have a consensus direction) F was positive in every round
measured. A baseline-centred deposit `max(F - F_fedavg, 0)` would make the mechanism
sign-robust and was prototyped, but showed no systematic benefit on real deltas and was
rejected rather than added on speculation. **Decide from the first real multi-round FL
run**: if `pheromone_entropy` sits at log(num_levels) across rounds, the colony is not
actually searching and this needs revisiting. `delta_mean_sq_norm` and
`pheromone_entropy` are both logged per round for exactly this check.

### Residual 2 -- the safety fallback cannot detect a degenerate fitness

`fallback_used` compares `F_best` against `F_fedavg` under the *same* fitness function,
so it reports 0 ("the colony beat FedAvg") even when the colony never searched at all --
confirmed across all 9 configs of the scale sweep, including the six where tau was
exactly uniform. This is not fixed and arguably cannot be fixed from inside F alone; it
is a reason to read `pheromone_entropy` rather than trust `fallback_used` as a health
signal. Recorded so no one later reads `fallback_used=0` as evidence the method worked.

### Still deferred to Phase 7, unchanged

Global shrinkage `s` is swept, not searched; `client_probe`'s broadcast/collect
round-trip is still unwired (`ClientProbeFitness` is the aggregation side only, and
`fitness_mode="client_probe"` is now rejected at construction rather than silently
accepted).

## Phase 3/6 -- `flwr run` rejects any `--run-config` key not declared in pyproject

Found 2026-09-17 while running FedACO end-to-end. `flwr run --run-config "alpha=0.3"`
fails with a bare `[code: 15] Invalid run configuration` that names no key: overrides are
only accepted for keys that already exist in `[tool.flwr.app.config]`.

This had already broken something. `pyproject.toml` carried a comment stating that
`alpha` / `classes-per-client` / `skew-sigma` were "deliberately absent from the smoke
default -- override alongside `regime` together, e.g. `--run-config "regime='dirichlet'
alpha=0.3"`". That command has never worked, which means **every non-IID federated run --
the entire point of the project -- was unreachable from the CLI**. All three are now
declared, along with every `aco-*` key.

Anything a future phase wants to sweep has to be declared in that table first; a comment
documenting a key is not enough.

## Phase 3 -- newer scikit-learn returns nan from roc_auc_score instead of raising

`eval/metrics.py` caught `ValueError` to report `auc_ovr_macro: None` when a class is
absent from a batch -- expected under label-skewed partitions. Newer scikit-learn no
longer raises: it emits `UndefinedMetricWarning` and returns `nan`, so the `except` never
fired and `nan` reached the result JSON.

Not cosmetic: `json.dumps` serializes that as a bare `NaN` token, which is **invalid JSON**
and is rejected by strict parsers -- so Phase 6 result files would have been unparseable
for exactly the partitions the project is about. Both paths now normalize to `None`.
Caught by `tests/test_models.py::test_compute_metrics_auc_handles_missing_class_gracefully`
and `tests/test_fl_app.py::test_evaluate_fn_checkpoints_after_every_round`, which were the
two failures in the suite before this was fixed.

## Phase 6 -- the simulation's default CPU budget is 2 cores *per client*, which caps K

Observed 2026-09-17 while running FedACO end-to-end on a 4-core CPU box. `flwr run` with
`num-clients=4` started Ray, created the `ClientAppActor`s, and then sat at round 0
indefinitely with every actor near 0% CPU -- no error, no timeout, just no progress. The
same run at `num-clients=2` completed all 3 rounds normally.

Verified, not inferred: `flwr.simulation.run_simulation.DEFAULT_SIMULATION_CONFIG`
has `client_resources_num_cpus = 2`. So 4 concurrent ClientApps request 8 cores on a
4-core machine. **The main sweep's K=20 would request 40 cores at this default.**

What is verified is the default and the two observations; the causal link (resource
starvation specifically, rather than something else that 2 clients happen to avoid) is
probable but not proven here -- it is consistent with the `clientapp-seconds: 0.0`
signature recorded in the Colab heartbeat entry above, though that instance was GPU.

**Before Phase 6 commits GPU-hours**: decide how client resources are sized. Either set
`client_resources` explicitly (a `[tool.flwr.federations]` section, which this project
currently does not have at all) or confirm the training environment has
`2 * num-clients` cores available. Do not assume the simulation degrades gracefully to
time-slicing when it does not -- it appears to stall silently, which in a long sweep is
indistinguishable from a slow round.
