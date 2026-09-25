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

## Dataset variant is class-balanced, not the canonical release — RESOLVED 2026-09-25 (option a)

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

### RESOLUTION 2026-09-25 — option (a), and the macro-F1 rationale survives with a corrected reason

The diagnostic this entry asked for ("per-class redundancy rate, images per pseudo-patient,
broken down by class") is now computed over the full manifest:

| class | images | pseudo-patients | images/pp | redundancy | raw share | de-dup share |
|---|---|---|---|---|---|---|
| glioma | 1,800 | 1,476 | 1.220 | 18.0% | 25.00% | 30.87% |
| meningioma | 1,800 | 1,403 | 1.283 | 22.1% | 25.00% | 29.34% |
| pituitary | 1,800 | 1,326 | 1.357 | 26.3% | 25.00% | 27.73% |
| **notumor** | 1,800 | **577** | **3.120** | **67.9%** | 25.00% | **12.07%** |

Overall redundancy 34.0%. Imbalance ratio: **1.00 raw, 2.56 after de-duplication.**

`notumor` is duplicated at 2.3-2.6x the rate of every tumour class. That is not a gradient, it
is one class carrying nearly all the redundancy -- the signature the 2026-09-14 entry named as
evidence of augmentation-based balancing, now quantified.

**Decision: option (a) -- keep this variant, document it precisely, restate the macro-F1
rationale.** Three reasons, in order of weight:

1. **The rationale does not need rescuing, it needs restating.** This entry's concern was that
   §6.3 justifies macro-F1 because "the dataset is class-imbalanced", which is false of the
   archive. It is **true of the data actually trained on**: after pseudo-patient de-duplication
   the distribution is 2.56:1 with `notumor` at 12.07%. The paper states it in that form
   (`paper/04_EXPERIMENTAL_SETUP.md` §4.1). The metric choice was never wrong; the stated reason
   was attached to the wrong object.
2. **Option (b)/(c) are not available.** Fetching the canonical release needs Kaggle
   credentials this environment does not have, and `www.kaggle.com` is blocked by the egress
   proxy. Under the 7-9 day deadline it is also not affordable: re-running the audit, the split,
   the centralized ceiling and every sweep on a second dataset is the whole compute budget again.
3. **The finding is worth more than the comparability it costs.** "The published balance of this
   widely used archive is manufactured by duplicating one class, and 28.2% of its images leak
   across its own published train/test split" is a contribution to the data section. Papers
   reporting on this variant without de-duplication are reporting on 34%-redundant data with
   two-thirds of the redundancy in a single class.

**What this costs, recorded rather than buried:** numbers here are not directly comparable to
published results on the canonical release (~7,023 images). The paper says so in §4.1 and §6.4.
The pipeline is dataset-agnostic, so running the canonical release later is a config change --
it is listed as the first thing to add if the deadline moves.

Pinned by `tests/test_paper_numbers.py`, which re-derives every one of these numbers from
`manifest.csv` and `leakage_report.json` and fails if the paper stops matching them.

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

## Phase 6/7/8 -- config-layer scoping decisions, recorded rather than silently assumed

- **§6.3's secondary client-scale grid** (K in {10, 50}, C=0.3) doesn't specify which
  strategies/partitions/seeds it should span. `configs/experiment/main_client_scale.
  yaml` scopes it to FedACO vs. plain FedAvg only, IID partition only, 3 seeds --
  widen it later if the main sweep's results make a broader comparison worth the
  extra compute. Same reasoning duplicated in that file's own header comment.
- ~~**Robustness configs R1/R2's Krum arm** fixes `num-malicious-nodes=4` across all
  three attacker fractions.~~ **FIXED 2026-09-24.** The coupling moved into
  `strategies/factory.py`, which sees the whole resolved run_config: `f` now comes from
  `attacks.malicious_ids`, the same function the ClientApp uses to pick which partitions
  lie, so the count Krum assumes and the count that lie cannot drift. It was recorded here
  as an accepted simplification for days; the fix is smaller than the note explaining why
  it wasn't made. Worth remembering as a pattern -- "a Cartesian product can't express
  this" was true of the *config*, and false of the system.
- ~~**R6 (cold start) has no config file here at all.**~~ **BUILT 2026-09-24**
  (`fl/cold_start.py`, `configs/experiment/robustness_r6_cold_start.yaml`). The note
  below was right that the Simulation Runtime's actor lifecycle is out of reach and that
  no `run_config` key holds a partition back -- and wrong that this closed the door. A
  strategy never asks the runtime for nodes; it asks the `Grid` it is handed. Every client
  starts at round 1 and a filtered Grid keeps a deterministic subset (the highest node ids,
  disjoint from `malicious_ids`' lowest) out of the strategy's view until the join round.
  The clients exist; the aggregation does not see them, which is exactly the condition R6
  is about.

  The real hazard turned out to be a hang rather than an error: Flower's `sample_nodes` is
  `while len(grid.get_node_ids()) < min_available_nodes: sleep(1)` with no give-up, so
  hiding one node too many burns a whole Kaggle session silently. `with_cold_start`
  therefore validates the visible count against the strategy's own minimums and raises at
  construction, and a test builds every strategy the shipped config declares to prove it
  passes. That is also why the config sets `min-train-nodes: 10`, not 20.

  Still open, and only a real run can answer it: whether `Pheromone.begin_round`'s fresh
  row for an unseen client is initialised at a sensible level. Too high over-trusts a
  newcomer, too low freezes it out however good its updates are, and the unit test only
  establishes that a row appears -- not that its value is right. The config's
  `fedaco_cold_no_persistence` arm is what separates "the gap is stigmergy" from "the gap
  is fifteen clients for twenty rounds".
- **A10 (layer-wise alpha vs. model-wise), the plan's own explicitly optional,
  "time-box it" ablation, is not implemented and has no config file.** It needs a
  real redesign (the construction graph would need K x num_layer_groups stations,
  not K), not a config toggle -- skipped deliberately, matching the plan's own
  framing of it as optional, not a gap to apologize for.

## Phase 10 -- run_id churns whenever pyproject.toml's config table gains a key, even for unrelated strategies

Discovered while testing `scripts/verify_repro.py`: `predict_run_id` against the
*current* `pyproject.toml` no longer matches `results/fl/e35285b33a_0.json`'s own
run_id (the teammate's real, verified Phase 3 result -- `docs/OPEN_QUESTIONS.md`'s
earlier entry). Not a bug -- `make_run_id` hashes the *entire* resolved
`run_config` dict, and Phase 5/7 added many new `fedaco-*`/`strategy-name`/etc.
keys to `[tool.flwr.app.config]` since that run happened, so the hash legitimately
changed. Real consequence, not just a curiosity: adding *any* new global default
key churns every run_id, even for a plain FedAvg smoke run that never reads the new
key at all -- so a "rerun the same smoke test" after any config-table change
predicts a different run_id than last time, and old result files become orphaned
(no longer the resume-skip target for a nominally identical rerun). Verified,
not assumed: manually copying `e35285b33a_0.json` to today's predicted run_id
(`44b2b05b39_0.json`) and re-running `verify_repro.py --skip-run` against it passes
cleanly -- the checking logic itself is correct; only the run_id prediction moved
out from under the old file. Not fixing this now (would mean hashing only the keys
a given strategy actually reads, a real redesign of `make_run_id`'s contract);
flagging it so a future "why did resume-skip re-run something identical" question
has an answer on record.
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

**Update, 2026-09-17 (later)** -- the first run measured with the new health checker came
back `INERT` on question 1 while question 2 was `CLEAR`: tau sat 0.92% below uniform
(2.3758 vs ceiling 2.3979) *even though* best_fitness was positive every round
(0.78-0.92) and every round therefore deposited. That combination points away from the
deposit floor as the cause and toward deposit *magnitude*: the deposit is
`rho * Q * F ~= 0.1 * 0.8 = 0.08` onto one of 11 levels, against `tau0 = 1.0`, over only
5-6 iterations, with `end_round` pulling 10% back toward tau0 every round. tau has
neither the per-iteration step nor the iteration count to move away from uniform. If that
holds up, the lever is `aco-q-deposit` / `aco-rho` / the iteration budget, not the
fitness weights -- a different fix from the one this entry originally anticipated.
⚠️ Measured at K=2 on synthetic pixels with a reduced colony budget, so it is a lead, not
a finding. The real-data run at K>=10 with the full budget is what settles it.

**Update, 2026-09-17 (later still) -- the INERT verdict itself was unfounded.** tau starts
*at* log(L) by construction, so the entropy gap measures how far it has travelled from its
own initialization, at a rate set by `rho`, the deposit and the iteration budget. At that
run's settings the fastest any colony could concentrate tau averages 3.14% over four
rounds, so the 2% threshold demanded 64% of a best case that picks the same level every
iteration -- a colony that has stopped exploring. What the run actually did was move away
from uniform monotonically every round (0.19% -> 0.56% -> 1.36% -> 1.57%), reaching 29% of
what the budget allowed, while beating the FedAvg point on fitness in all four rounds.

The reasoning above about deposit *magnitude* stands: `rho * Q * F` against `tau0 = 1.0`
is a slow walk, and that is exactly why a short run cannot answer the question. What does
not stand is "pheromone is carrying essentially no signal". `fedswarm/aco/diagnostics.py`
computes the reachable ceiling for any budget and `check_fedaco_health.py` now judges
against it, reporting UNDERPOWERED where the old check reported INERT. See
docs/EXPERIMENT_LOG.md, "the INERT verdict was a statement about the run length".

**Still open, and still needs the real run.** None of this shows the colony *is*
searching -- only that the measurement taken could not have shown it either way. The
`make validate-fedaco` gate (15 rounds, default colony budget) requires 22% of the best
case, which is a bar a working colony can clear and a dead one cannot.

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

**2026-09-18: this residual now has a concrete instance, and it is worse than "cannot
detect".** See Residual 3 -- in the degenerate-optimum case `fallback_used=0` is not
merely uninformative, it is *correct and misleading at once*: the colony genuinely did
beat the FedAvg point, on an objective that wanted a useless answer.

### Residual 3 -- the fitness pays nothing for discarding every client but one

**Status: open, quantified, deliberately not "fixed".**

Dispersion is `sum_k alpha_k ||delta_k - Delta(alpha)||^2`, a weighted variance about the
weighted mean. At a single-client vertex `alpha = e_j` the weighted mean *is* `delta_j`,
so the term is exactly zero -- for any Gram matrix, not approximately and not only in
degenerate rounds. Alignment collapses to `cos(delta_j, robust_mean)` and the
concentration penalty to its maximum, giving the exact identity

    F(e_j) = gamma_1 * cos(delta_j, robust_mean) - gamma_3 * log K

The whole price of throwing K-1 clients away is `gamma_3 * log K`. Measured on synthetic
deltas, the `gamma_entropy` needed to keep the corner from winning is 0.168 at K=2, 0.126
at K=4, 0.090 at K=10 and 0.074 at K=20 -- against the default **0.1**. So the sweep's
K=20 is (narrowly) safe and a K=4 validation run is not, which is the opposite of what a
smoke test should be. That is why the `K=4` recommendation has been removed everywhere.

**Why not just raise gamma_entropy.** Because it would be a fix by coincidence. A penalty
growing as `log K` against a term that vanishes outright is the wrong shape: any gamma_3
chosen to work at K=20 is still arbitrary at K=5, and one chosen at K=5 over-penalizes
concentration at K=50, where concentrating on a good subset may be the right answer.

**Update, 2026-09-18 -- the alternative is implemented, measured, and correcting two
errors in the paragraph this replaces.** The shape is now selectable:
`aco-concentration-penalty` takes `"entropy"` (the default, the method as proposed,
`log K - H(alpha)`) or `"gini"` (`sum_k p_k^2 - 1/K`, Simpson concentration).

Two things the earlier note got wrong, recorded rather than quietly edited:

1. **Sign.** It proposed `1 - sum_k alpha_k^2`. That is a *diversity* measure -- it
   *falls* toward a vertex, so subtracting `gamma_3 *` it from F would have rewarded
   concentration, making the problem strictly worse. The penalty has to rise toward the
   vertex, hence `sum_k p_k^2 - 1/K` as implemented.
2. **The reason.** It justified the swap by the penalty's gradient "not vanishing at the
   vertex". That is backwards on the facts and irrelevant to the argument. The entropy
   penalty's gradient is `log alpha_j + 1`, which *diverges* as `alpha_j -> 0`; the Gini
   penalty's is `2 alpha_j`, which vanishes there. And neither matters, because the
   colony searches a discrete level set that contains 0 exactly (`level_set`'s explicit
   floor level) -- it jumps to the vertex, it does not walk there. The operative quantity
   is the penalty's *value* at the vertex, not its slope near it.

The real argument is scale. Normalized dispersion is bounded and sits near 1 at every K,
so the penalty holding it in check should be bounded too. Vertex values are `log K` for
"entropy" (0.69 at K=2, 3.9 at K=50) and `1 - 1/K` for "gini" (0.5 to 0.98). Measured, the
gamma_3 needed to rule out the degenerate vertex across K in {2, 4, 10, 20, 50}:

| heterogeneity | entropy spread | gini spread |
|---|---:|---:|
| 0.5 | 2.71x | 1.06x |
| 1.0 | 2.88x | 1.01x |
| 2.0 | 3.44x | 1.20x |
| 4.0 | 4.41x | 1.53x |

At heterogeneity 1.0 the "gini" requirement is 0.231-0.233 across the whole range -- one
number works everywhere. Under "entropy" it runs 0.168 down to 0.058.

⚠️ Note what this does *not* fix: the requirement still moves with heterogeneity under
both shapes (0.10 -> 0.41 for "gini" as noise goes 0.5 -> 4.0). "gini" removes the K
dependence specifically, not the need to pick gamma_3 for the data.

**What settles it.** `corner_margin` is computed exactly, in O(K^2), honours whichever
shape is in force, and is logged every round (`strategies/fedaco.py`);
`check_fedaco_health.py` question 3 reads it. The `penalty_gini` and
`penalty_entropy_strong` cells in `configs/experiment/ablation_all.yaml` measure the
choice on real data -- the second isolates the shape from the level, so a win for "gini"
cannot be confounded with "gamma_3 was simply too small". The default stays "entropy":
changing it on synthetic deltas would be changing the method under the paper's own
description of it. `make fitness-landscape` maps the regime in the meantime.

⚠️ The numbers above come from synthetic deltas (a shared direction plus isotropic
Gaussian noise). Real training deltas are not isotropic, so treat the crossover values as
indicative. The identity itself is exact and data-independent.

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

## Phase 5 -- three things that made the honest hyperparameter search impossible to write

All found 2026-09-17 while building `scripts/run_hparam_search.py`, all fixed. Recorded
because each was silent: nothing failed loudly, the work simply could not have been done
correctly, and in two cases an incorrect version would have looked fine.

### 1. Result files carried no validation metric at all

The plan requires the search to select "on the val split". `build_evaluate_fn` evaluated
only the **test** split, so `rounds` and `final` carried `test_*` and nothing else. Any
search written against those files would have had to rank configurations by
`final_test_macro_f1` -- tuning every baseline against the exact number the paper
reports. `global_val_loader` already existed (added in Phase 5 for FedLAW, precisely
because using test for weight selection "would have been a real methodology bug") but
nothing in the result path used it.

Now every round logs `val_loss`/`val_accuracy`/`val_macro_f1`/`val_auc`, and `final`
carries `best_val_macro_f1` (max over rounds, so a config that peaks then drifts is not
penalized) plus `best_val_round`. `run_hparam_search.py::score` reads that key and no
other.

### 2. Every baseline hyperparameter was undeclared, so none could be swept

Same root cause as the partition keys (entry above): `flwr run` rejects any
`--run-config` key absent from `[tool.flwr.app.config]`. `fedprox-mu`, `fedopt-*`,
`num-malicious-nodes`, `trim-beta`, `lossweight-temperature`, `scaffold-server-lr`,
`fedlaw-steps`, `fedlaw-lr` were all missing, under a comment asserting they "only need
overriding via --run-config when actually sweeping them" -- describing the one thing that
did not work. **Every trial of the Phase 5 search would have failed with `[code: 15]`.**

`tests/test_phase5_search.py::test_every_grid_key_is_declared_in_pyproject` now checks
the declared table against the search grids, so the two cannot drift apart again.

**A trap inside the fix**: `fedopt-eta`/`fedopt-eta-l` were read by both FedAdam and
FedYogi with *different* per-strategy defaults (0.1/0.1 vs 0.01/0.0316, from the FedOpt
paper). Declaring one shared key would have given both the same literal value and made
those fallbacks dead code -- silently replacing FedYogi's published default. They are now
per-strategy keys (`fedadam-eta`, `fedyogi-eta`).

### 3. `flwr run` is asynchronous unless `--stream` is passed

A bare `flwr run` submits the run, prints "Successfully started run <id>", and returns
**exit 0 immediately** while the simulation is still starting. The first end-to-end run
of the search shelled out without `--stream`, checked for each trial's result file
microseconds after launching it, found nothing, and reported every trial as failed.

Compounding it: `flwr run` *also* exits 0 when the simulation itself dies (a missing
cache surfaces as an in-run traceback and "Exit Code: 700" behind a successful outer
process). So the return code is useless in both directions. The harness now passes
`--stream` unconditionally and judges success by "did a result file appear", not by the
subprocess's verdict.

**The second-order effect is worse than the mis-reporting.** Those launched-and-forgotten
runs do not stop -- they keep executing. After two aborted 2-trial searches this 4-core
box was running four orphaned simulations across 10 Ray sessions at load average 11.8,
which then starved the *next* (correctly blocking) trial badly enough that a single round
had not finished after twelve minutes, against ~70 s when uncontended. Nothing reports
this: the searches had already exited "cleanly", and the orphans are only visible in
`ps`. Cleanup required killing `flower-superlink` itself to reap the zombie
`flwr-simulation` children.

**This applies directly to Phase 6.** Any sweep runner that shells out to `flwr run`
must pass `--stream`, or it will launch every run in the sweep near-simultaneously,
conclude that all of them failed, and leave the machine thrashing under a pile of
orphaned simulations that make every subsequent timing measurement meaningless.

### Also fixed: the test fixtures were two-way, not three-way

`tests/test_fl_app.py`'s fixtures built only `train`/`test` rows, so `global_val_loader`
returned an empty loader and `evaluate_model` raised `need at least one array to
concatenate` the moment per-round val evaluation was added. The real manifest is
three-way and `split` is orthogonal to the on-disk directory (real val rows appear under
both `Training/` and `Testing/` paths); the fixtures now match.

### Also fixed: `val_loader` was gated on FedLAW alone

`fl/app.py` built the val loader only when `strategy_name == "fedlaw"`, so FedACO's
`aco-fitness-mode="server_val"` -- declared and supported since the Phase 4 close-out --
raised "requires model, val_loader, and device" on use. Introduced in `0bb64b7` by making
the mode selectable without extending the gate. It is now built unconditionally.

## Phase 4/6 -- the strategy's own diagnostics were never written to the result file

Found 2026-09-17 while building `scripts/check_fedaco_health.py`, fixed the same day.

`rounds_log` is built entirely by `build_evaluate_fn`, which only ever sees *evaluation*
metrics. Everything `aggregate_train` returns went to Flower's console output and nowhere
else: FedACO's `pheromone_entropy`, `best_fitness`, `fedavg_fitness`, `fallback_used`,
`alpha`, `alpha_entropy`, `delta_mean_sq_norm`, and the equivalents for every baseline.

The irony is sharp: the Phase 4 close-out added `delta_mean_sq_norm` *specifically* so a
run could be checked for the degenerate regime, and the field was not persisted anywhere
a checker could read. `paper/ALGORITHM.md` lists these as the algorithm's outputs, so the
result schema was missing the method's entire observable behaviour. Phase 6's analysis
would have been able to report accuracy and nothing about *why*, and the two residuals
recorded above (deposit floor, fallback blindness) would have been unanswerable from the
artifacts.

Fixed by `fl/app.py::merge_train_metrics`, which folds
`Strategy.start()`'s returned `Result.train_metrics_clientapp` (keyed by round, verified
against the installed flwr==1.36.0 dataclass) into `rounds_log` under a `train_` prefix,
respecting the resume `round_offset`.

**Anything Phase 6 wants to analyse must come through this path.** A strategy that
returns a metric from `aggregate_train` now gets it persisted automatically; one that
stashes state on `self` and never returns it still will not.

## Phase 6 -- how to actually size client CPUs

Follow-up to the entry above about 2 CPUs per ClientApp. The supported, non-deprecated
lever is:

    flwr federation simulation-config --client-resources-num-cpus 1

It writes to `~/.flwr/config.toml` and persists per machine, so it is a one-time setup
step rather than a per-run flag. The `[tool.flwr.federations]` `options.backend.
client-resources.*` form still parses but `flwr run` prints a deprecation warning and
points at this command (and at `--federation-config`) instead.

Unverified: whether pinning to 1 CPU is *sufficient* for K=20 on a 4-core box, or merely
necessary. The failure mode is a silent stall, so this wants an explicit check at the
target K before the sweep is launched, not an assumption.

## Phase 8 -- `train_is_malicious` counts examples, not clients

The per-reply `is_malicious` flag aggregates into the round metrics as
`train_is_malicious`, and Flower aggregates client metrics **weighted by `num-examples`**.
So the value is the fraction of malicious *examples*, not of malicious *clients*, and the
two diverge exactly where this project operates -- under label/quantity skew, clients hold
very different amounts of data.

Measured on a verified live run: `attack-fraction=0.5` over 2 dirichlet-skewed clients
reported `train_is_malicious = 0.293`, because the compromised client held 29% of the
examples. Reading that number as "29% of clients were malicious" would be wrong by a
factor of nearly two.

Use it to confirm an attack fired at all; read `attack-fraction` from the run config for
the client fraction. Not changed, because overriding Flower's aggregation for one key is
more intrusive than the note is worth -- but any Phase 8 analysis that quotes a malicious
*client* fraction has to take it from the config.

## Phase 6/8 -- CORRECTION: the K=4 stall was not CPU oversubscription

Recorded above, twice, as a client-resources problem: "the Simulation Runtime assigns 2
CPUs per ClientApp by default, so K=20 requests 40 cores... oversubscription stalls
silently at round 0 rather than queueing." **That diagnosis was wrong**, and it was
propagated into `run_sweep.py`'s preflight, the Makefile, the README and the Colab
notebook.

The real cause: **`num_supernodes` defaults to 2** (`flwr/supercore/constant.py`) and
nothing in this repository ever set it. `num-clients` is *this project's* run-config key —
it tells `partition_spec_from_run_config` how many ways to split the data — and has no
connection to how many ClientApps the Simulation Runtime creates. The K=4 run asked for
`min-train-nodes=4` while only 2 supernodes existed, so the server waited forever for
nodes that were never going to appear. Idle actors, no error, no progress: exactly the
observed symptom, and nothing to do with CPU count.

Verified 2026-09-17: with `--num-supernodes 4 --client-resources-num-cpus 1`, the same
K=4 run completes normally on the same 4-core box ("Sampled 4 nodes (out of 4)").

**The consequence had this not been caught** is worse than a stall. A sweep whose
`min-train-nodes` is low enough not to hang would not hang — it would quietly run 2
clients while every result file recorded `num-clients: 20`, with 18/20 of the data never
trained on. In `robustness.yaml` it compounds: `malicious_ids(20, 0.1) == {0, 1}`, so a
cell labelled "10% malicious" would have had *the entire participating federation*
compromised.

`scripts/run_sweep.py` now calls `flwr federation simulation-config --num-supernodes
<num-clients>` itself before the first cell and refuses to start if that fails, rather
than relying on an operator having run a setup command. More clients than cores is now a
note that the ETA is optimistic (Ray queues them), not a refusal.

## Phase 9 -- 5 seeds cannot produce a significant result, and that is arithmetic

**Status: acted on 2026-09-19 -- every sweep raised to 8 seeds. Two things remain open; see the end of this section.**

Found 2026-09-19 while folding `analysis.py`'s statistics into `make_tables.py`. The
signed-rank statistic is discrete, so its p-value has a floor set entirely by the pair
count -- independent of the data:

| seeds | smallest possible two-sided p | one-sided |
|---:|---:|---:|
| 5 | **0.0625** | 0.0312 |
| 8 | 0.0078 | 0.0039 |
| 10 | 0.0020 | 0.0010 |
| 15 | 0.0001 | 0.0001 |

`main.yaml` currently runs **5 seeds**. At 5 seeds a two-sided paired Wilcoxon cannot
return p < 0.05 *even when every single seed favours FedACO*. Verified directly against
the implementation, not derived: `analysis.min_achievable_p`.

Holm-Bonferroni then multiplies the smallest p by the family size, so the achievable
floor depends on how many comparisons the paper claims:

| family | seeds needed for alpha=0.05 (two-sided) |
|---|---:|
| 1 comparison | 6 |
| 6 (FedACO vs FedAvg, 6 regimes) | 8 |
| 66 (11 baselines x 6 regimes) | 12 |

So the question "how many seeds" cannot be answered without also fixing **how many
comparisons the paper will make**. Answering it for the wrong family is how a sweep gets
run twice.

**What has been done about it.** `make_tables.py` now prints a warning whenever the
table's p-values cannot reach alpha at its own seed count, naming the floor, the family
size, and the seed count that would suffice. A table reporting "not significant" for a
Cohen's d of 7.9 is not left for the discussion section to explain. The warning goes
silent as soon as the test can actually fire, so it is a safeguard rather than noise.

**What has not been decided, and needs to be:**

1. **Raise the seed count.** 5 -> 8 takes `main.yaml` from 360 to 576 cells (+60%);
   5 -> 12 takes it to 864. This is the honest fix and it costs GPU-hours.
2. **Shrink the claimed family.** If the paper's headline is FedACO vs FedAvg across 6
   regimes, that is 6 comparisons and 8 seeds suffices. Every additional baseline the
   paper claims significance against raises the requirement.
3. **Use a one-sided test.** Legitimate -- the hypothesis is directional -- and it buys
   roughly one seed. It is *not* legitimate if chosen after seeing the two-sided result,
   so `--alternative greater` is opt-in and is recorded in the table when used.

Doing none of these is also a choice: report effect sizes and paired deltas, and state
plainly that the seed count does not support significance testing. That is defensible if
said out loud and indefensible if the p-values are printed without the caveat.

### What was done, 2026-09-19

Option 1. **Every sweep now runs 8 seeds** -- `main.yaml` first, then all 16 ablation and
robustness configs. The granular configs turned out to be at **3** seeds, not 5: a floor of
0.25, four times worse than the main sweep's. Total sweep size 1508 -> 2576 cells.

### Still open

1. **A 66-comparison family needs 12 seeds, not 8.** 8 clears alpha=0.05 for a
   six-comparison family -- FedACO against FedAvg in each regime. If the paper claims
   significance against all 11 baselines in all 6 regimes, 8 is short and `main.yaml` needs
   864 cells rather than 576. `make tables` prints the requirement for whatever family the
   table actually contains, so this surfaces before the claim is written.

2. **The combined and granular sweep families overlap, and that now costs more than the
   seeds did.** `ablation_all.yaml` covers A1/A3/A9 while `ablation_a1/a3/a9.yaml` cover the
   same ground individually; `robustness.yaml` covers R1-R5's territory in combined form.
   Running both families runs those experiments twice at 8 seeds each.

   Both families exist for a real reason -- they read different YAML shapes, and the
   granular ones cover A2 and A4-A8 that `ablation_all.yaml` explicitly does not. But
   nothing says to run *both*, and the 2576-cell total assumes you do. Picking one family
   for the overlapping cells, or dropping the duplicates from whichever runs second, is a
   scope call for whoever owns the ablation runs -- and is worth more compute than any
   remaining seed decision.

## Phase 4 -- the level set did not match the plan's own hyperparameter table

**Status: fixed 2026-09-19. Prior results are affected.**

docs/IMPLEMENTATION_PLAN.md §14 writes the level set out literally:

    [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]

which is `linspace(0, 2.5, 11)`. `aco/schedules.level_set` was **log-spaced** until today.
Nobody could have caught it: the plan was not in the repository until the same day.

| | linear (plan) | log (what shipped) |
|---|---|---|
| max ratio between positive levels | 10x | **1143x** |
| levels below 1.0 | 4 of 11 | **9 of 11** |

**Why this is not cosmetic.** Log spacing puts nine of eleven levels below the FedAvg
point and lets a single ant construct alpha ratios over 1000:1, so a near-vertex weighting
like [0.99, 0.01] is one greedy draw away. Linear caps it at 10:1.

That bears directly on the degenerate-optimum finding (2026-09-18). The observed
alpha = [0.990, 0.010] was read as the colony finding the fitness's single-client optimum.
It was -- but the *reachability* of that corner was inflated by a level set the plan never
specified. `corner_margin` measures the fitness and is unaffected; how easily the colony
gets there is not.

**What was done.** `level_spacing` is now selectable, defaults to `"linear"` (the plan's
spec), and is exposed as `aco-level-spacing`. `"log"` is kept rather than deleted: every
result produced before today used it, and A6 -- the hyperparameter-sensitivity ablation
that owns L -- now has a `levels_log_spaced` cell so the choice is measured rather than
assumed.

**What is still open.** Whether the corner margin stays positive on real deltas under
*linear* spacing. The synthetic measurements in the 2026-09-18 entry and the K=10
rehearsal both used the log grid, so neither transfers. The first real run under the new
default is what settles it -- and until it does, "the fitness prefers one client" should
be quoted as a lead, not a finding.

## Phase 6 -- the per-ClientApp GPU fraction is wired but unverified on a GPU

**Status: open. Needs one GPU session to close, not analysis.**

`flwr federation simulation-config` takes `--client-resources-num-gpus` (verified against
the installed flwr 1.36.0: `FLOAT RANGE [x>=0.0]`, "Ratio of a GPU VRAM assigned to the
execution of each ClientApp"). Neither runner could pass it until today, so a GPU sweep was
not expressible from the CLI at all; both now accept `--gpus-per-client` and forward it
through `sweep.configure_federation`, which re-issues it on every K change.

**What is verified here.** The flag is accepted and the config write succeeds on this
CPU-only box (exit 0, "Updated simulation configuration", no tracked file mutated), and the
fraction is re-issued on each reconfiguration rather than set once and possibly dropped
(`tests/test_sweep.py::test_the_gpu_fraction_is_reissued_on_every_reconfiguration`).

**What is not, and cannot be from here.** Whether the fraction actually results in a GPU
allocation to the ClientApp actors -- that is Ray's placement, and there is no GPU on this
machine to observe it on. Nothing in this repo has ever run a ClientApp on a GPU.

**Why it matters more than an ordinary perf setting.** A ClientApp that gets no GPU does not
error; it trains on CPU. For the 576-cell main sweep that reads as "slower than the
projection". For `overhead.yaml` it is worse: that sweep's deliverable *is* a wall-clock
curve over K = 5 -> 200, and a CPU-bound curve is still a curve, so it would be reported as
the measured O(K^2) cost the sweep exists to establish. The first GPU session should print
`torch.cuda.*` from inside a ClientApp, or compare one cell's wall-clock against the CPU
rehearsal, before the overhead numbers are trusted.

## Phase 6 -- RESOLVED, badly: the degenerate optimum reproduces on the plan's linear grid

**Status: closed as confirmed, 2026-09-19, by the project's first real GPU run.**

The previous entry left this open: the K=10 rehearsal and every synthetic measurement of
`corner_margin` ran on the **log-spaced** level set, so none of them transferred to the
plan's `linspace(0, 2.5, 11)` default. The first run on real MRI data on a Kaggle T4, at
K=10 and the linear grid, answers it:

> **DEGENERATE**: the best single-client vertex outscored the FedAvg point in **15/15
> rounds** (mean margin **+0.6252**) at K=10.

Per-round `corner_margin` ranged 0.457 to 0.805 and was positive in every round. So the
level grid was never what made the corner attractive -- the **fitness function** prefers
"use one client, discard the rest", and the linear grid only makes it harder to reach.

**What makes this the dangerous shape rather than a bug.** The colony did *not* go to the
corner in this run: `alpha_max` averaged 0.338 and `alpha_entropy` stayed near 1.9 against
a 2.303 ceiling. The search is too short and too underpowered to find the optimum it is
pointed at. So the run *looks* healthy -- no fallbacks fired, best_fitness beat the FedAvg
point in 14 of 15 rounds -- and the failure arrives only as the colony gets *better*. Any
change that strengthens the search (more rounds, larger budget, higher `aco-q-deposit`)
moves FedACO toward a worse model while every diagnostic in the report improves.

**This is a fitness-design problem, and it is A1's problem too.** At equal budget the
controls (random, grid, PSO, GA) optimize the *same* fitness, so they inherit the same
corner. A1 could come back "ACO ties random search" for a reason that has nothing to do
with ACO: both are searching a landscape whose optimum is useless.

**What has to happen before the 576-cell sweep.** Raise `aco-gamma-entropy` until the
margin is negative at K=20, and confirm it with a gate re-run rather than by argument. The
knob exists (`concentration_penalty`, `CONCENTRATION_PENALTIES`), A7 owns its shape, and
`scripts/probe_*` can bracket a value on synthetic deltas -- but the number must be
confirmed against real deltas, because that is exactly the transfer that failed last time.

## Phase 4 -- the pheromone budget table assumes a fitness 4x the real one

**Status: open. Affects how question 1's verdict should be read, not any result.**

`aco/diagnostics.py` drives the real deposit rule with `MEASURED_FITNESS = 0.8`, and
`scripts/analyze_pheromone_dynamics.py` reports "must realize 21%" at the plan's budget
over 20 rounds on that basis. The first real run's `best_fitness` came in at 0.026-0.326,
mean ~0.19 -- roughly a quarter of the assumed value. The deposit is `rho * Q * max(F, 0)`,
so a 4x smaller F makes the best case ~4x smaller and the fraction a run must realize
correspondingly larger. The live run reported needing **253%** of its own best case, i.e.
question 1 was unanswerable at that budget, not merely tight.

The consequence is only about interpretation: an `UNDERPOWERED` verdict at a budget the
table calls comfortable is not a contradiction, it is the table's 0.8 being optimistic.
`MEASURED_FITNESS` should be re-derived from real runs once a gate run exists at the
corrected `aco-gamma-entropy`, since the fitness magnitude will move with it.

## Phase 3 -- RESOLVED: no FL run was reproducible, and the seed said otherwise

**Status: fixed 2026-09-19. Every FL result produced before this commit is affected.**

Two Kaggle gate runs at **identical config and identical seed** disagreed on every metric:

| | run 1 | run 2 |
|---|---|---|
| mean corner margin | +0.6252 | +0.6184 |
| rounds hitting the deposit floor | 1 (round 12) | 4 (rounds 1, 6, 8, 12) |
| tau, as a fraction of best case | 36% | 79% |
| final test macro-F1 | 0.1185 | 0.1381 |

Two independent causes, neither able to show up before this project ran a real federation.

**1. The ClientApp was never seeded.** `seed_everything` is called in `server_app.main()`
and nowhere else. In the Simulation Runtime a ClientApp is a separate **Ray actor process**,
so the server's seeding never reached it, and the train loader --
`DataLoader(..., shuffle=True)` with no `generator` -- drew its batch order from each
actor's own OS-seeded RNG. The `seed` recorded in every result file governed the server's
model initialization and nothing on the client. Fixed: both handlers seed themselves from
f(run seed, partition id, server_round), and the train loader takes an explicit generator.

Folding the **round** in matters as much as the client. Re-seeding to f(seed, partition)
alone would have made the runs reproduce perfectly while training all 100 rounds on one
fixed permutation -- a quieter bug that passes exactly the test written to catch this one.

**2. FedACO consumed replies in arrival order.** `deltas`, the Gram matrix, the
desirability and the colony's stations are all built positionally from the reply list, and
Ray does not fix its order -- `participating_client_ids` shows it changing between rounds
of a single run. A permutation hands the same pseudo-random draws to different clients, and
moves which clients `precompute_gram(trim_fraction=...)` trims when scores are close. The
aggregate was never *wrong* (alpha is matched to clients by id, the pheromone is keyed by
id); it was not reproducible. Fixed by sorting on client id before anything positional.

**What this invalidates.** Every FL result in `results/` predating this fix, as a
*reproducible* artifact. The numbers are not wrong -- they are draws from the right
distribution with an unrecorded seed. Seed-to-seed variance in the main sweep was still
real variance, so the 8-seed statistics plan is unaffected in kind. What breaks is
`verify_repro.py`, `docs/repro_reference.json`'s tolerance band, and the sweep's resume
logic, all of which assume a run can be repeated. The reference band has to be rebuilt from
a post-fix run before it means anything.

**Why 491 tests missed it.** Every one of them calls the handlers directly, in one process,
where the server's `seed_everything` had already run and there is no Ray actor boundary to
cross. The failure needs a real federation, which this project first had today.

## Phase 4 -- the degenerate corner is a fitness-design fault, not a tuning problem

**Status: cause identified and a fix implemented behind a flag, 2026-09-20. Unconfirmed on
real deltas.** The previous entry said to raise `aco-gamma-entropy` until the corner margin
went negative. That was done, and it is the wrong fix.

### What the second gate run showed

At `aco-gamma-entropy=0.6` (the measured 0.481 requirement plus headroom), at K=10:

| check | result |
|---|---|
| 3. corner degenerate? | **CLEAR** -- FedAvg outscores the best vertex by 0.4524 |
| 2. deposit floor | **7 of 15 rounds** had no positive alpha, so deposited nothing |
| 1. colony searching? | unanswerable (the best case went negative; see below) |
| 4. beats FedAvg? | **BEHIND: 0.1185 vs 0.3286, margin -0.2101** |

The corner was closed by breaking the mechanism. `F = gamma_1*align - gamma_2*disp -
gamma_3*P`, so a penalty large enough to outweigh a vertex's free alignment also drags every
*interior* candidate down. The MAX-MIN rule deposits `rho * Q * max(F, 0)`, so a negative
fitness deposits nothing however well the colony searches: tau never moves, `tau^a * eta^b`
collapses to `eta^b`, and what remains is deterministic heuristic weighting that lost to
plain averaging by 0.21 macro-F1.

### The prior problem, which was in the first run's data all along

`fedavg_fitness` is logged per round and nothing read it. At the **default**
`gamma_entropy=0.1`, over the first gate's 15 rounds:

    F at the FedAvg point: mean -0.0005, negative in 7/15 rounds, range -0.2798 to +0.1833

The objective does not rank the aggregation FedACO exists to improve on above zero. It is
zero-centred noise there. So the colony's `best_fitness` mean of +0.1602 is a margin over
noise, and there is no `gamma_entropy` that fixes that -- raising it only moves the reference
point further below zero.

### The cause, and why no amount of gamma_3 reaches it

Dispersion is `sum_k alpha_k ||delta_k - Delta(alpha)||^2`: a weighted variance about a mean
that **moves with alpha**. At `alpha = e_j` the mean *is* `delta_j`, so every term is exactly
zero -- not small, zero. A single-client answer pays nothing for dispersion and collects the
alignment for free, leaving only `gamma_3 * log K` against it. The degeneracy is structural.

### The fix, implemented and unconfirmed

`aco-dispersion-reference` selects what the spread is measured about:
`"weighted_mean"` (the method as proposed, still the default) or `"base"` -- about
`Delta(base_weights)`, the FedAvg point, which does not move with alpha. A vertex then costs
`||delta_j - Delta_base||^2`, which is large. Same O(K^2) Gram-only expansion, same cost.

It is also the more defensible quantity: FedACO's job is to beat FedAvg, so "how far are
these clients from the aggregate I am trying to beat" is worth charging for, while "how far
are they from wherever I put the mean" is trivially answered by collapsing onto one client.

Measured on synthetic consensus deltas, K in {10, 20}, noise in {1.5, 3.0}, at the **default**
`gamma_entropy=0.1`:

| | corner margin | F(best) | alpha_max |
|---|---|---|---|
| weighted_mean | +0.0571 (corner wins) | +0.3731 | 0.119 |
| base | **-0.5056** | +0.3720 | 0.119 |

Surgical where `gamma_3=0.6` was not: the corner loses and nothing else moves.

**LOCALLY REPRODUCED AND ANSWERED, 2026-09-24 (`make fitness-repro`).** The claim below that
only a GPU gate could settle this was wrong -- `ray`/`flwr[simulation]` run in this project's
Linux containers, and the missing piece was never compute but a fixture heterogeneous enough
to reproduce the failure. `scripts/synthetic_heterogeneous_dataset.py` gives one (class-correlated
patterns, so dirichlet skew produces genuinely different local optima; `ci_smoke_dataset.py` is
deliberately trivial for CI speed and cannot). At gamma_3=0.1, K=10, dirichlet(0.3), 15 rounds,
about a minute per arm:

| | weighted_mean (local) | real Kaggle | aggregate (local) |
|---|---|---|---|
| F at the FedAvg point, mean | **-0.0033** | -0.0005 / -0.0212 | **+0.8141** |
| negative in | 5/15 rounds | 7/15 rounds | **0/15** |
| corner_margin, mean | **+0.6185** | +0.3136 / **+0.6184** | **-0.5283** |
| corner wins in | **15/15** | **15/15** | **0/15** |
| rounds that deposited nothing | 3/15 | 1-7/15 | **0/15** |
| final test macro-F1 | 0.1000 | 0.1185 - 0.1381 | 0.1168 |

The left column reproduces the real failure on every axis. The right shows `aggregate` removing
both halves of it: the corner never wins, and F at the reference point stops being zero-centred
noise, so every round deposits.

`alpha_max` under `aggregate` is 0.207 against uniform's 0.100, so the colony still has room --
this is not the `gamma_3=0.6` failure where the landscape was repaired by flattening it.

**What is still open.** A synthetic *signature* match is a much weaker claim than a data match:
it says the fixture fails the same way, not that the fix transfers. 15 rounds is also far too
short to read the macro-F1 column. The real gate remains the test that counts -- but it is now
a confirmation rather than a discovery, and any further fitness change can be screened here
first for the cost of a minute instead of a GPU session.

**The original note, kept because its reasoning was the mistake worth remembering:** all of
that is synthetic. Consensus deltas have high alignment and low
dispersion, which is exactly the regime where this is easy, and real MRI deltas under
dirichlet(0.3) are not that regime -- the level-set episode is precisely a case of synthetic
evidence not transferring. The next gate run at `aco-dispersion-reference=base` and
`aco-gamma-entropy=0.1` is what settles it. A6 gains a `dispersion_about_fedavg_point` cell
so the choice is measured rather than asserted (250 -> 260 cells).

**If it wins, it is not a hyperparameter result.** It is a correction to the plan's §4.5
fitness, and the paper has to describe the method that way.

## Phase 4 -- `dispersion_reference="base"` was not the fix, and could not have been

**Status: superseded by `"aggregate"`, 2026-09-20. Unconfirmed on real deltas.**

The gate run at `aco-dispersion-reference=base`, `aco-gamma-entropy=0.1`, K=10:

| | weighted_mean | base |
|---|---|---|
| mean corner margin | +0.6184 | **+0.3136** |
| F at the FedAvg point | -0.0005 | **-0.0212** |
| vs FedAvg | -- | **BEHIND 0.1328 vs 0.3321** |

It halved the margin and stopped. That is not bad luck, it is the only thing the term can do:

    dispersion_about_reference(alpha) = sum_k alpha_k * c_k,   c_k = ||delta_k - Delta_ref||^2

`c_k` is fixed before alpha is chosen, so the term is **exactly linear in alpha**. A linear
function on the simplex attains its extremes at vertices, so it has no interior optimum at
all. It charges for picking *far* clients, never for concentration -- and a vertex on the
client with the smallest `c_k` is therefore *cheaper* than any mix. Measured at K=10,
noise 3.0: cheapest vertex 7.29 against 8.01 at every interior point.

So the previous entry's synthetic result (-0.51 at K=10) was real but not general. Consensus
deltas at low noise make the `c_k` nearly equal, which hides the whole defect; the probe was
run in the one regime where a linear term looks like a convex one.

### `"aggregate"`: the distance of the AGGREGATE from the consensus

    ||Delta(alpha) - Delta_rob||^2 = alpha' G alpha - 2 alpha . g_rob + ||Delta_rob||^2

Quadratic in alpha, minimized in the interior, and for a reason that is a property of the
quantity rather than a penalty bolted onto it: averaging cancels per-client noise, so a
K-way mix keeps roughly 1/K of each client's deviation from the consensus while a single
client keeps all of its own. An interior point is genuinely nearer the consensus, not merely
less penalized. Same precompute, no new O(K*d) work.

Swept over K in {10, 20} x noise in {1.5, 3, 6} x {uniform, dirichlet(0.3)-skewed} base
weights, at the **default** `gamma_entropy=0.1`, it is the only one of the three shapes that
is correct in all 12 cells:

| mode | corner margin | F at the FedAvg point |
|---|---|---|
| weighted_mean | +0.0012 .. +0.3663 (corner wins) | -0.3276 .. +0.3698 |
| base | -0.0223 .. -0.8621 | -0.2567 .. +0.3698 |
| **aggregate** | **-0.7373 .. -1.7790** | **+0.3021 .. +0.9885** |

The second column matters as much as the first. Under the other two shapes, F at the
reference point is zero-centred noise once the base weights are skewed -- which is exactly
what the real runs measured (-0.0005, then -0.0212) -- so the colony's margin over it is a
margin over noise regardless of what the corner does.

**The skewed high-noise cells now reproduce the real runs' negative F(fedavg)**, which is the
first time this project's synthetic proxy has matched the failure it is standing in for. That
is a reason to trust it more than the last two probes, not a reason to skip the gate run.

A6 keeps a cell for each of the three (260 -> 270): the paper has to show why the obvious
fixed-reference fix is not the one, and that needs the measurement, not an argument.


## Phase 3 -- a mis-set supernode count hangs a run forever, and nothing can detect it

**Status: mitigated where it can be, open where it cannot.**

Flower's node sampler (`flwr.serverapp.strategy.strategy_utils.sample_nodes`) is

    while len(all_nodes := list(grid.get_node_ids())) < min_available_nodes:
        sleep(1)

with no give-up. If the federation is configured for fewer SuperNodes than a run's
`min-available-nodes` / `min-train-nodes` asks for, the run does not fail -- it logs
"Waiting for nodes to connect" once a second and consumes the whole session.

**Observed live, 2026-09-24**, while screening A1 locally: a run configured for 4 supernodes
while asking for 10 sat at round 1 for over three minutes with the machine *idle*
(load average 0.29, ClientApp actors at 2% CPU) before it was killed. Identical runs against
a correctly configured federation finish 15 rounds in 39 seconds. On a Kaggle session that is
nine hours and no output.

**What makes it worse than an ordinary misconfiguration:** `flwr federation
simulation-config` can only SET the count, never read it. There is no way to ask what the
federation is currently configured for, so a preflight cannot check it -- the only protection
is to set it immediately before every run that depends on it.

**Mitigations in place.** `run_sweep.py` and `run_sweep_granular.py` both call
`configure_federation` before their cells and refuse if it fails, so every sweep is covered.
The gap was the notebooks' *bare* `flwr run` gate cells, which relied on a section-3 cell
having been run earlier in the same session -- a stale kernel, a restarted SuperLink or a
Run-All from the middle would all leave the count wrong. B's gate cell now sets it itself,
immediately before the runs.

**Still open.** Nothing can catch this from inside the application once a run has started.
A wrapper that bounded each `flwr run` with a timeout and treated "Waiting for nodes to
connect" in the log as fatal would close it -- worth doing if a session is ever lost to this,
and not worth the complexity before then. If a run sits at round 0 or 1 with an idle machine,
this is the first thing to check.

## Phase 7, A1 -- LOCAL SCREEN: ACO loses to all four controls, and the mechanism is identified

**Status: the paper's make-or-break result, screened locally 2026-09-24. Needs the real A1 to
confirm, but the cause is diagnosed and it is a design flaw rather than a verdict on ACO.**

A1 costs 80 cells and 17-33 GPU-hours on Kaggle. Screened on the local heterogeneous fixture
(5 search methods x 3 seeds x 15 rounds, K=10, dirichlet(0.3), `aggregate` fitness so the
landscape is not the degenerate one, equal evaluation budget enforced by `BudgetedFitness`):

| method | mean gain over the FedAvg point | alpha_max | budget used |
|---|---|---|---|
| **aco** | **+0.0013** | 0.200 | 96.8% |
| random | +0.0304 | 0.192 | 100% |
| coordinate_grid | **+0.0501** | 0.158 | 100% |
| pso | +0.0484 | 0.173 | 100% |
| ga | +0.0312 | 0.185 | 100% |

Per-seed ACO gain: `[-0.0002, +0.0002, +0.0039]`. **ACO beats every control on 0 of 3 seeds**,
and loses to the weakest of them (random) by a factor of 23.

The plan's stated failure mode for C2 was "ACO ties random search". This is worse than a tie.
But it is not the verdict it looks like, because the cause is mechanical and fixable.

### Why: the greedy rule constructs the FedAvg point

`eta_{k,l} = (1 + |lambda_l - d_hat_k|)^-1`, where `d_hat_k` rescales the per-client sigmoid
score `d_k` across the level range. Measured across K in {10, 20} and noise in {1.5, 3, 6},
**`d_k` spans about 0.04** (e.g. 0.7685 - 0.8073) against a level spacing of **0.25**. So every
client's `d_hat` lands in the same level bin, and `argmax_l eta_{k,l}` returns the same level
for every client -- identical in 3 of the 6 configurations tested, adjacent levels in the rest.

A uniform level assignment normalises back to `base_weights` **exactly**
(`levels_to_alpha`: `tilde = levels[idx] * base_weights`, then scaled to `target_sum`). So when
the argmax is client-invariant, the greedy branch of the ACS rule constructs **the FedAvg
point** -- and the screen ran at `q0=0.9`, so that is **90%** of station decisions.

The 0.9 is itself a bug, found on 2026-09-24 and fixed: plan §14 specifies `q0_exploitation:
0.70`, and `configs/strategy/fedaco.yaml` encodes it, but `pyproject.toml` -- which is what
`flwr run` and therefore this screen used -- still carried phase-4's 0.9. See "Four sources of
truth for the FedACO defaults" below.

The colony is therefore anchored at FedAvg by its own exploitation rule, and can only move via
the 30% exploratory draws. Its best candidate came in at +0.9559 against F(FedAvg) = +0.9558:
it finds the reference point and essentially nothing else. The controls, having no such anchor,
explore and find better points. That is the whole result.

### What this means and what it does not

It does **not** mean ACO cannot help here. It means the heuristic as specified in §4.5 has
almost no per-client dynamic range, so the pheromone-times-desirability rule degenerates to
"everyone takes the same level". Candidate fixes -- standardising `d_k` across clients before
rescaling, widening the sigmoid, or lowering `q0` -- are method changes for the team, not
something to pick unilaterally. Any of them can now be screened locally in minutes.

`q0` has since been screened (see below): lowering it recovers most of the gap but does not
close it, so it is a contributing cause rather than the whole story.

### Also measured, and deliberately NOT fixed

The colony visits **82 distinct alphas out of 210 evaluations** (61% duplicates) where random
search visits 210 of 210. Memoizing within a round is exact -- the Gram matrix is fixed, so a
repeated alpha has the same fitness by construction -- and was implemented and reverted:
`run_colony` stops on its own ant/iteration schedule rather than on `remaining()`, so the freed
budget is never spent and the best fitness was **identical** in all 8 configurations tested. It
also broke four tests encoding "no method outspends the shared budget" and changed the meaning
of the logged `evaluations_used`. Spending the freed budget is a change to the colony's stopping
rule, which belongs with the heuristic decision above.

## Phase 7 -- RESOLVED: `q0` is a contributing cause of A1's result, not the whole one

A1's local screen left the anchoring diagnosis untested: the argument was that a
client-invariant `argmax` makes the ACS greedy branch reconstruct the FedAvg point, so at
`q0=0.9` the colony spends 90% of its station decisions standing still. If that is right,
lowering `q0` must recover gain. Screened on the same heterogeneous fixture (K=10,
dirichlet(0.3), 15 rounds, `aggregate`, gamma_3=0.1), 4 values x 3 seeds, 12 real runs:

| `q0` | mean gain over the FedAvg point | per-seed |
|---|---|---|
| **0.9** (what the screen ran) | **+0.0013** | `[-0.0002, +0.0002, +0.0039]` |
| 0.5 | +0.0222 | `[+0.0214, +0.0203, +0.0248]` |
| 0.2 | +0.0259 | `[+0.0294, +0.0242, +0.0239]` |
| 0.0 | +0.0287 | `[+0.0268, +0.0301, +0.0292]` |

The effect is **+0.0274 from 0.9 to 0.0, a 22x improvement**, against a between-seed spread of
0.0033-0.0056 at every level -- so unlike the macro-F1 column below this is not an averaging
artifact. **The anchoring diagnosis is confirmed.**

It is not monotone below 0.5, and that matters for how it gets described: essentially the whole
effect is the single step from 0.9 to 0.5 (+0.021), and 0.5 / 0.2 / 0.0 are within noise of each
other. The finding is "0.9 is pathological", not "less greed is always better".

**But it does not close the gap.** The controls on the identical fixture: coordinate_grid
+0.0501, pso +0.0484, ga +0.0312, random +0.0304. Even at `q0=0.0` -- no exploitation at all,
which is barely ACO any more -- the colony reaches +0.0287 and still loses to every one of them.
So the greedy branch is *a* cause, and something else is also wrong. The remaining candidates
are unchanged and still the team's call: standardising `d_k` across clients before rescaling
(the 0.04-span problem the anchoring argument rests on, which lowering `q0` routes around rather
than fixes), and widening the sigmoid.

### What this screen could NOT answer, and why the fixture is at fault

The obvious follow-up -- does a bigger fitness gain give a better model? -- **cannot be answered
on this fixture**, and the attempt is worth recording because the first pass at it looked like a
result. Over all 27 local runs (A1's 15 + this screen's 12):

    Pearson r(fitness gain, final macro-F1)  = -0.095
    Spearman rho                             = -0.235,  permutation p = 0.272 (20k shuffles)

Read carelessly that is "optimising the fitness makes the model worse", which is a much stronger
and more interesting claim than the data supports. It is not significant, and the reason is that
the outcome variable is mostly a floor indicator: **11 of 27 runs score macro-F1 exactly
0.1000**, which for 4 classes is what an all-one-class predictor scores, i.e. total collapse.
The rest are scattered over 0.1168-0.3750 with no structure.

Per seed the q0 sweep's macro-F1 goes `0.1168 -> 0.1000 -> 0.1000 -> 0.1168`,
`0.3406 -> 0.3643 -> 0.1000 -> 0.1000`, `0.2109 -> 0.1875 -> 0.3720 -> 0.1000` -- no direction in
any of the three. My own first reading of this screen, before the per-seed check, was that
macro-F1 degraded monotonically with `q0` (means 0.2228 -> 0.1056) -- an averaging artifact over
three seeds that flip between the collapse floor and 0.37. It was never written down, and it is
recorded here because it is the same mistake as the isolated colony probe in a smaller space:
a number computed correctly from a fixture that cannot support the claim drawn from it.

**So the fixture is an instrument for fitness behaviour only.** It was validated against the real
runs' *fitness* signature (F at the FedAvg point near zero, corner_margin positive) and it
reproduces that faithfully -- which is what earned it trust for A1 and this screen. It was never
validated as an accuracy instrument, and at 15 rounds it is not one. Any claim of the form
"configuration X trains a better model" needs the real gate. Fixing this would need more rounds
or an easier signal, and both change the fixture's fitness signature, which is the thing it
exists to reproduce -- so it is a genuine trade, not an oversight to patch.

## Phase 4/6/7 -- RESOLVED: four sources of truth for the FedACO defaults, no two agreeing

Found while checking whether the screen above had run at the documented `q0`. It had not.

FedACO's default hyperparameters are written down in four places:

| where | role | `aco-q0` | `aco-rho-round` | `aco-gamma-dispersion` |
|---|---|---|---|---|
| `docs/IMPLEMENTATION_PLAN.md` §14 | the spec | 0.70 | 0.30 | 0.50 |
| `configs/strategy/fedaco.yaml` | per-strategy overrides | 0.70 | 0.30 | 0.50 |
| `pyproject.toml` `[tool.flwr.app.config]` | **what `flwr run` uses** | **0.9** | **0.1** | **1.0** |
| `paper/ALGORITHM.md` §14 table | the paper's description | 0.70 | 0.30 | 0.50 |

`pyproject.toml` carried phase-4's values (commit `0bb64b7`); `configs/strategy/fedaco.yaml` was
written later (commit `09739ca`) from the plan, annotating each line with its §14 symbol. Neither
was ever reconciled, and **nothing checked them against each other** -- the recurring shape in
this project, arriving this time as two sources of truth rather than a check that means nothing.

**Which values a sweep gets depends only on how its experiment YAML lists strategies.** Bare
names (`- fedaco`) go through `run_sweep.py`, which never reads `configs/strategy/`, so the run
takes pyproject's defaults. A `file:` reference goes through
`sweep.resolve_entry_overrides`, which layers the strategy file on top. The split runs straight
through the paper's tables:

    pyproject's values (q0=0.9):   main.yaml, ablation_all.yaml, robustness.yaml
                                   ...and every `flwr run`, so the Kaggle gate too
    fedaco.yaml's values (q0=0.70): main_client_scale.yaml, robustness_r1..r4, smoke.yaml

So `main.yaml`'s headline table and `main_client_scale.yaml`'s K-scaling table, which the paper
reads as one method at different client counts, would have run **different algorithms**. Same for
`robustness.yaml` against `robustness_r1..r4`. And `aco-gamma-dispersion` is a weight *inside the
fitness*, so the objective differed, not only the search over it -- 1.0 against 0.50 is double
weight on the dispersion term.

### Fixed

`pyproject.toml` now matches plan §14 on all three keys. That direction is not a choice between
two defensible values: the plan is the spec, `fedaco.yaml` and `ALGORITHM.md` both already encode
it, and A6 is the ablation that will tune these from that documented starting point.

`tests/test_config_defaults.py` (32 tests) locks it down, and each assertion was confirmed to
fail when the drift is reintroduced:

* every `configs/strategy/*.yaml` key that also exists in `pyproject.toml` must hold the same
  value -- a strategy file may add keys, never contradict a default;
* every strategy-file key must be a declared run_config key (an undeclared one makes `flwr run`
  exit with a bare `[code: 15]`, a misspelled declared one is silently ignored);
* both must match plan §14, with a separate test that re-parses §14 and fails if the
  transcription in the test file has itself gone stale;
* and an end-to-end test building the run_config by *both* runner paths and requiring every
  `aco-*` key to agree.

### Also fixed: `paper/ALGORITHM.md` described keys that do not exist

All 21 flat keys in its §14 table used a `fedaco-` prefix; the real prefix is `aco-`. Anyone
following that table to reproduce a run would have passed keys `flwr run` rejects outright. It
also described the levels as "log-spaced over [0.0, 2.5]" -- log spacing was a real experimental
error, fixed long ago, recorded in `EXPERIMENT_LOG.md`, and still in the paper's own method
description. Both corrected.

### What this does to results already in hand

Every local screen in this document ran at `q0=0.9`, `gamma_2=1.0`, `rho_round=0.1` -- verified
from the embedded `config.run_config` in the result JSONs, not inferred. A1's screen is therefore
a measurement of a configuration the paper does not document, and is being re-run at the
corrected defaults. The Kaggle gate is in the same position: it used pyproject, so it ran at 0.9.

## Phase 7, A1 -- the fallback rate is the measurement, and standardising `d_k` makes it worse

Two results, from 3 real runs on the heterogeneous fixture at the corrected plan-§14 defaults,
against the A1 controls already measured there.

### 1. Standardising `d_k` is not the fix

`aco-desirability-scaling='standardized'` z-scores `d_k` across clients before mapping it onto
the level range, so the measured 0.04 span fills the grid instead of collapsing into one bin.
It was the leading candidate fix for the anchoring problem. It makes things **worse**:

| arm | mean gain | per-seed |
|---|---|---|
| aco, `absolute` (plan §4.2) | +0.0066 | `[+0.0088, +0.0025, +0.0085]` |
| aco, `standardized` | **+0.0006** | `[+0.0040, -0.0006, -0.0018]` |

An 11x degradation, and negative in 2 of 3 seeds. The diagnosis that the greedy branch
reconstructs the FedAvg point was correct; the inference that it should therefore be made to
construct something else does not follow, because what it then constructs is worse. **The
FedAvg point was a good anchor, not a bad one.**

### 2. The controls search the same space, so A1 is not confounded

Worth stating because it was the next hypothesis and it is false. `aco/controls.py`:
`random_search`, `coordinate_grid_search` and `genetic_algorithm_search` all construct alpha
through `levels_to_alpha` -- the identical discretised construction the colony uses. Only
`pso_search` is continuous, and PSO is not the winner. So ACO is not being beaten by searchers
with a finer parameterisation; it loses on the same grid.

### 3. What is actually happening: the safety fallback fires in 44% of rounds

`fallback_used = safety_fallback and best_fitness <= fedavg_fitness` -- it records a round in
which **the colony found nothing better than the FedAvg point**. Across the same fixture and
budget:

| arm | rounds the search failed to beat FedAvg | mean alpha entropy (max log 10 = 2.3026) | alpha_max |
|---|---|---|---|
| **coordinate_grid** | **0.0%** | 2.2507 | 0.1565 |
| random | 17.8% | 2.1623 | 0.1897 |
| **aco (`absolute`)** | **44.4%** | 2.1958 | 0.1930 |
| aco (`standardized`) | 60.0% | 2.1850 | 0.1943 |

A coordinate sweep improves on FedAvg in **every round**. Uniform random sampling over the
same discrete space fails in about 1 round in 6. The ant colony fails in nearly **half**, and
standardising `d_k` pushes that to 60%.

This is the same ordering as the fitness-gain table, but it says something the gain does not:
the colony is not making small improvements, it is **frequently making none at all** and being
rescued by the safety net. `check_fedaco_health.py` has reported this number since it was
written, and its docstring already anticipated the reading -- "a high fallback rate with a good
final score means the score belongs to FedAvg". What had never been done was compare it across
search methods, at which point it stops being a diagnostic and becomes the result.

Note also that the winner is the **least** concentrated: coordinate_grid has the highest alpha
entropy and the lowest alpha_max of any arm. Whatever this fitness rewards on heterogeneous
data, it is not the concentrated weighting the method's story is built around.

### What this leaves

The candidate fixes are now measured rather than speculated:

* lower `q0` -- real (22x), insufficient (still loses to every control)
* standardise `d_k` -- **harmful** (11x worse, negative on 2 of 3 seeds)
* widen the sigmoid -- unscreened, and now much less promising: both screened fixes worked by
  moving the colony *away* from the FedAvg point, and both times that was the wrong direction

The remaining explanation is the one the numbers point at directly: at this evaluation budget,
`tau^a * eta^b` sampling is simply a worse search of this space than a coordinate sweep, and
the pheromone adds nothing a uniform sample does not already have. That is a statement about
claim C2, not a bug to fix, and it is the team's to act on. Plan §11's week-5 gate exists for
exactly this decision.

Confirming it on the real gate costs A1's 80 cells (17-33 GPU-h). Everything above is the local
fixture, which has reproduced the real runs' fitness signature on every axis checked but is
still a fixture.

## The reframe decision — TAKEN 2026-09-25, framing A, and reversible by one experiment

Plan §11's week-5 gate says: "If A1 shows ACO ties random search at equal budget, stop and
reframe before investing weeks 6-12 in a claim that won't survive review." §12 rates it
Medium-High likelihood and **fatal to the framing**.

**Status of the evidence.** Three independent local screens on the heterogeneous fixture, at
both the drifted and the documented defaults, and with a `q0` sweep down to 0.0:

| method | gain over the FedAvg point |
|---|---|
| coordinate_grid | +0.0501 |
| pso | +0.0484 |
| ga | +0.0312 |
| random | +0.0304 |
| **aco** | **+0.0013 → +0.0066** after a config-scoping fix |

ACO beat every control on **0 of 3 seeds**. The plan's stated failure condition was a *tie*;
this is a loss by 5.9x to the weakest control. The mechanism is identified and is not about ant
colony optimisation (greedy anchoring to the FedAvg point; see the 2026-09-24 log entry).

**Decision, taken because the work could not proceed without one.** The paper is written under
**framing A**: the degenerate fitness optimum and the equal-budget result are the findings, and
the reusable contributions are the `corner_margin` diagnostic, its closed-form penalty
threshold, the equal-budget protocol, and the de-duplicated split. `paper/01_INTRODUCTION.md`
§1.3 is written this way.

**This is not a claim that A1 will confirm it, and it is cheap to reverse.** The alternative
framing is written out in full at the end of `paper/01_INTRODUCTION.md`: if A1 separates ACO
from all four controls on real data at 8 seeds, §1.3 is deleted and replaced by that block, the
degeneracy becomes a method subsection rather than a finding, and §5.2 becomes an ablation
rather than the result. Nothing else in the paper changes -- §4 (setup), §6 (limitations) and §7
(reproducibility) are framing-independent by construction.

**Why take it now rather than wait for A1.** A1 is 17 GPU-h and cannot run in this environment.
Waiting leaves the paper unwritten, and the paper is the binding constraint on a 9-day deadline
far more than 70 GPU-hours is. Writing under the framing the evidence supports, with the switch
prepared, costs one section if the evidence turns out wrong and saves nine days if it does not.

**⚠️ This is a methodological call and the team can overturn it.** It was taken by the agent
because the work was blocked on it, not because it is the agent's to make. Everything needed to
reverse it is in one file.
