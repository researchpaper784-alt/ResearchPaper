# Experiment log

Chronological record of decisions, sweeps, failures, and abandoned directions.

---

## 2026-09-14 — Compute budget and training-environment decisions (pre-Phase-1)

### The problem

The implementation plan specifies ResNet-18 @ 224² as the main configuration and a full
factorial main sweep (~11 strategies × 6 partitions × 5 seeds ≈ 330 runs × 100 rounds).
Costed out against the available hardware, this is infeasible:

| Config | Est. per run (100 rounds) | Main sweep (330 runs) |
|---|---|---|
| ResNet-18 @ 224² (as planned) | ~2–3 h | ~800–1000 GPU-h |
| Small model @ 112², cached arrays, AMP | ~20–25 min | ~110–140 GPU-h |

At Kaggle's ~30 GPU-h/week free quota, the plan as written is ~30 weeks of quota for the
main table alone, before ablations (Phase 7) and robustness (Phase 8).

**These are estimates, not measurements.** They are derived from rough throughput
assumptions (ResNet-18 @224², batch 32, T4 with AMP ≈ 300–400 img/s training; ~11,400
image forward+backward passes per round at K=20, 2 local epochs) plus non-trivial
ray-actor and state-dict serialization overhead (20 clients × 44 MB × 2 transfers/round).
**Replace with a measured per-round wall-clock from the first Kaggle smoke run and update
this table.** Do not cite these numbers anywhere until measured.

### Decisions (author, 2026-09-14)

1. **Primary configuration is the small model at 112²**, used for the full main grid,
   all ablations (Phase 7), and all robustness tests (Phase 8). ResNet-18 @ 224² becomes a
   reduced secondary "does it hold with a larger backbone" table (~24 runs).

   *Justification for the paper:* from-scratch small CNNs are standard in the FL
   literature (the original FedAvg paper used a two-conv CNN), and the plan's own §12 risk
   register notes that a from-scratch backbone is the more honest FL setting — gains from
   aggregation methods are known to shrink behind a strong ImageNet-pretrained backbone.
   This is a defensible primary setting, not only a compute concession. State it plainly
   in the paper's experimental setup rather than burying it.

2. **Training runs on Kaggle** (primary), not Colab. The dataset is native to Kaggle
   (mounts at `/kaggle/input/...` with no download and no API credentials), the ~30
   GPU-h/week quota is predictable where Colab free is not, and "Save & Run All" provides
   real background execution. Colab remains available for overflow/interactive debugging.
   *Verify current quota and session-cap numbers in-account; they drift.*

3. **Division of labour between machines:**
   - **Local Mac (CPU):** Phase 1 entirely (dedup, manifest, partitioning — perceptual
     hashing ~7,000 JPEGs is a couple of minutes), plus Phase 9 analysis, figures, tables.
   - **Kaggle (GPU):** Phases 2–8 training only.
   - Code travels by git (required: `utils/provenance.py` records the git SHA, which is
     meaningless unless Kaggle runs a pinned, pushed commit). Result JSONs are small
     (~hundreds of KB for a whole sweep) and travel back the same way. Model checkpoints
     (~44 MB) stay ephemeral in `/kaggle/working` and matter only for intra-session-chain
     resume.

4. **Repo hosting deferred.** Author will decide GitHub public vs. private later. Until
   then the repo is local-only. **This is a blocker for Phase 3 onward** (Kaggle needs to
   clone from somewhere) but not for Phase 1.

5. **Notebook runner deferred to after Phase 1.**

### Consequences for Phase 1 (why this was decided first, cont. below)

- Preprocess and **cache decoded images as `uint8` arrays at 112²**, not JPEG paths
  re-decoded per round. Decoding ~11,400 images per round through PIL costs an estimated
  45 s/round single-threaded and is likely the single largest hidden cost in the
  simulation. The full cached dataset at 112² is ~215 MB and fits in RAM.
- Keep a **224² cache path** available for the secondary ResNet-18 table. The cache
  builder must be parameterized by resolution, not hardcoded.
- **Manifest paths must be relative to a dataset root supplied by an environment
  variable** (`FEDSWARM_DATA_ROOT`), never absolute — the same committed `manifest.csv`
  has to resolve against both the local download and Kaggle's `/kaggle/input/...` mount.

---

## 2026-09-14 — Phase 1.2 leakage audit: the dataset has severe train/test contamination

### What was received

The archive the author downloaded (`archive (2).zip`, 164 MB, 7,200 files) has the
Nickparvar naming convention (`Tr-gl_*`, `Te-pi_*`) and `Training/`/`Testing/` structure,
but is **perfectly class-balanced** — exactly 1400 per class in Training and 400 per class
in Testing. The widely-cited Nickparvar dataset is imbalanced (~1321/1339/1595/1457 train).
So this is a rebalanced variant, not the canonical release. Flagged; see Open question
below.

Observed from disk: 447 distinct image sizes; colour modes RGB 4129, L 3067, RGBA 3, P 1.

### Headline finding

| Measurement | Value |
|---|---|
| Images | 7,200 |
| Unique pseudo-patients (t=5) | **4,755** (34% of the dataset is redundant) |
| Exact byte-identical duplicates | 187 |
| Largest duplicate component | 28 images |
| **Cross-split leaked images (original split)** | **2,030 — 28.2% of the dataset** |
| Cross-split leaked components | 520 |
| phash/dhash agreement | 0.836 |

**28% of this dataset straddles its own train/test boundary.** Anyone who trains on the
provided split and reports test metrics is testing on training data. This is the
mechanism behind the 99%+ accuracies that saturate this dataset's literature.

### Threshold calibration (the plan requires this be chosen by inspection, not assumed)

phash with `hash_size=8` yields only even Hamming distances here, so thresholds pair up
(t=0≡1, t=2≡3, t=4≡5, ...).

| t | phash edges | components | leaked images | mixed-label components |
|---|---|---|---|---|
| 0 | 1,930 | 6,212 | **855** | 1 |
| 2 | 3,885 | 5,639 | 1,322 | 4 |
| **4/5 (chosen)** | **5,893** | **4,755** | **2,030** | **25** |
| 6 | 8,799 | 3,425 | 3,309 | 61 |
| 8 | 14,853 | 1,877 | 5,098 | 45 |
| 10 | 31,956 | 722 | 6,411 | 11 |

**Even at t=0 — bit-identical perceptual hashes — 855 images (11.9%) leak across the
split.** That is an indisputable lower bound requiring no threshold judgement at all, and
it is the number to quote if a reviewer disputes the calibration.

Visual inspection of sampled pairs (contact sheets regenerate via
`scripts/audit_threshold.py` into `data/processed/audit/`):

- **d=0:** 8/8 sampled pairs are unambiguous duplicates; 5/8 straddle Training/Testing
  (e.g. `Training/meningioma/Tr-me_846.jpg` vs `Testing/meningioma/Te-me_117.jpg` —
  same patient, same tumour, identical image). Zero false positives.
- **d=6:** degradation begins. 1/8 is a clear false positive (`Tr-no_297` vs `Tr-no_878`
  — different patients, different sequences entirely). Several others are the same
  patient's *adjacent slices*, which pseudo-patient grouping is meant to catch and are
  therefore true positives by our definition.
- Mixed-label components (a proxy for false merges) rise 25 → 61 between t=4 and t=6.

**Chosen: phash threshold 5 (≡4), hash_size 8.** Estimated false-positive rate at the
operating point: 0/8 at d≤4 in the inspected sample, with the first clear FP appearing at
d=6. This coincides with the plan's suggested default, now earned rather than assumed.

### Consequences

1. The paper must report leakage in the original split as a finding, with the t=0 lower
   bound (855) alongside the operating-point figure (2,030).
2. All splits are rebuilt at pseudo-patient level in Phase 1.3, making leakage zero by
   construction. Expect the resulting "clean" accuracy ceiling (Phase 2.1) to be
   **materially lower** than this dataset's published results — that gap is a feature, and
   the honest ceiling to measure every FL method against.
3. 4,755 pseudo-patients (not 7,200 images) is the real unit count for partitioning. At
   K=20 clients that is ~238 pseudo-patients each — still workable.
4. The 25 mixed-label components need a look before Phase 1.3: they are either residual
   threshold error or genuine dataset mislabeling (the SARTAJ component has documented
   glioma mislabeling).

### Open question raised

Whether to switch to the canonical imbalanced Nickparvar release. Arguments for: it is
what the literature uses, so results are comparable, and the plan's macro-F1 justification
rests on class imbalance which this variant has removed. Arguments against: this variant is
already downloaded and audited, and our re-split discards the provided split anyway. The
class-balance question matters more than it looks — see `docs/OPEN_QUESTIONS.md`.

---

## 2026-09-14 — Phase 1.4 partitioning: measured heterogeneity of each regime

Partitions are over the 3,330 **training pseudo-patients** (val/test stay global and
server-side). K=20, seed=0. Two diagnostics are recorded per partition and carried into
every result file: mean pairwise Jensen-Shannon divergence between client label
distributions (label skew) and the Gini coefficient of client sizes (size skew).

| Regime | JS divergence | size Gini | min client | max client |
|---|---|---|---|---|
| iid | 0.007 | 0.002 | 166 | 167 |
| dirichlet α=0.1 | 0.541 | 0.483 | 12 | 647 |
| dirichlet α=0.3 | 0.435 | 0.326 | 34 | 402 |
| dirichlet α=0.5 | 0.391 | 0.426 | 19 | 659 |
| dirichlet α=1.0 | 0.217 | 0.312 | 19 | 419 |
| pathological (2 cls) | 0.537 | 0.002 | 166 | 168 |
| quantity_skew σ=1.0 | 0.010 | 0.416 | 14 | 534 |

Sanity checks that the regimes behave as intended:

- `iid` is near-zero on both axes — a genuine control.
- Dirichlet JS divergence falls monotonically with α (0.541 → 0.435 → 0.391 → 0.217),
  which is the defining behaviour of the regime.
- **`pathological` and `quantity_skew` are orthogonal:** pathological has high label skew
  (0.537) with essentially no size skew (0.002), quantity_skew has the reverse (0.010 /
  0.416). That separation is what lets the paper attribute a gain to one kind of
  heterogeneity rather than "non-IID" in general, and it is why `quantity_skew` is the
  regime to watch — it attacks FedAvg's n_k weighting directly, which is precisely what
  FedSwarm replaces.
- Dirichlet size Gini is *not* monotone in α (0.326 at α=0.3 vs 0.426 at α=0.5) because α
  controls only label proportions; size skew is an uncontrolled by-product of the draw.
  Expect this to smooth across the 5 seeds. Do not describe Dirichlet α as controlling
  size skew in the paper.

### Deviation from the plan, recorded

The plan suggests reusing `flwr_datasets.partitioner.*`. Those partition rows of a Hugging
Face `Dataset`; our unit is a pseudo-patient row in a manifest and the required output is a
JSON-cacheable index assignment. Converting manifest → HF Dataset → partition → indices is
more code and more fragile than drawing the Dirichlet directly, and `quantity_skew` needs a
custom implementation regardless. All regimes are therefore implemented in
`src/fedswarm/data/partition.py` using the standard formulations (per-class Dir(α) draw;
FedAvg-style contiguous shards for pathological). Verified by unit tests including a
monotonicity check on α.

`source_shift` raises `NotImplementedError` pointing at `docs/OPEN_QUESTIONS.md` — still
blocked on the provenance-label decision.

---

## 2026-09-14 — source_shift provenance recovery: Figshare matched, SARTAJ/Br35H pending

### Verifying the three upstream sources actually exist

Before downloading anything, searched to confirm the slugs guessed in the original
`docs/OPEN_QUESTIONS.md` entry are real (that entry explicitly flagged them as
unverified). All three confirmed:

- [SARTAJ Bhuvaji's Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
  — confirmed, including the documented glioma-mislabeling issue the plan mentions.
- [Br35H :: Brain Tumor Detection 2020](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)
  — confirmed.
- [Figshare brain tumor dataset (Cheng et al.)](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
  — confirmed: 3,064 T1-weighted images from 223 patients, DOI 10.6084/m9.figshare.1512427.v5,
  meningioma 708 / glioma 1426 / pituitary 930. These exact counts independently match the
  PNG-file counts found weeks earlier in the oversized 903MB zip the author first
  downloaded by mistake -- confirms that zip's PNG component *was* this Figshare set.

### Figshare: downloaded and matched

Figshare has an open REST API (`api.figshare.com/v2/articles/1512427`), no login needed --
downloaded all 4 zip parts (~885MB total) directly, MD5-verified against the API's
reported checksums (all OK), extracted to 3,064 `.mat` files (MATLAB v7.3/HDF5 format,
needs `h5py` -- added to `pyproject.toml`).

**Bug found and fixed: MATLAB column-major storage.** First matching attempt (naive
min-max normalization, no transpose) got essentially zero matches: 3/3064 images matched
against the full 7,200-image manifest at distance <=5, versus Phase 1.2's own
within-dataset calibration where genuine duplicates sit at that threshold. Distance
distribution peaked at 12-16 bits (not ~32, so *some* signal existed, just badly offset)
-- ruled out a bad normalization choice by rendering the images (they looked like clean,
correctly-windowed MRI scans) and instead found the real cause: h5py reads HDF5 arrays
row-major, but MATLAB writes them column-major, so every image was being read transposed
(sideways) relative to its actual orientation. Confirmed the fix on a 300-file sample
before committing to the full run: 219 *exact* (distance-0) hash matches immediately
appeared. Full-dataset re-run: **1,829/7,200 manifest images (25.4%) matched to a Figshare
source**, median match distance 0, max 4 -- among matched images, 75% are byte-for-byte
identical hashes.

**Sanity check that passed cleanly:** `notumor` gets 3/1,800 matches (Figshare has no
healthy-scan class -- these 3 are almost certainly coincidental noise, not real matches).
`glioma` 332, `meningioma` 678, `pituitary` 816 -- all plausible given Figshare's
708/1426/930 source counts and this dataset's internal duplication.

### Still pending

SARTAJ and Br35H, both on Kaggle -- blocked on Kaggle API credentials (`~/.kaggle/kaggle.json`),
which the author is setting up. `src/fedswarm/data/source_provenance.py` already supports
both via `load_flat_image_dir()` (writes to the same schema as Figshare); only the download
step remains. **`source_shift` partitioning itself is not yet wired up** -- that's next
once all three sources are matched, since partitioning on ~25% coverage (Figshare only)
would leave 3/4 of clients' data unlabeled.

Module and 15 unit tests (synthetic `.mat`/image fixtures, including a test that plants an
asymmetric image and would fail without the transpose fix) committed; 87 tests total, all
green.

---

## 2026-09-14 — source_shift wired up; Kaggle auth needed a CLI upgrade

### Kaggle's newer bearer-token auth

The author's new Kaggle API token (`KGAT_...` prefix) didn't work with the pinned
`kaggle==1.6.17` CLI -- that version only implements the legacy `kaggle.json`
`{username, key}` scheme. Confirmed via web search and Kaggle's own `kaggle-cli` docs:
newer tokens use `KAGGLE_API_TOKEN` or `~/.kaggle/access_token`, supported from
`kaggle==2.x`. Upgraded the pin to `2.2.4` and it authenticated immediately. Token stored
at `~/.kaggle/access_token` (chmod 600), not `kaggle.json`.
`fedswarm.data.download.has_kaggle_credentials()` now checks all four forms (legacy
json/env-pair, new token file/env).

### SARTAJ and Br35H downloaded and matched

With auth working: SARTAJ (3,264 images, `Training|Testing/{glioma,meningioma,
no_tumor,pituitary}_tumor`) and Br35H (3,861 images, `yes/no/pred` + a nested
Mask-RCNN train/val/test split) downloaded via `kaggle datasets download`, extracted,
loaded through the existing `load_flat_image_dir()` path (no code changes needed --
built generically enough in the earlier Figshare-only checkpoint).

**Full three-source match, replacing the earlier Figshare-only 25.4%:**

| Source | Upstream images | Matched manifest images |
|---|---|---|
| Figshare | 3,064 | 1,531 |
| SARTAJ | 3,264 | 1,262 |
| Br35H | 3,861 | 1,222 |
| **Total matched** | | **4,015 / 7,200 (55.8%)** |

Median match distance 0 (exact hash), max 4, mean 0.04 -- overwhelmingly exact matches,
not threshold-boundary noise.

**Clean validation:** `notumor` reaches **100% match coverage** (1,222 Br35H + 578
SARTAJ, 0 Figshare) -- exactly consistent with class support (Figshare has no
healthy-scan class). `glioma`/`meningioma`/`pituitary` match against Figshare+SARTAJ
only, never Br35H -- meaning Br35H's contribution to this merged dataset is specifically
its `no` (healthy) folder, not its `yes` (tumor) folder. This is a real, falsifiable
property of how the merge was constructed, not an artifact of matching quality.

**At the pseudo-patient level (the actual partitioning unit), coverage is lower: 46.5%
overall, 46.9% within the training split specifically (1,563/3,330).** This is expected,
not a regression -- a duplicate cluster only needs ONE member to match for the whole
cluster to count as "matched" at the image level, but counts as exactly one pseudo-patient
either way. The remaining ~53% genuinely lack recoverable provenance, most likely from
further reprocessing (resizing/recompression) between the upstream sources and this
merged variant that pushes them past the phash threshold this method can bridge.

### source_shift regime implemented

`_source_shift()` in `partition.py`: clients are grouped by source, group sizes
proportional to each source's available pseudo-patient count (each source with any data
guaranteed >=1 client), and within a group that source's data is split evenly. Units with
no recovered provenance are **excluded from this regime specifically** (documented in
`n_units_excluded_no_provenance`), not folded into an arbitrary bucket that would blur
the cross-site signal the regime exists to isolate.

Real result at K=20: **1,563 training pseudo-patients placed, 1,767 excluded** (46.9%
coverage), JS divergence 0.247, size Gini 0.072. The class-composition figure
(`paper/figures/partition_source_shift.pdf`) shows exactly the expected structure:
Br35H-sourced clients are 100% notumor, Figshare/SARTAJ-sourced clients are a
glioma/meningioma/pituitary mix with zero notumor -- a clean, visually verifiable
consequence of Br35H's class support, not a tuning artifact.

**Consequence for the paper:** report `source_shift` results explicitly as computed over
the ~47%-coverage subset with known provenance, not the full training set. State the
coverage figure alongside any `source_shift` result. This is honest and defensible --
A reviewer asking "how do you know the source label" now has a real, quantified answer
instead of a silent assumption.

Hit and fixed a real performance bug while wiring this up: `build_partition`'s retry loop
(when a random split doesn't hit `min_client_size`) was re-reading and re-grouping the
~1.6MB source_provenance.csv from disk on every attempt, causing a test to spin for 90+
seconds instead of failing fast. Fixed with `@lru_cache` on the loader.

95 tests total (was 87), all green: 9 new source_shift-specific tests using synthetic
pseudo-patient->source dicts injected via `build_partition`'s new `source_labels`
parameter, independent of the real CSV.

---

## 2026-09-14 — fix: deterministic mode crashed the first real Colab training run

### What happened

The author's first Colab run of `scripts/run_experiment.py` (SimpleCNN, groupnorm,
seed=0) crashed with a non-zero exit code on the very first training attempt. Root cause,
found by re-reading `utils/seed.py` rather than guessing: `torch.use_deterministic_algorithms(True)`
was called without `warn_only=True`. Enabling determinism this way does not raise at
call time -- it raises **later**, the first time training hits an op with no deterministic
kernel for the active backend. `SimpleCNN`'s `AdaptiveAvgPool2d` is a well-known example
of an op whose backward pass historically lacks a deterministic CUDA implementation.

This bug existed since Phase 0.3 but was never triggered locally: this machine has no
CUDA, and the MPS-avoidance fix from Phase 2.1 meant every local run used CPU, where the
same op does not hit this limitation. **Colab was the first time this code path actually
ran on CUDA**, and it broke immediately.

### Fix

`seed_everything()` now calls `torch.use_deterministic_algorithms(True, warn_only=True)`,
matching the plan's own stated Phase 0.3 design intent ("catch, log a warning, and set
deterministic: false... rather than crashing") that the original implementation only
half-delivered -- it caught failures at the *enable* call, not from ops encountered
during actual training, which is where the real risk always was. Falls back to a
bare call for a hypothetical torch old enough to lack the `warn_only` kwarg.

4 new tests (99 total): two verify the warn_only call contract via mocking (can't
reproduce a CUDA-only non-deterministic op on this CPU-only machine), one verifies
graceful fallback on an old torch, and one trains `SimpleCNN` end-to-end under
deterministic mode as a direct regression guard on the actual model/layer that broke.

## 2026-09-16 — Phase 2.1 acceptance met: first real centralized ceiling, from Colab

### The numbers (5 seeds each, SimpleCNN@112; 3 of 5 seeds, ResNet18@224 pretrained)

| model | norm | n | test macro-F1 | test accuracy | mean best_epoch | mean wall_clock |
|---|---|---|---|---|---|---|
| simple_cnn | groupnorm | 5 | 0.9203 ± 0.0053 | 0.9211 ± 0.0052 | 41.4 | 450.8s |
| simple_cnn | batchnorm | 5 | 0.9300 ± 0.0127 | 0.9314 ± 0.0144 | 41.6 | 441.6s |
| resnet18 (pretrained) | groupnorm | 3 | 0.9701 ± 0.0075 | 0.9705 ± 0.0074 | 21.3 | 650.9s |

**Sanity check against Phase 1.2's leakage fix, per the plan's own acceptance
criterion:** none of these are implausibly close to 1.00 -- if they were, the
plan says to go back to Step 1.2. A believable, moderate ceiling (0.91-0.98) is
itself evidence the de-duplication held. This is now the ceiling every FL method
is measured against.

**BatchNorm centralized > GroupNorm centralized (0.9300 vs 0.9203) -- expected,
not a contradiction.** GroupNorm was chosen for the *federated* experiments
specifically because BatchNorm's running statistics aggregate badly across
non-IID clients (plan §2.1); nothing about that pathology applies to a single
machine training on the pooled data, where BatchNorm's per-batch statistics are
usually a mild edge over GroupNorm. Record both numbers as: BatchNorm centralized
is a normal, expected centralized-only data point, not the A9 ablation (already
corrected in the notebook, see the 2026-09-16 phase-2 commit) -- the real A9
comparison needs actual federated training (Phase 4/7).

**ResNet18 has only 3 of 5 seeds** (0, 1, 2) -- the session apparently stopped
before seeds 3-4. Not a blocker: it's the secondary/optional table, the notebook's
per-run checkpointing means seeds 3-4 can be added in any future session without
recomputing 0-2, and 3 seeds already show a tight spread (std 0.0075) suggestive
that 2 more won't move the picture much. Queue for later, don't block Phase 3 on it.

### Correcting my own earlier speculation: determinism mode is not the dominant cost here

When diagnosing "is this run going to hit my T4 quota," I named
`torch.use_deterministic_algorithms(True)` as the leading suspect, based on this
repo's own recorded *MPS* number (152s/epoch deterministic vs 26s/epoch not --
a 5.8x gap). The real CUDA numbers now available say that hypothesis does not
transfer to this hardware: 450s wall-clock over ~41 epochs for SimpleCNN@112 is
~11s/epoch, not dramatically slow for a T4 running deterministic-mode PyTorch.
**The actual driver of total wall-clock was epoch *count*, not epoch *speed*:**
several runs went 44-60 epochs (early_stop_patience=8) before stopping, and
10 SimpleCNN runs + 3 ResNet18 runs together cost ~1.8 GPU-hours, purely additive
across many fast-enough epochs. If quota is tight in a future sweep, the
higher-leverage lever is a smaller `max_epochs`/`early_stop_patience`, not
`training.deterministic=false` -- though that override still exists
(`scripts/run_experiment.py`) and costs nothing to leave in place.

### Provenance note: this run predates every Phase 2/3 fix pushed after it

`provenance.git_sha = 725002b` on every file here -- the Colab session cloned
before `2a615f8`/`bc5d53c`/`de4653a` were committed (and none were pushed to
origin until after this run started; see docs/OPEN_QUESTIONS.md if that gate
ever needs re-checking). Consequences, none of them blocking:
- `provenance.dirty: true` on every file -- exactly the DATASET_CARD.md
  re-verification bug fixed in `2a615f8`; harmless here since nothing was
  actually hand-edited, but the next Colab pull won't have this noise.
- No `epoch_time_s` per-round field (added in `bc5d53c`) -- wall_clock_s at the
  `final` level was enough to reconstruct the epoch-time story above, but
  future runs will have the finer-grained field.
- `flwr`/`flwr-datasets` are `null` in provenance -- expected, this notebook's
  centralized path never imports them (its own markdown says so).

Real environment, for the record: Linux, Tesla T4, CUDA 12.8, torch 2.11.0+cu128,
torchvision 0.26.0+cu128, numpy 2.1.3, Python 3.13.15 -- all newer than this
repo's Intel-macOS pins (torch==2.2.2, numpy<2), exactly as anticipated when the
compute-environment split was decided (2026-09-14 entry above): Colab's torch/
numpy, not the local pins, is what actually produced these numbers.

---

## 2026-09-17 — Phase 4 close-out: the dispersion term's scale was silently disabling the colony

### What prompted the measurement

An audit of what remained in Phase 4 flagged `DataFreeFitness.evaluate` as mixing terms
of different units: `alignment` is a cosine in [-1, 1] and the entropy penalty is in
[0, log K], but `dispersion` (`gram.py::weighted_dispersion`) is a raw sum of squared
distances, in units of ||Delta||^2. The initial hypothesis — that dispersion would
always dominate and hold F negative — was **wrong at the default config**, and measuring
it first rather than patching on the hypothesis is what found the real shape of the
problem.

### Measured: real SimpleCNN deltas, K=20, 112px, d=390,404

At the declared defaults (`local-lr=0.01`, `local-epochs=1`) the three terms are in fact
well balanced, because a <1M-parameter CNN at lr=0.01 produces per-client delta norms
below 1 (so the *square* is sub-unit, not large):

| round | mean \|\|delta_k\|\| | align@FedAvg | disp@FedAvg | F@FedAvg | tau spread after colony |
|---|---|---|---|---|---|
| 1 | 0.793 | +0.969 | 0.619 | +0.345 | 3.2e-01 |
| 2 | 0.442 | +0.958 | 0.210 | +0.744 | 5.8e-01 |
| 3 | 0.471 | +0.982 | 0.192 | +0.786 | 6.2e-01 |

### The actual failure: it dies as soon as you leave that one config

Sweeping the `(lr, local_epochs)` grid Phases 6–7 will vary, K=8, same model/data:

| lr | epochs | disp@FedAvg | F@FedAvg | frac of candidates with F>0 | tau spread |
|---|---|---|---|---|---|
| 0.01 | 1 | 0.40 | +0.56 | 0.80 | 4.2e-01 |
| 0.01 | 2 | 1.38 | -0.43 | 0.16 | 7.3e-03 |
| 0.01 | 5 | 2.63 | -1.65 | 0.07 | **0** |
| 0.05 | 1 | 13.4 | -12.4 | 0.00 | **0** |
| 0.05 | 2 | 38.5 | -37.5 | 0.00 | **0** |
| 0.05 | 5 | 85.7 | -84.7 | 0.00 | **0** |
| 0.1 | 1 | 47.4 | -46.4 | 0.00 | **0** |
| 0.1 | 2 | 149.7 | -148.8 | 0.00 | **0** |
| 0.1 | 5 | 329.3 | -328.4 | 0.00 | **0** |

A 660x swing in `disp`. `colony.py` deposits `rho * Q * max(F, 0)`, so once F is negative
for every candidate **every deposit is exactly zero**: tau evaporates uniformly across
all levels, `tau^a * eta^b` reduces to `eta^b`, and FedACO degenerates into deterministic
heuristic-greedy weighting — no colony search, no cross-round stigmergy, i.e. none of
the mechanism the paper's contribution rests on. The declared default is one local epoch
away from this, and E=5 is a standard FedAvg setting.

**The safety fallback cannot catch it.** It compares `F_best` against `F_fedavg` under the
same degenerate fitness, and `F_best > F_fedavg` held in **all 9** configs above,
including the fully-dead ones. A broken run reports `fallback_used=0`, `best_fitness` a
large negative number, and otherwise looks healthy.

### Fix, and the two candidates that were tested

Divide dispersion by `trace(G)/K` (`GramPrecompute.mean_sq_norm`). It is a per-round
constant, independent of alpha, so it rescales the term uniformly and **cannot change the
ranking dispersion induces among candidates** — it only fixes the weight against
alignment. Over the same grid:

| variant | disp@FedAvg range | tau spread |
|---|---|---|
| unnormalized | 0.40 – 266.7 (660x) | zero in 6/9 configs |
| **/ trace(G)/K** | **0.763 – 0.933 (1.22x)** | **nonzero in 9/9** |

A baseline-centred deposit (`max(F - F_fedavg, 0)`) was also tested on top and **rejected**:
tau spread went up in some configs and down in others with no systematic benefit, so it
would have been a knob the evidence does not support.

**Consequence to be aware of when reading FedACO numbers:** normalization puts dispersion
(~0.86) on the same footing as alignment (~0.96), so with gamma_2 = 1.0 the two are now
roughly co-equal. This changes behavior at the default config (F>0 fraction 0.80 -> 0.50,
colony best 0.67 -> 0.34). gamma_2 was **left at 1.0** — the value `paper/ALGORITHM.md`
specifies — rather than retuned, because silently changing a stated method hyperparameter
is not a bug fix. It is now sweepable as `aco-gamma-dispersion` if Phase 7 wants it.

Guarded by `test_fitness_is_scale_invariant_under_normalization` (F is exactly invariant
to scaling all deltas by a constant — verified to fail without the fix) and
`test_pheromone_still_carries_signal_when_updates_are_large` (tau spread 0.0 before,
5.3e-02 after, at a 100x delta scale). `delta_mean_sq_norm` is now logged per round as
the diagnostic, to be read alongside `pheromone_entropy`.

### Measured ACO overhead, at the real scale

K=20, SimpleCNN@112, d=390,404, 360 images/client, CPU:

| budget | aco_time | of which Gram precompute | vs K-sequential round (56.8s) | vs fully-parallel round (2.8s) |
|---|---|---|---|---|
| round 0 (30 ants x 10 iters) | 144.9 ms | 60.3 ms | 0.255% | 5.10% |
| final round (10 ants x 4 iters) | 110.8 ms | 58.9 ms | 0.195% | 3.90% |

`test_overhead` previously asserted 15% "to absorb timing noise", a margin that had never
been measured; the actual ratio in that test is ~1.8%, so it has been tightened to the
plan's own 5% bar (§4.8).

### FedACO verified end-to-end in a live `flwr run` (plumbing only)

Phase 4's last open item. `flwr run . --run-config "strategy-name='fedaco' num-clients=2
num-rounds=3 regime='dirichlet' alpha=0.3 ..."` on a 4-core CPU box: **exit 0,
`status: "completed"`, 3/3 rounds, 213.2 s wall-clock**, result JSON written through the
normal `utils/results.py` path with every FedACO metric present. This is the first time
any Phase 4 or Phase 5 code has executed inside the real Flower runtime rather than
against hand-built Messages.

**⚠️ The accuracy numbers from this run are meaningless and must never be reported.** The
raw JPEGs are not available in that environment (no Kaggle credentials), so the run used
a *synthetic pixel cache* generated against the real Phase-1 manifest: real
manifest/partitioning/splits, fabricated images. It validates plumbing, not accuracy. The
run wrote to a scratch directory, not `results/`.

Per-round colony behavior (K=2, so read as a smoke trace, not evidence):

| round | alpha | best_F | fedavg_F | fallback | pheromone_entropy | delta_mean_sq_norm | aco_ms |
|---|---|---|---|---|---|---|---|
| 1 | [0.990, 0.010] | 0.7384 | 0.5950 | 0 | 2.3939 | 1.8827 | 65.1 |
| 2 | [0.004, 0.996] | 0.8960 | 0.7218 | 0 | 2.3758 | 0.4956 | 32.7 |
| 3 | [0.990, 0.010] | 0.8984 | 0.5905 | 0 | 2.3668 | 0.3803 | 24.3 |

The colony beat the FedAvg point every round (`best_F > fedavg_F`, no fallback), and F
stayed comfortably positive, so the deposit floor never engaged -- the normalization is
doing its job on live data.

**Two things to watch, both flagged rather than acted on:**

1. `pheromone_entropy` is 2.3939 / 2.3758 / 2.3668 against a maximum of
   log(11) = 2.3979 -- i.e. 0.2%, 0.9%, 1.3% below uniform. It is declining, so pheromone
   *is* accumulating structure, but it is contributing very little to selection so far.
   This is the exact diagnostic Residual 1 in docs/OPEN_QUESTIONS.md says to watch. Three
   rounds at K=2 is far too small to conclude anything; re-check on the first real
   multi-round run at K=20.
2. `alpha` saturates at the extremes of the level grid and flips between rounds
   (0.99/0.01 -> 0.004/0.996 -> 0.99/0.01). With gamma_3 = 0.1 the entropy penalty is
   weak relative to alignment. At K=2 this is close to degenerate by construction, but if
   the same saturation shows up at K=20 the entropy weight needs revisiting.
