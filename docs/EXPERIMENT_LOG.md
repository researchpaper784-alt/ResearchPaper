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

---

## 2026-09-17 — Phase 5: the honest hyperparameter search and the IID acceptance gate

Phase 5's two remaining items, both of which had been blocked on "needs a live FL run"
since Phase 3. Building them surfaced three separate reasons the first one could not have
been written correctly before today; all are recorded in docs/OPEN_QUESTIONS.md.

### What was delivered

`scripts/run_hparam_search.py` — per-strategy grids, a shared `--budget` (default 8), one
`flwr run` per trial, resumable. Selection reads `best_val_macro_f1` and nothing else.
`scripts/check_iid_band.py` — the IID acceptance gate, exiting nonzero on FAIL.
`make hparam-search`, `make hparam-search-plan`, `make iid-band`.

### The prerequisite nobody had noticed

Result files carried **no validation metric at all** — `build_evaluate_fn` evaluated the
test split only. An "honest search on the val split" written against those files would
have had to rank configurations by `final_test_macro_f1`, i.e. tune every baseline
against the number the paper reports. `global_val_loader` had existed since Phase 5's
FedLAW work, added specifically because using test for weight selection "would have been
a real methodology bug", but nothing in the result path used it.

Every round now logs `val_loss`/`val_accuracy`/`val_macro_f1`/`val_auc`, and `final`
carries `best_val_macro_f1` (max over rounds, so a config that peaks then drifts is not
punished) and `best_val_round`. Cost: one extra forward pass over 735 val images per
round, ~52% of the existing test pass.

### The search would have failed on every single trial

Every baseline hyperparameter — `fedprox-mu`, `fedopt-*`, `num-malicious-nodes`,
`trim-beta`, `lossweight-temperature`, `scaffold-server-lr`, `fedlaw-steps`,
`fedlaw-lr` — was undeclared in `[tool.flwr.app.config]`, under a comment asserting they
"only need overriding via --run-config when actually sweeping them". `flwr run` rejects
undeclared keys outright, so not one of them could be swept. Same root cause as the
partition keys found during the Phase 4 close-out; that fix covered the keys being
touched at the time and left these.

A trap inside the fix, worth recording because it would have been invisible: `fedopt-eta`
was read by both FedAdam and FedYogi with *different* published defaults (0.1 vs 0.01,
from the FedOpt paper). One declared key would have given both the same literal value and
silently replaced FedYogi's default. Split into per-strategy keys.

### `flwr run` is asynchronous, and the failure mode compounds

A bare `flwr run` submits the run and returns exit 0 immediately. The search's first
end-to-end attempt checked for each result file microseconds after launching the trial,
found nothing, and reported every trial failed. Worse, those launched-and-forgotten runs
keep executing: two aborted 2-trial searches left four orphaned simulations across 10 Ray
sessions at load average 11.8 on a 4-core box, which starved the next (correctly
blocking) trial so badly that one round had not finished in twelve minutes against ~70 s
uncontended. Clearing it required killing `flower-superlink` to reap the zombies.

The harness now passes `--stream` unconditionally and judges a trial by "did a result
file appear", never by the subprocess's exit code — which is uninformative in both
directions, since `flwr run` also exits 0 when the simulation itself dies.

### Two self-inflicted bugs caught by the tests written alongside

- `find_existing` matched trials on the swept keys alone, so a search at seed 1 would
  have instantly "resumed" seed 0's results and produced a second seed that was a
  duplicate of the first — quietly destroying the seed variance every error bar depends
  on. Now matches the full configuration, ignoring path-only differences.
- `diagnose` scanned line-by-line against a flat marker set including bare "error", so it
  reported Ray's `FutureWarning: ... turn off this error message` as the reason a trial
  failed. Now priority-ordered, warnings excluded.

### Acceptance gate, verified both ways

`check_iid_band.py` on hand-built result sets: a clustered IID picture (5 strategies,
3 seeds, spread 0.0116 against seed noise 0.0057) returns **PASS**; a picture with FedACO
0.086 clear of the field returns **FAIL** with exit 1 and points at the three things that
actually cause it (partition not really IID, unmatched search budgets, selection on
test). A spread that exceeds the band but sits inside 2x seed noise returns
**INCONCLUSIVE** rather than either verdict, because with few seeds those are genuinely
not distinguishable.

⚠️ The 0.05 band is **this script's default, not a figure from the implementation plan**,
which specifies the check but no threshold in any material available in this repo.
Recorded rather than baked in silently, per CLAUDE.md.

### Search verified end-to-end against the live Flower runtime

`run_hparam_search.py --strategy fedprox --budget 2 --num-clients 2 --num-rounds 1`, two
real `flwr run` subprocesses on a CPU box:

| trial | `fedprox-mu` | `best_val_macro_f1` | `final_test_macro_f1` |
|---|---|---|---|
| 1 | 0.001 | 0.1132 | 0.1119 |
| 2 | 0.01 | **0.1169** | — |

Selected `fedprox-mu=0.01` on val, `trials_scored: 2`, `budget_shortfall: 0`, exit 0.
What this establishes, none of which held this morning: a baseline hyperparameter
actually reaches the strategy through `--run-config`; the result file carries a val
metric; val and test are genuinely different numbers (0.1132 vs 0.1119 on the same run);
and the harness blocks until each run finishes instead of racing it.

**⚠️ The scores are meaningless as science.** This ran against the synthetic pixel cache
(real Phase-1 manifest, fabricated images) because the raw JPEGs need Kaggle credentials
this environment does not have. Only the mechanism is verified. The real search is
GPU-hours in the author's training environment and has not been run.

---

## 2026-09-17 — the FedACO health check, and a gap it exposed before it could run

Built `scripts/check_fedaco_health.py` to answer the three questions that decide whether
Phase 6 is worth building as designed: is the colony searching, has the deposit floor
engaged, and does FedACO beat FedAvg without the fallback carrying it.

### The strategy's diagnostics were never persisted

Writing the checker immediately exposed that it had nothing to read. `rounds_log` is
built entirely by `build_evaluate_fn`, which only sees evaluation metrics, so everything
`aggregate_train` returned -- `pheromone_entropy`, `best_fitness`, `fallback_used`,
`delta_mean_sq_norm`, `alpha` -- went to Flower's console and nowhere else.

The Phase 4 close-out had added `delta_mean_sq_norm` *specifically* so a run could be
checked for the degenerate regime, and it was not in the result file. `ALGORITHM.md`
lists these as the algorithm's outputs. Phase 6's analysis would have had accuracy and no
way to say anything about the method's behaviour. Fixed by `merge_train_metrics`, folding
`Strategy.start()`'s `Result.train_metrics_clientapp` into the round log under a `train_`
prefix, respecting the resume offset. Verified on a live run: 14 `train_*` fields now
persist per round.

### First measured verdict (synthetic pixels — a lead, not a finding)

4 rounds, K=2, dirichlet α=0.3, reduced colony budget:

| check | verdict |
|---|---|
| Is the colony searching? | **INERT** — tau 0.92% below uniform (2.3758 vs ceiling 2.3979) |
| Has the deposit floor engaged? | **CLEAR** — best_fitness 0.78–0.92, every round deposited |
| Beats FedAvg? | no baseline run yet; fallback fired in 0% of rounds |

**That combination is informative.** tau is flat *despite* positive fitness and a deposit
every round, which rules out the `max(F, 0)` floor — the cause the open residual in
docs/OPEN_QUESTIONS.md anticipated — and points at deposit *magnitude* instead:
`rho * Q * F ≈ 0.08` onto one of 11 levels, against `tau0 = 1.0`, over 5–6 iterations,
with `end_round` pulling 10% back toward tau0 each round. If it holds, the lever is
`aco-q-deposit` / `aco-rho` / the iteration budget, not the fitness weights.

> **Superseded, 2026-09-17 (later).** The INERT verdict was wrong — not the reasoning
> above about deposit magnitude, which was right, but the verdict it was attached to. At
> this run's own budget no colony could have scored above ~3.1%, so the 2% threshold
> demanded 64% of a theoretical maximum. τ was in fact moving monotonically away from
> uniform every round. See "the INERT verdict was a statement about the run length" below.

Caveats that matter: K=2, synthetic pixels, and a deliberately reduced budget
(20 ants / 6 iterations vs the 30/10 default). All three push toward a flat tau on their
own. This is a lead for the real run to settle, not a result.

### Measured timing, for the compute table

48.1 s per round at K=2 / 1 local epoch on CPU (192 s over 4 rounds) → **1.34 hours for a
100-round run at that shape**. Not yet the K=20 GPU number the compute-budget table needs,
but the first real measurement of any kind against a table that says in bold "these are
estimates, not measurements" and "do not cite these numbers until measured".

### How to run the real thing

`make validate-fedaco K=4` (or the last three cells of `notebooks/colab_fl_smoke.ipynb`),
then `make health`. Client resources have to be pinned first — the target does it —
because the Simulation Runtime assigns 2 CPUs per ClientApp by default and oversubscribing
stalls silently rather than queueing.

> **Superseded, 2026-09-18 — drop the `K=4`.** At K=4 the fitness optimum is degenerate:
> dispersion is exactly zero at a single-client vertex and `gamma_entropy * log K` is too
> small to pay for it, so a correctly working colony returns a one-client answer and every
> other signal reads as success. Run the default K=10 or higher. See "the fitness pays
> nothing for throwing clients away" below.

> **Also superseded** — the CPU-oversubscription claim in the last sentence was retracted
> on 2026-09-17; the stall was `num_supernodes` defaulting to 2, not oversubscription.

---

## 2026-09-17 — Phase 6: the main sweep runner

`scripts/run_sweep.py` + `configs/experiment/main.yaml`. Expands (strategy x regime x
seed) into cells, runs each as one `flwr run`, and indexes completed runs in
`results/manifest.jsonl` — the resume index CLAUDE.md specifies — so an interrupted sweep
continues instead of recomputing.

The full matrix is **12 strategies x 6 regimes x 5 seeds = 360 cells x 100 rounds**.

### Four design choices that are failure-driven, not stylistic

Each corresponds to something that actually went wrong earlier in the session
(docs/OPEN_QUESTIONS.md); removing any of them reintroduces a silent failure.

1. **`--stream` is passed unconditionally.** A bare `flwr run` returns exit 0 immediately
   while the simulation is still starting. Without this, the sweep would launch all 360
   runs near-simultaneously, conclude every one failed, and leave the machine under
   orphaned simulations that keep executing after it "finished".
2. **Exit codes are ignored entirely.** `flwr run` also exits 0 when the simulation dies,
   so it lies in both directions. A cell counts as done only when a result file with
   `status: "completed"` matching its full config exists.
3. **Preflight refuses to start.** K=20 requests 40 CPUs at the Simulation Runtime's
   default 2/ClientApp, and oversubscription stalls at round 0 with idle actors and no
   error rather than queueing. The runner checks before the first cell and refuses,
   naming the `flwr federation simulation-config --client-resources-num-cpus 1` fix.
4. **Result matching includes seed and regime**, not just strategy. Matching on too
   little is how a sweep silently reuses one seed's result for another and destroys the
   variance every error bar depends on — the same bug that hit the Phase 5 search, caught
   there and guarded here before it could happen at scale.

Seed is the innermost loop, so an interruption leaves whole (strategy, regime) groups
finished rather than one seed of everything: a partial sweep is still analysable for what
it covered.

### Cost projection, and what it is worth

`make main-plan` prints a projection from a measured per-round cost rather than an
assumption. At today's only datapoint (48.1 s/round, K=2, 1 local epoch, CPU) the full
matrix projects **481 hours**. That number is nearly meaningless as a quote — it is a CPU
measurement at K=2 for a sweep at K=20 — and the runner says so in its own output rather
than presenting it as authoritative. It becomes real after a run at the target shape;
`--seconds-per-round` takes the better number once one exists.

This is still the first cost figure in the project derived from a measurement at all. The
2026-09-14 compute table above remains estimates, and still says in bold not to cite them.

---

## 2026-09-17 — Phase 7 ablation config and Phase 9 table builder

### `variants`: a fourth sweep axis

An ablation is one strategy with one thing changed, not a different sweep shape, so
`run_sweep.py` gained an optional `variants` axis rather than Phase 7 getting its own
runner. A cell is now (strategy x regime x variant x seed). The variant is omitted from
the cell label when it is the implicit `default`, so any sweep config without a
`variants:` key produces exactly the labels it did before — an existing manifest keeps
matching and nothing already computed is re-run. Variant overrides beat regime overrides
on a key collision, since a regime that also set the ablated key would otherwise silently
cancel the ablation.

### `configs/experiment/ablation_all.yaml` — 150 cells

**Only the ablations this repository actually names.** The implementation plan is not in
the repo, so A1 (pheromone persistence), A3 (fitness mode) and A9 (norm) are the only
numbered ones recoverable; A2 and A4–A8 are deliberately absent rather than invented,
because labelling a guessed ablation with the plan's number would be worse than the gap.
Alongside them are the knobs docs/OPEN_QUESTIONS.md explicitly defers to Phase 7: the
global shrinkage `s` sweep, the safety fallback off, and gamma_2 — the dispersion weight
left at its specified 1.0 during the Phase 4 close-out rather than retuned.

`default` is included as a variant so the ablations have a control inside the same sweep;
differencing against numbers from a different sweep would fold in unrelated variation.

Two of these are worth their compute regardless of how FedACO scores. `a1_persistence_none`
is the test `aco/pheromone.py` names in its own docstring — remove persistence and the
swarm framing "should show a measurable drop, or the framing is decorative".
`no_safety_fallback` separates "the colony helped" from "the fallback protected it",
which the main table structurally cannot: with the fallback on, FedACO silently *is*
FedAvg whenever the colony loses.

### `scripts/make_tables.py` — Phase 9 aggregation

Aggregates over seeds per (strategy, regime, variant) into Markdown + CSV. The arithmetic
is mean and std; the work is in not overstating a number:

- **Incomplete cells are marked, never dropped.** A cell with 2 of 5 seeds shows n=2 and a
  warning. Silently averaging whatever finished is how one strategy's mean over 5 seeds
  ends up beside another's over 2, rendered identically.
- **Deltas are paired by seed** where both arms ran the same ones, and the table records
  whether pairing was possible rather than quietly falling back to mean-to-mean.
- **FedACO rows carry `fallback` and `τ entropy` next to the score**, with a note on how
  to read them. A macro-F1 column alone cannot show that the colony never searched, or
  that the fallback turned FedACO into FedAvg for most of the run.
- **Absent metrics render as `—`, not 0.** A baseline has no pheromone; printing 0.000
  would read as "maximally concentrated", the opposite of "not applicable".
- Exits nonzero when any cell is incomplete, so `make tables` cannot silently produce a
  quotable table from a half-finished sweep.

Verified against this session's real result files: it correctly flagged all three cells as
n=1 of 5, produced a paired FedACO-vs-FedAvg delta of −0.0091, and surfaced FedACO's
fallback rate (0%) and τ entropy (2.373, against a 2.398 ceiling) beside the score.

`make main-plan` / `make ablations-plan` / `make tables` / `make tables-ablation`.
`make figures` still points at a `scripts/make_figures.py` that does not exist.

---

## 2026-09-17 — Phase 9 figures

`scripts/make_figures.py`, the last target in the Makefile that pointed at a file which
did not exist. Four figures, each picked by what its data has to say: convergence (line,
faceted by regime), final comparison (dot plot with error bars), ablation deltas
(diverging bars), colony health (line against a threshold).

### Emphasis instead of twelve hues

The sweep has 12 strategies. A categorical palette carries about 8 before adjacent hues
stop being distinguishable, and generating a 9th is a documented anti-pattern rather than
a stylistic preference. The paper's claim is FedACO against FedAvg, so those two take the
first two categorical slots and the other ten render as one muted gray field — which is
also a more honest picture of the argument than twelve equal-weight lines.

The two hues (`#2a78d6`, `#eb6834`) were run through the palette validator rather than
eyeballed: all-pairs, light surface, they clear the lightness band, chroma floor, CVD
separation (ΔE 24.7 worst, against a floor of 8), normal-vision separation (33.6, floor
15) and the 3:1 contrast floor, with no warnings — so no relief rule is owed. The
diverging pair for the ablation arms (`#2a78d6` / `#e34948`) passes the same checks.

### Three layout defects found by rendering and looking

Every one of these passed the palette validator, which checks color and not geometry.
They were only visible in the rendered PNG:

1. **Convergence endpoint labels collided and overflowed.** FedACO and FedAvg converge to
   within a hair of each other in exactly the regimes that matter, so the two value
   labels printed on top of one another; at the right edge they ran off the panel. Now
   nudged apart when closer than 4.5% of the y-range, with the x-limit extended 10% to
   give them room.
2. **Ablation x-tick labels ran together** into one unreadable string — five 6-character
   labels in a narrow panel at default tick density. Capped at 4 ticks.
3. **Colony health stacked three regime labels on the same point.** The first draft
   overlaid regimes in one panel and separated them by alpha, which is a weak channel
   that collapses entirely when two curves coincide. Rebuilt as small multiples, one
   panel per regime, one hue.

### The figure that exists to fail

`colony_health.png` plots pheromone entropy against its log(L) ceiling, annotated with
how far below uniform each regime sits. It is the only figure here that can show the
method's mechanism did not run — convergence, comparison and ablation figures would all
look perfectly healthy on a run whose colony never searched. Its y-range is anchored to
the ceiling rather than auto-scaled, so a curve sitting a hair under uniform is not
stretched into looking like a dramatic descent.

Rendered and inspected against 240 synthetic result files covering 12 strategies, 3
regimes, 5 variants and 5 seeds. **Those files are fabricated to exercise the plotting
code and are not results**; they were written to a scratch directory, never to `results/`.

`make figures` / `make figures-ablation`. Every Makefile target now points at a file that
exists.

---

## 2026-09-17 — Phase 8: adversarial clients

`src/fedswarm/fl/attacks.py` + `configs/experiment/robustness.yaml` (225 cells).

### The gap this closes

**There was no way to make a client behave maliciously anywhere in the codebase.** Three
of this project's baselines — Krum, FedTrimmedAvg and FedMedian — are robust-aggregation
rules whose entire justification is surviving Byzantine clients, and nothing could put one
in front of them. Every number they would have earned in the main table comes from an
entirely honest federation, where they have nothing to do but lose a little accuracy to
their own trimming.

`num-malicious-nodes`, which reads like the missing piece, is *Krum's own hyperparameter*
— how many Byzantine clients Krum should assume are present. It makes nobody misbehave.
The two are easy to confuse and were.

FedACO has a stake as well: its desirability heuristic scores clients on alignment with
the robust consensus and on update magnitude against the median, both explicitly anomaly
signals, and `test_planted_bad_client` already asserts the unit-scale version of the
claim. This is what lets it be made about a real run.

### Two kinds of attack, kept apart

`sign_flip`, `scaled` and `gaussian` poison the **update** after honest training — the
threat model robust aggregation addresses. `label_flip` poisons the **data** before it, so
the client sends a perfectly well-formed update computed from a lie, and aggregation-level
defences have far less grip. A defence that handles one says nothing about the other, so a
robustness claim has to name which it was tested against.

Who is compromised is deterministic (the lowest `ceil(f·K)` partition ids), not resampled
per round: a rotating attacker set measures something quite different from a fixed
compromised subset, and a seeded-random choice would make the result depend on a seed the
analysis never sees.

### A decorator silently bound to the wrong function

The first live attacked run exposed a bug worth recording for its failure mode rather than
its cause. The new `_poisoned_loader` helper was inserted between `@client_app.train()`
and `train_handler`, so **Flower registered the helper as the train handler**. It was
called with a `Message`, every client reply failed with `'Message' object is not
iterable`, and the global model never moved off its random initialization.

`flwr run` still **exited 0 and wrote a complete-looking result file**. The only visible
symptom was a test macro-F1 frozen at 0.054 — on a short run, indistinguishable from a
hard problem. This is the third distinct time this session that `flwr run`'s exit code has
been actively misleading. Fixed, and guarded by
`test_client_app_decorators_are_bound_to_the_real_handlers`, since nothing in the suite
previously tested *which* function a decorator captured.

### Verified live

`attack='sign_flip' attack-fraction=0.5 attack-scale=2.0`, K=2, 3 rounds: clients trained,
no errors, and the result file records the attack config alongside a per-round
`train_is_malicious` confirming the poisoned path executed.

⚠️ That value came back **0.293, not 0.5** — Flower aggregates client metrics weighted by
`num-examples`, so it is the fraction of malicious *examples*, not *clients*; the
compromised client held 29% of the data. Recorded in docs/OPEN_QUESTIONS.md. Any Phase 8
analysis quoting a malicious client fraction must take it from `attack-fraction`.

`make robustness-plan` / `make robustness` / `make tables-robustness`.

---

## 2026-09-17 — code review of Phases 4–9, and one diagnosis retracted

A deliberate adversarial pass over the whole session's diff, before any sweep spends
compute. Fourteen findings; twelve fixed, two recorded. Several were severe enough that
the sweeps would have produced confident, wrong numbers rather than failing.

### The one that would have invalidated everything

**`num_supernodes` defaults to 2 and nothing in this repository ever set it.**
`num-clients` is *this project's* run-config key — it tells the partitioner how many ways
to split the data — and has no connection to how many ClientApps the Simulation Runtime
creates. Left unset, `make main` would have run 360 cells that record `num-clients: 20`
while only partition ids 0 and 1 ever received a ClientApp: 18/20 of the data never
trained on, every result file mislabelled. In `robustness.yaml` it compounds —
`malicious_ids(20, 0.1) == {0, 1}` — so a cell labelled "10% malicious" would have had the
*entire participating federation* compromised.

**This also retracts an earlier diagnosis in this log and in docs/OPEN_QUESTIONS.md.** The
K=4 stall recorded as CPU oversubscription was not that. Only 2 supernodes existed while
`min-train-nodes=4` waited for 4 that were never created. With `--num-supernodes 4` the
same run completes on the same 4-core box. The wrong cause had been propagated into the
sweep runner's preflight, the Makefile, the README and the Colab notebook; all corrected.
`run_sweep.py` now configures the federation itself and refuses to start if that fails.

### Other findings that would have corrupted results silently

- **The hyperparameter search would have reported FedAvg with FedACO's score.**
  `find_existing` matched on base config and overrides but not on `strategy-name`, and
  `--strategy all` iterates `sorted(GRIDS)` so fedaco runs first. Every strategy with an
  empty grid — fedavg, median, fednova — would have "resumed" fedaco's result and never
  run. The reference baseline every claim is relative to would have been a copy of the
  method under test.
- **Booleans rendered as `True`, which is not valid TOML.** `flwr run` parses the
  assembled `--run-config` with tomli and rejects the whole string, so all 15 cells of the
  `no_safety_fallback` ablation — the one that separates "the colony helped" from "the
  fallback protected it" — would have died with a bare `[code: 15]`.
- **Every robustness variant collapsed to `default` in the tables.** `_variant_of` knew
  only the FedACO ablation knobs, so 9 variants × 5 seeds per strategy keyed into one
  per-seed dict and the published table would have averaged a clean control together with
  a 30%-sign-flip run and reported it as n=5.
- **A variantless control matched any sibling variant's result**, so the `default`/`clean`
  cell would have been marked complete, never run, and every ablation delta differenced
  against another ablation.
- **Update poisoning crashed on GPU** — CPU `full_state` minus CUDA `model.state_dict()` —
  so 5 of the 9 robustness variants would have failed in the project's stated training
  environment while the non-attack path worked.
- **FedACO dropped every client-reported metric**, because its `aggregate_train` returns a
  hand-built MetricRecord. For the strategy under test in the robustness sweep there was
  no way to confirm from the result file that the attack fired.
- **Krum ran every attacked cell at `f=0`**, the setting under which its Byzantine
  guarantee does not hold. "Krum broke at 30%" would have been a statement about a
  misconfigured Krum.
- **A resumed run re-seeded the colony from round 1** and restarted its ant-budget decay,
  because `Strategy.start()` renumbers rounds — so a resumed cell was not the same
  experiment as an uninterrupted one.
- **`label_flip` froze augmentation** for poisoned clients only, confounding the attack's
  effect with a reduced-augmentation effect.
- Figures plotted rounds 0-based against 1-based round numbers everywhere else; the README
  status text contradicted its own command table; `find_result` re-parsed every result
  file per cell (~130,000 reads before a resumed 360-cell sweep starts, now cached).

### Fixing one introduced another, caught by running it

The variant discriminator first rejected candidates that merely *carried* another
variant's key. Every variant key is declared in pyproject — it must be, or `flwr run`
rejects it — so it appears in every resolved config with its default, and the control was
handed its own result and reported "no result file". Now discriminates by *value*.
Verified on a live K=4 sweep: both variants complete, resume recognises both, and the
table shows them as distinct rows.

### Recorded, not fixed

`format_run_config`, `_same` and `diagnose` are duplicated between `run_sweep.py` and
`run_hparam_search.py` — which is how the boolean-TOML bug came to exist in two places.
Both copies are fixed; the duplication itself wants extracting into a shared module.
**Done in the entry below**, which also found a third copy of the bug class.

## 2026-09-17 — the duplicated runner helpers, and the drift they were already hiding

The review's one recorded-not-fixed finding, closed: `format_run_config`, `_same` and
`diagnose` now live once, in `src/fedswarm/utils/runner.py`, alongside the `flwr run`
invocation itself and the result-file reader.

Extracting them turned up what the finding predicted but had not looked for — **the two
copies of `diagnose` had already diverged, and the sweep's was the broken one.** It
filtered log noise with `"arn" not in line.lower()`, a stand-in for "not a warning" that
also discards every line containing *learn*:

```python
# scripts/run_sweep.py, before
lines = [line.strip() for line in output.splitlines() if "arn" not in line.lower()]
```

So `ValueError: learning rate must be positive` — the likeliest way a tuning cell dies,
and the reason `diagnose` exists at all — was thrown away, and the sweep printed "no
diagnostic line found in output" for exactly the failures it most needed to explain. The
search's copy had been fixed weeks of commits earlier to filter `Warning`/`warn`
explicitly. Nobody had ever seen the two side by side. The shared copy is the search's.

Four helpers were shared, not three, because the fourth is where the worst failure lives:
`run_flwr` now owns the `flwr run` invocation, so `--stream` and `FEDSWARM_REPO_ROOT` are
passed from one place. Dropping either is silent — without `--stream` every run is
launched asynchronously and reported as failed while the orphans keep computing; without
the env var every run fails to find the image cache from *inside* the simulation, where
the outer process still exits 0. Neither had a test. Both do now
(`tests/test_runner.py`), along with an identity check that the two scripts hold the
*same* function objects, which is what stops the copies drifting a third time.

The result-file reader is shared too, across five scripts, with one deliberate
difference preserved as a parameter: the runners read a flat `--output-dir` (matching a
result to the wrong sweep would corrupt resume), the analysis scripts `rglob` all of
`results/` (a main sweep and an ablation sweep live in separate subdirectories).

Behaviour otherwise unchanged: 288 tests pass, and `make_tables.py`,
`check_fedaco_health.py` and `check_iid_band.py` produce the same output as before the
change when run against the same result files.


## 2026-09-17 — the INERT verdict was a statement about the run length

The health check's question 1 — "is the colony searching?" — compares τ's entropy against
an absolute threshold: within 2% of the ceiling log(L) and the colony is declared INERT,
"carrying essentially no signal". It returned INERT on the project's only FedACO run, and
that verdict was about to send Phase 4 back to rewrite the deposit rule.

**τ starts at the ceiling.** `tau0` sits on every level of every row, so the entropy gap
is not a property of the colony alone — it is the distance τ has travelled from its own
initialization, and the travel rate is fixed by `rho`, the deposit magnitude
`rho * Q * max(F, 0)` and the iteration budget. Comparing a four-round run against an
absolute threshold measures the run length.

`fedswarm/aco/diagnostics.py` bounds it: drive the real `run_colony` and `Pheromone` with
`q0=1.0` and a constant fitness, so one level wins at *every* iteration. Nothing
concentrates τ faster. At the smoke run's own settings (K=2, iters 6→4 over 4 rounds,
F≈0.87) that ceiling is **3.14%** mean gap — so the 2% threshold required the colony to
realize **64%** of a best case that has stopped exploring altogether.

What the run actually did, from the same result file the verdict was read from:

| round | τ gap | best case | α | best F | FedAvg F |
|---:|---:|---:|---|---:|---:|
| 1 | 0.19% | 0.74% | [0.990, 0.010] | 0.780 | 0.663 |
| 2 | 0.56% | 2.44% | [0.009, 0.991] | 0.917 | 0.651 |
| 3 | 1.36% | 4.51% | [0.998, 0.002] | 0.901 | 0.708 |
| 4 | 1.57% | 4.88% | [0.038, 0.962] | 0.891 | 0.713 |

τ moves away from uniform **every round, monotonically**, reaching 29% of what the budget
allowed. And the colony beat the FedAvg point on fitness in all four rounds. "Carrying
essentially no signal" was not what the data said.

### The bound was wrong first, and a real colony beat it

The first version disabled the global-best deposit to keep the derivation clean. That
made it not a bound: `colony.py` deposits the global best on top of the iteration best
every `global_best_every=5` iterations, so a real colony at the default setting
concentrated τ *faster than its own supposed ceiling* — 3.13% against a claimed maximum of
2.41%. Caught by running real colonies against it before believing it;
`test_no_real_colony_beats_the_bound` now runs four of them at every commit. The corrected
bound folds the global-best deposit in and reads `aco-global-best-every` from the run.

### What changed

- `fedswarm/aco/diagnostics.py`: `best_case_gaps` (the bound, driven through the real
  update) and `closed_form_entropy` (an independent derivation, checked against
  `run_colony` at seven iteration counts so neither can drift unnoticed).
- `check_fedaco_health.py` question 1 now judges against the run's own ceiling. INERT
  means τ covered <15% of the available distance — scale-free, so it means the same thing
  in round 2 of a smoke test as in round 90 of the sweep. A run where clearing the
  absolute threshold would take >60% of the best case reports **UNDERPOWERED**: τ is
  moving, the run is too short to call it, and that is not evidence either way.
- `scripts/analyze_pheromone_dynamics.py` prints what each budget makes reachable, so the
  question "can this run answer the question?" is asked before the run, not after.

Making the check budget-aware risks making it unfalsifiable, so the test that matters is
`test_a_genuinely_dead_colony_is_still_inert_in_a_short_run`: τ pinned at uniform in a
four-round run is still INERT. The bar moved; it did not disappear.

**The open question is unchanged and still needs the real run.** None of this says the
colony is searching — only that the one measurement taken could not have shown it either
way. What it does say is that the reduced smoke budget cannot answer question 1 at all, so
`make validate-fedaco` should run long enough (or with a larger `aco-iters-start` /
`aco-q-deposit`) that the threshold is reachable. The projection table says the plan's
default budget clears it from round 1.


## 2026-09-18 — the fitness pays nothing for throwing clients away

Reading the α column of the one FedACO run, looking for something else:

| round | α | best F | FedAvg F | fallback |
|---:|---|---:|---:|---:|
| 1 | [0.990, 0.010] | 0.780 | 0.663 | 0 |
| 2 | [0.009, 0.991] | 0.917 | 0.651 | 0 |
| 3 | [0.998, 0.002] | 0.901 | 0.708 | 0 |
| 4 | [0.038, 0.962] | 0.891 | 0.713 | 0 |

The colony put essentially all the weight on one client every round, and flipped which
one. `alpha_entropy` averaged 0.06 against a ceiling of log 2 = 0.69. Every other signal
said this was going well: the fallback never fired, and the colony beat the FedAvg point
on fitness by a comfortable margin in all four rounds.

**It was going well. The target is degenerate.** Dispersion is
`sum_k alpha_k ||delta_k - Delta(alpha)||^2` — a weighted variance about the weighted
mean. At `alpha = e_j` the mean *is* `delta_j`, so every term is zero. Not small: zero,
for any Gram matrix whatsoever. Alignment collapses to `cos(delta_j, robust_mean)` and the
concentration penalty to its maximum, so

    F(e_j) = gamma_1 * cos(delta_j, robust_mean) - gamma_3 * log K

exactly. Verified against `DataFreeFitness.evaluate` at three K with non-default gammas
(`test_vertex_fitness_has_a_closed_form`). The entire cost of discarding K-1 clients is
`gamma_3 * log K`, and that guard grows only logarithmically while the thing it guards
against vanishes outright.

### It bites at small K, which is exactly where validation runs live

`F(best vertex) - F(FedAvg point)` on synthetic deltas (a shared consensus direction plus
isotropic noise; `noise` is the heterogeneity knob). Positive means the degenerate answer
wins:

| noise | K=2 | K=4 | K=6 | K=10 | K=20 | K=50 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | −0.049 | −0.107 | −0.144 | −0.192 | −0.258 | −0.348 |
| 1.0 | **+0.047** | **+0.036** | **+0.014** | −0.023 | −0.079 | −0.164 |
| 2.0 | **+0.108** | **+0.108** | **+0.093** | **+0.053** | −0.011 | −0.100 |
| 4.0 | **+0.135** | **+0.125** | **+0.113** | **+0.061** | −0.022 | −0.130 |

and the `gamma_entropy` each K needs to keep the corner from winning at noise 1.0:
**0.168 at K=2, 0.126 at K=4, 0.090 at K=10, 0.074 at K=20** — against the project's
default of **0.1**.

So the sweep's K=20 is safe and `make validate-fedaco K=4`, the command the README told
people to validate it with, was not. A smoke test in the one regime the real run is not
in. The K=4 recommendation is removed from the README, the Makefile and PROGRESS_REVIEW,
and the Makefile's `K ?=` now carries the reason.

⚠️ The deltas in that table are synthetic and isotropic; real training deltas are not, so
the crossover values are indicative. The vertex identity above is exact and holds for any
data — which is why the fix is to measure it rather than to trust the table.

### What changed

- `fitness.corner_margin(gram, base_weights, config)`: `F(best vertex) - F(FedAvg point)`,
  exact and O(K^2) because the vertex side needs no search. Logged as `corner_margin`
  every round by `strategies/fedaco.py`.
- `check_fedaco_health.py` gains question 3, "is the fitness optimum degenerate?", which
  reports it. A DEGENERATE verdict prints *before* the other three, because while it holds
  they cannot be read: a colony that searches well is supposed to find the corner.
  MARGINAL covers the K=20 high-heterogeneity case, where it loses by ~0.01 — negative,
  but not a comfortable pass.
- `scripts/analyze_fitness_landscape.py` (`make fitness-landscape`) maps the regime
  against K, heterogeneity and gamma_entropy before a run is spent on it.

### What this does not settle

Whether the margin is positive on *real* brain-MRI deltas at the sweep's K, and what
`gamma_entropy` should be. The honest fix is not to pick a bigger number here — it is that
a penalty growing as `log K` against a term that vanishes is the wrong shape, and a
dispersion floor or a `1 - sum_k alpha_k^2` concentration penalty would be the right one.
That is a change to the method as proposed, not a bug fix, so it is not being made on
synthetic evidence. The first real multi-round run at K>=10 decides it; `corner_margin` is
logged so that run answers the question by itself.

### And the validation notebook still had the num_supernodes bug

Checked while removing the `K=4` recommendation, because the Colab notebook is the path
someone would actually run this on. `notebooks/colab_fl_smoke.ipynb` set
`--client-resources-num-cpus 1` and **not** `--num-supernodes`, while passing
`min-train-nodes=K`. With `num_supernodes` defaulting to 2 that run hangs at round 0
forever -- the exact stall the 2026-09-17 entry retracted the CPU-oversubscription
diagnosis for, still live in the one file that would have reproduced it. The entry said
the correction had been propagated to five files; the notebook got the CPU half of it and
not the half that mattered.

Fixed, along with the retracted diagnosis still being stated as fact in the same cell's
warning text, and the "three questions" list (now four). The notebooks were also not valid
nbformat -- eight code cells had no `outputs` field, so `ruff` could not parse the file at
all and had been reporting a schema error instead of the E401 inside it. Both notebooks
are valid now and `ruff check .` is clean repo-wide for the first time.

### The run that started this was not a search pathology

Worth stating plainly, because it changes how the earlier entries read: α = [0.99, 0.01]
was the colony finding the optimum it was given, at a K it should never have been run at
(the `num_supernodes=2` bug put it there). Two separate diagnostics — the INERT verdict
withdrawn yesterday and this one — both turned out to be artifacts of that same K=2.


## 2026-09-18 (later) — the penalty shape, implemented as an ablation rather than a decision

Yesterday's entry named the degenerate optimum and stopped there, on the grounds that
fixing it would be changing the method on synthetic evidence. That was right about the
default and wrong about stopping: the fix can be *implemented and left off*, so the first
real run measures it instead of prompting a second round trip through the GPU budget.

`aco-concentration-penalty` now selects the shape — `"entropy"` (default, the method as
proposed, `log K - H(alpha)`) or `"gini"` (`sum_k p_k^2 - 1/K`). `corner_margin` reads the
shape rather than assuming `log K`, and `configs/experiment/ablation_all.yaml` gains two
cells: `penalty_gini` and `penalty_entropy_strong`. The second exists so a win for the
first cannot be confounded with "gamma_3 was simply too small" — both run at 0.25, so only
the shape differs. Ablation sweep goes 150 → 180 cells.

### The argument is scale, and it measures cleanly

Normalized dispersion is bounded and sits near 1 at every K, so a penalty holding it in
check should be bounded too. Vertex values: `log K` runs 0.69 → 3.9 over K ∈ {2..50};
`1 - 1/K` runs 0.50 → 0.98. The gamma_3 each shape needs to rule out the degenerate
vertex, over that same range:

| heterogeneity | entropy | gini | entropy spread | gini spread |
|---|---|---|---:|---:|
| 0.5 | 0.070 → 0.026 | 0.097 → 0.103 | 2.71× | 1.06× |
| 1.0 | 0.168 → 0.058 | 0.233 → 0.232 | 2.88× | **1.01×** |
| 2.0 | 0.256 → 0.074 | 0.355 → 0.297 | 3.44× | 1.20× |
| 4.0 | 0.294 → 0.067 | 0.408 → 0.266 | 4.41× | 1.53× |

At heterogeneity 1.0, one gamma_3 covers K=2 through K=50 under "gini" to within 1%.

⚠️ And what it does not fix: the requirement still moves with heterogeneity under both
shapes — 0.10 to 0.41 for "gini" as noise goes 0.5 → 4.0. "gini" removes the *K*
dependence, not the need to pick gamma_3 for the data. Saying otherwise would oversell it.

### Two errors in yesterday's own proposal

Yesterday's OPEN_QUESTIONS note proposed `1 - sum_k alpha_k^2` and justified it by the
gradient "not vanishing at the vertex". Writing the implementation made both wrong:

- **Sign.** `1 - sum alpha^2` is a *diversity* measure; it falls toward a vertex.
  Subtracting `gamma_3 *` it from F would have *rewarded* concentration — strictly worse
  than the bug it was meant to fix. The penalty must rise toward the vertex:
  `sum_k p_k^2 - 1/K`.
- **The reason.** Entropy's gradient is `log alpha_j + 1`, which *diverges* as
  `alpha_j -> 0`; Gini's is `2 alpha_j`, which vanishes there — the opposite of what was
  claimed. And it is beside the point: the colony searches a discrete level set containing
  0 exactly, so it jumps to the vertex rather than walking there. Only the penalty's
  *value* at the vertex matters.

Both corrected in place in docs/OPEN_QUESTIONS.md, with the error stated rather than
edited away. The measured claim above replaces the hand-wave.

### A new variant-overlap case, checked before trusting it

`penalty_gini` and `penalty_entropy_strong` are the first two variants in this project
that share an override *value* (`aco-gamma-entropy=0.25`). That is the exact shape of the
resume bug that hit in September — a control handed a sibling's result file. Verified on
real expanded cells and pinned by
`test_two_variants_sharing_a_key_value_are_still_told_apart`: each of the three cells
resolves to its own result. The by-value discriminator handles it, which is what it was
rewritten for.

324 tests pass, `ruff check .` clean.

## 2026-09-19 -- the analysis stack, and a seed count that cannot reach significance

Person A's third item: `analysis.py` (Wilcoxon, Holm-Bonferroni, Cohen's d, bootstrap CI)
existed but the Makefile wired `make_tables.py`, which did mean/std/n and no test. Folded
the statistics in -- `add_significance` pairs by seed (reusing the pairing `add_deltas`
already did), tests each row against the baseline, and applies Holm once across the whole
table rather than per regime.

**Then the integration turned up something worse than a missing feature.** The
signed-rank p-value has a floor set by the pair count alone. At `main.yaml`'s 5 seeds the
smallest possible two-sided p is **0.0625** -- so the planned analysis could not have
produced a significant result at alpha=0.05 under any data whatsoever. On synthetic
best-case input (FedACO ahead on every seed, Cohen's d = 7.91) the table reports:

| strategy | macro-F1 | Delta vs fedavg | p (Holm) | d |
|---|---|---:|---:|---:|
| fedaco | 0.9090 +/- 0.0032 | +0.0250 | 0.250 | +7.91 |

A d of 7.9 reported as p = 0.25. Without an explanation, that row reads as "no effect" --
the single most misleading number the paper could contain, and it would have been
discovered while writing the discussion section.

`make_tables.py` now prints the reason: the floor, the family size, the best achievable
adjusted p, and the seed count that would clear alpha. At 8 seeds the same data gives
p = 0.008 and the warning goes silent, so it is falsifiable rather than always-on.

The seed-count decision is recorded as open in docs/OPEN_QUESTIONS.md, because it is a
compute-budget question for whoever runs the main sweep and not a code fix: 5 -> 8 seeds
is 360 -> 576 cells, and clearing alpha over the full 66-comparison family needs 12.

Two of my own errors, for the record. I asserted "8 seeds" in a test where the correct
answer for a single-comparison family was 6, then "12 seeds" where a 6-regime table is a
6-comparison family needing 8. The test now asserts the *relationship* -- a larger family
needs more seeds -- because the count depends on how many claims the table makes and is
precisely the thing that is easy to miscount by hand.

440 tests pass, ruff clean.

## 2026-09-19 -- main.yaml raised to 8 seeds

Acting on the seed-count finding in the entry above. `main.yaml` goes from 5 to 8 seeds:
**360 -> 576 cells**, a 60% increase in the sweep's compute.

The reason is arithmetic, not taste. At 5 seeds the signed-rank test's smallest possible
two-sided p is 0.0625, so the sweep could not have produced a significant result at
alpha=0.05 under any data. 8 seeds moves the floor to 0.0078, which clears 0.05 after
Holm-Bonferroni across a six-comparison family (FedACO vs FedAvg in each regime).

`make_tables.py`'s `--expected-seeds` default moves to 8 with it, so a complete run is not
measured against a stale expectation.

⚠️ **Two things this does not cover, deliberately left as decisions:**

1. **A 66-comparison family still needs 12 seeds (864 cells).** If the paper claims
   significance against all 11 baselines in all 6 regimes rather than against FedAvg, 8 is
   not enough. `make tables` prints the requirement for whatever family the table actually
   contains, so this surfaces before the claim is written rather than after.
2. **`ablation_all.yaml` and `robustness.yaml` are still at 5 seeds.** Their comparisons
   (FedACO variants against the default, methods under attack) have their own family sizes
   and their own compute costs, and raising them was not part of this change. They carry
   the same floor, so any significance claim from those sweeps has the same problem.

The projected cost at 576 cells is ~770 hours on the only per-round measurement this
project has -- 48.1 s/round at K=2 on CPU, which docs/EXPERIMENT_LOG.md's compute table
says in bold not to cite. That number needs replacing with a real GPU measurement from the
validation run before anyone plans around it.

## 2026-09-19 (later) -- the ablation and robustness sweeps raised to 8 seeds

Completing the change the previous entry left open. All 16 ablation and robustness configs
now run 8 seeds.

The granular configs were at **3**, not 5 -- a floor of **0.25**, four times worse than the
main sweep's. A2-A9 and R1-R5 could not have distinguished anything from anything.

| sweep family | cells before | cells at 8 seeds |
|---|---:|---:|
| `ablation_all` | 180 | 288 |
| `robustness` | 225 | 360 |
| granular `ablation_a1-a9` | 365 | 920 |
| granular `robustness_r1-r5` | 162 | 432 |
| **ablations + robustness** | **932** | **2000** |
| `main` (raised earlier) | 360 | 576 |
| **grand total** | **1508** | **2576** |

A 71% increase in total sweep size.

### ⚠️ The duplication is now the bigger cost, not the seeds

The combined and granular families **overlap**. `ablation_all.yaml` covers A1/A3/A9, and
`ablation_a1/a3/a9.yaml` cover the same ground individually; `robustness.yaml` is a
combined adversarial + partial-participation sweep over the ground R1-R5 cover separately.
Running both families runs those experiments twice, at 8 seeds each.

Both families exist for a real reason -- they read different YAML shapes and the granular
ones cover A2 and A4-A8 that `ablation_all.yaml` explicitly does not (the merge entry of
2026-09-18 records why). But nothing says to run *both*, and the total above assumes you
do. Picking one family for the overlapping ablations, or dropping the duplicated cells from
whichever is run second, is worth more compute than any seed decision at this point.

Not decided here: that is a scope call for whoever owns the ablation runs.

### Still not covered

A 66-comparison family (all 11 baselines x 6 regimes) needs **12** seeds, not 8. 8 clears
alpha=0.05 for a six-comparison family -- FedACO against FedAvg in each regime. If the
paper claims significance against every baseline, this is still short. `make tables` prints
the requirement for whatever family the table actually contains, so it surfaces before the
claim is written rather than after.

440 tests pass, ruff clean; all 16 configs dry-run clean through their own runners.

## 2026-09-19 (later) -- the sweep plan, deduplicated: 2,576 cells -> 1,586

Acting on three decisions at once: keep the 8-seed floor only where a p-value is
actually claimed, revert it elsewhere, and resolve the two overlapping sweep families.

### Seeds: 8 where a claim rests on them, 5 everywhere else

`main.yaml` and `ablation_a1.yaml` stay at 8. Those are the two places the paper says
"significantly": the headline FedACO-vs-FedAvg comparison, and the make-or-break control
asking whether ACO beats random search at equal budget. At 5 seeds neither could produce
p < 0.05 under any data.

Everything else went back to 5. The other ablations and the robustness sweeps are read as
*descriptions* -- which component carries the result, how performance degrades under
attack -- and effect sizes and degradation curves carry that without a significance test.
Raising them cost ~640 cells and strengthened no claim in the paper.

### The overlap was worse than duplicated compute

Matching the two ablation families by *run-config key* rather than by label found five
duplicated variants -- and a collision:

| key varied | `ablation_all` called it | granular called it |
|---|---|---|
| `aco-persistence` | **A1** | **A2** |
| `aco-fitness-mode` | A3 | A3 |
| `model-norm` | A9 | A9 |
| `aco-target-sum` | shrinkage | A7 |
| `aco-gamma-dispersion` | gamma_dispersion | A4 |

**`ablation_all`'s "A1" was persistence; `ablation_a1.yaml` is the search-method control.
Both wrote to `results/fl/ablation`.** "A1" in a results table meant two different
experiments depending on which runner produced it, and nothing downstream could have told
them apart. The ~176 duplicated cells were the smaller half of that problem.

`robustness.yaml` was the same shape: its label_flip, sign_flip, gaussian and
partial-participation variants all duplicated R1/R2/R3, which sweep *three* attacker
fractions where the combined file sampled one or two -- the finer grid strictly contains
the coarser one, so removing them loses nothing.

Both combined files now keep only what no granular config covers:

- `ablation_all`: the safety-fallback ablation and the concentration-penalty shape pair.
  288 -> 60 cells.
- `robustness`: the `scaled` magnitude-only attack. 360 -> 75 cells. Kept deliberately --
  it is the one attack an alignment-based heuristic cannot see, since FedACO's `a_k` is a
  cosine and is blind to a scaled update, so only the norm ratio `r_k` can catch it.
  Dropping it would remove the single case that separates the method's two heuristics.

### A live bug found while updating the Makefile

`make ablations-granular` globbed `configs/experiment/ablation_a*.yaml`, which also
matches `ablation_all.yaml` -- a different YAML shape the granular runner cannot parse
(`KeyError: 'partitions'`). The target would have died partway through the ablation set.
Narrowed to `ablation_a[0-9].yaml`.

### The plan now

| | cells |
|---|---:|
| main (8 seeds) | 576 |
| ablation_all | 60 |
| robustness | 75 |
| ablation_a1 (8 seeds) | 80 |
| ablation_a2-a9 | 525 |
| robustness_r1-r5 | 270 |
| **total** | **1,586** |

Down from 2,576 with both families at 8 seeds. Run both targets in each family -- they no
longer overlap.

440 tests pass, ruff clean.

## 2026-09-19 (later) -- the Kaggle sweep notebook, and 33 Makefile paths that never worked there

Built `notebooks/kaggle_main_sweep.ipynb` for person B's workstream, and found while testing
its commands that **every `make` target was unusable on Kaggle or Colab.** The Makefile
hardcoded `.venv/bin/python` in 29 places and `.venv/bin/flwr` in 4; notebook runtimes install
into the system Python and have no `.venv` at all, so `make validate-fedaco` would have failed
instantly with "No such file or directory" -- on the exact machine the gate has to run on.

Now `PY ?= .venv/bin/python` / `FLWR ?= .venv/bin/flwr`, overridable:
`make validate-fedaco PY=python FLWR=flwr`. Default behaviour locally is unchanged (verified).

### The notebook runs the plan's gates before the sweep, deliberately

Plan §11 puts A1 at week 5 as a go/no-go on claim C2, before the main sweep in weeks 7-8, and
the risk register rates "ACO ties equal-budget random search" Medium-High / fatal. The
notebook therefore runs, in order: the 20-minute mechanism check (`validate-fedaco` + `health`),
then A1's 80 cells, and only then the 576-cell sweep -- with both gates marked as stop points.
Ordering them the other way would spend the larger budget on a question the paper might no
longer be able to ask.

It is also built for a multi-session run, because 57,600 rounds does not fit Kaggle's ~9h GPU
cap: results are restored from the previous version's output at the top, the sweep skips
completed cells, and the last section says what to save and how to resume. No Kaggle API
credentials are needed -- the dataset mounts from the sidebar.

Two build errors of my own worth recording, both caught by `ruff` rather than by reading:
nbformat needs each `source` line to keep its trailing newline (splitting without them
concatenated the whole notebook into one 3,000-character syntax error), and a multi-line
`!python -c "..."` cell does not parse as either shell or Python -- it is now a plain Python
cell.

440 tests pass, ruff clean repo-wide.

## 2026-09-19 (later) -- the plan is in the repository, and it immediately found a deviation

`docs/IMPLEMENTATION_PLAN.md`. It existed from the start and was never committed, which
had two concrete costs: Phase 7's A2 and A4-A8 were recorded as "not reconstructable from
what the repo names" and left unbuilt for days, and the level set diverged from §14's
explicit hyperparameter table with nothing to catch it.

Reading §14 against `aco/schedules.level_set` took about a minute and found the second.
The plan specifies `linspace(0, 2.5, 11)`; the code shipped log-spaced -- **1143x max ratio
between positive levels against linear's 10x, and nine of eleven levels below the FedAvg
point instead of four.**

`level_spacing` is now selectable and defaults to "linear". "log" is kept, because every
result before today used it and A6 owns L; A6 gains a `levels_log_spaced` cell so the
switch is measured rather than assumed. All 440 tests passed unchanged under the new
default, which says the suite never pinned the old grid -- the deviation was invisible to
it as well.

**This partially qualifies the 2026-09-18 degenerate-optimum entry.** The observed
alpha = [0.990, 0.010] was the colony finding the fitness's single-client optimum, and
`corner_margin` measures the fitness itself, so that part stands. But how *reachable* the
corner was is a property of the level grid, and the grid was not the plan's. The synthetic
sweeps and the K=10 rehearsal both ran on the log grid. Whether the margin stays positive
under linear spacing on real deltas is now open, and recorded as such.

Two amendments to the plan itself, recorded rather than silently followed: §9.1's Wilcoxon
+ Holm at the stated 5 seeds cannot return a significant result at any data (floor 0.0625),
which is why main.yaml runs 8; and §0.3's naming collision was resolved by renaming to
FedSwarm.

440 tests pass, ruff clean.

## 2026-09-19 (later still) -- the LaTeX tables are wired

Plan §9.3's deliverable: `paper/tables/*.tex`, booktabs, best-in-bold, significance
markers. The capability already existed in `src/fedswarm/tables.py` and was reachable from
nothing -- the Makefile's three table targets all call `scripts/make_tables.py`, which
emitted `.md` and `.csv` only.

`make_tables.py` now emits `.tex` alongside them, from the *same* rows, rendering through
`fedswarm.tables.main_results_table` rather than a second LaTeX emitter. The plan's
instruction is "never hand-type a number into the paper"; two formatters drifting apart is
the same failure one step later. All three targets (`tables`, `tables-ablation`,
`tables-robustness`) pick it up with no Makefile change.

Ablation tables are labelled by *variant* rather than strategy -- every row of an ablation
sweep is FedACO, so labelling by strategy prints a column of identical "fedaco" rows and
loses the thing being ablated.

One adapter detail worth not "fixing" later: `main_results_table`'s `method=` parameter is
"the row that gets no marker". Its own convention marks the baselines ("fedaco vs. this
baseline"); ours marks the method ("this row differs significantly from fedavg"), which is
what our caption states and what `rows` carries. So the row left unmarked is the
baseline's, and passing `baseline` into `method=` looks inverted and is not. Commented at
the call site.

### §9.3's acceptance criterion is now a test

> *"Deleting `paper/figures/` and `paper/tables/` and re-running restores them
> byte-for-byte identical."*

`test_regenerating_the_table_is_byte_for_byte_identical` checks exactly that. A timestamp,
a dict iteration order or a set ordering anywhere in the path would break it, and the
breakage would surface only as noise in the paper's diff. Verified by hand too: identical
md5 across a delete-and-regenerate.

Also pinned: markers appear at 8 seeds (p=0.008 earns `**`) and are absent at 5 (floor
0.0625 earns nothing). A table that marked both would assert significance the test never
found.

One of my own test expectations was wrong and is worth recording: I asserted the escaped
string `no\_safety\_fallback` would appear, but `_variant_of` names that variant
`no-fallback` with a hyphen, so there was no underscore to escape. The output was correct
and the test was not. It now asserts escaping on `dirichlet_0.3`, which does carry one.

446 tests pass, ruff clean.

## 2026-09-19 (later still) -- the num_supernodes bug, a third time, in B's scaling sweeps

Found while rehearsing person B's remaining commands rather than by reading code.
`scripts/run_sweep_granular.py` never set `num_supernodes` at all -- and unlike `main.yaml`,
the configs it drives **vary `num-clients` across cells**: `overhead.yaml` sweeps K = 5, 10,
20, 50, 100, 150, 200; `main_client_scale.yaml` 10 -> 50; `robustness_r5` likewise.

`num_supernodes` is a *federation* setting. No `--run-config` key can carry it, and it
defaults to 2. So every cell above K=2 would have either hung at round 0 forever
(`min-train-nodes` above the supernode count) or silently trained 2 clients while recording
100.

**The second failure mode is the dangerous one here, and it is specific to which sweeps
these are.** `overhead.yaml` and `main_client_scale.yaml` exist to produce the measured
ACO-time-vs-K curve that plan §1.3's claim C3 rests on -- "a measured curve matching K^2 is
far more convincing than a complexity assertion" (§8, R5). Measured at a fixed 2 supernodes
while labelling cells K=5..200, that curve comes out **flat**. A flat curve reads as
evidence *for* the overhead being negligible. The bug would have manufactured support for
the claim it was meant to test.

This is the third time this project has been bitten by `num_supernodes`: once in the
strategy path (2026-09-17, retracting a CPU-oversubscription diagnosis), once in the Colab
notebook (2026-09-18, still live after the first fix was called propagated), and now in the
granular runner.

### The fix, and the proof

`fedswarm.sweep.run_sweep` now resolves each cell's `num-clients` against pyproject's
defaults and reconfigures the federation when the value changes -- only when it changes, so
`main.yaml`'s 576 constant-K cells cost one subprocess call, not 576. A failed
reconfiguration refuses the cell (`status: failed_federation_config`) rather than running
it, because both failure modes are worse than stopping.

Verified live, not just by unit test: the federation was deliberately pre-set to 2
supernodes, then a two-cell sweep at K=2 and K=6 was run against the synthetic dataset.

    num-clients=2   completed   clients that actually trained: 2  ids=[1, 0]
    num-clients=6   completed   clients that actually trained: 6  ids=[1, 2, 3, 0, 4, 5]

`train_participating_client_ids` matches the recorded `num-clients` in both. Before the fix
the K=6 cell hangs at round 0 indefinitely.

453 tests pass, ruff clean.

## 2026-09-19 (later still) -- the first GPU run: two bugs only a GPU could show, and the gate's verdict

Person B ran GATE 1 on a Kaggle T4. FedACO, K=10, 15 rounds, dirichlet alpha=0.3, seed 0.
It completed and wrote `results/fl/f89d6200b0_0.json`. Three things came out of it.

### Bug 1: evaluation ran the model on CPU while the batches were on the GPU

`eval/evaluator.evaluate` took a `device`, moved the batches onto it, and left the weights
wherever they were. Every other call site compensated -- `local_train`, `build_evaluate_fn`,
`ServerValFitness`, `run_experiment` all call `.to(device)` themselves -- and
`fl/app.py::evaluate_handler` did not. On CPU both halves are the same device, so 457 tests
and every CPU run this project has ever done passed. On the T4:

    RuntimeError: Input type (torch.cuda.FloatTensor) and weight type
                  (torch.FloatTensor) should be the same

**The run did not fail.** Only the client-side federated evaluation did, once per client
per round, and the ServerApp's own centralized evaluation is a different code path that was
already correct. So the result file was written, marked `completed`, with

    Aggregated ClientApp-side Evaluate Metrics: {}

A silently empty metric block, in a file that otherwise looks finished, across 678 cells.
Fixed in `evaluate` itself rather than at the call site, so the contract is whole: a
function handed a device now moves both halves onto it.

### Bug 2: the server never told clients which round it was, so R4's DP noise never changed

The first run's train metrics carried `server_round = -1` in every round -- line 549's
default, meaning the key was absent. The ServerApp builds `train_config` once, before round
1, and nothing put a round number in it. Scaffold wrote one, spelled `server-round`, which
no reader uses.

The cost is R4's. The Gaussian mechanism seeds its noise with
`seed*104729 + partition*1000003 + server_round`. With that term pinned at 0, **every round
drew the same noise** -- a fixed per-client perturbation the model trains around, not DP
noise. R4 would have reported FedACO as far more noise-robust than it is, and no field in
the result file would have shown why. A comment at that line asserted the server sent the
key; it did not, and the comment is corrected rather than deleted.

Fixed by wrapping `configure_train` in the factory. Seven of the twelve strategies are
Flower built-ins with no subclass here, so a per-strategy fix would have covered the five
custom ones and left every baseline the paper compares against broken.

### The gate's verdict: question 3 is DEGENERATE, and that is the finding

| | |
|---|---|
| 1. Is the colony searching? | **UNDERPOWERED** -- tau at 36% of best case; SEARCHING would need 253% |
| 2. Deposit floor engaged? | **PARTIAL** -- best_fitness <= 0 in 1 of 15 rounds |
| 3. Fitness optimum degenerate? | **DEGENERATE** -- single-client vertex beat FedAvg 15/15, mean +0.6252 |
| 4. Beats FedAvg? | **NO BASELINE** -- the FedAvg run had not landed when the check ran |

Question 3 closes the open question from this morning's level-set entry, and closes it the
wrong way: the corner wins on the **linear** grid, on **real** deltas. The grid was never
what made it attractive. Recorded in full in docs/OPEN_QUESTIONS.md, including why this is
also A1's problem -- the controls optimize the same fitness, so "ACO ties random search"
could be reported for a reason that has nothing to do with ACO.

### The first honest compute number, and the first evidence for claim C3

**7.5s per round** at K=10, 1 local epoch, image-size 112, on a T4. The estimates in this
file's compute table are superseded by it. Extrapolated to the real sweeps -- and it is an
extrapolation, since main.yaml runs K=20:

| | at 7.5s/round (K absorbed) | at 15s/round (linear in K) |
|---|---|---|
| main.yaml (576 x 100) | 120 GPU-h | 240 GPU-h |
| A1 (80 x 100) | 17 GPU-h | 33 GPU-h |
| everything B owns | **139 GPU-h** | **279 GPU-h** |

At Kaggle's ~30 GPU-hours per week that is 5-9 weeks of wall-clock for one person, which is
a scoping fact the plan's week-7-8 slot does not currently reflect.

**C3 ("negligible aggregation overhead") has its first real measurement and it holds:**
`aco_time_ms` + `gram_time_ms` averaged 198 ms against 7,500 ms per round -- **2.6% of
wall-clock** at K=10. The O(K^2) curve still needs `overhead.yaml` to make the claim over
K, but the constant is small where it has been measured.

471 tests pass, ruff clean.

## 2026-09-19 (later still) -- the gate now names the number instead of giving advice

The first GPU gate's most useful output was also its least actionable: "raise
`aco-gamma-entropy` until the margin is negative". Closing that would have cost another GPU
session to measure a quantity the round had already built the inputs for.

**It needs no search.** The margin is *linear* in gamma_entropy:

    margin(g3) = [g1*max_cos - g3*P_max] - [g1*align(base) - g2*disp(base) - g3*P(base)]
               = A - g3 * (P_max - P(base))

with `A` free of g3. One evaluation at the current value fixes the whole line, so the zero
crossing is `g3* = g3 + margin(g3) / (P_max - P(base))`. `aco/fitness.required_gamma_entropy`
computes it in closed form over the same Gram matrix, `strategies/fedaco.py` logs it every
round, and `check_fedaco_health.py` reports the **max across rounds** -- gamma_entropy is set
once per run, so the value that clears the average round leaves the worst rounds degenerate.

Against a replica of B's real per-round margins, the check now prints:

    **Set `aco-gamma-entropy` to at least 0.450** (currently 0.1); that is the largest
    zero-crossing over 15 rounds, computed in closed form from each round's own Gram matrix.

0.450 is a **lower bound** for that run, because the replica assumes uniform base weights
(`P(base) = 0`). FedACO's reference point is the num-examples-weighted average, which under
dirichlet(0.3) is already concentrated, so the real `P(base) > 0`, the real denominator is
smaller, and the real requirement is higher. The live run computes it exactly.

Twelve round-trip tests (K in {2,4,10,20} x three noise levels) feed the returned value back
into the real `corner_margin` and assert it lands on zero -- deliberately not a test of the
algebra against itself, which would survive someone adding a term to the fitness.

### --strict now fails on DEGENERATE, and the notebook gates A1 on it

A1 costs 80 cells and 17-33 GPU-hours to ask "does the colony beat equal-budget random
search". Its four controls optimize the *same* fitness, so on a degenerate landscape they
inherit the same useless optimum and a tie says nothing about ACO -- while the plan reframes
the paper on exactly that result. `--strict` previously passed whenever the colony was
SEARCHING, which is the one case where a degenerate optimum hides best: the colony is
working, and working toward the corner.

The notebook's gate cell now runs the check through `subprocess.run` rather than `!`, because
a failing `!` line prints its error and the notebook carries straight on -- which is how the
first gate run left A1 one Run-All away from spending that budget.

### A6 was sweeping seven hyperparameters and not the one that matters

`aco-gamma-entropy` is the only term standing against the corner, and A6 did not touch it.
Now three cells: 0.1 (default), 0.5 (brackets the K=10 requirement with headroom), 1.0
(deliberately past where the penalty should start costing accuracy, so the sweep shows the
cost and not only the fix). A6 goes 220 -> 250 cells.

**The default stays 0.1.** Raising it repairs the corner but charges *any* concentration,
including the genuine down-weighting of a straggler or an adversary that R1-R3 need FedACO to
perform. That is a measured trade-off, not a bug fix, and changing the default silently would
make every future result incomparable with the plan's stated configuration.

### Check 4's ambiguity, closed in the output rather than in the next session

"NO BASELINE" could not distinguish a FedAvg run that had not finished from one whose result
file the check failed to match -- and the first gate printed it while a FedAvg run was
visibly starting in the same cell. `config["strategy"]` is populated correctly, so the
matching was never broken, but the log could not establish that. The check now lists every
strategy present in the results directory before running its four questions.

491 tests pass, ruff clean.

## 2026-09-19 (last) -- three ways the gate would have wasted the next GPU session

Found while making the second gate attempt easy to run, not by testing the mechanism.

**1. `--strict` blocked on UNDERPOWERED, which is not a failure.** It means *this run cannot
tell* -- the gate is 15 rounds by design and tau has not had time to leave uniform by a
margin the threshold can see. Since the notebook now gates A1 on this exit code, someone who
had just fixed the corner would still be refused, with the only remedy being to lengthen a
gate whose whole purpose is to be short. A1 itself runs at 100 rounds, where the question is
answerable. UNDERPOWERED and NO DATA are now non-blocking and say so in the output; INERT
and DEGRADING still block, because those are real negative verdicts about the mechanism.

A gate that cannot tell inconclusive from failing is a gate nobody can pass.

**2. The check could report the previous attempt.** Changing `aco-gamma-entropy` changes the
config hash, so a re-run at a new value writes a *new* file beside the old one -- both
completed, both 15 rounds. `max()` returns the first maximal element, so the check could
read the stale attempt and report the old verdict against the new setting: you change the
value, re-run, and the report says nothing changed. Now tiebroken on mtime, and the header
prints the `aco-gamma-entropy` of the run it actually read.

**3. The gate's gamma was buried in a run-config string.** It is the one number that changes
between attempts, so it is now a named `GAMMA_ENTROPY` constant at the top of the notebook's
gate cell, defaulting to 0.6 -- the measured 0.481 requirement with ~25% headroom.

### A note on fixtures

The three `--strict` tests needed runs that genuinely produce UNDERPOWERED, SEARCHING and
INERT. Check 1's verdict is not a function of tau's entropy alone: it is tau's movement as a
*fraction* of what that run's own deposit budget makes reachable, and the deposit is
`rho * Q * max(F, 0)`. So a near-uniform tau reads INERT at a healthy fitness (budget there,
unused) and UNDERPOWERED at a small one (budget never there). The fixtures were found by
probing the real function; the first attempt assumed entropy alone decided it and wrote a
test that asserted the opposite of what it set up.

499 tests pass, ruff clean.

## 2026-09-24 -- person C's workstream, and two real defects found preflighting it

Built `notebooks/kaggle_ablations_robustness.ipynb` for Phases 7-8. It reuses person B's
setup cells (clone/pin, pip, dataset probe, restore, cache) by reading them out of the built
notebook rather than copying the text -- those five absorbed a dozen fixes and a second
hand-maintained copy would drift.

**C's workstream is larger than B's: 990 cells, 99,000 rounds, ~212 GPU-hours** (or ~420 if
per-round cost scales with K; the gate measured K=10 and these run K=20).

| family | cells | rounds |
|---|---|---|
| Ablations A2-A9 | 585 | 58,500 |
| `ablation_all` | 60 | 6,000 |
| Robustness R1-R5 | 270 | 27,000 |
| `robustness` | 75 | 7,500 |

### Two defects, both found before anyone spent a GPU-hour

**1. Krum was mis-tuned in two thirds of the cells it appears in.** `robustness_r1` and
`robustness_r2` hardcoded `num-malicious-nodes: 4` (20% of 20 clients) across attacker
fractions of 10%, 20% and 30%, because a Cartesian-product grid cannot express "this
strategy-specific value must track that partition-specific one". Krum was therefore correctly
tuned in one cell per sweep and mis-tuned in the other two -- and Krum is the Byzantine
baseline the paper's "resilience for free" claim is measured against, so a mis-tuned Krum
flatters FedACO in a way no reader could detect from the tables.

The coupling now lives in `strategies/factory.py`, which sees the whole resolved run_config.
`f` comes from `attacks.malicious_ids` -- the same function the ClientApp uses to decide which
partitions actually lie -- so the number Krum assumes and the number that lie cannot drift.
Verified: f = 0/2/4/6 at fractions 0/10/20/30% at K=20, `attack="none"` gives 0 whatever the
fraction (the clean arm), an explicit value still wins (`robustness.yaml` sets 2 and 6 per
variant and keeps working), and Flower's `num_closest = max(1, n - f - 2)` is satisfied at
every cell. This had been recorded in OPEN_QUESTIONS as an accepted simplification; it was
fixable, and the fix is smaller than the note explaining why it wasn't.

**2. Seventeen configs documented a runner that cannot load them.** There are two config
shapes and two runners: `base_overrides`/`strategies`/`partitions` loads only in
`run_sweep_granular.py`, `common`/`strategies`/`regimes`/`variants` only in `run_sweep.py`.
Every granular config's `# Run:` line named `run_sweep.py`, which fails immediately with
`ValueError: ... is missing required key 'regimes'`. Copy-pasteable, wrong, and only
discoverable by someone with a session running. Fixed in all 17 and pinned by a test that
checks the documented command against the config's own shape.

Preflight that came back clean: all 15 of C's configs dry-run; every run-config key across all
22 configs is declared in pyproject (an undeclared key is rejected with a bare `[code: 15]`
naming nothing); all 1,646 assembled `--run-config` strings parse under `tomllib`, which is
what flwr uses; and only R5 varies K, on the granular runner that reconfigures per cell.

### The notebook splits C's work by prerequisite, which is the point of it

*Section 4, no prerequisites (385 cells).* R1-R5 and A9 compare FedACO against Krum /
Trimmed-Mean / Median / FedAvg under attack, DP noise and straggling, or control for a
BatchNorm confound. Those comparisons mean what they say whatever the colony is doing
internally, and R1-R2 are the paper's second contribution.

*Section 5, gated (605 cells, ~130 GPU-hours).* A2, A4-A8 and `ablation_all` measure the
colony's own machinery. On a fitness with no usable optimum they describe the failure rather
than the method. The cell runs `check_fedaco_health.py --strict` and refuses on a DEGENERATE
optimum or an INERT colony; UNDERPOWERED is inconclusive and does not block.

**A3 is exempt and should run first.** 20 cells, ~4 GPU-hours, and it is the cheapest
experiment that can localise the blocking problem: the same colony against `data_free` (the
proposed surrogate) and `server_val` (real macro-F1 on a server-held split). If `server_val`
works and `data_free` does not, the fault is the surrogate rather than the search -- a
different paper-level conclusion to "ACO does not help", reached for 4 GPU-hours instead of
A1's 33.

527 tests pass, ruff clean.

## 2026-09-24 (later) -- the check that was missing when gamma_entropy=0.6 shipped

`aggregate` was shipped on two numbers: the corner margin and F at the FedAvg point. Both
looked right. Those are exactly the two numbers that looked right for `gamma_entropy=0.6`,
which then collapsed alpha onto the FedAvg point and made FedACO a slower FedAvg. Neither can
see whether the colony has anywhere left to go.

A real `run_colony` under all three shapes, K=10, noise 3.0, dirichlet(0.3)-skewed base
weights, gamma_entropy=0.1, 5 seeds:

| mode | corner | F(base) | improvement over FedAvg | alpha_max | \|alpha-base\|_1 |
|---|---|---|---|---|---|
| weighted_mean | +0.2501 | +0.0096 | +0.0391 | 0.420 | 0.470 |
| base | -0.1915 | +0.0096 | **+0.0084** | 0.458 | 0.156 |
| **aggregate** | **-0.8924** | **+0.4157** | **+0.1554** | 0.317 | 0.355 |

Under `aggregate` the colony gains **4x** what it finds under the method as proposed and
**18x** what it finds under `base`, with an alpha that is neither collapsed onto the FedAvg
point (uniform would be 0.100) nor sitting on a vertex. Entropy 1.635 against a 2.303 ceiling:
genuinely non-uniform, not concentrated.

**A second, independent reason `base` was the wrong fix**, and a consequence of the same
linearity that made it unable to rule out the corner: a linear term added to an objective
tilts it without curving it, so the corner gets charged and the interior gets levelled. It
moved alpha the least of the three (0.156) and found almost nothing (+0.0084). It failed in
both directions at once.

Both properties are now tests rather than a probe I ran once, because "the landscape is sane"
and "the landscape is useful" are different claims and this project has now twice shipped the
first while believing it had checked the second.

Still synthetic. But this is the skewed high-noise regime that reproduces the real runs'
F(fedavg) < 0, so it is the closest proxy available -- and the gate run is what settles it.

529 tests pass, ruff clean.

## 2026-09-24 (later still) -- R6 exists: cold start, via a filtered Grid

R6 (clients joining after round 20) was the one entry in the plan's robustness table with no
config file at all, and `docs/OPEN_QUESTIONS.md` explained why: which SuperNodes exist is
Flower's Simulation Runtime's business, and no `run_config` key says "partition 15 must not
participate before round 20".

Both true, and neither closes the door. **A strategy never asks the runtime for nodes -- it
asks the `Grid` it is handed.** So every client starts at round 1 and a filtered Grid keeps a
deterministic subset out of the strategy's view until the join round. The clients exist; the
aggregation does not see them, which is precisely the condition R6 is about: does the pheromone
initialise sensibly for a client id it meets at round 21 when every other id has twenty rounds
of deposit behind it?

`fl/cold_start.py`, `configs/experiment/robustness_r6_cold_start.yaml` (25 cells, 2,500
rounds, ~5 GPU-h). Late joiners are the **highest** node ids, deliberately disjoint from
`attacks.malicious_ids`' lowest, so a cell combining an attack with a cold start delays
different clients than it compromises instead of quietly testing one thing twice.

### The hazard was a hang, not an error

Flower's own sampler:

    while len(all_nodes := list(grid.get_node_ids())) < min_available_nodes:
        sleep(1)

No give-up. Hide one node too many and the run does not fail -- it logs a line a second
forever, which on Kaggle consumes the whole nine-hour budget and writes nothing. Nothing
downstream can catch it, because nothing is raised. So `with_cold_start` checks the visible
count against the strategy's own `min_available_nodes` / `min_train_nodes` /
`min_evaluate_nodes` / `ceil(visible * fraction_train)` and raises at construction, and a test
builds every strategy the shipped config declares to prove that config passes its own guard.
That is also why the config sets `min-train-nodes: 10` rather than 20, with the reason written
above the key.

Evaluation is filtered too. Leaving it unfiltered would report a late joiner's local-val
metrics on a model its data never touched, for twenty rounds.

### What R6 can and cannot settle

It sits in the **gated** half of person C's notebook, not the free half, and that placement is
the honest one: the plan says "cross-round pheromone should shine here", and that claim needs a
pheromone that moves. The two FedAvg arms are not gated -- they establish what a late joiner
costs a memoryless aggregator, which is a paper number whatever the colony is doing, and
without them a FedACO gap cannot be told apart from "fifteen clients for twenty rounds is
simply less data". The `fedaco_cold_no_persistence` arm is the third leg: if the gap survives
with persistence off, it was never about stigmergy.

Person C's workstream is now 1,015 cells / 101,500 rounds / ~217 GPU-hours.

547 tests pass, ruff clean.

## 2026-09-24 (last) -- verify_repro was checking a non-reproducible number against a band narrower than its variance

Phase 10's acceptance criterion is "runs the smoke config and checks final metrics fall within
a recorded tolerance band". The band, `docs/repro_reference.json`, was seeded from run
`e35285b33a_0` at git_sha `9b01a462...` -- **before** client-side seeding existed. At that
commit `seed_everything` ran only in `server_app.main()`, and a ClientApp is a separate Ray
actor process, so every client's train loader shuffled from OS entropy.

So the reference value was one draw from an unrecorded distribution. Its `tolerance_abs` was
0.15, documented as "wide enough to absorb float-determinism drift". The drift was not
float-sized: two Kaggle runs of an identical FedAvg config, same seed, returned **0.1363 and
0.3286** -- a 0.19 swing, wider than the band itself. The check was comparing a
non-reproducible quantity against a tolerance narrower than its own variance, and passing or
failing at random.

Post-fix it is worse rather than better. Runs reproduce now, and a deterministic value has no
reason to land near an old random draw -- so the check would fail for correct runs. Either way
`verify_repro: PASSED` never meant what the plan says it means, and a reproducibility claim in
a paper resting on it is not recoverable once published.

**`load_reference` now refuses a band that does not carry `seeded_after_commit` equal to the
determinism-fix commit**, with the reason and the remedy in the error. `verify_repro.py` exits
2 on that refusal rather than pretending to a verdict, and gains `--seed-reference` to write a
fresh band from a completed run in one command instead of hand-edited JSON.

Two deliberate choices in that:

- **Seeding is its own mode, not a fallback.** Re-seeding whenever the check fails would turn
  every regression into a new reference and the check into a no-op. B's notebook prints the
  command and does not run it.
- **The regenerated tolerance is 0.05, not 0.15.** The old value was sized to absorb variance
  the code no longer has. With the clients seeded, the same config and seed reproduce, so the
  band only needs to cover genuine cross-platform float drift -- and a band three times wider
  than the effect it measures cannot catch a broken pipeline, which is its only job.

The stale file is marked rather than deleted, so the numbers any pre-fix result was checked
against stay on record. Its two existing tests now pass `allow_stale=True` and check the
file's *shape*, which is what they were really for; verifying against it is what the refusal
covers.

**This is the third instance this session of the same defect class**: a check that runs, passes,
and means nothing. The others were the client-side evaluation whose metrics came back silently
empty, and the `seed` recorded in every result file that governed only the server. All three
were invisible precisely because something was reported as working.

552 tests pass, ruff clean.

## 2026-09-24 (last, really) -- the LaTeX table dropped both of the markdown's warnings

Following the pattern from the reproducibility band -- a check that runs, passes and means
nothing -- into the one place it matters most: the artifact that goes in the paper.

`make_tables.py` emits markdown, CSV and LaTeX from one `rows` list, deliberately, so two
formatters cannot drift. They had drifted anyway, because `to_latex` built its DataFrame from
`strategy, partition, mean, std, n` and dropped two things the markdown carried.

**1. Incomplete cells were indistinguishable from complete ones.** Built from a sweep with 3 of
8 fedaco seeds:

    markdown:  | fedaco | default | 3 ⚠️ | 0.5100 ± 0.0100 | ...
               ⚠️ **1 cell(s) have fewer seeds than expected** ... Do not quote anything

    latex:     fedaco & 0.510 $\pm$ 0.010 \\

Character-for-character identical to a complete cell. The markdown is read by whoever ran the
sweep; **the LaTeX is what goes in the paper**, and the main sweep runs over many sessions, so
a partially-filled table is its normal state for weeks. The plan's own rule is "never
hand-type a number into the paper", which makes this emitter the number.

**2. The caption advertised a significance threshold the test could not reach.** The same
table's markdown says "these p-values cannot reach 0.05 at this seed count -- the smallest
possible two-sided p at n=3 is 0.2500". Its LaTeX caption said
`($^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$)`. A reader of the paper had no way to know the
legend promised something no data could deliver.

Both now live in the caption, because a caption is the only thing that travels with a table
someone pastes into a draft. After the fix, the same partial sweep:

    \caption{... $^{\dagger}$~1 cell(s) are built from fewer seeds than planned and are not
    comparable to the rest: fedaco/iid (3 of 8). Do not quote these numbers. No significance
    markers are shown: at $n=3$ paired seeds the signed-rank test's smallest attainable
    two-sided $p$ is 0.2500 ... so $p<0.05$ is unreachable at any data ...}

    fedaco & 0.510 $\pm$ 0.010$^{\dagger}$ \\

A complete, adequately-powered sweep gets its legend back and carries no dagger -- verified,
because a warning that is always present is wallpaper.

`power_note` (markdown) and `_latex_power_note` now share `_power_floor`. Two formatters
computing "is this table underpowered" separately is precisely how one ended up warning about
what the other advertised, and a test asserts they agree at n in {3, 5, 6, 8, 10}.

**Fourth instance of the same defect class this session.** The others: client-side evaluation
failing on GPU while the run reported `completed` with an empty metrics block; the `seed` in
every result file that governed only the server; and the reproducibility band seeded from a
non-reproducible run. Every one was invisible because something reported success.

557 tests pass, ruff clean.

## 2026-09-24 (the important one) -- the FL harness runs on THIS box, and it found another dropped half

`ray` and `flwr.simulation` are installed in this container, on Linux x86_64. Every note in this
repo saying the harness cannot run locally is about the *Intel-macOS* dev machine and has been
quietly false here for the whole session. So a real `flwr run .` went end to end on the
synthetic smoke dataset, and three things came out of it.

### 1. The determinism fix is confirmed empirically, not structurally

Two runs, identical config, `seed=0`, separate processes:

| | run 1 | run 2 |
|---|---|---|
| final_test_macro_f1 | 0.3333333333 | 0.3333333333 |
| train_best_fitness | 0.8654202819 | 0.8654202819 |
| train_corner_margin | -0.0249431129 | -0.0249431129 |
| train_alpha | [0.25714287, 0.25714287, 0.22857143, 0.25714287] | identical |

**11 of 11 metrics bit-for-bit identical.** Until now the only evidence was a test asserting
that `seed_everything` appears inside the handlers -- which proves the call exists, not that it
works. The same run also shows `train_server_round: 3.0` (was -1) and
`participating_client_ids: [0, 1, 2, 3]` (sorted), so both halves of the 2b7c479 fix are
confirmed on real output.

### 2. Client-side evaluation was computed every round and thrown away

`Result` carries `evaluate_metrics_clientapp` beside `train_metrics_clientapp` (verified
against the installed dataclass). `merge_train_metrics` folded in the first. **Nothing read the
second.** So every client's `evaluate_handler` ran a forward pass over its local-val split,
every round, and the output reached Flower's console and nowhere else.

Wasted compute is the small cost -- a per-client forward pass per round across ~1,700 planned
cells. The real cost is that **it is why the GPU bug survived three sessions.** The fields that
vanished were never in the result file, so a `completed` result with every client's evaluation
failing was byte-comparable to a healthy one. A field that is absent cannot be checked; one
recorded and empty can.

It also lost a capability the code went out of its way to build.
`per_class_confusion_counts` is additive across clients *precisely* so a correct pooled
macro-F1 can be recomputed under label skew -- the plan flags naive averaging of per-client
macro-F1 as wrong, and label skew is what R1/R2 measure. Those counts were computed and
discarded. `merge_evaluate_metrics` now folds them in under `client_eval_`; a real run confirms
19 fields land in the result file, per-class counts included.

### 3. CI asserted `status == "completed"` and nothing else

Which is exactly the assertion the GPU bug satisfied. `scripts/check_smoke_result.py` now
checks that a result is *complete* rather than merely finished -- client-side evaluation
present, `train_server_round` counting from 1, per-round records existing, a populated `final`
-- and CI runs the smoke a **second time** and requires the two to agree exactly.

That second run is the part no unit test can replace. The unseeded-client bug needs two real
processes to show, which CI has and a developer's `pytest` does not. Comparison is bit-for-bit,
not a tolerance band: same machine, same build, same seed, so there is no drift to absorb and a
tolerance would hide the one failure it exists for -- the shipped reproducibility band was 0.15
wide while the variance it was meant to absorb was 0.19.

Each check was verified against a mutated copy of the real result: the device bug, the -1 round
number, a run that aggregated nothing, and two runs differing at the same seed are all caught,
and a healthy result passes.

**Fifth instance of the same defect class, and the one that explains the others**: a subsystem
whose output goes nowhere cannot be checked, so every bug inside it is invisible by
construction.

572 tests pass, ruff clean.

## 2026-09-24 (the one that mattered) -- the fitness question, reproduced and answered locally

Three Kaggle sessions were spent establishing that the data-free fitness has no usable optimum
on heterogeneous data. Each one cost a GPU session to obtain one row of numbers. That was
avoidable, and the reason it was not avoided is worth stating plainly: the repo's notes say the
FL harness cannot run locally, and that is true of the **Intel-macOS dev machine** and false of
every Linux container this project has used. `ray` and `flwr.simulation` were installed here the
whole time.

The missing piece was never compute. It was a fixture heterogeneous enough to fail the same way.
`ci_smoke_dataset.py` cannot: 40 images, labels alternating by parity, pixel content unrelated
to the label -- nothing there produces client disagreement, and a FedACO run on it reports
`fedavg_fitness = +0.86` and a negative corner margin, which is the opposite of the regime under
investigation.

`scripts/synthetic_heterogeneous_dataset.py` gives each class a distinct low-frequency pattern,
so a client holding {0,1} has a genuinely different local optimum from one holding {2,3} and its
update points elsewhere. Difficulty is tunable, and it had to be: the first attempt
(amplitude 0.35 / noise 0.18) was trivially separable, reached final macro-F1 **1.0000**, and
reproduced the corner problem but not the zero-centred-fitness problem -- a fixture easier than
the task it stands in for reproduces only the failures that do not depend on difficulty, and
there is no way to know in advance which those are.

At amplitude 0.15 / noise 0.45, K=10, dirichlet(0.3), 15 rounds, gamma_3=0.1, **about a minute
per arm**:

| | weighted_mean (local) | real Kaggle | aggregate (local) |
|---|---|---|---|
| F at the FedAvg point, mean | **-0.0033** | -0.0005 / -0.0212 | **+0.8141** |
| negative in | 5/15 rounds | 7/15 rounds | **0/15** |
| corner_margin, mean | **+0.6185** | +0.3136 / **+0.6184** | **-0.5283** |
| corner wins in | **15/15** | **15/15** | **0/15** |
| rounds that deposited nothing | 3/15 | 1-7/15 | **0/15** |
| final test macro-F1 | 0.1000 | 0.1185 - 0.1381 | 0.1168 |

The left column reproduces the real failure on every axis, including a corner margin that agrees
with the Kaggle run to three decimals -- coincidence in the last digit, but the signature matches
throughout. The right column shows `aggregate` removing both halves: the corner never wins, and
F at the reference stops being zero-centred noise, so every round deposits. `alpha_max` is 0.207
against uniform's 0.100, so the colony still has room -- this is not the gamma_3=0.6 failure
where the landscape was repaired by flattening it.

`make fitness-repro` runs all three modes and the health check on each.

**What this does and does not establish.** A synthetic *signature* match is much weaker than a
data match: it says the fixture fails the same way, not that the fix transfers. 15 rounds is far
too short to read the macro-F1 column. The real gate is still the test that counts -- but it is
now a confirmation rather than a discovery, and every further fitness change can be screened
here for a minute instead of a GPU session.

**The cost of the wrong belief.** Five GPU sessions, three of them spent obtaining numbers this
box produces in three minutes. The belief was written down, inherited, and never retested after
the platform changed.

572 tests pass, ruff clean.

## 2026-09-24 -- and the incomplete-cell warning would have fired on every correct sweep

One commit after adding the LaTeX dagger for cells built from too few seeds, none of the three
`make_tables.py` Makefile targets passed `--expected-seeds` at all -- so all three used the
script's default of 8.

`main.yaml` runs 8 seeds, so `make tables` was right by accident. The ablation and robustness
families run **5 by design** -- a degradation curve carries its meaning through effect sizes,
and the 8-seed floor exists only where a p-value is claimed (main.yaml's headline comparison
and ablation_a1's control). So `make tables-ablation` and `make tables-robustness` would have
daggered every cell of a correctly completed sweep and printed "Do not quote these numbers"
beneath it.

That is 855 of C's ~1,015 cells. A warning that fires on almost everything is wallpaper, which
is the failure I wrote a test against in the same commit that created the condition for it.

Fixed per target, and pinned by a test that reads the seed counts **out of the configs** rather
than from a remembered number -- these counts have already changed twice (5 -> 8 for the two
families a p-value comes from, then back to 5 for the rest), and a hardcoded assertion would
have been the third thing to drift.

The test also checks something the fix cannot: that every config writing to one results
directory agrees on its seed count. `results/fl/ablation` is the exception -- A2-A9 write 5
seeds there and ablation_a1 writes 8 -- so one `--expected-seeds` cannot be right for both, and
the target takes 5 with the asymmetry written down rather than silently flagging 585 cells to
avoid mis-marking 80.

573 tests pass, ruff clean.

## 2026-09-24 -- lost ten minutes to the hazard I had just written a guard for

While screening A1 locally, a run sat at round 1 for three and a half minutes with the machine
**idle** -- load average 0.29, ClientApp actors at 2% CPU -- when identical runs finish 15
rounds in 39 seconds. Four ClientApp actors existed; the run asked for `min-available-nodes=10`.

That is Flower's `while len(grid.get_node_ids()) < min_available_nodes: sleep(1)`, the
unbounded wait I documented in `fl/cold_start.py` two commits earlier and guarded against for
R6 specifically. It is not specific to R6 at all: **any** run whose node requirements exceed
the configured supernode count hangs forever, and `flwr run` never errors.

The script I wrote for the A1 screen did not re-issue `simulation-config`, inheriting whatever
the previous experiment had left. That is exactly the shape of B's gate cell, which relies on
section 3 having run earlier in the session -- and a stale kernel, a restarted SuperLink or a
Run-All from the middle all leave it wrong. On Kaggle it costs nine hours and produces nothing.

`flwr federation simulation-config` can only SET the count, never read it, so no preflight can
check it after the fact. The only protection is setting it immediately before each run that
depends on it, which B's gate cell now does. The sweeps were already covered --
`run_sweep.py` and `run_sweep_granular.py` both call `configure_federation` and refuse if it
fails. C's notebook uses only those runners, so it has no bare `flwr run` to fix.

Worth recording as a pattern, not just an incident: I documented this failure mode, wrote a
guard for one narrow case of it, and then walked into the general case within the hour. The
guard was scoped to where I was looking rather than to where the hazard was.

## 2026-09-24 -- A1 screened locally: ACO loses to every control, and we know why

The plan's make-or-break experiment for claim C2, run on the local fixture instead of 17-33
Kaggle GPU-hours. 5 search methods x 3 seeds, equal budget, `aggregate` fitness.

**ACO gains +0.0013 over the FedAvg point. coordinate_grid gains +0.0501, pso +0.0484, ga
+0.0312, random +0.0304. ACO beats every control on 0 of 3 seeds.** The plan's feared outcome
was "ACO ties random search"; this is a loss to random by 23x.

### The mechanism, which makes it fixable rather than fatal

`eta_{k,l} = (1 + |lambda_l - d_hat_k|)^-1` with `d_hat_k` spreading the per-client score `d_k`
over the level range. Across K in {10,20} x noise in {1.5,3,6}, **`d_k` spans ~0.04** against a
level spacing of **0.25** -- so every client's argmax lands on the same level (identical in 3 of
6 configurations, adjacent in the rest).

And a uniform level assignment normalises back to `base_weights` exactly. So the greedy branch
of the ACS rule constructs **the FedAvg point**, 70% of the time at `q0=0.7`. The colony is
anchored to the reference it is supposed to improve on: best candidate +0.9559 against
F(FedAvg) +0.9558.

The controls have no such anchor and explore freely. That is the entire result -- not a
statement about ant colony optimisation, but about a heuristic with 16% of a level's worth of
dynamic range driving a rule that is greedy 70% of the time.

### A fix I implemented and reverted

The colony visits 82 distinct alphas per 210 evaluations; random search visits 210 of 210. So
61% of the colony's budget buys answers it already has, and A1 was partly measuring which
method repeats itself least. Memoizing within a round is exact -- fixed Gram matrix, so a
repeated alpha has the same fitness by construction.

It changes nothing. `run_colony` stops on its own ant/iteration schedule rather than on
`remaining()`, so the freed budget is never spent: best fitness was **identical** across all 8
configurations tested. It also broke four tests encoding "no method outspends the shared
budget" and silently changed what the logged `evaluations_used` means. Reverted.

Recording it because the temptation was to keep it -- it was written, it was correct, and it
looked like progress. A change that breaks a real invariant's tests and improves nothing is
churn regardless of who wrote it.

### What the team has to decide

Candidate heuristic fixes -- standardising `d_k` across clients before rescaling, widening the
sigmoid, lowering `q0`, or letting the colony spend its freed budget -- are changes to the
method §4.5 specifies, not bug fixes. Any of them can now be screened locally in minutes rather
than a GPU session. That choice is not mine to make.

573 tests pass, ruff clean.

## 2026-09-24 -- the isolated colony probe cannot screen A1's fixes, and saying so is the finding

Having diagnosed why ACO loses A1 (the greedy branch constructs the FedAvg point), the obvious
next step was to screen candidate fixes cheaply with an isolated colony probe -- synthetic Gram
matrix, one round, no FL -- rather than 12 more real runs.

**It does not work, and the reason is worth recording.** In the probe, over 6 (K, noise) cases
x 3 seeds at budget 300:

    control: coordinate_grid    +0.0000 gain over F(base)
    control: random             -0.0243
    aco q0=0.9                  +0.0002   (beats grid on 15/18)

Nothing finds anything better than the FedAvg point. But the real A1 run has
coordinate_grid at **+0.0501** and pso at **+0.0484** -- the probe does not reproduce the
phenomenon it was built to screen. Three differences, and any of them is enough: uniform base
weights where the real runs use num-examples weights under dirichlet(0.3) skew; synthetic
consensus deltas where the real ones come from actual training; and one isolated round where the
real runs carry pheromone across fifteen.

On that landscape the FedAvg point is already near-optimal, so every method's gain is ~0 by
construction and the probe ranks noise. It also reported "standardizing d_k makes things worse"
(-0.05) -- a conclusion with no support, on a fixture that cannot express the effect.

**The general lesson, third time this session:** a cheap proxy has to be validated against the
phenomenon before its rankings mean anything. The heterogeneous *dataset* fixture earned trust
by reproducing the real failure signature on every axis first. This probe skipped that step, and
its numbers looked precise enough to act on.

Screening the candidate fixes therefore needs real runs. `aco-q0` is already a config key so it
needs no code change; standardizing `d_k` would need a flag, and is only worth adding if q0
alone does not account for the anchoring.

## 2026-09-24 (later) -- A1 re-screened at the plan's real defaults: the ranking does not move

The defaults reconciliation (see `OPEN_QUESTIONS.md`, "four sources of truth") meant every local
screen so far had run at `q0=0.9`, `gamma_2=1.0`, `rho_round=0.1` instead of plan §14's
0.70 / 0.50 / 0.30 -- verified from the embedded `config.run_config`, not inferred. Since `q0`
was the axis the anchoring diagnosis rests on, A1 had to be re-screened before any of it could be
believed. 15 runs, same fixture, same equal-budget harness, only the three corrected defaults:

| method | mean gain over the FedAvg point | per-seed | macro-F1 |
|---|---|---|---|
| **aco** | **+0.0066** | `[+0.0088, +0.0025, +0.0085]` | 0.2700 |
| random | +0.0231 | `[+0.0256, +0.0189, +0.0249]` | 0.1000 |
| ga | +0.0295 | `[+0.0341, +0.0233, +0.0311]` | 0.1670 |
| pso | +0.0355 | `[+0.0402, +0.0286, +0.0377]` | 0.2248 |
| **coordinate_grid** | **+0.0390** | `[+0.0410, +0.0350, +0.0412]` | 0.1000 |

**ACO beats every control on 0 of 3 seeds, and loses to the best by 5.9x.** The ordering is
identical to the drifted screen -- coordinate_grid > pso > ga > random > aco -- and ACO's
per-seed spread (0.0025-0.0088) does not overlap any control's.

Every method's absolute gain fell (e.g. coordinate_grid +0.0501 -> +0.0390), which is expected
and not a result: `gamma_2` halved, so F is on a different scale. **Only the within-screen
ranking transfers between the two screens**, and it is unchanged.

So the config bug was real, worth fixing, and **not the cause of A1's failure**. ACO went from
+0.0013 to +0.0066 -- a 5x improvement on its own terms, still 5.9x short of the weakest thing
it needs to beat. Combined with the `q0` sweep, which reached +0.0287 at `q0=0.0` and still lost
to every control: greedy anchoring is a real contributing mechanism and fixing it does not
rescue the claim.

**What this means for plan §11's week-5 gate.** The plan's stated failure condition was "ACO
ties random search at equal budget -- stop and reframe". Three independent local screens now say
it does not tie random search, it loses to it, at both the drifted and the documented defaults.
The gate's instruction was to reframe before spending weeks 6-12, and roughly 80% of the
project's remaining 351-701 GPU-hours sits behind it.

Confirming this on the real gate costs A1's 80 cells (17-33 GPU-h). The local fixture has
reproduced the real runs' fitness signature on every axis checked so far, but it is a fixture,
and a claim this consequential should not rest on it alone.

606 tests pass, ruff clean.

## 2026-09-24 (later still) -- claim C3 holds, but not for the reason it claims

`overhead.yaml` goes to K=200, which needs 200 Ray actors and the real dataset, so it cannot
run here. But C3 is about *server-side* work, and that is directly measurable:
`scripts/bench_aggregation_overhead.py` times `precompute_gram` plus `run_colony` on K x d
delta tensors at d=390,404 (simple_cnn's state dict, image-size independent).

**Why this proxy is trustworthy where the isolated colony probe was not.** The colony probe
tried to *rank search methods by fitness* and failed because its synthetic landscape did not
reproduce the real one. This measures the wall-clock of a deterministic computation on arrays
of a given shape, and `deltas @ deltas.T` does identical arithmetic whether the deltas came
from training or a generator. It also runs on the right device: `precompute_gram` builds its
output with `device=deltas.device`, the deltas arrive as Flower message arrays, and FedACO's
`device` argument is used only by `fitness_mode="server_val"` -- so aggregation is on CPU in
real runs too. And it is anchored: the project's one real measurement is 198 ms at K=10 on a
T4 box; this benchmark gives 121-131 ms, a ratio of 0.61-0.66x on a different host CPU.

### The claim's substance holds, and improves with K

| K | overhead (gram + colony) | share of a round |
|---|---|---|
| 5 | 117 ms | 3.13% |
| 10 | 131 ms | 1.74% |
| 20 | 204 ms | 1.36% |
| 50 | 434 ms | 1.16% |
| 100 | 757 ms | 1.01% |
| 200 | 1467 ms | 0.98% |

(Round cost extrapolated as 7500 ms x K/10 from the measured K=10 point -- an assumption, so
the percentages are softer than the overhead numbers, which are measured.)

### The O(K^2) shape does not appear

| K | gram ms | ms/K | ms/K^2 | implied GFLOP/s |
|---|---|---|---|---|
| 5 | 15.8 | 3.155 | 0.6310 | 1.2 |
| 20 | 86.7 | 4.333 | 0.2167 | 3.6 |
| 100 | 593.4 | 5.934 | 0.0593 | 13.2 |
| 200 | 1355.8 | 6.779 | 0.0339 | 23.0 |
| 400 | 3624.3 | 9.061 | 0.0227 | 34.5 |
| 800 | 9442.0 | 11.803 | 0.0148 | 52.9 |

`ms/K^2` falls **43x** across the range while `ms/K` rises only 3.7x, and the implied GFLOP/s
climbs monotonically from 1.2 to 52.9 -- the arithmetic is never the limit. Least squares:
**linear R^2 = 0.9722, quadratic R^2 = 0.9689.** Effectively tied, because both approximate a
curve that sits between them and closer to linear.

The reason is memory traffic. `deltas @ deltas.T` is O(K^2 d) arithmetic but only O(Kd) bytes
read, and at K=200 that matrix is 312 MB. The build is bandwidth-bound long before it is
compute-bound, and it has still not crossed over at K=800 (1.25 GB).

The colony is separately flat: 90-188 ms across K=5-200, a 1.3x rise for a 40x rise in K. The
Gram trick makes each fitness evaluation O(K), so the fixed 30x10 budget should give O(A*I*K) --
but at these sizes the per-evaluation Python loop dominates the vector work entirely.

### What this means for the paper

O(K^2) is still the right *asymptotic* statement, and C3's honest claim -- negligible overhead
-- holds more strongly than the plan expected. But plan §9.2's figure 5 asks for a fitted c*K^2
curve overlaid on the measurements, which `figures.plot_overhead_vs_k` implements, and that
overlay would assert a shape the data does not show: a reader who plots `ms/K^2` sees it fall
43x. The function's docstring now records the measurement and the options; which fit the figure
shows is a presentation decision, so it is documented rather than changed unilaterally.

Confirming the constant on the real hardware is still worth `overhead.yaml`'s 7 cells (~0.1
GPU-h) -- the cheapest sweep in the project.

637 tests pass, ruff clean.

## 2026-09-24 (end of day) -- running the Phase 9 chain for the first time found four faults

Plan §9.3's acceptance criterion is "`make figures tables` regenerates every artifact from
`results/` with no manual steps. Deleting `paper/figures/` and `paper/tables/` and re-running
restores them byte-for-byte identical." Nothing had ever run it. An 8-cell local sweep
(fedavg/fedaco x dirichlet(0.3)/iid x 2 seeds, 6 rounds on the synthetic fixture) put real
result files through `run_sweep_granular.py -> aggregate_results -> make_tables -> make_figures`
and every one of the following needed the whole chain to be visible. None is catchable by a
unit test on a fixture.

### 1. `_variant_of` compared each knob against a hardcoded default

The worst of the four, and self-inflicted: reconciling `aco-gamma-dispersion` to plan §14's
0.50 made **every** run read as a `g2=0.5` variant -- FedAvg included, because pyproject
supplies the key to every resolved config. `figure_convergence` and `figure_comparison`, the
paper's two headline figures, filter to `variant == "default"`, so **both silently skipped
with a complete set of valid results present**, and `add_deltas` (which also keys on
"default") left every comparison column empty. A correct change to a value, made in the right
place, broke the two figures that matter most, and the only symptom was a "skipped (no data)"
line that looks exactly like a sweep that has not run yet.

`_variant_of` now reads `pyproject.toml` instead of literals, and covers the keys added since
it was written (`aco-q0`, `aco-gamma-entropy`, `aco-desirability-scaling`,
`aco-dispersion-reference`). Five tests, including one that moves a default and asserts the
label follows.

### 2. `aggregate_results.py` accepted names that matched nothing

`--partitions totally_bogus --baselines nonexistent` printed "(no overlapping-seed comparisons
found)", wrote both CSVs and exited **0**. This one is not cosmetic: **Holm-Bonferroni corrects
each p-value by the size of the comparison family**, so a name that matches nothing shrinks the
family and every surviving p-value is corrected *less* aggressively -- results come out looking
**more** significant. Plan §9.1's warning ("with 5 seeds and a dozen comparisons, uncorrected
p-values manufacture significance; a reviewer who checks will find it") arriving through a typo.

The label spelling invites it: the config says `regime: dirichlet`, the analysis frame says
`dirichlet_0.3`, so `--partitions dirichlet` matches nothing. Now fatal, listing what is
present.

### 3. The granular runner checked a directory it never told the run about

`run_sweep` tested `is_completed(run_id, output_dir)` against its `--output-dir` argument while
`execute_run` never put `output-dir` in the run_config. With both at the default `results/fl`
they coincided; pass anything else -- and `run_sweep.py`'s own docstring says "`output-dir` is
routinely outside: a Colab run writes to Drive, a Kaggle run to ..." -- and every cell records
`status: "unknown"` (the status this module reserves for `flwr run` exiting 0 while the round
died), **and resume stops working**: a sweep restarted after a session timeout silently re-runs
every cell it had already finished. All 8 cells here recorded "unknown" with the results sitting
in `results/fl`. `resolve_output_dir` now returns one directory used for both sides.

### 4. `partition_stats` proved itself

Not a fault -- the fix from earlier today, confirmed on real runs: js_divergence 0.4238 for
dirichlet(0.3) against 0.0238 for iid, the right ordering and the right magnitudes, and
`gain_vs_heterogeneity.png` rendered for the first time.

### The criterion now holds, and is a command

    make verify-phase9

10 artifacts (3 tables, 5 figures, 2 CSVs) generated twice and compared byte-for-byte: all
identical. Byte-for-byte is the right bar, not a tolerance -- both passes read the same JSON
with the same code on one machine, so any difference is non-determinism in the artifact
pipeline itself (an unsorted glob, dict iteration order, an embedded timestamp), and each of
those makes "did this number change?" unanswerable across a re-run.

### 5. The robustness tables had no delta column at all

Found while checking that `_variant_of` labels every declared sweep cell correctly. It does --
but that surfaced a pre-existing consequence: `robustness_r1/r2/r3` set an attack in
`base_overrides` and `main_client_scale.yaml` sets `fraction-train: 0.3`, so **every** row in
those sweeps carries that mark and none is "default". `add_deltas` keyed its baseline on
`variant == "default"`, found nothing, and left `delta = None` on every row -- so four sweeps
printed an empty comparison column, which renders as the same "—" as a cell with no paired
seeds. Nothing distinguishes "no baseline existed" from "no seeds overlapped".

`add_deltas` now matches the baseline within the same variant first and falls back to the
default variant. Within-variant is also the comparison those sweeps intend: FedACO against
FedAvg **at the same attack level**. Matched against an unattacked control instead, a FedACO
row at 30% sign-flip would read -0.35 rather than +0.05 -- reporting the attack's damage as
the method's deficit. Four tests, including that one.

662 tests pass, ruff clean.

## 2026-09-24 (end) -- every arm of every sweep C owns executes; 96/96, nothing failed

`strategy_from_run_config` accepting a config is necessary and not sufficient: it builds the
strategy, not the round. `scripts/preflight_sweep_configs.py` runs one cell of every declared
arm at 2 rounds, K=6, image-size 32 against the synthetic fixture, so the strategy, attack,
cold-start and aggregation paths all execute with real Flower messages. Judged by whether a
result JSON appeared, not by the exit code, since `flwr run` exits 0 when the simulation dies.

    96 arms, 0 failed

    robustness_r1_label_flip         4    ablation_a1    5    ablation_a6      27
    robustness_r2_update_attack      4    ablation_a2    3    ablation_a7       4
    robustness_r3_stragglers         2    ablation_a3    2    ablation_a8       2
    robustness_r4_dp_noise           2    ablation_a4    4    ablation_a9       4
    robustness_r5_client_scaling     4    ablation_a5    5    ablation_all      4
    robustness_r6_cold_start         5    robustness    15

**R6's five arms matter most**: it is the newest config, written today, and its failure mode is
a silent infinite hang rather than an error -- Flower's `sample_nodes` waits in a
`while ...: sleep(1)` loop that never gives up, so a cold start hiding one node too many
consumes a whole Kaggle session and writes nothing. That path now has a real run behind it.

A6's 27 arms all executing is the second most useful result: at 200 distinct cells and 42-83
GPU-h it is the largest single sweep in the project, and a config error in arm 27 would have
been found three-quarters of the way through a session.

This says the code paths run. It says **nothing** about whether the numbers mean anything --
rounds, K and image size are all cut, and the fixture is synthetic. It is 70 minutes of CPU
standing between C and 214-427 GPU-hours.

### Two bugs in the preflight itself, both found by running it

* It globbed only patterns containing `*`, so `configs/experiment/ablation_a[0-9].yaml` -- the
  natural way to name A1-A9, and the pattern in the Makefile target -- was treated as a literal
  filename and died on FileNotFoundError after 36 arms had passed. Now globs on any
  metacharacter, refuses a pattern matching nothing, and checks `flwr` is on PATH before
  reconfiguring the federation.
* No resume. This container reaped the detached process three times; each restart redid every
  arm already checked, so the walk never reached the ablations. An arm that wrote a result is
  an arm that ran -- the same test `sweep.is_completed` applies per cell.

670 tests pass, ruff clean.

## 2026-09-25 -- the project resized to 9 days, and two bugs in yesterday's preflight

The stated deadline moved to 7-9 days. The recorded remaining cost was **351-701 GPU-hours**
(B 139-279, C 214-427), which at Kaggle's ~30 GPU-h per week per account is 12-23 weeks on one
account and 4-8 across three. No ordering of that fits nine days.

**The binding constraint was never the compute.** 351-701 hours was the price of proving claim
C2, and three independent local screens say C2 is false: ACO gains +0.0013, then +0.0066 after
a config-scoping fix, against random search's +0.0304, coordinate grid's +0.0501, PSO's +0.0484
and a GA's +0.0312 -- beating every control on **0 of 3 seeds**. Roughly 80% of the remaining
hours exist only to support the framing that is failing. Sized to the claim the evidence can
actually carry, the experimental core is ~90 GPU-h.

### The resized set: 254 cells for C, 80 + 8 for B

| config | cells | GPU-h | replaces | was |
|---|---|---|---|---|
| `main_reduced` | 144 | 30.0 | `main.yaml` (576) | 120-240 |
| `robustness_r1_reduced` | 40 | 8.3 | `robustness_r1_label_flip` (60) | 25-50 |
| `robustness_r2_reduced` | 40 | 8.3 | `robustness_r2_update_attack` (120) | 50-100 |
| `ablation_a2_reduced` | 30 | 6.2 | `ablation_a2` (60) | 25-50 |
| **C total** | **254** | **52.9** | | 214-427 |
| `gate_fitness` | 8 | 0.25 | new | -- |
| `ablation_a1_reduced` | 80 | 16.7 | `ablation_a1` (80) | 17-33 |

Every cut is named in the config header it belongs to, so the paper states them rather than a
reviewer finding them. The cuts: K=20 -> 10 (the one size this project can *quote* rather than
extrapolate, since 7.5s/round was measured at K=10), 2 local epochs -> 1, six regimes -> three
(a monotone heterogeneity ladder, which is what figure 9 and claim C1 need), twelve strategies
-> six, R1/R2's middle attacker fractions, and A6/A4/A5/A7/A8/R3-R6 entirely.

**A1's cell count is unchanged at 80, deliberately.** Its five search methods, two partitions
and eight seeds *are* the experiment; the saving comes from K and local epochs alone. 8 seeds
is load-bearing here more than anywhere: at 5 the signed-rank floor of 0.0625 makes
"inconclusive, sample too small" arithmetically certain, and that is the one outcome this
comparison cannot afford to report.

**What the trim gained rather than cost.** `main.yaml` ran 1 local epoch; R1/R2/A2/A1 ran 2.
So no pairing of the full configs was a controlled comparison -- "FedACO under attack" and
"FedACO clean" differed in local training as well as in the attack, and A1's rows were not
comparable to the main table's at all. All six reduced configs now agree with `main_reduced` on
K, local epochs, rounds, image size, lr, batch size and model, pinned by a test.

**One coupling the trim introduced.** `robustness_r2_reduced` ships with no clean arm and uses
the one `robustness_r1_reduced` writes to the same `results/fl/robustness`. This works because
`attack: none` and `fraction-train: 1.0` are already pyproject defaults, so that arm resolves
to the default variant -- which is what `add_deltas` falls back to. Run r1 before r2 and do not
trim r1's clean arm, or every R2 delta is None, which renders as the same "—" as a cell with no
paired seeds. Pinned by four tests.

### Bug 1: the preflight ran one partition per config and called the rest ok

`arms()` paired every strategy with `partitions[0]` and no other partition. Any partition axis
carrying a *code path* was therefore executed for its first value only, while the report printed
"ok" for the whole config. **Yesterday's "96 of 96 arms execute" was 96 of a possible 137.**

Never executed, across a month of this script's existence: `robustness_r2_update_attack`'s
`sign_flip` arms -- and `gaussian` inflates the update norm where `sign_flip` preserves it and
reverses the direction, two different paths through `aco/heuristics.py`, so the norm-ratio
heuristic was preflighted against half of what it claims to detect. Also R4's higher DP sigmas,
R3's straggler fractions, and every regime but the first in `robustness.yaml` and `ablation_all`.

Found while writing `robustness_r1_reduced.yaml`, whose natural partition order puts `clean`
first: label-flipping would not have executed once, and the report would have read 4/4 ok for a
config whose entire purpose is the attack.

Now every strategy still runs against `partitions[0]`, and each remaining partition runs once
against one strategy -- **linear in partitions, not multiplicative**, so A6 goes 27 arms to 28
rather than 27xP and the walk stays CPU-minutes. The probe is FedACO where the config has it,
because pairing an attacked partition with `fedavg` would execute the attack and skip the code
that reads it. `robustness_r1_reduced` declares the attacked partition first so all four
strategies meet it -- Krum derives its `f` from the cell's own `attack-fraction`, and a
mis-derivation is invisible with no attackers present.

### Bug 2: the resume key ignored the config, so changed arms reported "already ran"

The resume added *yesterday* keyed on `config_stem + arm_label`. Neither carries what the arm
resolves to. So reordering `robustness_r1_reduced.yaml`'s partitions -- which keeps every
strategy label identical -- produced, within the hour:

    skip    fedavg             (already ran)
    skip    krum               (already ran)
    skip    trimmed_mean       (already ran)
    skip    fedaco             (already ran)

from results computed with `attack: none`. Label-flipping had not run once and the report read
4/4 ok. This is the September sweep bug -- a control handed a sibling's result file -- reproduced
inside the mechanism added to prevent lost work, one day later.

The key now includes an 8-character hash of the resolved config via the same `make_run_id` the
real runners use; the two orderings fingerprint `389b6bb4` and `e9db62e6`. Two tests: that the
same label under two partitions gets different keys, and that no two arms of a config ever
collide (checked on A6, whose axes deliberately share a centre value).

**Both bugs are the same shape, and it is the shape this repository keeps producing: a check
that runs, passes, and means nothing.** The count is now eleven instances. What is new here is
that one of them was introduced by me yesterday and caught today only because an unrelated
config edit happened to expose it -- a resume that silently accepts stale work fails in exactly
the direction that looks like success.

### The 137-arm result: every arm of every sweep in the project executes

    137 arms ran, 0 failed

    robustness_r1_reduced      5    ablation_a1        6    ablation_a6      28
    robustness_r2_reduced      5    ablation_a2        6    ablation_a7       5
    ablation_a2_reduced        4    ablation_a3        3    ablation_a8       3
    robustness_r1_label_flip   6    ablation_a4        6    ablation_a9       5
    robustness_r2_update       9    ablation_a5        7    ablation_all      6
    robustness_r3_stragglers   4    robustness        15
    robustness_r4_dp_noise     5
    robustness_r5_scaling      4
    robustness_r6_cold_start   5

41 of those 137 had never run before today, because they are the partitions the old
`parts[0]` rule skipped. The newly covered ones that matter: `robustness_r2_update`'s
`sign_flip_10/20/30pct` (the norm-ratio heuristic's second failure mode), R4's higher DP
sigmas, R3's larger straggler fractions, and the non-first regimes of `robustness.yaml`,
`ablation_all` and every A-config.

**A third reporting bug, found in this run's own output.** The summary printed "147 arms ran"
against 137 distinct arms. `robustness_r1_reduced` and `robustness_r2_reduced` were named
explicitly *and* matched again by the `robustness_r*.yaml` glob later in the same argument
list -- the natural way to say "these two first, then the rest". The second walk was
correctly skipped by resume, but skips count toward `passed`, so the headline overstated
coverage by ten. Path resolution is now an order-preserving dedupe in its own tested
function. A pass count that overstates coverage is this script's own failure mode, which
makes it the third instance today of the same shape.

**What this does and does not establish**, restated because the number is easy to misread:
2 rounds, K=6, image-size 32, synthetic fixture. It says every declared arm's strategy,
attack, cold-start, persistence and aggregation path executes with real Flower messages. It
says nothing about whether any number means anything. It is ~100 minutes of CPU standing in
front of ~70 GPU-hours.

### The 9-day set itself: 18 arms, 0 failed

The 137-arm walk covered the full-size configs. The six the 9-day plan actually runs had
verified *cell counts* and nothing else -- `main_reduced`, `gate_fitness` and
`ablation_a1_reduced` had never executed at all.

    18 arms ran, 0 failed

    gate_fitness           4    default, gamma_entropy_0.45, dispersion_aggregate, fedavg
    ablation_a1_reduced    6    aco, random, coordinate_grid, pso, ga, + dirichlet_0.3
    main_reduced           8    fedavg, fedprox, krum, trimmed-mean, fedlaw, fedaco, + 2 regimes

**Both candidate fitness fixes execute**, which is the result worth having here: `gamma_3 =
0.45` and `aco-dispersion-reference: aggregate` are the two things `gate_fitness.yaml` exists
to compare, and a config error in either would have been found on a Kaggle GPU with the gate's
whole purpose being to be cheap. `main_reduced`'s `fedlaw` arm also runs -- it is the closest
prior work and the one baseline whose absence would leave the contribution uncontested.

155 arms across every sweep config in the project, 0 failed. Still 2 rounds, K=6,
image-size 32, synthetic fixture: the code paths run, the numbers mean nothing.

### Where this leaves the project

Everything that can be built without a GPU or the real images is built. What remains:

| | work | GPU-h | owner |
|---|---|---|---|
| 1 | **The framing decision** -- diagnostic paper or reframe | 0 | the team |
| 2 | **The dataset-variant decision** (§"DECISION NEEDED", open since 2026-09-14) | 0 | the team |
| 3 | `gate-fitness`, then `a1-reduced` | 17 | B |
| 4 | `main-reduced`, `r1-reduced`, `r2-reduced`, `a2-reduced` | 53 | C |
| 5 | Phase 9 on the first real results | 0 | A |
| 6 | **The paper** -- `paper/` holds only ALGORITHM.md; 4 of §13's 8 sections have no file | 0 | unassigned |

Items 1, 2 and 6 need no compute and none of them is assigned. Item 6 is the one to worry
about: 70 GPU-hours is a config change, and eight pages with two related-work sections is
nine days of somebody's attention.

## 2026-09-25 (later) -- the paper, and the dataset question closed by its own diagnostic

`paper/` held `ALGORITHM.md` and nothing else; four of plan §13's eight sections had no file and
no owner. The paper, not the 70 GPU-hours, is the binding constraint on a 9-day deadline -- a
config change buys the compute, and eight pages with two related-work sections does not.

Written: `01_INTRODUCTION`, `02_RELATED_WORK`, `04_EXPERIMENTAL_SETUP`, `06_LIMITATIONS`,
`07_REPRODUCIBILITY`, plus `00_ABSTRACT` and `05_RESULTS` as scaffolds that name the command
filling each slot, and a `README` mapping every section and figure to its producer. §4, §6 and
§7 are framing-independent by construction, so they stand whichever way A1 goes.

**Citations are the one thing that could not be finished.** The plan names FedAAW, FedLAW,
FedNolowe and DaWa with no bibliographic detail and this container has no literature access.
Eighteen `[CITE: ...]` markers say exactly what each one needs. A test asserts they still exist,
because silently deleting one ships an unsourced claim about prior work -- and inventing an
author or venue is the fastest possible desk reject.

### The dataset-variant question, open since 2026-09-14, closed by the diagnostic it asked for

That entry asked for "per-class redundancy rate (images per pseudo-patient, broken down by
class)" as the test of whether the archive's perfect balance was manufactured. Computed:

| class | images | pseudo-patients | images/pp | redundancy | de-dup share |
|---|---|---|---|---|---|
| glioma | 1,800 | 1,476 | 1.220 | 18.0% | 30.87% |
| meningioma | 1,800 | 1,403 | 1.283 | 22.1% | 29.34% |
| pituitary | 1,800 | 1,326 | 1.357 | 26.3% | 27.73% |
| **notumor** | 1,800 | **577** | **3.120** | **67.9%** | **12.07%** |

Imbalance ratio **1.00 raw, 2.56 de-duplicated**. `notumor` carries 2.3-2.6x the redundancy of
every tumour class -- not a gradient but one class holding nearly all of it.

**This closes the question in the useful direction.** The concern was that plan §6.3 justifies
macro-F1 because "the dataset is class-imbalanced", which is false of the archive. It is **true
of the data actually trained on**: 2.56:1 with `notumor` at 12.07%. The metric was never the
problem; the stated reason was attached to the archive instead of to the de-duplicated split.
Decision recorded as option (a) -- keep this variant, document it precisely, restate the reason.
Options (b) and (c) need Kaggle credentials this environment lacks against a blocked host, and
would cost the entire compute budget a second time.

`docs/IMPLEMENTATION_PLAN.md:559` now carries the correction inline, because a spec sentence
known to be false about the data in use is exactly the stale source-of-truth this project keeps
getting caught by.

### A prose section cannot be generated, so it gets a test instead

Plan §9.3 forbids hand-typing a number into the paper, and generated tables obey it. §4.1-4.3
are prose quoting measured counts from `manifest.csv` and `leakage_report.json`, and nothing
connected the two -- re-running the audit at a different pHash threshold would leave the paper
asserting the old figures with no test failing.

`tests/test_paper_numbers.py` re-derives every quoted dataset number from its source and asserts
the paper still states it, including the derived 28.2% leakage figure and the *comparative*
claims (notumor duplicated at >2x every tumour class; the de-duplicated data still materially
imbalanced) rather than only the values -- a value check passes while the argument built on it
stops holding. It also asserts `05_RESULTS.md` contains no four-decimal metric value, which is
the rule the whole file exists to protect.

### The framing decision, taken rather than left open

Recorded in full in `docs/OPEN_QUESTIONS.md`. The paper is written under framing A (the
degenerate optimum and the equal-budget loss as the findings) because three screens say C2 fails
and the work could not proceed without a choice. The alternative framing is written out at the
end of `paper/01_INTRODUCTION.md`, so if A1 reverses the result the switch is deleting one
section and pasting another. **It is a methodological call the team can overturn**; it was taken
because the writing was blocked on it, not because it belongs to the agent.

724 tests pass, ruff clean.

## 2026-09-25 (end) -- the novelty claim is refuted by the literature, and it corroborates our result

Resolving the paper's `[CITE]` markers against a live literature index turned up three papers that
change what this project can claim. Recorded in full with metadata in `paper/REFERENCES.md`.

**1. Swarm optimization of the aggregation weight vector is already published.** Plan §1.2 states
that swarm intelligence in FL sits at the systems layer and that "applying an ACO metaheuristic
directly to the aggregation weight vector ... is the open slot". False:

* **Adp-FL-PSO** (Srinivas et al., NMITCON 2025) -- "an enhanced Particle Swarm Optimization (PSO)
  algorithm is applied to **the server to compute the optimal aggregation weights**".
* **FedPSO** (Park et al., Sensors 2021, 119 citations) -- replaces FedAvg's weight aggregation
  with PSO, targeting communication cost.

**2. Gradient-based adaptive weights without proxy data is CVPR 2025 work.** **FedAWA** (Shi et
al., CVPR 2025, 49 citations) adapts aggregation weights from client update vectors, needs no
proxy dataset, and up-weights clients whose updates align with the global direction -- the same
signal as our alignment term. Our four stated differentiators (no server data, no convexity bound,
multi-dimensional signal, no policy training) are satisfied by FedAWA on the first two and
arguably the third.

**3. The refutation corroborates our own negative result.** The published swarm work on
aggregation weights uses **PSO**. A benchmark of nine swarm algorithms for FL client selection
(Khan et al., 2024) found Grey Wolf ahead of both ACO and PSO. Our A1 screens measure
**PSO +0.0484 against ACO +0.0013-0.0066** on identical fitness and budget. Three independent
signals, same direction: ACO is not the strong choice here.

This is the most useful thing that happened today. The project's framing was "we are first to
search alpha with a swarm method"; that was never true, and holding it would have been found by
any reviewer who reads CVPR. What survives is defensible and is what the paper now claims:

1. the first **equal-budget, equal-fitness** comparison across ACO, PSO, GA, coordinate grid and
   random search for aggregation-weight search -- which tells you *which* search matters, where
   the existing papers each validate one method against FedAvg;
2. `corner_margin` and its closed-form penalty threshold, which apply to **any** of these methods
   including the published PSO ones -- the degenerate single-client optimum is a property of the
   objective, not of ACO;
3. cross-round stigmergy. FedAWA, Adp-FL-PSO and FedPSO are all **stateless between rounds**.
   This is the one structural novelty left, and it rests entirely on ablation A2.

**A2 is therefore promoted from secondary ablation to the experiment carrying the contribution**,
and it is in C's 9-day set at 30 cells / ~6 GPU-h. `ablation_a2_reduced.yaml` was written this
morning for a different reason and turns out to be the most important cell block C runs.

Two actions recorded for the team, neither doable here: **Adp-FL-PSO should become a baseline**,
being the direct competitor rather than FedLAW; and one further finding worth using rather than
just citing -- FedLAW (Li et al., 2023, 146 citations) reports that aggregation weights need not
sum to 1 and that client *coherence* governs which clients matter, which is prior work on the same
signal as our alignment term and should be discussed in §2.1 rather than listed.

Also resolved: **Krum** (Blanchard et al., 2017, 3,028 citations), whose $O(n^2(d+\log n))$
complexity is worth stating beside our own $O(K^2 d)$ precompute -- the robust baseline has the
same quadratic client dependence we are asked to justify. And **Xie et al., 2019** ("Fall of
Empires"), which breaks Krum and coordinate-wise median with inner-product manipulation attacks:
our R2 `sign_flip` arm preserves the update norm and reverses direction, so it *is* an
inner-product manipulation, and §5.3 should cite it rather than present the difficulty as new.

Eleven markers remain open (FedNolowe, DaWa, FedProx, SCAFFOLD, the "FedACo" collision paper, the
dataset's originating publication, and the client-drift analysis). The search rate-limited. None
may be filled from memory.

724 tests pass, ruff clean.

## 2026-09-25 (handover) -- the three decisions confirmed, and A2 moved to the front

The project owner confirmed all three provisional decisions: **framing A**, **dataset option
(a)**, and the **342-cell / ~70 GPU-h scope**. They are no longer agent calls pending review;
`docs/OPEN_QUESTIONS.md` records both as CONFIRMED and the paper should be written as though they
hold rather than hedged. What can still move is the evidence -- A1 and A2 -- not the decisions.

**`make c-all` reordered so `a2-reduced` runs first.** This follows from the citation work rather
than from preference. FedAWA (CVPR 2025), Adp-FL-PSO and FedPSO all optimise aggregation weights,
and all three are stateless between rounds, so cross-round pheromone persistence is the only
structural novelty left in the project. A2 -- persistence none / decayed / full -- is the only
experiment that tests it, at 30 cells and ~6 GPU-h.

Spending 30 GPU-h on `main_reduced`'s headline table before knowing whether the one remaining
novelty exists is the wrong order. If A2 says persistence buys nothing, the paper's contribution
reduces to the equal-budget comparison and the `corner_margin` diagnostic, and the main table
becomes supporting evidence for a methods note rather than the centrepiece -- which changes what
is worth running next.

New order: `gate_fitness` (0.25 GPU-h, which fix) -> `a2_reduced` (6, is there a contribution)
-> `a1_reduced` (17, is the framing right) -> `main_reduced` (30) -> `r1` (8) -> `r2` (8).
The first three are 23 GPU-h and settle every open question the paper's structure depends on.

724 tests pass, ruff clean.
