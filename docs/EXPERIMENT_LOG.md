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
