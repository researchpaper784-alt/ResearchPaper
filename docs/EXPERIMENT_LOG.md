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
