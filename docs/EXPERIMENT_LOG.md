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

### Consequences for Phase 1 (why this was decided first)

- Preprocess and **cache decoded images as `uint8` arrays at 112²**, not JPEG paths
  re-decoded per round. Decoding ~11,400 images per round through PIL costs an estimated
  45 s/round single-threaded and is likely the single largest hidden cost in the
  simulation. The full cached dataset at 112² is ~215 MB and fits in RAM.
- Keep a **224² cache path** available for the secondary ResNet-18 table. The cache
  builder must be parameterized by resolution, not hardcoded.
- **Manifest paths must be relative to a dataset root supplied by an environment
  variable** (`FEDSWARM_DATA_ROOT`), never absolute — the same committed `manifest.csv`
  has to resolve against both the local download and Kaggle's `/kaggle/input/...` mount.
