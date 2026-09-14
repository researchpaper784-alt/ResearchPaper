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
