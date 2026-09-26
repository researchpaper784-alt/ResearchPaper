# 4. Experimental setup

Every number in this section is measured and reproducible from the repository. Sources are
named inline so a reviewer can check each one.

## 4.1 Dataset, and a balance that is not real

We use the four-class brain tumour MRI classification dataset (glioma, meningioma,
pituitary, no-tumour). **The archive obtained on 2026-09-14 is a rebalanced variant, not the
canonical release**, and the distinction matters enough to state before anything else.

| | archive as published | after pseudo-patient de-duplication |
|---|---|---|
| images / components | 7,200 | 4,755 pseudo-patients |
| glioma | 1,800 (25.00%) | 1,476 (30.87%) |
| meningioma | 1,800 (25.00%) | 1,403 (29.34%) |
| pituitary | 1,800 (25.00%) | 1,326 (27.73%) |
| no-tumour | 1,800 (25.00%) | **577 (12.07%)** |
| imbalance ratio (max/min) | **1.00** | **2.56** |

*Source: `data/processed/manifest.csv`, `data/processed/split_stats.json`.*

The published archive is *exactly* balanced — 1,800 images per class. That balance is an
artifact of duplication, and it is concentrated in one class:

| class | images per pseudo-patient | redundancy |
|---|---|---|
| glioma | 1.220 | 18.0% |
| meningioma | 1.283 | 22.1% |
| pituitary | 1.357 | 26.3% |
| **no-tumour** | **3.120** | **67.9%** |

Overall redundancy is 34.0%. No-tumour is duplicated at 2.3–2.6× the rate of every tumour
class — which is precisely what balancing this dataset by oversampling would produce. We
therefore report results on the **de-duplicated** distribution, which is imbalanced at 2.56:1,
and we note that any result reported on this archive without de-duplication is computed on
data that is one-third redundant with two-thirds of that redundancy inside a single class.

> **Consequence for the primary metric.** We report **macro-F1**. The usual justification —
> "the dataset is class-imbalanced" — is *false of the archive* and *true of the data actually
> trained on*. We state it in the second form. Macro-F1 weights the 12.07% no-tumour class
> equally with the 30.87% glioma class, which both matches the clinical asymmetry (a missed
> tumour and a false alarm are not equally costly) and refuses the accuracy inflation that a
> 2.56:1 prior permits.
>
> Numbers from this variant are **not directly comparable** to published results on the
> canonical imbalanced release (~7,023 images). We do not claim otherwise, and the pipeline is
> dataset-agnostic so the canonical release can be run as a robustness note.

## 4.2 De-duplication and leakage, at the pseudo-patient level

That this dataset leaks is **not our finding**. Image-level de-duplication of it has been
published at least three times — a 3,522-scan leakage-free benchmark [CITE: MTA-Swin, Lu et al.,
J. Medical Systems 2026], deduplicated splits of the same archive with a 4.17-point accuracy drop
[CITE: WICA-Net-M, Khan et al., Computers 2026], and a pHash before/after protocol [CITE:
Saifullah et al., ICTAI 2025] — against the broader background of slice-selection bias in this
dataset [CITE: Wallis et al., Medical Image Analysis 2022] and leakage-inflated accuracy in
medical imaging generally [CITE: Tampu et al., Scientific Data 2022]. We report our audit for
completeness and reproducibility, not as a contribution:

| finding | count |
|---|---|
| images scanned | 7,200 |
| unique SHA-256 | 7,013 |
| exact duplicate images | 187 |
| near-duplicate edges (pHash, threshold 5) | 5,893 |
| connected components → pseudo-patients | 4,755 |
| components with >1 image | 1,103 |
| largest component | 28 images |
| **components leaking across the original Training/Testing split** | **520** |
| **images leaking across the original split** | **2,030** |
| components spanning more than one class label | 25 |

*Source: `data/processed/leakage_report.json`.*

**2,030 of 7,200 images (28.2%) appear on both sides of the dataset's own published split.**

### What image-level de-duplication leaves behind

Two of those papers state the remaining problem themselves: *"patient-level leakage remains
unresolved because patient identifiers are unavailable."* [CITE: Khan et al. 2026; also Sabuj et
al. 2026] Image-level de-duplication — keep an image unless it is a near-duplicate of one
already kept — guarantees that no two retained images are near-duplicates. **It does not
guarantee they are unrelated.** Images $A$ and $C$ can both survive while each is a
near-duplicate of a dropped $B$; they are then split independently, and one can land in train
while the other lands in test.

We treat near-duplication as a graph and split on its **connected components**: every image
joined to another by any chain of near-duplicate edges belongs to one pseudo-patient, and a
pseudo-patient is never divided across train/val/test or across federated clients. The
difference is the transitive part of the graph, and on this dataset it is not small:

| | count |
|---|---|
| same-component image pairs | 7,182 |
| of which **not** direct near-duplicates (chain-linked only) | **1,289** |
| components that are not cliques | **235** of 1,103 |
| images a greedy image-level pass retains **that are chain-linked to another retained image** | **394 of 4,968 (7.9%)** |
| related pairs it then splits independently | **256** (range 241–275 over 5 orderings) |

*Source: `data/processed/component_vs_image_dedup.json`, produced by
`scripts/measure_component_vs_image_dedup.py` from the committed manifest's `phash` and
`sha256` columns, which exactly reproduces the committed partition. pHash threshold 5.*

Component-level splitting makes the last two rows zero by construction:
`n_components_spanning_splits` is 0 and `tests/test_partition.py::test_no_cross_split_leakage`
asserts it.

**What it costs, stated rather than hidden.** Chain-linked is not the same as same-patient. A
pHash chain can join genuinely different slices that merely look alike — the 25 components
spanning two class labels show chains *do* sometimes cross a boundary that the same patient could
not. Component-level splitting is therefore **conservative**: it may over-group, leaving 4,755
independent units where the greedy image-level pass reports ~4,968, and coarser units make
stratification harder. We accept that trade in the direction that cannot inflate a test score.

The 25 mixed-label components are not silently discarded: this dataset's SARTAJ portion has
documented glioma mislabeling, so these are either a too-loose near-duplicate threshold or
genuine label noise, and we report the count rather than resolving it by assumption.

Splits are 70/10/20 by pseudo-patient, stratified by class. Maximum per-class deviation from
the global distribution is **0.389 percentage points** (`split_stats.json`), against a stated
tolerance of 2.

## 4.3 Centralized ceiling

Federated numbers are meaningless without the non-federated ceiling on the *same*
de-duplicated split.

| model | norm | seeds | test macro-F1 | test accuracy |
|---|---|---|---|---|
| SimpleCNN @112 | GroupNorm | 5 | 0.9203 ± 0.0053 | 0.9211 ± 0.0052 |
| SimpleCNN @112 | BatchNorm | 5 | 0.9300 ± 0.0127 | 0.9314 ± 0.0144 |
| ResNet-18 (ImageNet-pretrained) | GroupNorm | 3 | 0.9701 ± 0.0075 | 0.9705 ± 0.0074 |

*Measured on a Colab T4, 2026-09-16; recorded in `docs/EXPERIMENT_LOG.md`.*

None of these is implausibly close to 1.00. Plan §2.1 makes that the acceptance criterion for
the de-duplication: a ceiling of 0.91–0.98 is itself evidence the leakage fix held. All
federated results below use **SimpleCNN with GroupNorm** — GroupNorm because BatchNorm's
running statistics are themselves a federated-averaging problem, and ablation A9 controls for
that choice rather than assuming it.

## 4.4 Federated configuration

| setting | value | why |
|---|---|---|
| clients K | 10 | the one size with a measured per-round cost (7.5 s/round on a T4) |
| rounds | 100 | |
| local epochs | 1 | |
| optimizer | SGD, lr 0.01, batch 32 | tuned for FedAvg and reused for every strategy |
| image size | 112 | |
| participation | full (`fraction-train = 1.0`) | R3 varies it |
| seeds | 8 | see §4.6 |
| partition regimes | IID, Dirichlet α=0.3, Dirichlet α=0.1 | a monotone heterogeneity ladder |

*Source: `configs/experiment/main_reduced.yaml`.*

Each client's shard is split 90/10 into local-train and local-val. A global validation split
is held at the server **only** for hyperparameter selection and for the optional `server_val`
fitness mode; the default method does not use it, which is the point of the comparison against
FedLAW.

## 4.5 Baselines

FedAvg, FedProx, Krum, Trimmed-Mean, FedLAW, and the proposed method. Each baseline received
its own hyperparameter search on the validation split with a budget matched to the proposed
method's (`scripts/run_hparam_search.py`; every search recorded in `docs/EXPERIMENT_LOG.md`).

FedLAW is the load-bearing baseline: it is the closest prior work — server-side *learned*
aggregation weights — so it is the comparison that decides whether an ACO-searched weight
vector buys anything a gradient-learned one does not. Krum and Trimmed-Mean serve twice, as
standard baselines and as the Byzantine-robust reference set in §5.3.

**Scope stated plainly:** six strategies, not the twelve the repository implements, and three
partition regimes, not six. This is a compute constraint and not a methodological claim.
`configs/experiment/main.yaml` holds the full grid (576 cells, 120–240 GPU-hours) and
`main_reduced.yaml` documents each cut at the point it is made. The regimes dropped were
pathological (2 classes/client), quantity-skew and source-shift; quantity-skew is the most
costly omission, since it attacks FedAvg's $n_k$ weighting directly.

## 4.6 Statistics

Eight seeds, and this is an arithmetic requirement rather than a preference. The Wilcoxon
signed-rank test's two-sided p-value has a floor set by the pair count alone: at 5 seeds the
smallest attainable p is **0.0625**, so a 5-seed study cannot report significance at α=0.05
*under any data whatsoever* — we measured a synthetic best case at Cohen's d = 7.91 reporting
p = 0.25. Eight seeds move the floor to 0.0078.

We test the proposed method against FedAvg in each of the three regimes: a **three-comparison
family**, corrected with Holm–Bonferroni across the whole table rather than per regime. The
remaining four baselines are reported descriptively with effect sizes and bootstrap CIs, not
as significance claims — six strategies × three regimes would be a 15-comparison family that
8 seeds does not clear. `scripts/make_tables.py` prints the seed requirement for whatever
family the table it just built actually contains; that output, not this paragraph, is the
authority.
