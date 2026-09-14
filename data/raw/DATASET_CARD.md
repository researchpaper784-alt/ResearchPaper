# Dataset card — Brain Tumor MRI

**Every number in this file was computed from disk** by
`python -m fedswarm.data.download --verify`. Nothing here is copied from the dataset's
Kaggle page, a blog post, or the implementation plan. Re-run that command to regenerate.

| Field | Value |
|---|---|
| Source URL | https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset |
| Kaggle slug | `masoudnickparvar/brain-tumor-mri-dataset` |
| Verified (UTC) | 2026-09-14T06:28:05.522767+00:00 |
| Scanned path | `data/raw/brain-tumor-mri` |
| Archive SHA256 | `882817250048c78ef7a759cf23e540d7b581f2327b16663c9d3db12f5d2ffdb4` |
| Archive size (bytes) | 164670110 |
| License | **UNVERIFIED — read it from https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset and record it here before the paper cites this dataset.** |

## Provenance

This dataset is a merge of three sources, per its Kaggle description: the figshare brain
tumor dataset (Cheng et al.), the SARTAJ dataset, and Br35H. That multi-source structure
is what Phase 1.4's `source_shift` partitioning regime exploits to simulate realistic
cross-site feature shift.

**Known issue to carry into the paper's limitations:** the SARTAJ component has documented
mislabeling in the glioma class. Phase 1.2 records this but does not attempt to fix it.

## Per-class, per-split counts (counted from disk)

| Split | glioma | meningioma | notumor | pituitary | **Total** |
|---|---|---|---|---|---|
| Training | 1400 | 1400 | 1400 | 1400 | **5600** |
| Testing | 400 | 400 | 400 | 400 | **1600** |
| **Total** | **1800** | **1800** | **1800** | **1800** | **7200** |

⚠️ These are the counts of the **original, as-distributed split**, before de-duplication.
Phase 1.2 rebuilds the split at the pseudo-patient level; the counts that belong in the
paper are the post-de-duplication ones in `data/processed/leakage_report.json`.

## Observed image sizes

447 distinct sizes observed. Top 20 shown.

| Size (WxH) | Count |
|---|---|
| `512x512` | 5014 |
| `225x225` | 338 |
| `630x630` | 90 |
| `201x251` | 57 |
| `228x221` | 51 |
| `232x217` | 50 |
| `442x442` | 48 |
| `236x236` | 48 |
| `150x198` | 44 |
| `200x252` | 43 |
| `428x417` | 42 |
| `227x222` | 39 |
| `173x201` | 36 |
| `206x244` | 35 |
| `256x256` | 33 |
| `192x192` | 31 |
| `201x250` | 29 |
| `218x231` | 29 |
| `215x234` | 28 |
| `227x262` | 27 |

## Observed colour modes

| PIL mode | Count |
|---|---|
| `RGB` | 4129 |
| `L` | 3067 |
| `RGBA` | 3 |
| `P` | 1 |

## Unreadable files

None — every file opened cleanly.
