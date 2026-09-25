# Amazon ML Challenge — Business Entity Resolution
## Problem summary + phase-by-phase solution plan

Source: `amazon_ml_challenge_problem_statement_copy.pdf` (8 pages).
This document is a distilled restatement of the problem plus an executable plan. Every
rule in Part 1 comes from the PDF; Part 2 onwards is the proposed approach.

---

# Part 1 — Problem summary

## 1.1 The task

Business identity records arrive from **3 independent sources** with **no shared
identifiers** and heavy field noise. Decide which records refer to the same real-world
business (classic **Entity Resolution**).

- **Source 1 is the deduplicated reference source.** For every Source 1 record, find all
  matching records in **Source 2** and **Source 3**.
- A Source 1 entity may match **zero, one, or many** S2/S3 records.
- S2↔S3 linkage is not asked for directly — the unit of prediction is always one S1 entity.

## 1.2 Data

All files are **tab-separated** (`.tsv`) — addresses and ID lists contain commas, so a
missing `sep="\t"` silently collapses each line into one column.

```python
df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
```

Source files (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`) columns:

| Column | Notes |
|---|---|
| `entity_id` | Unique per record; prefix `S1-` / `S2-` / `S3-` encodes the source |
| `business_name` | Abbreviations, legal suffixes, typos, transliterations |
| `business_address` | Partial addresses, format variation, missing components, landmarks |
| `country` | Training covers **US** and **India**; test **additionally contains France** |

There is **no `source` column** — source comes from the ID prefix and the file.

Ground truth (`dataset/train/train_ground_truth.tsv`):

| Column | Notes |
|---|---|
| `source1_entity_id` | An S1 record |
| `matched_entity_ids` | Comma-separated S2/S3 IDs; **empty** when the entity has no matches |

**Country is an open set of string labels.** Do not hard-code, filter, or one-hot to
`{US, India}`. Every test entity — France included — must appear in the submission.

### Noise patterns to expect

- **Names:** `Corp`↔`Corporation`, `Pvt`↔`Private`, `Ltd`↔`Limited`, legal-suffix
  inconsistency, DBA/trade names, `&`↔`and`, word-order transposition, typos.
- **Addresses:** `Rd`↔`Road`, `St`↔`Street`, transliteration variants, missing components
  (no PIN, no state), landmark references (`Near SBI ATM`), municipal numbering formats,
  component reordering.

## 1.3 Outputs

Two tab-separated files in `output/`:

**`matching_results.tsv`** — the only leaderboard-scored file.

```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

**`candidate_pairs.tsv`** — the candidate set fed into the matching model, i.e. **the last
blocking/filtering stage, whatever the model actually scores** (not an early raw pass).
Not leaderboard-scored; used to audit blocking quality (recall ceiling, reduction ratio)
and verify the pipeline.

```
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
```

Rules for **both** files:

1. Exactly one row per Source 1 entity in the test set — every entity present.
2. Empty `matched_entity_ids` / `candidate_entity_ids` for no matches / no candidates.
3. No duplicate IDs within a list; no duplicate `source1_entity_id` rows.
4. Only `S2-`/`S3-` IDs that exist in the test set. Self-matches to S1 are rejected.
5. Final matches should be a **subset of candidates** (the validator warns otherwise).

Validate locally before every submission, from the `student_resource/` directory:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

`PASS` (exit 0) means safe to submit; otherwise a numbered issue list (exit 1). It does
**not** compute a score.

## 1.4 Scoring — macro-averaged F<sub>0.5</sub>

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

- Computed **per Source 1 entity**, then **averaged over all S1 entities** in the
  evaluation set. **Singletons are included in the average.**
- An entity with no true matches scores **1.0** if you predict an empty list and **0.0**
  if you predict anything.
- Precision-heavy: a false merge costs roughly 2× a missed link.
- Worked example from the PDF: predicted `[S2-00047, S2-00193, S3-00812]`, truth
  `[S2-00047, S3-00812]` → P = 2/3, R = 1.0, F<sub>0.5</sub> = 0.714.

**Consequence for design:** the decision layer matters as much as the model. Adding a
4th uncertain match to a 3-true-match entity costs more than omitting a true one, and
every wrongly-merged singleton is a full 1.0 lost.

## 1.5 Constraints, leaderboard, integrity

- Output format must be exact — failed validation = not evaluated. A correct upload shows
  **SCORED** with its F<sub>0.5</sub>.
- **Final model must be MIT / Apache-2.0 licensed and ≤ 8 billion parameters.**
- **Public leaderboard** = subset of test during the challenge; **private leaderboard** =
  the remaining portion, revealed at the end and used for **final rankings**. You always
  submit predictions for the full test set; the split is applied at scoring time.
- No ground truth for test — hold out a validation split from training and score it
  yourself with the F<sub>0.5</sub> formula.
- **Strictly prohibited:** external databases, APIs, or services for entity lookup —
  commercial ER APIs, government business registries, **geocoding APIs**, any internet
  data augmentation. All pipelines are reviewed; violations mean immediate
  disqualification.

### Final submission package (every team, zipped)

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # same file uploaded to the leaderboard
│   └── candidate_pairs.tsv         # your blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # all source code
│       ├── README.md               # end-to-end reproduction steps
│       └── requirements.txt        # pinned dependencies
└── Documentation_template.md       # filled-in methodology write-up
```

The methodology document must cover: methodology, candidate generation/blocking strategy,
model architecture and feature engineering, plus anything else relevant. No page limit —
depth over brevity.

## 1.6 Official tips

Invest in blocking (it caps recall) · use string-similarity features (Jaccard,
Levenshtein, TF-IDF cosine) · attend to country-specific address patterns · tune the
precision/recall trade-off for F<sub>0.5</sub> · do not neglect singletons · validate the
output format before submitting.

---

# Part 2 — Solution architecture

A three-stage pipeline, which is also exactly what the two required output files describe:

```
 normalize  →  block (recall-oriented)  →  score pairs (precision-oriented)  →  decide
   Phase 2         Phase 3                     Phases 5–6                     Phase 7
                      │                                                           │
                      └────────→ candidate_pairs.tsv          matching_results.tsv ┘
```

Design commitments:

1. **Recall lives in blocking, precision lives in the decision layer.** Blocking targets
   ≥ 98% recall ceiling on validation; the classifier + thresholds spend that recall.
2. **Country is never a learned category.** It is used only relationally (same-country
   flag, and to select which normalization rules fire), so France behaves like any unseen
   label rather than falling off a one-hot cliff.
3. **One row per S1 entity, always** — generated from the S1 file itself, not from the
   candidate set, so entities with zero candidates still emit an empty row.
4. **Everything is a function of the provided data + an offline pretrained model.** No
   network calls at inference; no gazetteers scraped from the internet.

---

# Part 3 — Phase plan

Effort figures assume one person working roughly full days; run phases 5–7 as a loop.

## Phase 0 — Scaffolding (½ day)

Create the exact submission layout up front so nothing has to be reshuffled later:

```
business_entity_resolution/
├── src/
│   ├── io_utils.py        # strict TSV read/write, ID-list parsing
│   ├── normalize.py       # Phase 2
│   ├── blocking.py        # Phase 3
│   ├── pairs.py           # Phase 4
│   ├── features.py        # Phase 5
│   ├── model.py           # Phase 6
│   ├── decide.py          # Phase 7
│   ├── score.py           # Phase 8 (macro F0.5)
│   └── run_pipeline.py    # CLI: --split {train,val,test}
├── README.md
└── requirements.txt
```

- Pin every dependency; fix seeds everywhere (`numpy`, model, any sampler).
- One config file (YAML) holding thresholds, top-k, feature flags — so a leaderboard
  submission can be traced back to a config.
- Write `src/io_utils.py` first and use it everywhere: `sep="\t"`,
  `dtype=str`, `keep_default_na=False` (so an empty address is `""`, not `NaN`),
  and `quoting=csv.QUOTE_NONE` on write so no ID list ever gets quoted.

**Exit criteria:** `run_pipeline.py --split test` produces a format-valid
`matching_results.tsv` of all-empty predictions that passes `validate_submission.py`.
That is a guaranteed-scoreable baseline (it already earns the singleton credit) and proves
the I/O contract before any modelling.

## Phase 1 — Data audit (1 day)

Measure, do not assume. Record every number in a notebook that the methodology doc can cite.

- Row counts per source per country; `country` value set and exact spellings/casing.
- Missing/empty rates for `business_name`, `business_address`.
- Ground-truth distribution: fraction of S1 entities with 0 / 1 / 2 / 3+ matches;
  separately for S2 and S3. **The singleton rate sets the baseline score** — if 30% of
  entities are singletons, the all-empty submission scores 0.30.
- Do matches ever appear twice from the same source (S1 ↔ two S2 records)? This decides
  whether the decision layer can assume at most one match per source.
- Country agreement: does a true match ever cross countries? If effectively never,
  same-country becomes a hard blocking constraint; if it happens, keep it a soft feature.
- Character-level audit: scripts present (Latin/Devanagari/accents), case, punctuation,
  digit patterns in addresses (PIN vs ZIP vs postcode), and the most frequent tokens per
  country — these frequent tokens are what the normalization map should target.
- Sanity-check the ground truth: do all referenced IDs exist in the S2/S3 train files?

**Exit criteria:** a written noise taxonomy with frequencies, and the measured
all-empty baseline score on a validation split.

## Phase 2 — Normalization (1 day)

A single deterministic function per field, applied identically to all three sources.
Keep raw values too — some features (exact-string equality, digit sequences) work better
pre-normalization.

**Name pipeline:** Unicode NFKC → casefold → strip accents → `&`→`and` → remove
punctuation → collapse whitespace → expand legal/abbreviation forms (`pvt`→`private`,
`ltd`→`limited`, `corp`→`corporation`, `co`→`company`, `inc`, `llc`, `llp`, `plc`,
`sarl`/`sas`/`sa` for France) → emit **two** variants: `name_full` (suffixes expanded,
kept) and `name_core` (suffixes stripped). Also emit a sorted-token form for
transposition-insensitive comparison, and an alphabetised-character/initials signature.

**Address pipeline:** same base cleanup → expand street-type abbreviations
(`rd`→`road`, `st`→`street`, `ave`, `blvd`, `apt`, `flr`, `bldg`, `opp`, `nr`→`near`) →
split off **numeric tokens**: house/plot numbers and postal codes (5-digit US ZIP,
6-digit Indian PIN, 5-digit French code postal — detected by **pattern**, never by a
downloaded postcode list) → drop landmark lead-ins (`near`, `opposite`, `behind`,
`beside`) into a separate landmark bag rather than deleting them → emit `addr_core` plus
`addr_numbers` (a set) and `addr_postcode`.

**Transliteration:** a small, data-derived rule set for the variants actually seen in
training (`sri`/`shri`, `ph`/`f`, doubled consonants, `-ee`/`-i` endings). Prefer a
phonetic key (Double Metaphone on the Latin form) over hand-listing pairs.

Two compliance notes worth recording in the methodology doc:
- Hand-written abbreviation maps are standard text normalization, not external lookup.
  Keep them small, in-repo, and derived from observed training tokens — that is the
  defensible position under the fair-play rule.
- **No geocoding, and no downloaded postcode/city gazetteer.** Postal codes are handled
  as regex-detected digit patterns only.

**Exit criteria:** `normalize.py` is pure, unit-tested on hand-picked hard examples from
Phase 1, and idempotent (`f(f(x)) == f(x)`).

## Phase 3 — Blocking / candidate generation (2 days — the highest-leverage phase)

This stage sets the recall ceiling, and its output *is* `candidate_pairs.tsv`.

Use a **union of cheap recall-oriented indexes**, each producing top-k per S1 entity:

1. **Char n-gram TF-IDF on `name_core`** (`TfidfVectorizer(analyzer="char_wb",
   ngram_range=(3,5))`), sparse matrix product S1 × (S2∪S3), top-k by cosine. Robust to
   typos and suffix noise, and language-agnostic — the main workhorse.
2. **Word TF-IDF on `name_full + addr_core`** — catches long multi-word names where
   char n-grams saturate.
3. **Token inverted index with IDF pruning:** index rare tokens only (skip the top ~1%
   most frequent), then take S2/S3 records sharing ≥1 rare name token or ≥2 address tokens.
4. **Exact-key blocks:** postcode + first-3-chars-of-name; `addr_numbers` overlap +
   name-initials; phonetic key of `name_core`. These rescue the typo-heavy cases the
   vector methods rank low.
5. **Optional dense pass:** embed `name + address` with a small multilingual sentence
   encoder and add ANN top-k. Adds cross-script/transliteration recall and is where
   France coverage is most likely to come from. **Verify the license on the model card
   (MIT or Apache-2.0) and the parameter count (≤ 8B) before adopting it** — small
   multilingual E5 / MiniLM-class encoders are the usual fit, but confirm rather than
   assume.

Restrict the pool per S1 entity to the **same country** only if Phase 1 showed
cross-country matches are effectively absent — and implement it as "same country first,
fall back to all countries when a block is empty" so an unseen label like France can never
produce an empty candidate set through a bookkeeping accident.

**Instrument this stage — both numbers belong in the methodology doc:**

- **Recall ceiling** = fraction of true (S1, S2/S3) pairs present in the candidate set.
  Target ≥ 98%; anything blocking misses is permanently unreachable.
- **Reduction ratio** = 1 − |candidates| / (|S1| × |S2∪S3|), plus mean/median/p99
  candidates per entity.

Tune `k` on the recall/size curve and stop where recall flattens; typical landing zone is
tens of candidates per entity, not hundreds.

**Exit criteria:** recall ceiling ≥ 98% on validation with a candidate list small enough
to feature-engineer in minutes, and a written `candidate_pairs.tsv` that passes the
validator.

## Phase 4 — Training pairs (½ day)

- Split **by S1 entity** into train/validation (e.g. 80/20), **stratified by match count**
  (0, 1, 2, 3+) and by country so both folds carry US and India in proportion. Splitting
  by entity — never by pair — is what keeps a pair's own entity out of both folds.
- Keep the **full S2/S3 pools** available to both folds and score only the fold's S1
  entities. That mirrors test conditions, where every S1 entity competes against the
  whole pool.
- Labels: run the Phase 3 blocker on the training fold; every candidate pair present in
  the ground truth is positive, every other candidate is negative. **Train on blocked
  negatives, not random ones** — random negatives are trivially separable and produce a
  model that collapses at inference time.
- Handle imbalance by keeping all positives and, if needed, capping negatives per entity
  at the hardest N (highest blocking score), or use class weights. Record the ratio.
- Also record, per entity, whether the blocker missed a true match — those entities are
  the ceiling loss and should not be blamed on the classifier later.

**Exit criteria:** a reproducible `pairs_train.parquet` / `pairs_val.parquet` with
`(s1_id, cand_id, label, country, blocking_source)` and no entity in both folds.

## Phase 5 — Features (1–2 days)

Per candidate pair, a compact numeric vector. Group them so ablations are cheap:

**Name similarity:** Jaro-Winkler, normalized Levenshtein, token-set and token-sort ratio,
token Jaccard, char 3-gram Jaccard, TF-IDF cosine (word and char), longest-common-substring
ratio, initials match, rare-token overlap weighted by IDF, phonetic-key equality, and
`name_core` exact equality.

**Address similarity:** the same battery on `addr_core`, plus targeted signals that carry
most of the discriminative power:
- postcode exact match / prefix-match length / one side missing,
- `addr_numbers` set overlap (house/plot numbers are near-identifying),
- street-token Jaccard after removing numbers,
- landmark-bag overlap,
- both-addresses-empty and one-empty indicators.

**Context / relational features** — these are what make the decision layer precise:
- `same_country`, and which source the candidate came from (`is_S2` / `is_S3`) as a flag,
- **rank and score of this candidate within its S1 entity** from each blocking index,
- **margin to the best competing candidate** for the same S1 entity, and to the best
  candidate from the *other* source,
- the candidate's own "popularity": how many S1 entities it appears as a candidate for
  (a record that looks like everything is usually a generic name),
- IDF-weighted name rarity of the S1 entity itself (common names need stronger evidence),
- field-missingness flags for both records.

**Optional semantic features:** cosine of the Phase 3 sentence embeddings on name, on
address, and on the concatenation. Cheap once the vectors exist and they generalise across
scripts better than character overlap.

Keep every feature **symmetric and country-agnostic**: no feature may be "is India", only
"same country" or "postcode pattern length matched". Verify by computing feature
distributions per country and confirming the model's behaviour does not hinge on a label.

**Exit criteria:** a feature matrix, a one-line description per feature for the
methodology doc, and a quick mutual-information / permutation-importance ranking.

## Phase 6 — Matching model (1–2 days)

**Baseline (do this first):** gradient-boosted trees on the pair features — LightGBM
(MIT) or XGBoost (Apache-2.0), both comfortably inside the license and parameter
constraints. Tabular GBDTs are the right default here: they handle heterogeneous
similarity features, mixed missingness, and monotone-ish signals with almost no tuning,
train in minutes, and give calibrated-enough probabilities plus readable importances for
the write-up. Use grouped CV (group = S1 entity) for hyperparameters.

**Upgrade path, in order of expected value per unit of effort:**

1. **Better decision layer** (Phase 7) — usually beats a better model on F<sub>0.5</sub>.
2. **Embedding features** from a small multilingual encoder added to the GBDT.
3. **Fine-tuned cross-encoder** on `"name [SEP] address"` pairs, trained on the blocked
   pairs, and *ensembled with* (not substituted for) the GBDT — average or stack the two
   scores. A small cross-encoder (tens to hundreds of millions of parameters) is where the
   remaining hard typo/transliteration cases get resolved, and it stays far inside the
   8B limit.
4. **Listwise / per-entity model** that sees all candidates of one S1 entity at once
   (e.g. LambdaMART-style ranking objective), since the metric is itself per-entity.

For any pretrained checkpoint: record name, revision, license, and parameter count in the
methodology doc, and confirm MIT/Apache-2.0 and ≤ 8B on the model card. Cache weights in
the package or document the exact download, since the graders re-run the pipeline.

**Exit criteria:** pair-level AUC/PR-AUC on validation, plus the score distribution split
by positive/negative — the shape of that overlap is what Phase 7 tunes against.

## Phase 7 — Decision layer (1 day, highest score-per-hour after blocking)

Turning scores into ID lists is where F<sub>0.5</sub> is won. Tune all of it against the
**macro F<sub>0.5</sub> scorer**, never against pair-level accuracy.

Rules to tune jointly:

1. **Global threshold `τ`** — sweep 0.01→0.99 and plot macro F<sub>0.5</sub>. Expect the
   optimum well above 0.5 because the metric is precision-heavy.
2. **Singleton rule** — if no candidate reaches `τ`, emit an empty list. Consider a
   second, higher bar for entities whose best score is only marginally above `τ`:
   predicting nothing scores 1.0 on a true singleton but 0.0 on a real match, so the
   break-even depends on the measured singleton rate from Phase 1.
3. **Per-source thresholds** (`τ_S2`, `τ_S3`) if the sources differ in noise level.
4. **Cap per entity** — at most `k` matches, and/or at most one per source if Phase 1
   showed that pattern. A relative-margin rule ("keep candidates within δ of the best")
   usually beats a fixed cap.
5. **Mutual-best consistency** — prefer pairs where the S1 entity is also the candidate's
   best S1 (one-to-one-ish pressure reduces false merges on generic names).
6. **Transitivity check** — if an S2 and an S3 record are both matched to one S1 entity
   but are wildly dissimilar to each other, drop the weaker one.

Tune on validation, then confirm the chosen operating point is not a knife edge: plot the
score curve and pick a **plateau**, not a spike. A threshold that is optimal by 0.002 on
validation will not survive the private leaderboard.

**Exit criteria:** a config-stored operating point, with the validation F<sub>0.5</sub>
curve saved for the write-up.

## Phase 8 — Evaluation harness (½ day, build it before Phase 7)

Implement the metric exactly as specified — per entity, macro-averaged, singletons
included:

```python
def f_beta_half(pred: set[str], true: set[str]) -> float:
    if not true and not pred:
        return 1.0
    if not true or not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return (1.25 * p * r) / (0.25 * p + r)

def macro_score(preds: dict[str, set[str]], truth: dict[str, set[str]]) -> float:
    # every S1 entity in the evaluation set contributes, matched or not
    return sum(f_beta_half(preds.get(e, set()), t) for e, t in truth.items()) / len(truth)
```

Then build the diagnostics that tell you *where* the score is lost:

- Score decomposed into **blocking loss** (true pair never a candidate) vs **model loss**
  (candidate scored below τ) vs **false merges** (predicted, not true) vs
  **singleton errors**. Each maps to a different phase — this is the only reliable way to
  choose what to work on next.
- Per-country breakdown (US vs India), and per-match-count breakdown (singletons, 1, 2, 3+).
- **France proxy:** hold out one country entirely (train on US, validate on India, and
  vice versa) to estimate how much the pipeline degrades on an unseen country. If the drop
  is large, push normalization and embeddings harder and lean less on country-correlated
  patterns. Also assert at the code level that no branch reads `country == "..."` for
  anything but rule selection with a generic fallback.
- A **leaderboard log**: one row per upload — config hash, validation score, public score.
  Watching the two diverge is the earliest warning of overfitting to the public subset.

## Phase 9 — Inference, packaging, validation (½ day)

- Run the full pipeline on test with **train-fitted** vectorizers/models; never refit
  IDF on test alone if the training corpus statistics are what the model expects (and if
  you do fit on the combined corpus, do it identically for the validation runs).
- Emit both TSVs from the **S1 test file's full entity list**, sorted, with empty strings
  for no-match rows, no quoting, and a trailing newline.
- Assert before writing: matches ⊆ candidates; all IDs exist in the test S2/S3 files; no
  `S1-` in either list; no duplicates; row count == S1 test row count.
- Run `utils/validate_submission.py` and require `PASS` — treat a non-PASS as a build
  failure, not a warning.
- Record end-to-end wall-clock and peak memory (the README must let a grader reproduce).

## Phase 10 — Documentation and final zip (½ day)

Fill in `Documentation_template.md` covering, at minimum, what the PDF asks for:
methodology, blocking strategy **with measured recall ceiling and reduction ratio**, model
architecture and the full feature list, the decision-layer rules and how thresholds were
chosen, the validation protocol and scores (including the held-out-country experiment),
ablations, the pretrained-model license/parameter table, and an explicit fair-play
statement that no external data source, geocoder, or lookup API was used.

Then assemble exactly the required tree, verify the zip extracts to that structure, and
confirm `requirements.txt` recreates a working environment from scratch.

---

# Part 4 — Suggested schedule

| Day | Focus | Deliverable |
|---|---|---|
| 1 | Phases 0–1 | Scaffolding + all-empty valid submission + data audit |
| 2 | Phase 2 | Normalization, unit-tested |
| 3–4 | Phase 3 | Blocking at ≥ 98% recall ceiling + `candidate_pairs.tsv` |
| 5 | Phases 4 + 8 | Splits, blocked pairs, macro F<sub>0.5</sub> scorer |
| 6–7 | Phases 5–6 | Features + GBDT baseline; **first real leaderboard upload** |
| 8 | Phase 7 | Threshold/decision tuning — usually the biggest single jump |
| 9–10 | Phase 6 upgrades | Embeddings, cross-encoder, ensemble |
| 11 | Phase 8 | Error decomposition, held-out-country check, ablations |
| 12 | Phases 9–10 | Final inference, validation, documentation, zip |

Compress by merging days 3–4 and 9–10 if the deadline is tighter; never compress Phase 7.

---

# Part 5 — Risks and pre-mortem

| Risk | Mitigation |
|---|---|
| Blocking misses true pairs → hard recall ceiling | Union of complementary indexes; measure ceiling every change; treat < 98% as a Phase 3 bug |
| Recall-chasing tanks precision under F<sub>0.5</sub> | Tune only against macro F<sub>0.5</sub>; expect τ well above 0.5 |
| Singletons ignored | Explicit no-match rule; report the singleton-only sub-score separately |
| France (unseen country) collapse | No country one-hot; relational features only; held-out-country validation; blocking fallback so no entity gets an empty pool by accident |
| Public-leaderboard overfitting | Trust local validation; pick threshold plateaus; log public vs local per upload |
| Format rejection wastes a submission | Validator wired into the pipeline as a hard gate from day 1 |
| License / parameter-count violation | Model table with license + parameter count, verified on each model card before adoption |
| Accidental fair-play violation | No network at inference; abbreviation maps small, in-repo, derived from training tokens; no geocoding or postcode gazetteers; stated explicitly in the write-up |
| Unreproducible final package | Pinned requirements; seeded runs; fresh-clone reproduction rehearsed before zipping |

---

# Part 6 — First three things to do now

1. **Phase 0 + the all-empty submission.** It establishes the I/O contract, passes the
   validator, and gives a real baseline number (the singleton rate) to beat.
2. **Phase 1 audit, specifically the match-count distribution.** Everything in Phase 7 —
   caps, singleton rules, thresholds — is calibrated off those frequencies.
3. **Phase 3 char n-gram TF-IDF blocker with a recall-ceiling readout.** One afternoon,
   and it tells you immediately whether the rest of the plan has enough recall to work with.
