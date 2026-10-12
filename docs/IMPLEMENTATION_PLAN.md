> **Added to the repository 2026-09-19.** This document existed from the start but was not
> in the repo, and its absence had concrete costs: Phase 7's ablations A2 and A4-A8 were
> recorded as "not reconstructable from what the repo names" and left unbuilt for days, and
> the level set in `aco/schedules.py` diverged from §14's explicit hyperparameter table
> (log-spaced instead of linear) with nothing to catch it. Anything here outranks a
> reconstruction in `docs/OPEN_QUESTIONS.md`; where the two disagree, this is the spec and
> the other is a note about not having had it.
>
> Two places where the plan itself needs amending, both recorded in docs/OPEN_QUESTIONS.md
> rather than silently followed:
>
> - **§9.1's statistics are not achievable at the stated 5 seeds.** A two-sided Wilcoxon
>   signed-rank test at n=5 has a p-value floor of 0.0625, so Wilcoxon + Holm as specified
>   cannot return a significant result under any data. `main.yaml` runs 8 seeds, which is
>   consistent with the risk register's ">= 5 seeds" and makes §9.1 executable.
> - **§0.3's naming collision was resolved by renaming** to FedSwarm (see CLAUDE.md).
>   "FedACO" survives in code identifiers and in this document.

---

# FedACO — Implementation Plan

**Project:** *FedACO: Adaptive Aggregation Optimization in Federated Learning using Swarm Intelligence for Brain Tumor Diagnosis*

**Audience:** This document is written to be handed directly to Claude Code (or any capable coding agent) as the master specification. Every phase states **WHAT** to build, **WHY** it exists (the research justification — do not skip these, they determine design choices), and **HOW** to implement it, ending with a machine-checkable **ACCEPTANCE** criterion.

**Target configuration (decided):**

| Decision | Choice |
|---|---|
| Task | Brain tumor MRI classification (4-class) |
| FL framework | Flower (`flwr`), simulation mode |
| Compute | Single GPU (Colab / Kaggle / one local card) |
| Scope | Full research repo: baselines, ablations, statistics, paper-ready figures & tables |

---

## 0. Read this first (instructions to the implementing agent)

### 0.1 Ground rules

1. **Work phase by phase.** Do not start Phase *N+1* until Phase *N*'s acceptance check passes. Each phase ends in a committed, runnable state.
2. **Every phase gets a git commit** with a message of the form `phase-N: <summary>`. Tag milestones (`v0.1-data`, `v0.2-fedavg`, `v0.3-fedaco`, …).
3. **No silent fabrication.** If a number, dataset property, or API signature cannot be verified, stop and say so in `docs/OPEN_QUESTIONS.md` rather than guessing. A research paper dies on one unverifiable number.
4. **Determinism is a feature, not a nicety.** Every run takes a seed; every result file records the full config + git SHA + library versions. See Phase 6.
5. **Write the test at the same time as the code.** Phases 4 and 7 in particular are unfalsifiable without unit tests — an optimizer that silently does nothing still produces plausible-looking curves.
6. **Prefer boring, inspectable code** over clever abstractions. This repo will be read by reviewers and by the author six months from now.

### 0.2 ⚠️ Critical API warning — read before writing any Flower code

Flower changed its public API substantially. **Most code you have seen for Flower is obsolete.** As of this writing the latest release is **`flwr` 1.36.0** (Sept 2026), and the **Message API** has been the way since **1.21**.

**Obsolete (do NOT use):**

```python
# ❌ DEAD API — do not write this
import flwr as fl
class FlowerClient(fl.client.NumPyClient):
    def fit(self, parameters, config): ...
class MyStrategy(fl.server.strategy.Strategy):
    def aggregate_fit(self, server_round, results, failures): ...
fl.simulation.start_simulation(client_fn=..., ...)
```

**Current (DO use):**

```python
# ✅ Current Message API
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, Strategy

app = ClientApp()

@app.train()
def train(msg: Message, context: Context) -> Message:
    model = build_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    loss = local_train(model, ...)
    content = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({"train_loss": loss, "num-examples": n}),
    })
    return Message(content=content, reply_to=msg)
```

The `Strategy` abstract base class you must subclass has exactly these abstract methods:

```python
from collections.abc import Iterable
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp import Grid

class Strategy(ABC):
    @abstractmethod
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]: ...

    @abstractmethod
    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]: ...

    @abstractmethod
    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]: ...

    @abstractmethod
    def aggregate_evaluate(
        self, server_round: int, replies: Iterable[Message]
    ) -> MetricRecord | None: ...

    @abstractmethod
    def summary(self) -> None: ...

    # concrete — do not override
    def start(self, grid, initial_arrays, num_rounds=3, timeout=3600,
              train_config=None, evaluate_config=None, evaluate_fn=None) -> Result: ...
```

Note the renames: `fraction_fit` → `fraction_train`, "clients" → "nodes", strategies live in `flwr.serverapp.strategy` (**not** `flwr.server.strategy`).

**Mandatory first action of Phase 0:** pin an exact version in `pyproject.toml`, install it, and run
`python -c "import flwr, inspect; from flwr.serverapp.strategy import Strategy; print(flwr.__version__); print(inspect.getsource(Strategy))"`
Paste the real signatures into `docs/FLOWER_API_NOTES.md` and **code against what you actually see**, not against this document. If the installed API differs from the above, the installed API wins — update this plan's assumptions in `docs/OPEN_QUESTIONS.md`.

### 0.3 ⚠️ Naming collision — flag to the author

A 2025 PRCV/Springer paper already uses the acronym **FedACo** ("Adaptive Collaboration with Fine-Grained Aggregation for Personalized Federated Learning"). Same acronym, different expansion and different method.

Record this in `docs/OPEN_QUESTIONS.md` and surface it to the author. Options: (a) keep FedACO and add an explicit differentiation sentence + citation in Related Work; (b) rename — e.g. **FedSwarm**, **ACO-Agg**, **PherAgg**, **FedPher**. Do not silently proceed as if the name is free.

### 0.4 Repository layout (create this in Phase 0)

```
fedaco/
├── README.md
├── CLAUDE.md                      # agent working notes & conventions
├── pyproject.toml
├── LICENSE                        # MIT or Apache-2.0
├── .gitignore                     # data/, results/, *.ckpt, wandb/
├── Makefile                       # make setup / data / test / smoke / main / figures
├── configs/
│   ├── base.yaml
│   ├── data/          {iid, dirichlet_0.1, dirichlet_0.3, dirichlet_0.5, pathological_2}.yaml
│   ├── model/         {resnet18, simplecnn, efficientnet_b0}.yaml
│   ├── strategy/      {fedavg, fedprox, fedadam, fedyogi, scaffold, fednova,
│   │                   fedloss, fedlaw, krum, trimmed_mean, fedaco}.yaml
│   └── experiment/    {smoke, main, ablation_*, robustness_*, overhead}.yaml
├── src/fedaco/
│   ├── data/          download.py  dedup.py  partition.py  datasets.py  transforms.py
│   ├── models/        factory.py   resnet.py  simple_cnn.py
│   ├── fl/            client_app.py  server_app.py  task.py  serialization.py
│   ├── strategies/    base.py  fedprox.py  scaffold.py  fednova.py  fedloss.py
│   │                  fedlaw.py  robust.py  fedaco.py
│   ├── aco/           colony.py  pheromone.py  heuristics.py  fitness.py
│   │                  gram.py  schedules.py  controls.py
│   ├── eval/          metrics.py  evaluator.py  calibration.py
│   └── utils/         seed.py  logging.py  checkpoint.py  config.py  provenance.py
├── scripts/           run_experiment.py  run_sweep.py  aggregate_results.py
│                      make_figures.py  make_tables.py  verify_repro.py
├── tests/             test_partition.py  test_gram.py  test_aco.py  test_strategy.py
│                      test_determinism.py  test_metrics.py
├── notebooks/         00_data_audit.ipynb  01_results_explorer.ipynb
├── results/           (gitignored; every run writes a self-describing JSON)
├── paper/             figures/  tables/  ALGORITHM.md
└── docs/              FLOWER_API_NOTES.md  OPEN_QUESTIONS.md  EXPERIMENT_LOG.md
```

---

## 1. Research framing — what this project must prove

Read this section before writing code. It determines what the software has to be able to measure.

### 1.1 The problem

Vanilla FedAvg forms the global model as a fixed, data-size-proportional convex combination:

$$w_{t+1} \;=\; \sum_{k \in S_t} \frac{n_k}{\sum_j n_j}\, w_k^{t+1}$$

Under non-IID client data — the normal case in multi-hospital medical imaging, where scanner vendor, field strength, protocol and case mix all differ — these weights are *provably suboptimal*. Client updates point in conflicting directions; a client whose update opposes the consensus still contributes in proportion to its dataset size, dragging the global model and slowing convergence (client drift).

### 1.2 The gap FedACO claims

The literature on adaptive aggregation weights splits into:

- **Analytically derived weights** (e.g. FedAAW) — minimize a convergence upper bound. Elegant, but the bound relies on assumptions (smoothness, bounded dissimilarity) that don't hold for deep CNNs.
- **Learned weights** (e.g. FedLAW) — gradient-learn the weights on a server-side proxy validation set. Strong, but requires the server to hold data, which is often the exact thing federation is avoiding.
- **Heuristic weights** (loss-based, e.g. FedNolowe; class-contribution weighting) — cheap, but greedy and one-dimensional.
- **RL-optimized weights** (e.g. DaWa) — powerful but needs many episodes to train a policy, and transfers poorly across deployments.

Swarm intelligence has appeared in FL, but at the **systems layer**: ACO for scheduling model-update transmission and PSO for edge-device selection; ACO for federated *feature selection*. **Applying an ACO metaheuristic directly to the aggregation weight vector, with pheromone persisting across communication rounds as a memory of client reliability, is the open slot.**

### 1.3 The three claims the experiments must support

| # | Claim | Experiment that proves it | Where it can fail |
|---|---|---|---|
| **C1** | FedACO converges faster and to higher macro-F1 than FedAvg and standard baselines under non-IID partitions | Phase 6 main table + convergence curves, 5 seeds, significance tests | Gains vanish with a pretrained backbone; gains within seed noise |
| **C2** | The gain comes from **ACO specifically**, not merely from "optimizing the weights somehow" | Phase 7 equal-fitness-budget comparison vs. random search, grid search, PSO, GA | **This is the most likely reviewer objection and the most likely place the paper dies.** Build it early, not last |
| **C3** | FedACO's server-side cost is negligible relative to a round, and it degrades gracefully | Phase 7 overhead table (§4.6 Gram trick), Phase 8 robustness | A naive implementation is 100× slower than FedAvg — the Gram-matrix formulation in §4.6 is what prevents this |

### 1.4 Honest limitations to state in the paper (design for them now)

- FedACO, like FedLAW, Krum and every other update-inspecting method, **requires per-client updates at the server and is therefore incompatible with plain secure aggregation.** Do not hide this. Address it: run a DP-noise robustness experiment (Phase 8.4) and discuss compatibility with partial/clustered SecAgg as future work.
- Metaheuristics carry hyperparameters. Phase 7's sensitivity study must show the method is not knife-edge.
- Single-dataset results generalize weakly. Keep the data layer swappable so a second dataset can be dropped in if a reviewer demands it.

---

## 2. Phase 0 — Environment, scaffolding, provenance

### Step 0.1 — Repository and dependency skeleton

**What.** Create the layout in §0.4 with `pyproject.toml`, `Makefile`, `.gitignore`, `LICENSE`, `CLAUDE.md`, and an empty-but-importable `src/fedaco` package. Initialize git.

**Why.** A research repo whose structure is decided on day one survives forty experiment variants. One decided in week six does not. The `results/` contract (Phase 6) in particular cannot be retrofitted.

**How.**

- Python 3.11 (`requires-python = ">=3.11,<3.13"`). Use `uv` if available, else `venv` + `pip`.
- Pin exactly, then record actual resolved versions in `docs/FLOWER_API_NOTES.md`:
  `flwr[simulation]`, `flwr-datasets[vision]`, `torch`, `torchvision`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `pyyaml`, `omegaconf` (or `hydra-core`), `matplotlib`, `seaborn`, `tqdm`, `rich`, `pillow`, `imagehash`, `pytest`, `ruff`, `mypy`.
- Install PyTorch with the CUDA build matching the target GPU; guard for CPU fallback.
- `Makefile` targets: `setup`, `data`, `test`, `lint`, `smoke`, `main`, `ablations`, `figures`, `tables`, `clean`.
- `CLAUDE.md` records: package manager, how to run one experiment, where results go, the Flower API warning from §0.2, commit conventions.

**Acceptance.** `make setup && make lint && python -c "import fedaco; print(fedaco.__version__)"` succeeds. `git log` shows one commit.

### Step 0.2 — Verify and document the live Flower API

**What.** Run the introspection command from §0.2. Write `docs/FLOWER_API_NOTES.md` containing: installed `flwr` version; the real source of `Strategy`; the real signature of `FedAvg.__init__` and `FedAvg.aggregate_train`; the real `ArrayRecord` ↔ `torch.state_dict` conversion helpers; the constructor signature of `flwr_datasets.FederatedDataset` and of `DirichletPartitioner` / `PathologicalPartitioner` / `IidPartitioner`.

**Why.** Every downstream phase depends on these signatures. Ten minutes here prevents a full day of subtle breakage in Phase 4, and it gives you a ground-truth reference to code against instead of recalled API shapes.

**How.**

```python
import inspect, flwr, flwr_datasets
from flwr.serverapp.strategy import Strategy, FedAvg
from flwr.app import ArrayRecord
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
for obj in (Strategy, FedAvg, ArrayRecord, FederatedDataset, DirichletPartitioner):
    print("=" * 70, obj.__name__)
    print(inspect.signature(obj.__init__) if inspect.isclass(obj) else "")
    print(inspect.getsource(obj)[:6000])
```

Also run the official quickstart-pytorch example end-to-end unmodified to confirm the install works before you build on it.

**Acceptance.** `docs/FLOWER_API_NOTES.md` exists and contains real pasted signatures. The unmodified Flower PyTorch quickstart completes ≥ 2 rounds locally.

### Step 0.3 — Provenance and reproducibility utilities

**What.** `utils/seed.py` (seeds `random`, `numpy`, `torch`, CUDA; sets `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG`), `utils/provenance.py` (captures git SHA, dirty flag, `pip freeze`, GPU name, CUDA version, hostname, UTC timestamp), `utils/config.py` (OmegaConf load + CLI override + config hash), `utils/logging.py` (rich console + JSONL file log).

**Why.** Reviewers increasingly ask for reproducibility artifacts, and you will personally need to answer "which commit produced Table 3?" many times. A config hash also lets the sweep runner skip already-completed runs on resume — essential on a single GPU with session timeouts.

**How.** `provenance.capture() -> dict` is called once per run and embedded in every result JSON. Deterministic mode may raise on some CUDA ops — catch, log a warning, and set `deterministic: false` in the recorded provenance rather than crashing.

**Acceptance.** `pytest tests/test_determinism.py` passes: two runs of a 30-second training loop with the same seed produce bitwise-identical loss sequences (or, on a non-deterministic op, identical to within 1e-5 with the warning recorded).

---

## 3. Phase 1 — Data layer

> This phase decides whether your results are believable. Brain-tumor-MRI papers routinely report 99%+ accuracy that evaporates under a proper split. Do not inherit that problem.

### Step 1.1 — Acquire the dataset

**What.** Download the 4-class brain tumor MRI dataset (classes: `glioma`, `meningioma`, `notumor`, `pituitary`), which ships as `Training/` and `Testing/` directories of JPEGs. Primary source: the Kaggle "Brain Tumor MRI Dataset" (Nickparvar), which is itself a merge of the Figshare (Cheng et al.) set, the SARTAJ set, and Br35H.

**Why.** It's public, small enough for a single GPU, has enough classes for a meaningful macro-F1, and its multi-source provenance gives a natural, *honest* story for simulating cross-site heterogeneity.

**How.**

- `data/download.py` supports two paths: Kaggle API (`kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset`) and a manual-zip fallback with a documented `data/raw/` drop location. Never commit the data.
- Write `data/raw/DATASET_CARD.md` recording: exact source URL, download date, SHA256 of the archive, license, and **the per-class file counts you actually observe** — count them from disk with a script, do not copy numbers from a blog post or from this document.
- Record the observed image size distribution and colour mode (many are 512×512 grayscale saved as 3-channel JPEG).

**Acceptance.** `python -m fedaco.data.download --verify` prints a per-class, per-split count table computed from disk and writes `DATASET_CARD.md`. Counts are reproducible across two runs.

### Step 1.2 — Data audit and de-duplication ⚠️

**What.** Detect and remove (a) exact duplicates, (b) near-duplicates via perceptual hashing, and critically (c) **cross-split leakage** — images appearing in both `Training/` and `Testing/`. Produce `notebooks/00_data_audit.ipynb` and `data/processed/leakage_report.json`.

**Why.** **This is the single most important credibility step in the project.** The merged dataset is known to contain near-duplicate slices — consecutive slices from the same patient volume are visually near-identical — and the SARTAJ component has documented mislabeling in the glioma class. If near-duplicate slices from one patient straddle the train/test boundary, you are testing on training data and every number in your paper is inflated. A reviewer who runs a five-minute hash check will find this.

**How.**

- Exact: SHA256 of raw bytes.
- Near: `imagehash.phash` (and `dhash` as a cross-check) at hash_size 8–16; flag pairs with Hamming distance ≤ 5; tune the threshold by manually inspecting ~50 flagged pairs and record the chosen threshold and its false-positive rate in the report.
- Build a duplicate graph, take connected components, and treat **each component as one pseudo-patient**. This is your unit of splitting from here on.
- Rebuild the split **at the pseudo-patient level**, not the image level: no component may appear in more than one of train/val/test, and no component may be split across FL clients.
- Report, in the paper's data section and in `leakage_report.json`: number of exact duplicates, number of near-duplicate components, number of cross-split leaks found in the original split, and the resulting clean counts.
- Record but do not attempt to fix the SARTAJ glioma mislabeling — cite it as a known limitation, and (optional, strong) run a sensitivity check on a subset with the suspect files excluded.

**Acceptance.** `pytest tests/test_partition.py::test_no_cross_split_leakage` passes: the intersection of pseudo-patient IDs between any two splits is empty. `leakage_report.json` reports how many leaks existed in the original split.

### Step 1.3 — Canonical splits and preprocessing

**What.** Produce a stable, versioned manifest: `data/processed/manifest.csv` with columns `path, label, pseudo_patient_id, split, phash, source_component`. Splits: `train` / `val` / `test` at roughly 70/10/20 by pseudo-patient, stratified by class.

**Why.** A CSV manifest decouples *which files* from *how they are loaded*, so partitioning, augmentation and model code never re-derive the split. It also makes the split itself a citable artifact you can ship with the paper.

**How.**

- Preprocessing: convert to grayscale then replicate to 3 channels (so ImageNet-pretrained backbones work), resize with aspect-preserving pad to 224×224, normalize with ImageNet statistics for pretrained models and with dataset statistics for from-scratch models (compute and record both).
- Train-time augmentation: random resized crop (scale 0.8–1.0), horizontal flip, ±10° rotation, mild brightness/contrast jitter. **No vertical flip** — MRI orientation is anatomically meaningful. Eval-time: resize + center crop only.
- The `val` split is **global and held at the server** only for the optional `server_val` fitness mode and for hyperparameter selection. Be explicit in the paper about when it is and is not used; the default FedACO mode must not need it.

**Acceptance.** `manifest.csv` exists; class distribution per split is within 2 percentage points of the global distribution; `tests/test_partition.py` verifies stratification and the no-leak property.

### Step 1.4 — Federated partitioning

**What.** `data/partition.py` producing client-index assignments for five regimes:

| Regime | Parameterization | Purpose |
|---|---|---|
| `iid` | uniform random | control / upper reference |
| `dirichlet` | α ∈ {0.1, 0.3, 0.5, 1.0} | label skew, standard in the literature |
| `pathological` | 2 classes per client | extreme label skew |
| `quantity_skew` | client sizes ~ lognormal / power law | tests the data-size prior FedAvg relies on |
| `source_shift` | partition by original source component (Figshare / SARTAJ / Br35H) | **realistic cross-site feature shift** — the most clinically honest setting |

Client counts K ∈ {10, 20, 50}; participation fraction C ∈ {0.3, 1.0}.

**Why.** Dirichlet is the field's lingua franca and you need it for comparability. `source_shift` is the one that makes this a *medical* FL paper rather than a CIFAR paper with MRI pasted in: real heterogeneity between hospitals is feature/acquisition shift, not just label imbalance. `quantity_skew` directly attacks FedAvg's $n_k$ weighting, which is exactly what FedACO replaces — expect your largest gains there, and say so.

**How.**

- Use `flwr_datasets.partitioner.{IidPartitioner, DirichletPartitioner, PathologicalPartitioner}` where they fit; implement `quantity_skew` and `source_shift` yourself against the same interface.
- **Partition pseudo-patients, never individual images.**
- Every partition is a pure function of `(regime, params, K, seed)`; cache to `data/processed/partitions/{hash}.json`.
- Each client's local data is further split 90/10 into local-train / local-val (the local-val split is needed for the `client_probe` fitness mode in §4.5 and for per-client early stopping).
- Emit per-partition diagnostics: class-distribution heatmap (clients × classes), client size histogram, and a scalar heterogeneity index (mean pairwise Jensen–Shannon divergence between client label distributions). Log the index into every result file so plots can be drawn against *measured* heterogeneity rather than against the nominal α.

**Acceptance.** For every regime: partitions are disjoint, cover all training pseudo-patients, are reproducible under a fixed seed, and no client has zero samples. `make figures-data` writes the heatmaps to `paper/figures/partition_*.pdf`.

---

## 4. Phase 2–4 — Models, the FL harness, and the FedACO core

### Step 2.1 — Model factory and centralized reference

**What.** `models/factory.py` exposing `build_model(name, num_classes, pretrained)` for `resnet18` (ImageNet-pretrained and from-scratch), `simple_cnn` (~4 conv blocks, <1M params), and optionally `efficientnet_b0`. Then train each **centrally** on the pooled training set and record the result as the performance ceiling.

**Why.** Two reasons, both load-bearing. (1) The centralized number is the ceiling every FL method is measured against — "FedACO recovers 97% of centralized macro-F1 where FedAvg recovers 91%" is a far stronger sentence than a bare accuracy. (2) It is your debugging oracle: if centralized training doesn't reach sensible performance, the bug is in data or model, not in federation, and you will otherwise burn days hunting it in the wrong place.

**How.**

- Replace the final FC with a `num_classes`-way head; keep BatchNorm but **record that BN is a known FL pathology**. Offer `--norm {bn,gn}` and use **GroupNorm for the main FL experiments** (BN running statistics aggregate badly under non-IID and will confound your aggregation results). Run a small BN-vs-GN comparison and report it — it's a cheap, useful ablation.
- Training: AdamW or SGD+momentum, cosine schedule, early stopping on val macro-F1, mixed precision, batch size to fit the GPU.
- Save `results/centralized/{model}_{seed}.json` in the same schema as FL results.

**Acceptance.** Centralized ResNet-18 (pretrained, GroupNorm) reaches a stable test macro-F1 on the **de-duplicated** split, recorded with 5 seeds (mean ± std). Whatever that number is, it is now the ceiling. If it is implausibly close to 1.00, return to Step 1.2 — you still have leakage.

### Step 3.1 — Flower ClientApp

**What.** `fl/client_app.py` with `@app.train()` and `@app.evaluate()` handlers, plus `fl/task.py` holding model construction, local train/eval loops, and dataloader creation from the partition manifest.

**Why.** The client is where every strategy's per-client behavior lands. Making it strategy-agnostic — it reads its instructions from `msg.content["config"]` — means adding FedProx or SCAFFOLD later touches only the server and one conditional term, not the whole harness.

**How.**

- `train` handler: load arrays into the model, run `local_epochs` of SGD, return updated arrays plus a `MetricRecord` containing **everything the server might need for weighting**: `num-examples`, `train_loss_before`, `train_loss_after`, `local_val_loss`, `local_val_acc`, `local_val_f1`, `update_norm`, `num_batches`, `client_id`.
- Support optional proximal term (`mu`) and optional control variates (SCAFFOLD) via config flags, so one client serves all strategies.
- `evaluate` handler: return loss plus per-class counts so the server can compute macro-F1 globally rather than averaging per-client F1 (which is wrong under label skew — a client holding two classes has an undefined 4-class macro-F1).
- Resource config for simulation: `backend_config = {"client_resources": {"num_cpus": 2, "num_gpus": 0.25}}` — tune `num_gpus` to the card so several virtual clients share it.

**Acceptance.** A 2-client, 2-round smoke run completes and the returned metrics dict contains every field listed above.

### Step 3.2 — FedAvg reproduction (the harness baseline)

**What.** `fl/server_app.py` with `@app.main()`, driving the built-in `FedAvg` strategy, plus centralized server-side evaluation on the global test set each round via `evaluate_fn`, plus round-level logging to `results/`.

**Why.** Before you can claim FedACO beats FedAvg, your FedAvg must be a *fair* FedAvg — properly tuned, not a strawman. Reviewers assume a strawman by default. Build it first, tune it honestly, and record the tuning.

**How.**

- `strategy.start(grid=grid, initial_arrays=arrays, num_rounds=R, train_config=..., evaluate_fn=...)`.
- Tune FedAvg's local learning rate, local epochs and batch size with a small grid **on the val split**, and record the grid in `docs/EXPERIMENT_LOG.md`. Use the winner for every strategy unless a strategy has its own hyperparameters, in which case give it an equally sized budget.
- Log per round: global test loss/accuracy/macro-F1/AUC, mean client train loss, the selected client set, wall-clock, and aggregation wall-clock separately.
- Add `utils/checkpoint.py`: save strategy state + arrays + RNG state every `k` rounds; `--resume` restores. **Non-negotiable on Colab/Kaggle** — sessions die mid-sweep.

**Acceptance.** FedAvg on the IID partition converges to within a small, explicitly recorded gap of the centralized ceiling. FedAvg on Dirichlet α=0.1 is visibly worse and noisier — if it is not, your partitioner isn't producing real heterogeneity; go back to Step 1.4.

---

### Step 4 — FedACO: the contribution

> Implement this only after Step 3.2's acceptance passes. Write `paper/ALGORITHM.md` alongside the code; the pseudocode in the paper and the code must be kept in sync, and the easiest way is to write them together.

#### 4.1 Formulation

Let $S_t$ be the clients selected in round $t$, $|S_t| = K$. Let $\Delta_k = w_k^{t+1} - w_t$ be client $k$'s update. Aggregation is

$$w_{t+1} \;=\; w_t + \sum_{k \in S_t} \alpha_k \Delta_k, \qquad \alpha_k \ge 0,\ \ \textstyle\sum_k \alpha_k = s$$

FedAvg fixes $\alpha_k = n_k/\sum_j n_j$ and $s = 1$. **FedACO searches for $\alpha$ each round using ant colony optimization**, optionally allowing a global shrinkage $s \in [s_{\min}, 1]$ (a sum below 1 acts like weight decay and is known to help generalization — make it a config flag and ablate it).

#### 4.2 Construction graph and pheromone

Discretize the weight space. Define $L$ multiplier levels $\Lambda = \{\lambda_1,\dots,\lambda_L\}$ applied to the FedAvg weight — e.g. $L = 11$ levels log-spaced over $[0, 2.5]$ with $\lambda = 1$ always included, so $\lambda_l = 1\ \forall l$ recovers FedAvg exactly.

- The construction graph has $K$ decision stations (one per client); at station $k$ an ant picks exactly one level $l$, giving $\tilde{\alpha}_k = \lambda_l \cdot n_k / \sum_j n_j$. Normalize to the target sum.
- **Pheromone matrix** $\tau \in \mathbb{R}^{K \times L}$, $\tau_{k,l}$ = accumulated desirability of giving client $k$ level $l$.
- **Heuristic desirability** $\eta_{k,l}$ from server-side signals that cost no extra privacy:
  - $a_k = \cos(\Delta_k, \bar\Delta_{\text{rob}})$ — alignment with a robust (trimmed-mean) consensus direction;
  - $r_k = \|\Delta_k\| / \operatorname{median}_j \|\Delta_j\|$ — drift/magnitude anomaly;
  - $v_k$ — client-reported local validation improvement;
  - $q_k = n_k / \max_j n_j$ — the data-size prior.

  Combine: $d_k = \sigma\!\big(\beta_1 a_k - \beta_2|\log r_k| + \beta_3 v_k + \beta_4 q_k\big)$, rescale $d_k$ into $[\lambda_1, \lambda_L]$ as $\hat d_k$, and set $\eta_{k,l} = \big(1 + |\lambda_l - \hat d_k|\big)^{-1}$.
- **Transition rule** (Ant Colony System, pseudo-random-proportional):

  with probability $q_0$, $\;l^* = \arg\max_l \tau_{k,l}^{a}\,\eta_{k,l}^{b}$;
  otherwise sample $l$ with $p(l\mid k) \propto \tau_{k,l}^{a}\,\eta_{k,l}^{b}$.

**Why this design.** Two things must be true for the paper to hold up. First, the heuristic $\eta$ must use only information the server already has (updates and reported scalars) — otherwise FedACO costs more privacy than FedAvg and the contribution is compromised. Second, including $\lambda = 1$ in the level set means **the FedAvg solution is always reachable**, so ACO can only improve on it within the surrogate objective. That, plus the fallback in §4.7, makes "no worse than FedAvg" close to a structural property rather than an empirical hope.

#### 4.3 Cross-round pheromone persistence — *the "Adaptive" in FedACO*

**What.** Carry $\tau$ from round $t$ to $t+1$ with decay $\tau \leftarrow (1-\rho_{\text{round}})\,\tau + \rho_{\text{round}}\,\tau_0$, keyed by **client identity**, not by position in $S_t$.

**Why.** This is the core novelty and the reason ACO — rather than PSO or a GA — is the right metaheuristic here. Pheromone is *stigmergy*: persistent shared memory deposited in the environment. Carried across rounds, it becomes an accumulated record of which clients have historically produced useful updates, so the colony starts each round from an informed prior instead of from scratch. PSO and GA have no natural equivalent. **This is the sentence your Related Work section is built around, and the ablation in Phase 7 that removes cross-round persistence is the one that must show a drop.** If it doesn't, the "swarm intelligence" framing is decorative and you should know that early.

**How.** Store $\tau$ in a dict keyed by `client_id`; initialize rows for newly seen clients at $\tau_0$; apply an extra decay to clients absent for many rounds; expose `--pheromone-persistence {none,full,decayed}` as an ablation axis.

#### 4.4 Fitness — three pluggable modes

Implement all three behind one interface `Fitness.evaluate(alpha) -> float`. Default is `data_free`.

| Mode | Definition | Assumption | Role |
|---|---|---|---|
| `data_free` *(default)* | Surrogate from update geometry only (§4.5) | none beyond FedAvg | The method as proposed |
| `server_val` | Macro-F1 of $w(\alpha)$ on a small server-held val set | server holds data | Upper-bound reference; direct comparison to FedLAW-style methods |
| `client_probe` | Broadcast top-$M$ candidates in round $t+1$'s config; clients report local-val loss for each; fitness applied with one round's delay | one extra small payload | Realistic middle ground; strong ablation |

⚠️ **Cost asymmetry — plan for it.** Only `data_free` benefits from the $O(K^2)$ Gram trick in §4.6. `server_val` must *materialize* a full model per candidate and run a forward pass over the val set, so it is orders of magnitude more expensive; `client_probe` costs an extra broadcast payload. Therefore: run `data_free` at the full colony budget, and run `server_val` / `client_probe` only in the A3 ablation with a **heavily reduced budget** (e.g. $A=8$, $I=3$, evaluated on a 256-image val subset). Report the budget difference in the ablation table — do not compare a 300-evaluation `data_free` run against a 24-evaluation `server_val` run without saying so.

**Why three.** It converts the biggest reviewer objection ("your server needs data, so this isn't really federated") into a table. You show the data-free surrogate, you show how much is left on the table versus the oracle-ish `server_val`, and you show a practical delayed variant. Three modes cost perhaps 150 extra lines and buy an entire ablation subsection.

#### 4.5 The data-free surrogate objective

$$F(\alpha) \;=\; \gamma_1 \underbrace{\frac{\langle \Delta(\alpha),\, \bar\Delta_{\text{rob}}\rangle}{\|\Delta(\alpha)\|\,\|\bar\Delta_{\text{rob}}\|}}_{\text{consensus alignment}} \;-\; \gamma_2 \underbrace{\sum_k \alpha_k \|\Delta_k - \Delta(\alpha)\|^2}_{\text{weighted dispersion}} \;-\; \gamma_3 \underbrace{\big(\log K - H(\alpha)\big)}_{\text{concentration penalty}}$$

where $\Delta(\alpha) = \sum_k \alpha_k \Delta_k$, $\bar\Delta_{\text{rob}}$ is a coordinate-wise trimmed mean of $\{\Delta_k\}$, and $H(\alpha)$ is the entropy of the normalized weights.

**Why each term.** Alignment rewards weightings that move the global model along the direction most clients agree on (drift reduction). Dispersion penalizes weightings dominated by clients far from the resulting consensus (outlier and Byzantine resistance). The entropy term prevents collapse onto one or two clients, which would destroy the federation's purpose and overfit the surrogate. Together they encode "move decisively in the agreed direction without letting any one site take over."

#### 4.6 ⚠️ The Gram-matrix trick — do not skip this

A naive implementation evaluates $F(\alpha)$ by materializing $\Delta(\alpha)$ over $d \approx 11\text{M}$ parameters. With $A$ ants × $I$ iterations that is $A\!\cdot\!I\!\cdot\!K\!\cdot\!d$ flops per round — hundreds of times the cost of the round itself. FedACO would be unpublishable on efficiency grounds.

**Fix:** every term above depends on the updates only through inner products. Precompute the Gram matrix **once per round**:

$$G \in \mathbb{R}^{K\times K}, \qquad G_{ij} = \langle \Delta_i, \Delta_j\rangle \qquad \text{(cost } O(K^2 d)\text{, one pass)}$$

Then each fitness evaluation is $O(K^2)$, **independent of model size**:

- $\|\Delta(\alpha)\|^2 = \alpha^\top G\,\alpha$
- $\langle \Delta(\alpha), \bar\Delta\rangle = \tfrac{1}{K}\,\alpha^\top G \mathbf{1}$ (use the trimmed-mean coefficient vector in place of $\tfrac{1}{K}\mathbf{1}$ for $\bar\Delta_{\text{rob}}$)
- $\sum_k \alpha_k\|\Delta_k - \Delta(\alpha)\|^2 = \sum_k \alpha_k\big(G_{kk} - 2(G\alpha)_k + \alpha^\top G\alpha\big)$

With $K=20$, $A=30$, $I=10$: 300 evaluations × ~400 flops ≈ **negligible**, versus one $O(K^2 d)$ Gram pass that is itself cheaper than a single client's local epoch. Only the *winning* $\alpha$ is ever materialized into a real model. **This is what makes claim C3 true, and it is worth a short subsection of the paper in its own right.**

Implement in `aco/gram.py`; store $\Delta_k$ as flattened float32 (or float16 with float32 accumulation) tensors on GPU; compute $G$ with one `torch.matmul` on the stacked $[K \times d]$ matrix, chunked over $d$ if memory is tight.

#### 4.7 Pheromone update, bounds, schedules, and safety fallback

- **Evaporation + deposit** (MAX–MIN Ant System): $\tau_{k,l} \leftarrow (1-\rho)\tau_{k,l} + \rho\,Q\,F(\alpha^{\text{best}})\cdot\mathbb{1}[(k,l) \in \alpha^{\text{best}}]$, depositing from the iteration-best and, every $T$ iterations, the global-best.
- **Bounds** $\tau \in [\tau_{\min}, \tau_{\max}]$ to prevent premature stagnation; track and log **pheromone entropy** per round as a convergence diagnostic (it makes a genuinely nice figure).
- **Adaptive colony budget:** ants $A_t$ and iterations $I_t$ decay with the round index (aggregation choice matters most early); stop early when pheromone entropy falls below a threshold or the global best hasn't improved for $p$ iterations. Log the realized budget.
- **Safety fallback:** compute $F(\alpha_{\text{FedAvg}})$ every round. If $F(\alpha^{\text{best}}) \le F(\alpha_{\text{FedAvg}})$, use the FedAvg weights and increment a counter. Report the fallback rate in the paper — a low rate is evidence the search is doing real work; a high rate is an honest negative signal you need to see, not hide.

#### 4.8 The strategy class

**What.** `strategies/fedaco.py` defining `FedACO(Strategy)` (or subclassing `FedAvg` if the installed class exposes a clean override point — check `docs/FLOWER_API_NOTES.md`).

**How.**

- `configure_train`: sample $\lceil C\cdot N\rceil$ nodes from `grid`; build one `Message` per node carrying `arrays` and a `ConfigRecord` (`local_epochs`, `lr`, `server_round`, and for `client_probe` mode, the candidate $\alpha$ list).
- `aggregate_train`:
  1. Drop replies where `msg.has_error()`; log failures.
  2. Extract $\Delta_k$ and the metric scalars; **hold $\Delta_k$ on GPU as a stacked matrix**.
  3. Compute $G$ (`aco/gram.py`) and heuristic desirabilities (`aco/heuristics.py`).
  4. Run the colony (`aco/colony.py`) → $\alpha^{\text{best}}$, plus diagnostics.
  5. Apply the safety fallback.
  6. Materialize $w_{t+1} = w_t + \sum_k \alpha_k\Delta_k$; return `(ArrayRecord, MetricRecord)` where the metrics carry `alpha_entropy`, `alpha_max`, `fallback_used`, `aco_time_ms`, `gram_time_ms`, `best_fitness`, `fedavg_fitness`, `pheromone_entropy`, and the full $\alpha$ vector.
- Keep all ACO logic in `aco/` and make `strategies/fedaco.py` a thin adapter, so `aco/` can be unit-tested and reused by the PSO/GA controls in Phase 7 **against the identical fitness function**.

**Acceptance — unit tests (write these; the method is unfalsifiable without them):**

1. `test_gram_equivalence`: for random $\Delta$ and random $\alpha$, Gram-computed fitness matches the naive full-vector computation to within 1e-4.
2. `test_fedavg_recoverable`: with pheromone forced uniform, $q_0 = 1$, and $\eta$ forced to peak at $\lambda = 1$, FedACO's output equals FedAvg's output to within float tolerance.
3. `test_planted_bad_client`: with one client's update set to pure noise or a negated consensus direction, FedACO assigns it a weight materially below its FedAvg weight.
4. `test_pheromone_persists`: pheromone state at round $t+1$ is a decayed function of round $t$ and is keyed by client id under a changing participation set.
5. `test_simplex`: returned $\alpha$ is non-negative and sums to the configured target within 1e-6.
6. `test_overhead`: ACO wall-clock per round is below a configured fraction (e.g. 5%) of total round wall-clock in the smoke config.

---

## 5. Phase 5 — Baselines

**What.** Implement or wire up, all sharing the ClientApp and evaluation harness:

*Aggregation / optimization baselines:* `FedAvg` (built-in), `FedProx` (proximal $\mu$), `FedAdam` / `FedYogi` (server optimizers), `SCAFFOLD` (control variates), `FedNova` (normalized averaging).

*Weight-selection competitors — the ones that matter most:* loss-based weighting (FedNolowe-style normalized-loss weights), FedLAW-style learnable weights on the server val set, and the robust aggregators `Krum` / `Trimmed-Mean` / `Median`.

**Why.** The optimization baselines establish that you are competitive with the standard toolkit. The **weight-selection competitors are the real comparison** — they attack the same problem FedACO attacks, and a reviewer will consider your paper incomplete without them. The robust aggregators do double duty: they are weight-selection competitors *and* the natural comparison for the Byzantine robustness experiments in Phase 8.

**How.** Prefer built-ins where the installed `flwr` provides them (check `docs/FLOWER_API_NOTES.md`); implement the rest in `strategies/`. Give every baseline an honest hyperparameter search on the val split with a budget matched to FedACO's, and record every search in `docs/EXPERIMENT_LOG.md`. Where a public reference implementation exists, sanity-check your version reproduces its reported behavior on a standard setting.

**Acceptance.** Every strategy runs the smoke config end to end and writes a result JSON in the common schema. On the IID partition, all strategies land within a narrow band of each other (a baseline that is wildly worse on IID is broken, not outclassed).

---

## 6. Phase 6 — Experiment orchestration

### Step 6.1 — Config system and the result contract

**What.** Composable YAML configs (`base` + `data` + `model` + `strategy` + `experiment`) with CLI overrides. Every run writes exactly one self-describing JSON:

```jsonc
{
  "run_id": "<config_hash>_<seed>",
  "config": { /* fully resolved */ },
  "provenance": { "git_sha": "...", "dirty": false, "packages": {...}, "gpu": "...", "timestamp": "..." },
  "partition_stats": { "js_divergence": 0.41, "client_sizes": [...], "class_matrix": [[...]] },
  "rounds": [ { "round": 1, "test_acc": ..., "test_macro_f1": ..., "test_auc": ...,
                "per_class_recall": {...}, "train_loss": ..., "alpha": [...],
                "alpha_entropy": ..., "fallback_used": false,
                "aco_time_ms": ..., "round_time_s": ... } ],
  "final": { "test_macro_f1": ..., "best_round": ..., "rounds_to_target": {"0.85": 23} },
  "status": "completed"
}
```

**Why.** One flat, complete record per run means analysis scripts never need to know how a run was produced, sweeps can skip completed `run_id`s on resume, and "which commit produced Table 3" is answerable a year later. Embedding `partition_stats` lets you plot gains against *measured* heterogeneity, which is a much more convincing figure than plotting against nominal Dirichlet α.

### Step 6.2 — Sweep runner with resume

**What.** `scripts/run_sweep.py` expanding an experiment config into a run list, executing sequentially, skipping completed `run_id`s, checkpointing every $k$ rounds, and writing a `results/manifest.jsonl` index.

**Why.** On a single GPU the main table is roughly (11 strategies × 5 partitions × 5 seeds) runs. Colab and Kaggle sessions expire. Without resume you will lose a full sweep at least once — everyone does.

**How.** Write a `.lock` file per in-flight run; on startup, treat locks older than a timeout as crashed and re-queue. `--dry-run` prints the run list and an estimated GPU-hour total. Order runs so that **one complete seed of every configuration finishes first** — that way a partial sweep still yields a full (if noisy) table.

### Step 6.3 — The main experiment

**What.** Full factorial: strategies × {iid, dirichlet 0.1/0.3/0.5, pathological, quantity_skew, source_shift} × {K=20, C=1.0} × 5 seeds, plus a reduced grid for K ∈ {10, 50} and C = 0.3. Primary metric **macro-F1**; secondary accuracy, one-vs-rest AUC, per-class recall (report glioma recall explicitly — it is the clinically costly class), rounds-to-target-macro-F1, total communication, and aggregation overhead.

**Why macro-F1 as primary.** ⚠️ *Corrected 2026-09-25: the archive obtained for this project is exactly class-**balanced** (1,800/class), so the sentence below is false of it as written. It is true of the data actually trained on -- after pseudo-patient de-duplication the distribution is 2.56:1 with `notumor` at 12.07%, because the archive's balance is manufactured by duplicating that class (67.9% redundancy against 18-26%). The metric choice stands; the reason must be stated against the de-duplicated distribution. See `docs/OPEN_QUESTIONS.md` "Dataset variant ... RESOLVED" and `paper/04_EXPERIMENTAL_SETUP.md` §4.1.* The dataset is class-imbalanced and the clinical cost of a missed tumor is asymmetric. Accuracy would let a method look good by exploiting the majority class. Fix the primary metric *now*, before seeing results, and report it everywhere — choosing the metric after the fact is the most common way honest researchers accidentally p-hack.

**Acceptance.** `make main` runs to completion (possibly across several resumed sessions); `results/manifest.jsonl` contains one entry per planned run with `status: completed`.

---

## 7. Phase 7 — Ablations and sensitivity

> **Build ablation A1 as soon as FedACO runs — before the full main sweep.** It is the claim most likely to fail, and failing fast is worth days.

| ID | Ablation | What it isolates | Why it matters |
|---|---|---|---|
| **A1** | **Equal-fitness-budget controls:** replace the colony with random search, coordinate grid search, PSO, and a GA — *identical* fitness function, *identical* number of evaluations | Whether ACO specifically helps | **The make-or-break experiment (claim C2).** If ACO ties random search at equal budget, you do not have an ACO paper — you have an adaptive-weighting paper, and you should reframe honestly rather than overclaim |
| **A2** | Pheromone persistence: `none` / `decayed` / `full` | Whether cross-round stigmergy helps | Justifies ACO over memoryless metaheuristics; this is the paper's conceptual hook (§4.3) |
| **A3** | Fitness mode: `data_free` / `server_val` / `client_probe` | Cost of the privacy-preserving surrogate | Pre-empts "your server needs data" |
| **A4** | Surrogate terms: drop $\gamma_1$, $\gamma_2$, $\gamma_3$ individually | Which signal drives the gain | Turns a hand-designed objective into an evidence-backed one |
| **A5** | Heuristic components: drop alignment / norm-ratio / val-improvement / size prior | Value of each server-side signal | Shows the heuristic isn't cargo cult |
| **A6** | Hyperparameter sensitivity: $a$, $b$, $\rho$, $q_0$, $A$, $I$, $L$ (one-at-a-time around the default) | Robustness to tuning | Metaheuristic papers are rejected for knife-edge sensitivity. Report a heatmap; the flatter the better |
| **A7** | Shrinkage: $\sum\alpha = 1$ vs. learned $s \le 1$ | Generalization effect of weight shrinking | Connects cleanly to known results on global weight shrinking |
| **A8** | Adaptive vs. fixed colony budget | Efficiency of the schedule | Supports the overhead claim (C3) |
| **A9** | Normalization: BatchNorm vs. GroupNorm | Confound control | Shows your gains are aggregation, not a BN artifact |
| **A10** | Layer-wise $\alpha$ (per layer group) vs. model-wise | Extension | Optional; if it works, it's a strong final subsection. Time-box it |

**How.** Each ablation is an experiment config; all controls in A1 must call the *same* `Fitness` object and be capped by the *same* evaluation counter — implement the budget cap inside `Fitness` itself so no control can cheat by accident. A1 controls live in `aco/controls.py`.

**Acceptance.** Every ablation writes results in the common schema; `make ablations` reproduces all tables from `results/`.

---

## 8. Phase 8 — Robustness and stress tests

| ID | Test | Setup | Why |
|---|---|---|---|
| **R1** | Label-flipping attackers | 10%, 20%, 30% of clients flip labels | The dispersion term in §4.5 should down-weight them automatically. Compare against Krum / Trimmed-Mean / Median |
| **R2** | Gaussian / sign-flip update attackers | same fractions | Tests the norm-ratio heuristic |
| **R3** | Stragglers & partial participation | C ∈ {0.1, 0.3, 0.5}; random dropouts mid-round | Cross-round pheromone should shine here — it remembers clients that are currently absent |
| **R4** | DP noise | Gaussian noise on client updates at several $\sigma$ | Answers "does your method survive differential privacy?" — asked of every FL paper that inspects updates |
| **R5** | Client-count scaling | K ∈ {10, 20, 50, 100} | Verifies the $O(K^2)$ overhead claim empirically; plot measured ACO time vs. K against the $K^2$ curve |
| **R6** | Cold start / new clients | clients joining after round 20 | Tests the pheromone-initialization policy for unseen clients |

**Why.** R1–R2 convert a side effect of your design into a *second* contribution ("FedACO provides Byzantine resilience for free"), which materially strengthens the paper. R4 is the standard rebuttal-killer. R5 is the empirical half of claim C3 — a measured curve matching $K^2$ is far more convincing than a complexity assertion.

**Acceptance.** All robustness experiments run; `paper/figures/robustness_*.pdf` generated; any case where FedACO *loses* is recorded in `docs/EXPERIMENT_LOG.md` and stated in the paper's limitations. Do not quietly drop losing configurations.

---

## 9. Phase 9 — Analysis, statistics, and paper artifacts

### Step 9.1 — Aggregation and statistics

**What.** `scripts/aggregate_results.py` loads `results/*.json` into a tidy DataFrame and computes, per (strategy, partition): mean ± std over seeds; paired comparisons of FedACO against each baseline using **Wilcoxon signed-rank** over seeds (do not assume normality with n=5); **Holm–Bonferroni correction** across the family of comparisons; and **Cohen's d** effect sizes. Also compute rounds-to-target-macro-F1 with bootstrap CIs.

**Why.** With 5 seeds and a dozen comparisons, uncorrected p-values manufacture significance. A reviewer who checks will find it. Reporting corrected p-values plus effect sizes is both more honest and more persuasive — a large effect size with a marginal p-value is a better story than the reverse, and only one of them survives scrutiny.

**Guardrail:** if FedACO's improvement over the best baseline is smaller than the seed-to-seed standard deviation, **say so plainly** and pivot the paper's emphasis toward the settings where the effect is real (likely `quantity_skew`, `source_shift`, high heterogeneity, and the Byzantine settings). A narrow, well-supported claim publishes; an overreaching one does not.

### Step 9.2 — Figures

`scripts/make_figures.py`, all vector PDF, colorblind-safe palette, consistent styling, ≥ 10pt fonts at final print size:

1. Convergence curves (macro-F1 vs. round), mean ± 95% CI band, one panel per partition regime.
2. Final macro-F1 bar chart with error bars, grouped by partition.
3. **α heatmap over rounds** (clients × rounds) for one representative run, with true client "quality" annotated — this is the figure that shows the method *doing something interpretable*, and it will be the one reviewers remember.
4. Pheromone entropy and α entropy vs. round (search-convergence diagnostic).
5. Aggregation overhead vs. K, with the fitted $K^2$ curve overlaid.
6. Robustness: macro-F1 vs. attacker fraction, FedACO vs. Krum / Trimmed-Mean / FedAvg.
7. Sensitivity heatmap over the two most important ACO hyperparameters.
8. Partition diagnostic heatmaps (from Phase 1).
9. Gain vs. measured Jensen–Shannon heterogeneity — a scatter across all runs with a fitted trend, showing the gain grows with real heterogeneity.

### Step 9.3 — LaTeX tables

`scripts/make_tables.py` emits `paper/tables/*.tex` with `booktabs`: main results (strategies × partitions, mean ± std, best in bold, significance markers), ablation tables, overhead table, robustness table. **Generate them from `results/` — never hand-type a number into the paper.**

**Acceptance.** `make figures tables` regenerates every artifact from `results/` with no manual steps. Deleting `paper/figures/` and `paper/tables/` and re-running restores them byte-for-byte identical.

---

## 10. Phase 10 — Reproducibility, documentation, release

**What.**

- `README.md`: one-paragraph method summary, the headline result, install, `make` targets, a 10-minute smoke command, and a table mapping every paper figure/table to the command that produces it.
- `paper/ALGORITHM.md`: formal pseudocode (Algorithm 1: FedACO round; Algorithm 2: ant construction; Algorithm 3: pheromone update), complexity analysis, and the full default hyperparameter table.
- `scripts/verify_repro.py`: runs the smoke config and checks final metrics fall within a recorded tolerance band.
- `tests/`: full suite green; add the §4.8 tests plus end-to-end smoke tests for every strategy.
- `docs/EXPERIMENT_LOG.md`: chronological record of every sweep, including failures and abandoned directions.
- CI (GitHub Actions): lint + unit tests + 2-round smoke on CPU.
- Release: tag `v1.0`, archive to Zenodo for a DOI, and cite that DOI in the paper.

**Why.** Reproducibility checklists are now mandatory at most relevant venues, and the figure→command mapping table is the single highest-leverage thing you can put in a README for a reviewer.

**Acceptance.** A clean clone on a fresh machine reaches a passing `make test && make smoke` using only the README.

---

## 11. Suggested execution order and time budget

Assumes one GPU and a part-time author; adjust freely.

| Week | Phases | Milestone |
|---|---|---|
| 1 | 0, 1.1–1.2 | Repo runs; **data audit done and leakage quantified** |
| 2 | 1.3–1.4, 2.1 | Manifest, partitions, centralized ceiling |
| 3 | 3.1–3.2 | FedAvg reproduced and honestly tuned; checkpoint/resume working |
| 4–5 | 4 | FedACO implemented; all §4.8 unit tests green |
| 5 | **7 / A1 only** | **Equal-budget control run — go/no-go on claim C2** |
| 6 | 5 | All baselines running |
| 7–8 | 6 | Main sweep (run overnight; resume as needed) |
| 9 | 7 | Remaining ablations |
| 10 | 8 | Robustness suite |
| 11 | 9 | Statistics, figures, tables |
| 12 | 10 | Packaging, README, release, paper draft |

**The week-5 gate is the important one.** If A1 shows ACO ties random search at equal budget, stop and reframe before investing weeks 6–12 in a claim that won't survive review.

---

## 12. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Gains vanish with an ImageNet-pretrained backbone | High | High | Report both pretrained and from-scratch; from-scratch is the honest FL setting. Emphasize high-heterogeneity regimes |
| ACO ties equal-budget random search (C2 fails) | **Medium-High** | **Fatal to the framing** | Ablation A1 at week 5. If it fails: reframe around cross-round pheromone memory (A2) and Byzantine robustness (R1–R2), which random search cannot provide |
| Data leakage inflates all numbers | High if unchecked | Fatal | Phase 1.2 de-duplication at pseudo-patient level; report the leak count found |
| Seed variance swamps the effect | Medium | High | ≥ 5 seeds, Wilcoxon + Holm, report effect sizes, narrow the claim to regimes where it holds |
| ACO overhead dominates round time | Low (with §4.6) | High | Gram-matrix formulation; `test_overhead` unit test; R5 scaling curve |
| Flower API drift breaks the harness | Medium | Medium | Pin exact version; `docs/FLOWER_API_NOTES.md`; CI smoke test |
| Colab/Kaggle session loss | High | Medium | Checkpoint/resume + sweep manifest from day one |
| Acronym collision with existing FedACo | Certain | Low-Medium | Differentiate and cite, or rename (§0.3) — author's decision |
| Reviewer: "incompatible with secure aggregation" | Certain | Medium | State it as a limitation; run R4 (DP noise); discuss clustered SecAgg as future work |

---

## 13. Paper outline ↔ artifact mapping

| Paper section | Produced by |
|---|---|
| 1. Introduction | §1.1–1.2 of this plan |
| 2. Related work — adaptive aggregation | §1.2 (FedAAW, FedLAW, FedNolowe, DaWa) + the FedACo differentiation from §0.3 |
| 2b. Related work — swarm intelligence in FL | ACO for transmission scheduling / PSO for device selection; ACO for federated feature selection; swarm methods for communication efficiency |
| 3. Method | `paper/ALGORITHM.md` (Algorithms 1–3), §4.1–4.7 |
| 3.x Complexity & efficiency | §4.6 Gram formulation + R5 measured curve |
| 4. Experimental setup | Phase 1 data card + leakage report, Phase 6 config |
| 5.1 Main results | Table 1, Figures 1–2 |
| 5.2 Ablations | Tables 2–3, Figures 4, 7 (A1–A10) |
| 5.3 Robustness | Table 4, Figure 6 (R1–R6) |
| 5.4 Efficiency | Overhead table, Figure 5 |
| 5.5 Interpretability | Figure 3 (α heatmap), Figure 9 (gain vs. heterogeneity) |
| 6. Limitations | §1.4 + any losing configurations from Phase 8 |
| 7. Reproducibility | Phase 10, Zenodo DOI |

---

## 14. Default hyperparameters (starting point — tune in A6)

```yaml
fedaco:
  levels:                 [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
  alpha_exponent_a:       1.0     # pheromone influence
  beta_exponent_b:        2.0     # heuristic influence
  rho_evaporation:        0.10
  q0_exploitation:        0.70    # ACS pseudo-random-proportional
  n_ants:                 30      # decayed by budget schedule
  n_iterations:           10
  tau_init:               1.0
  tau_min:                0.01
  tau_max:                10.0
  pheromone_persistence:  decayed # {none, decayed, full}
  rho_round:              0.30    # cross-round decay toward tau_init
  fitness_mode:           data_free   # {data_free, server_val, client_probe}
  gamma1_alignment:       1.0
  gamma2_dispersion:      0.50
  gamma3_entropy:         0.10
  weight_sum:             1.0     # or 'learned' for shrinkage (A7)
  safety_fallback:        true
  budget_schedule:        cosine  # A_t, I_t decay with round
  early_stop_patience:    3

federated:
  num_clients:            20
  fraction_train:         1.0
  num_rounds:             100
  local_epochs:           2
  local_lr:               0.01
  local_batch_size:       32
  optimizer:              sgd
  momentum:               0.9

model:
  name:                   resnet18
  pretrained:             true      # also run false
  norm:                   groupnorm # BN confounds FL aggregation
  num_classes:            4
  image_size:             224

seeds:                    [0, 1, 2, 3, 4]
primary_metric:           test_macro_f1
```

---

## 15. First five commands for the implementing agent

```bash
# 1. Scaffold (Phase 0.1)
mkdir -p fedaco && cd fedaco && git init
# ... create the layout from §0.4 ...

# 2. VERIFY THE FLOWER API BEFORE WRITING ANY FL CODE (Phase 0.2) — do not skip
python -c "import flwr, inspect; from flwr.serverapp.strategy import Strategy, FedAvg; \
print(flwr.__version__); print(inspect.getsource(Strategy)); print(inspect.getsource(FedAvg)[:8000])" \
  | tee docs/FLOWER_API_NOTES.md

# 3. Data: download, audit, DE-DUPLICATE (Phase 1.1–1.2)
python -m fedaco.data.download --verify
python -m fedaco.data.dedup --phash-threshold 5 --report data/processed/leakage_report.json

# 4. Establish the ceiling (Phase 2.1)
python scripts/run_experiment.py --config configs/experiment/centralized.yaml --seed 0

# 5. Reproduce FedAvg, then build FedACO on top (Phase 3.2 → 4)
python scripts/run_experiment.py --config configs/experiment/smoke.yaml \
  --strategy fedavg --data dirichlet_0.5 --seed 0
```

---

## 16. Sources consulted for this plan

- [Flower Strategy source (`flwr.serverapp.strategy.strategy`)](https://flower.ai/docs/framework/_modules/flwr/serverapp/strategy/strategy.html) — abstract method signatures in §0.2
- [Upgrade to Message API — Flower Framework](https://flower.ai/docs/framework/how-to-upgrade-to-message-api.html) — the 1.21 API change
- [Flower Strategy Abstraction](https://flower.ai/docs/framework/explanation-flower-strategy-abstraction.html) — custom strategy guidance
- [flwr_datasets partitioner API](https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html) — partitioner class names
- [flwr release history (latest 1.36.0, Sept 2026)](https://libraries.io/pypi/flwr)
- [Brain Tumor MRI Dataset (Kaggle, Nickparvar)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [Original figshare brain tumor dataset (Cheng et al.)](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
- [Revisiting Weighted Aggregation in Federated Learning with Neural Networks (FedLAW)](https://consensus.app/papers/details/e3b1614377c75e02bae69befe98cdc9a/)
- [Federated Learning With Adaptive Aggregation Weights for Non-IID Data in Edge Networks (FedAAW)](https://consensus.app/papers/details/fe72b05ef602535a9667152c1dd2c5c9/)
- [FedNolowe: normalized loss-based weighted aggregation](https://consensus.app/papers/details/d794ba62a24c586d95a5146b36c11c61/)
- [Optimizing Hierarchical Federated Learning: A Reinforcement Learning Approach (DaWa)](https://consensus.app/papers/details/c163941fdca25558a864671f2e476a28/)
- [A Hybrid Swarm Intelligence Approach for Optimizing MLLM Deployment in Edge-Cloud FL (PSO + ACO at the systems layer)](https://consensus.app/papers/details/3a8a1bf1af025bcbb10807d331eae6de/)
- [Fed-MGACO: federated multi-label feature selection via game-theoretic evolutionary ACO](https://consensus.app/papers/details/38566e4442005e5db5aba243ba834a6f/)
- [Decentralized Federated Learning With Distributed Aggregation Weight Optimization](https://consensus.app/papers/details/e4af10857bb05da6a5dedaa3a4755ff3/)
- [FedACo: Adaptive Collaboration with Fine-Grained Aggregation for Personalized FL](https://link.springer.com/chapter/10.1007/978-981-95-4987-0_1) — **acronym collision, see §0.3**
