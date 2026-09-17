# FedSwarm

**FedSwarm: Adaptive Aggregation Optimization in Federated Learning using Swarm Intelligence for Brain Tumor Diagnosis**

> Renamed from the original working title "FedACO" to avoid a naming collision with a
> 2025 PRCV/Springer paper that already uses the acronym "FedACo" for an unrelated method.
> See [docs/OPEN_QUESTIONS.md](docs/OPEN_QUESTIONS.md).

FedSwarm replaces FedAvg's fixed, data-size-proportional aggregation weights with weights
searched each round by an Ant Colony Optimization (ACO) metaheuristic over client updates,
with pheromone persisting across communication rounds as a memory of client reliability.
Evaluated on 4-class brain tumor MRI classification under realistic non-IID federated
partitions (label skew, quantity skew, and cross-source feature shift).

Status: **Phases 0–5 built.** Data pipeline, centralized ceiling, Flower FL harness,
FedACO, the baseline strategies, the per-baseline hyperparameter search
(`make hparam-search`) and the IID acceptance gate (`make iid-band`) are all implemented
and tested. The centralized ceiling is measured on real data (SimpleCNN@112: 0.9300
macro-F1 BatchNorm / 0.9203 GroupNorm; ResNet-18: 0.9701 over 3 seeds — see
[docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)).

**No federated results exist yet.** The FL harness is verified end-to-end, but the
hyperparameter search and the IID gate have only been exercised on plumbing; both need a
real training environment with the dataset to produce numbers. The main sweep (Phase 6),
ablations (Phase 7), robustness (Phase 8) and analysis (Phase 9) are not built —
`results/` is empty by design (gitignored), and `make main`/`figures`/`tables` refer to
scripts that do not exist yet.

## Setup

Requires Python 3.11 or 3.12 and [`uv`](https://github.com/astral-sh/uv).

```bash
make setup
```

This creates `.venv/` and installs the project in editable mode. Note: `flwr[simulation]`
(via its `ray` dependency) has no wheel for Intel macOS. Local `make setup` installs plain
`flwr` for API work and unit tests; install the `simulation` extra in the actual training
environment (Colab/Kaggle, both Linux):

```bash
uv pip install -e ".[simulation]" --python .venv/bin/python
```

## Repository layout

See [CLAUDE.md](CLAUDE.md) for working conventions and [docs/FLOWER_API_NOTES.md](docs/FLOWER_API_NOTES.md)
for the verified, installed Flower API this codebase targets.

## Figure/table → command map

Every artefact the paper cites, and the command that produces it. Each command is
resumable and skips work already done, so re-running one after an interruption costs only
what is missing.

| Artefact | Command | Phase |
|---|---|---|
| Partition heterogeneity figures | `make figures-data` | 1 |
| Centralized ceiling table | `make test` then `scripts/summarize_centralized.py` | 2 |
| Per-baseline hyperparameter search | `make hparam-search` (`make hparam-search-plan` to cost it) | 5 |
| IID narrow-band acceptance gate | `make iid-band` | 5 |
| Main sweep, all cells | `make main` (`make main-plan` first — it prints the cost) | 6 |
| Ablation sweep | `make ablations` (`make ablations-plan` first) | 7 |
| `paper/tables/main.{md,csv}` | `make tables` | 9 |
| `paper/tables/ablation.{md,csv}` | `make tables-ablation` | 9 |
| `convergence.png`, `comparison.png`, `colony_health.png` | `make figures` | 9 |
| `ablations.png` | `make figures-ablation` | 9 |

### Before the sweep

```bash
make validate-fedaco K=4    # two short runs: FedACO and FedAvg, same seed and partition
make health                 # does FedACO's mechanism actually do anything?
```

`make health` answers three questions the accuracy columns cannot: whether the colony is
searching at all (pheromone entropy against its `log(L)` ceiling), whether the deposit
floor has engaged, and whether the safety fallback — not the colony — is carrying any
margin over FedAvg. A sweep run before that check can produce 360 cells of numbers about
a mechanism that was never running. It costs about twenty minutes against the sweep's
several hundred hours.

⚠️ **Set client resources once per machine first.** The Flower Simulation Runtime assigns
**2 CPUs per ClientApp**, so `K=20` requests 40 cores; oversubscription does not queue, it
stalls at round 0 with idle actors and no error. `make validate-fedaco` runs this for you:

```bash
flwr federation simulation-config --client-resources-num-cpus 1
```
