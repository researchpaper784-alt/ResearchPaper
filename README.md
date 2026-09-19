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

Status: **All nine phases have working infrastructure.** Data pipeline, centralized
ceiling, Flower FL harness, FedACO, the baselines and their honest hyperparameter search,
the main sweep, the ablation and robustness sweeps, and the tables and figures. The
centralized ceiling is measured on real data (SimpleCNN@112: 0.9300 macro-F1 BatchNorm /
0.9203 GroupNorm; ResNet-18: 0.9701 over 3 seeds — see
[docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)).

⚠️ **No federated results exist yet.** Every FL run so far has been against a synthetic
pixel cache, because the raw images need Kaggle credentials the build environment does not
have — so the pipeline is verified end to end and not one number in it is. `results/` is
empty by design (gitignored). Run `make validate-fedaco && make health` before
committing compute to any sweep: it is about twenty minutes, and it says whether FedACO's
mechanism is doing anything at all.

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
| Robustness sweep (attacks, dropout) | `make robustness` (`make robustness-plan` first) | 8 |
| `paper/tables/robustness.{md,csv}` | `make tables-robustness` | 8 |

### Before the sweep

```bash
make pheromone-budget       # can a run of this length answer question 1 at all?
make fitness-landscape      # is the fitness optimum degenerate at this K?
make validate-fedaco        # two short runs: FedACO and FedAvg, same seed and partition
make health                 # does FedACO's mechanism actually do anything?
```

`make health` answers four questions the accuracy columns cannot: whether the colony is
searching at all (pheromone entropy against its `log(L)` ceiling), whether the deposit
floor has engaged, whether the fitness optimum is degenerate, and whether the safety
fallback — not the colony — is carrying any margin over FedAvg. A sweep run before that check can produce 576 cells of numbers about
a mechanism that was never running. It costs about twenty minutes against the sweep's
several hundred hours.

⚠️ **Don't shrink K for the validation run.** Dispersion is a weighted variance, so it is
*exactly zero* at a single-client vertex — the fitness pays nothing for discarding every
client but one, and the only guard is `gamma_entropy * log K`, which weakens as the
federation shrinks. On synthetic deltas that guard needs `gamma_entropy > 0.17` at K=2 and
`> 0.13` at K=4 against the default `0.1`, so a K=4 smoke test sits in the regime where
the fitness optimum is degenerate while the K=20 sweep it is validating does not. A
correctly working colony there returns a one-client answer and every other signal reads as
success. `corner_margin` is logged per round and question 3 reads it;
`make fitness-landscape` maps the regime before the run.

Question 1 is judged against what the run's own budget makes reachable, not against a
fixed gap, and `make pheromone-budget` prints that ceiling for each budget. τ starts *at*
the entropy ceiling and walks away from it at a rate set by the deposit and the iteration
count, so a short run cannot distinguish "not searching" from "hasn't moved yet" — an
earlier version of the check reported INERT on a four-round run whose best possible score
was barely above the threshold (docs/EXPERIMENT_LOG.md, 2026-09-17). Shorten `ROUNDS` and
the check quietly stops meaning anything; the budget table says when.

⚠️ **The Simulation Runtime creates 2 clients unless told otherwise.** `num-clients` is
this project's own key — it decides how many ways the *data* is partitioned. How many
ClientApps actually exist is Flower's `num_supernodes`, which defaults to **2**. Set
unequal, a run silently trains on 2 partitions while its result file records 20, and a
`min-train-nodes` above the supernode count simply hangs at round 0 with idle actors and
no error. `make validate-fedaco` and `scripts/run_sweep.py` both configure this for you;
to do it by hand:

```bash
flwr federation simulation-config --num-supernodes 20 --client-resources-num-cpus 1
```
