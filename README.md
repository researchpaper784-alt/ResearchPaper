# FedSwarm

> ### 👋 New to this project? Read [`HANDOVER.md`](HANDOVER.md) first.
> It covers current status, the three decisions already taken and how to reverse them,
> exactly what to run in what order, and the landmines that have each already cost
> someone a day.

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
| `alpha_heatmap.png` (plan §9.2 fig 3) | `make figures` | 9 |
| `overhead_vs_k.png` (fig 5) | `make overhead` then `make figures` | 8 |
| `robustness.png` (fig 6) | `make robustness-granular` then `make figures` | 8 |
| `sensitivity.png` (fig 7) | needs a factorial sweep — see note below | 7 |
| `gain_vs_heterogeneity.png` (fig 9) | `make main` then `make figures` | 9 |
| Robustness sweep (attacks, dropout) | `make robustness` (`make robustness-plan` first) | 8 |
| `paper/tables/robustness.{md,csv}` | `make tables-robustness` | 8 |

`make figures` writes whichever of the nine it has data for and names the rest on a
`skipped (no data)` line, so a missing figure says which sweep it is waiting on rather than
disappearing.

**`sensitivity.png` has no sweep yet.** Plan §9.2 asks for a heatmap over two ACO
hyperparameters, but `ablation_a6.yaml` is a *one-at-a-time* sweep: each arm moves one knob
off a shared centre, so its cells form a cross, not a grid, and `figure_sensitivity` refuses
to image it (a KxK cross fills only 2K-1 cells, and the colour scale would be read across
the holes). Producing that figure needs a small factorial sweep over the two knobs A6 shows
to matter most — worth declaring only after A6 has run and identified them.

### Check a sweep config runs before spending GPU hours on it

```bash
python scripts/synthetic_heterogeneous_dataset.py --out-dir /tmp/fx --num-images 600 --image-size 32
make preflight-configs FIXTURE=/tmp/fx
```

Runs one cell of every declared arm at 2 rounds and K=6. `strategy_from_run_config` rejecting a
bad config is necessary and not sufficient — it builds the strategy, not the round, and a config
can construct cleanly then die at round 1. The worst case does not raise at all: a
`min-train-nodes` above the supernode count makes Flower's `sample_nodes` wait in a
`while ...: sleep(1)` loop that never gives up, so the run consumes the whole session and writes
nothing. A minute per arm on CPU against several hundred GPU-hours.

### Which sweeps are worth running right now

Two answers, depending on the deadline.

**If you have 7–9 days**, the full set does not fit and no ordering of it does: it is
351–701 GPU-hours, which at Kaggle's ~30 GPU-h per week per account is 12–23 weeks on one
account and 4–8 across three. Run the reduced set — **342 cells, ~70 GPU-h**, in this order:

| order | make target | cells | GPU-h | who |
|---|---|---|---|---|
| 1 | `gate-fitness` | 8 | 0.25 | B — which fitness fix closes the corner, ~15 min |
| 2 | `a1-reduced` | 80 | 16.7 | B — the go/no-go on claim C2 |
| 3 | `main-reduced` | 144 | 30.0 | C — the headline table |
| 4 | `r1-reduced` | 40 | 8.3 | C — **before** `r2-reduced`, see below |
| 5 | `r2-reduced` | 40 | 8.3 | C |
| 6 | `a2-reduced` | 30 | 6.2 | C |
| | `b-all`, `c-all` | **342** | **~70** | |

Every cut is named in the config header it belongs to, so the paper states them rather than
a reviewer finding them. `r1-reduced` must run before `r2-reduced`: R2 ships with no
unattacked arm and uses the `clean` arm R1 writes to the same `results/fl/robustness`, so
trimming R1 leaves every R2 delta `None` — which renders as the same "—" as a cell with no
paired seeds. A1 keeps all 80 cells deliberately; its five search methods, two partitions
and eight seeds are the experiment, and only its per-cell cost was cut.

**If the deadline moves**, the full sweeps are still the right thing to run, and they do not
all have the same prerequisites:

| tier | sweeps | cells | GPU-h | condition |
|---|---|---|---|---|
| 1 | R1–R6, `robustness.yaml`, A9, A3 | 430 | 90–179 | none — run these first |
| 2 | A4, A5, A7 | 175 | 36–73 | survive a reframe, but only interpretable once the colony's mechanism is sound |
| 3 | A2, A6, A8, `ablation_all` | 340 | 88–175 | ACO-specific; dropped entirely if claim C2 is reframed |

Tier 2 is the distinction worth being careful about: those three ablate the *fitness* and
*desirability* terms, which a reframe keeps — but if the colony is not searching (the safety
fallback currently fires in 44% of rounds, where a coordinate sweep improves in every one) then
every arm collapses toward the same behaviour and the ablation measures noise rather than the
term it names. `scripts/check_fedaco_health.py --strict` is the gate on that.

### On a GPU box, set `GPUS` before any sweep

```bash
make main GPUS=0.05        # 1/num-clients lets every ClientApp share one card
```

Ray hides the GPU from any actor requested with `num_gpus=0`, so with no fraction set every
ClientApp trains on **CPU** while the ServerApp keeps the GPU. Server-side evaluation and
every logged metric look completely normal — the only symptom is wall-clock, against a cost
projection that a small-core box makes optimistic anyway. For `main.yaml` that is 576 cells ×
100 rounds of CPU training: it does not finish inside any Kaggle quota and no result file says
why. Both runners now refuse to start in that situation and name the fraction to pass; leave
`GPUS` empty on CPU, or pass `--gpus-per-client 0` to ask for CPU deliberately.

### Before the sweep

**On Colab or Kaggle**, pass `PY=python FLWR=flwr` to any `make` target — those runtimes
install into the system Python and have no `.venv`. `notebooks/kaggle_main_sweep.ipynb` is
the ready-to-run Kaggle notebook for the main sweep; it does this for you, restores results
from a previous session, and runs the plan's two gates before the 576-cell sweep.

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
