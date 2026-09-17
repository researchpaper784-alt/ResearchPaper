.PHONY: setup data test lint smoke validate-fedaco health pheromone-budget hparam-search hparam-search-plan iid-band main main-plan ablations ablations-plan robustness robustness-plan tables-robustness figures-data figures figures-ablation tables tables-ablation clean

setup:
	uv venv --python 3.12 .venv
	uv pip install -e ".[dev]" --python .venv/bin/python

data:
	.venv/bin/python -m fedswarm.data.download --verify

test:
	.venv/bin/python -m pytest -q

lint:
	.venv/bin/python -m ruff check src tests scripts

# Phase 3 FL smoke run (2 clients, 2 rounds -- the plan's Step 3.1 acceptance size,
# set as the [tool.flwr.app.config] defaults in pyproject.toml). Needs the `simulation`
# extra (`ray`, no Intel-macOS wheel -- Colab/Kaggle only, see docs/FLOWER_API_NOTES.md),
# so this only actually runs there, not on this machine.
smoke:
	.venv/bin/flwr run . --stream

# Phase 5 -- give every baseline an honest hyperparameter search on the val split, with
# a budget matched to FedACO's. Selection reads best_val_macro_f1 only; it never touches
# the test metric. --dry-run prints the trial plan without spending any compute.
hparam-search:
	.venv/bin/python scripts/run_hparam_search.py --strategy all --budget 8

hparam-search-plan:
	.venv/bin/python scripts/run_hparam_search.py --strategy all --budget 8 --dry-run

# The real-data validation run: the first federated training this project will have done
# on actual images rather than a synthetic cache. Two runs (FedACO and FedAvg, same seed
# and partition) so the comparison is like-for-like, then the health check. Everything
# downstream -- the Phase 6 sweep's shape, its compute budget, and whether FedACO's
# mechanism is worth sweeping at all -- depends on what this says.
# K defaults to 10. Override on a small box: `make validate-fedaco K=4`.
# ROUNDS is load-bearing for the health check, not just for accuracy: tau starts uniform
# and walks away from it at a rate set by the deposit and the iteration budget, so a short
# run cannot answer "is the colony searching?" at all. At ROUNDS=15 with the default
# colony budget, question 1 needs the colony to realize 22% of the fastest concentration
# the rule permits -- a fair bar. At ROUNDS=4 it needs 38%, and with a reduced
# `aco-iters-start` it passes 60% and stops meaning anything. Run `make pheromone-budget`
# before shortening this. (docs/EXPERIMENT_LOG.md, 2026-09-17.)
K ?= 10
ROUNDS ?= 15

validate-fedaco:
	# `num-clients` partitions the DATA; `num_supernodes` is how many ClientApps Flower
	# actually creates and defaults to 2. Unequal, a run trains on 2 partitions while
	# recording K, and a min-train-nodes above the supernode count hangs at round 0 with
	# idle actors and no error -- which this project previously misdiagnosed as CPU
	# oversubscription (docs/OPEN_QUESTIONS.md). Both are set here.
	.venv/bin/flwr federation simulation-config --num-supernodes $(K) --client-resources-num-cpus 1
	FEDSWARM_REPO_ROOT=$(PWD) .venv/bin/flwr run . --stream --run-config \
	  "strategy-name='fedaco' num-clients=$(K) min-train-nodes=$(K) min-evaluate-nodes=$(K) \
	   min-available-nodes=$(K) num-rounds=$(ROUNDS) local-epochs=1 regime='dirichlet' alpha=0.3 seed=0"
	FEDSWARM_REPO_ROOT=$(PWD) .venv/bin/flwr run . --stream --run-config \
	  "strategy-name='fedavg' num-clients=$(K) min-train-nodes=$(K) min-evaluate-nodes=$(K) \
	   min-available-nodes=$(K) num-rounds=$(ROUNDS) local-epochs=1 regime='dirichlet' alpha=0.3 seed=0"
	.venv/bin/python scripts/check_fedaco_health.py --results-dir results/fl

health:
	.venv/bin/python scripts/check_fedaco_health.py --results-dir results/fl

# What a given colony budget makes *reachable* on question 1, before spending a run on it.
pheromone-budget:
	.venv/bin/python scripts/analyze_pheromone_dynamics.py

# Phase 5 -- the IID acceptance gate. Under IID there is little heterogeneity for an
# aggregation rule to exploit, so a large spread means something other than the method is
# driving the number. Exits nonzero on FAIL.
iid-band:
	.venv/bin/python scripts/check_iid_band.py --results-dir results/fl

# Always run `make main-plan` first: it prints the cell count and a cost projection, and
# refuses nothing, so it is the cheapest way to find out that 360 cells x 100 rounds is
# more compute than you have before committing to it.
main-plan:
	.venv/bin/python scripts/run_sweep.py --config configs/experiment/main.yaml --dry-run

main:
	.venv/bin/python scripts/run_sweep.py --config configs/experiment/main.yaml

# Phase 7. Only the ablations this repo actually names (A1 persistence, A3 fitness mode,
# A9 norm) plus the knobs docs/OPEN_QUESTIONS.md defers here; A2 and A4-A8 are not
# reconstructable because the implementation plan is not in the repo. See the config.
ablations-plan:
	.venv/bin/python scripts/run_sweep.py --config configs/experiment/ablation_all.yaml --dry-run

ablations:
	.venv/bin/python scripts/run_sweep.py --config configs/experiment/ablation_all.yaml

# Phase 8. Robustness: adversarial clients (fl/attacks.py) and partial participation.
# 225 cells. Three baselines here -- krum, trimmed-mean, median -- are robust-aggregation
# rules whose entire justification is this sweep; every number they have so far was earned
# against an entirely honest federation.
robustness-plan:
	.venv/bin/python scripts/run_sweep.py --config configs/experiment/robustness.yaml --dry-run

robustness:
	.venv/bin/python scripts/run_sweep.py --config configs/experiment/robustness.yaml

tables-robustness:
	.venv/bin/python scripts/make_tables.py --results-dir results/fl/robustness

figures-data:
	.venv/bin/python scripts/make_partition_figures.py

# Phase 9. Four figures to paper/figures/. Each is skipped (and named as skipped) when
# the sweep feeding it has not run, rather than rendered blank.
figures:
	.venv/bin/python scripts/make_figures.py --results-dir results/fl/main

figures-ablation:
	.venv/bin/python scripts/make_figures.py --results-dir results/fl/ablation --only ablation

# Phase 9. Aggregates over seeds and writes Markdown + CSV to paper/tables/. Exits
# nonzero when any cell has fewer seeds than expected, so an incomplete sweep cannot be
# quoted from by accident.
tables:
	.venv/bin/python scripts/make_tables.py --results-dir results/fl/main

tables-ablation:
	.venv/bin/python scripts/make_tables.py --results-dir results/fl/ablation

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
