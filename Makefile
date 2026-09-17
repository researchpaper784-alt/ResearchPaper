.PHONY: setup data test lint smoke validate-fedaco health hparam-search hparam-search-plan iid-band main ablations figures-data figures tables clean

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
K ?= 10
ROUNDS ?= 15

validate-fedaco:
	# The Simulation Runtime assigns 2 CPUs per ClientApp by default, so K=10 would
	# request 20 cores -- more than Colab (2) or Kaggle (4) has, and it does not degrade
	# gracefully: a 4-client run on this 4-core box sat at round 0 indefinitely with every
	# actor idle, no error (docs/OPEN_QUESTIONS.md). Pin it to 1 first. This writes to
	# ~/.flwr/config.toml and persists, so it only needs running once per machine.
	.venv/bin/flwr federation simulation-config --client-resources-num-cpus 1
	FEDSWARM_REPO_ROOT=$(PWD) .venv/bin/flwr run . --stream --run-config \
	  "strategy-name='fedaco' num-clients=$(K) min-train-nodes=$(K) min-evaluate-nodes=$(K) \
	   min-available-nodes=$(K) num-rounds=$(ROUNDS) local-epochs=1 regime='dirichlet' alpha=0.3 seed=0"
	FEDSWARM_REPO_ROOT=$(PWD) .venv/bin/flwr run . --stream --run-config \
	  "strategy-name='fedavg' num-clients=$(K) min-train-nodes=$(K) min-evaluate-nodes=$(K) \
	   min-available-nodes=$(K) num-rounds=$(ROUNDS) local-epochs=1 regime='dirichlet' alpha=0.3 seed=0"
	.venv/bin/python scripts/check_fedaco_health.py --results-dir results/fl

health:
	.venv/bin/python scripts/check_fedaco_health.py --results-dir results/fl

# Phase 5 -- the IID acceptance gate. Under IID there is little heterogeneity for an
# aggregation rule to exploit, so a large spread means something other than the method is
# driving the number. Exits nonzero on FAIL.
iid-band:
	.venv/bin/python scripts/check_iid_band.py --results-dir results/fl

main:
	.venv/bin/python scripts/run_sweep.py --config configs/experiment/main.yaml

ablations:
	.venv/bin/python scripts/run_sweep.py --config configs/experiment/ablation_all.yaml

figures-data:
	.venv/bin/python scripts/make_partition_figures.py

figures:
	.venv/bin/python scripts/make_figures.py

tables:
	.venv/bin/python scripts/make_tables.py

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
