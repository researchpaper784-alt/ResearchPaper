.PHONY: setup data test lint smoke smoke-all validate-fedaco health pheromone-budget fitness-landscape hparam-search hparam-search-plan iid-band main main-plan main-client-scale ablations ablations-plan ablations-granular robustness robustness-plan robustness-granular overhead tables-robustness figures-data figures figures-ablation tables tables-ablation verify-repro fitness-repro clean

# Interpreter paths, overridable. The default is the local `uv` venv from `make setup`
# (CLAUDE.md), but Colab and Kaggle install into the system Python and have no .venv at
# all -- every target here was unusable there until these became variables. On a notebook
# runtime:
#
#     make validate-fedaco PY=python FLWR=flwr
#
PY ?= .venv/bin/python
FLWR ?= .venv/bin/flwr


setup:
	uv venv --python 3.12 .venv
	uv pip install -e ".[dev]" --python $(PY)

data:
	$(PY) -m fedswarm.data.download --verify

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts

# Phase 3 FL smoke run (2 clients, 2 rounds -- the plan's Step 3.1 acceptance size,
# set as the [tool.flwr.app.config] defaults in pyproject.toml). Needs the `simulation`
# extra (`ray`, no Intel-macOS wheel -- Colab/Kaggle only, see docs/FLOWER_API_NOTES.md),
# so this only actually runs there, not on this machine. Real Colab CPU-runtime recipe:
# notebooks/colab_fl_smoke.ipynb.
smoke:
	$(FLWR) run . --stream

# Every strategy, tiny scale, one partition -- "does everything still run" (Phase 6/10).
smoke-all:
	$(PY) scripts/run_sweep_granular.py --config configs/experiment/smoke.yaml

# Phase 5 -- give every baseline an honest hyperparameter search on the val split, with
# a budget matched to FedACO's. Selection reads best_val_macro_f1 only; it never touches
# the test metric. --dry-run prints the trial plan without spending any compute.
hparam-search:
	$(PY) scripts/run_hparam_search.py --strategy all --budget 8

hparam-search-plan:
	$(PY) scripts/run_hparam_search.py --strategy all --budget 8 --dry-run

# The real-data validation run: the first federated training this project will have done
# on actual images rather than a synthetic cache. Two runs (FedACO and FedAvg, same seed
# and partition) so the comparison is like-for-like, then the health check. Everything
# downstream -- the Phase 6 sweep's shape, its compute budget, and whether FedACO's
# mechanism is worth sweeping at all -- depends on what this says.
# K defaults to 10. Overriding it *down* is not free, and not only for statistics: the
# data-free fitness pays nothing for a single-client answer (dispersion is a weighted
# variance and is exactly zero at a simplex vertex), and the only guard is
# `gamma_entropy * log K`. On synthetic deltas that guard needs gamma_entropy > 0.17 at
# K=2 and > 0.13 at K=4, against the default 0.1 -- so a K=4 validation run sits in the
# regime where the fitness optimum is degenerate while the K=20 sweep it is validating
# does not. `make fitness-landscape` maps it; `corner_margin` measures it per round.
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
	$(FLWR) federation simulation-config --num-supernodes $(K) --client-resources-num-cpus 1
	FEDSWARM_REPO_ROOT=$(PWD) $(FLWR) run . --stream --run-config \
	  "strategy-name='fedaco' num-clients=$(K) min-train-nodes=$(K) min-evaluate-nodes=$(K) \
	   min-available-nodes=$(K) num-rounds=$(ROUNDS) local-epochs=1 regime='dirichlet' alpha=0.3 seed=0"
	FEDSWARM_REPO_ROOT=$(PWD) $(FLWR) run . --stream --run-config \
	  "strategy-name='fedavg' num-clients=$(K) min-train-nodes=$(K) min-evaluate-nodes=$(K) \
	   min-available-nodes=$(K) num-rounds=$(ROUNDS) local-epochs=1 regime='dirichlet' alpha=0.3 seed=0"
	$(PY) scripts/check_fedaco_health.py --results-dir results/fl

health:
	$(PY) scripts/check_fedaco_health.py --results-dir results/fl

# What a given colony budget makes *reachable* on question 1, before spending a run on it.
pheromone-budget:
	$(PY) scripts/analyze_pheromone_dynamics.py

# Where (K, gamma_entropy, heterogeneity) makes the fitness prefer a single-client answer.
fitness-landscape:
	$(PY) scripts/analyze_fitness_landscape.py

# Phase 5 -- the IID acceptance gate. Under IID there is little heterogeneity for an
# aggregation rule to exploit, so a large spread means something other than the method is
# driving the number. Exits nonzero on FAIL.
iid-band:
	$(PY) scripts/check_iid_band.py --results-dir results/fl

# Always run `make main-plan` first: it prints the cell count and a cost projection, and
# refuses nothing, so it is the cheapest way to find out that 576 cells x 100 rounds is
# more compute than you have before committing to it.
main-plan:
	$(PY) scripts/run_sweep.py --config configs/experiment/main.yaml --dry-run

main:
	$(PY) scripts/run_sweep.py --config configs/experiment/main.yaml

main-client-scale:
	$(PY) scripts/run_sweep_granular.py --config configs/experiment/main_client_scale.yaml

# Phase 7. The two families are now disjoint (2026-09-19). A1-A9 live in their own
# per-ablation configs and are driven by `ablations-granular`; `ablation_all.yaml` keeps
# only what no A-number covers -- the safety-fallback ablation and the concentration-
# penalty shape. Five variants were removed from it because a granular config already
# varied the same run-config key, and one of those had `ablation_all` labelling
# persistence "A1" while ablation_a1.yaml is the ACO-vs-random-search control, both
# writing to results/fl/ablation. Run BOTH targets; they no longer overlap.
ablations-plan:
	$(PY) scripts/run_sweep.py --config configs/experiment/ablation_all.yaml --dry-run

ablations:
	$(PY) scripts/run_sweep.py --config configs/experiment/ablation_all.yaml

ablations-granular:
	for f in configs/experiment/ablation_a[0-9].yaml; do \
		$(PY) scripts/run_sweep_granular.py --config "$$f" || exit 1; \
	done

# Phase 8. Same split. R1-R5 live in their own configs (`robustness-granular`);
# `robustness.yaml` keeps only the `scaled` magnitude-only attack, which no granular
# config covers and which is the one case an alignment-based heuristic cannot see --
# FedACO's a_k is a cosine and is blind to it, so only the norm ratio r_k can catch it.
# The granular configs sweep three attacker fractions where the combined file sampled
# one or two, so the finer grid strictly contains the coarser one. Run both.
robustness-plan:
	$(PY) scripts/run_sweep.py --config configs/experiment/robustness.yaml --dry-run

robustness:
	$(PY) scripts/run_sweep.py --config configs/experiment/robustness.yaml

robustness-granular:
	for f in configs/experiment/robustness_r*.yaml; do \
		$(PY) scripts/run_sweep_granular.py --config "$$f" || exit 1; \
	done

# The dedicated, denser K sweep for the O(K^2) overhead curve (plan Sec 9.2, figure 5).
overhead:
	$(PY) scripts/run_sweep_granular.py --config configs/experiment/overhead.yaml

tables-robustness:
	$(PY) scripts/make_tables.py --results-dir results/fl/robustness

figures-data:
	$(PY) scripts/make_partition_figures.py

# Phase 9. Four figures to paper/figures/. Each is skipped (and named as skipped) when
# the sweep feeding it has not run, rather than rendered blank.
figures:
	$(PY) scripts/make_figures.py --results-dir results/fl/main

figures-ablation:
	$(PY) scripts/make_figures.py --results-dir results/fl/ablation --only ablation

# Phase 9. Aggregates over seeds and writes Markdown + CSV to paper/tables/. Exits
# nonzero when any cell has fewer seeds than expected, so an incomplete sweep cannot be
# quoted from by accident.
tables:
	$(PY) scripts/make_tables.py --results-dir results/fl/main

tables-ablation:
	$(PY) scripts/make_tables.py --results-dir results/fl/ablation

# Phase 10 -- runs the smoke config and checks final metrics against a recorded
# tolerance band (scripts/verify_repro.py).
# The project's blocking question, answerable locally in about a minute per arm.
#
# `ray`/`flwr[simulation]` DO run in this project's Linux containers and on Kaggle -- the
# "cannot run locally" notes are about the Intel-macOS dev machine. So the fitness question
# does not need a GPU session: it needs data heterogeneous enough to reproduce the failure,
# which `scripts/synthetic_heterogeneous_dataset.py` provides and `ci_smoke_dataset.py`
# (deliberately trivial, for CI speed) cannot.
#
# Reproduces, at gamma_3=0.1, K=10, dirichlet(0.3), 15 rounds:
#   weighted_mean  corner_margin +0.6185, positive 15/15; F(FedAvg point) -0.0033, negative 5/15
#   real Kaggle    corner_margin +0.6184, positive 15/15; F(FedAvg point) -0.0005, negative 7/15
# and shows `aggregate` fixing both (corner -0.5283 positive 0/15; F(FedAvg) +0.8141).
#
# ⚠️ A synthetic *signature* match is much weaker than a data match. Confirm anything found
# here on the real gate before it goes in the paper.
fitness-repro:
	$(PY) scripts/synthetic_heterogeneous_dataset.py --out-dir /tmp/fedswarm_het --num-images 600 --image-size 32
	$(FLWR) federation simulation-config --num-supernodes 10 --client-resources-num-cpus 1
	for mode in weighted_mean base aggregate; do \
		echo "=== $$mode ==="; \
		$(FLWR) run . --stream --run-config "strategy-name='fedaco' num-clients=10 min-client-size=3 min-train-nodes=10 min-evaluate-nodes=10 min-available-nodes=10 num-rounds=15 local-epochs=1 num-classes=4 image-size=32 local-batch-size=8 regime='dirichlet' alpha=0.3 seed=0 aco-dispersion-reference='$$mode' aco-gamma-entropy=0.1 cache-dir='/tmp/fedswarm_het/cache' manifest-path='/tmp/fedswarm_het/manifest.csv' partition-cache-dir='/tmp/fedswarm_het/parts' output-dir='/tmp/fedswarm_het/$$mode' checkpoint-dir='/tmp/fedswarm_het/ckpt_$$mode'" || exit 1; \
		$(PY) scripts/check_fedaco_health.py --results-dir /tmp/fedswarm_het/$$mode; \
	done

verify-repro:
	$(PY) scripts/verify_repro.py

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
