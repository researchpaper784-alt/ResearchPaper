.PHONY: setup data test lint smoke smoke-all main main-client-scale ablations robustness overhead figures-data figures tables verify-repro clean

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
# so this only actually runs there, not on this machine. Real Colab CPU-runtime recipe:
# notebooks/colab_fl_smoke.ipynb.
smoke:
	.venv/bin/flwr run . --stream

# Every strategy, tiny scale, one partition -- "does everything still run" (Phase 6/10).
smoke-all:
	.venv/bin/python scripts/run_sweep.py --config configs/experiment/smoke.yaml

# Phase 6, Step 6.3 -- the full factorial (11 strategies x 7 partitions x 5 seeds).
# Meant to run across several resumed sessions; resume/lock is scripts/run_sweep.py's job.
main:
	.venv/bin/python scripts/run_sweep.py --config configs/experiment/main.yaml

main-client-scale:
	.venv/bin/python scripts/run_sweep.py --config configs/experiment/main_client_scale.yaml

# Phase 7 -- A1 (the make-or-break equal-budget control) through A9.
ablations:
	for f in configs/experiment/ablation_a*.yaml; do \
		.venv/bin/python scripts/run_sweep.py --config "$$f" || exit 1; \
	done

# Phase 8 -- R1/R2/R3/R4/R5 (R6 has no config file yet, see docs/OPEN_QUESTIONS.md).
robustness:
	for f in configs/experiment/robustness_r*.yaml; do \
		.venv/bin/python scripts/run_sweep.py --config "$$f" || exit 1; \
	done

# The dedicated, denser K sweep for the O(K^2) overhead curve (plan §9.2, figure 5).
overhead:
	.venv/bin/python scripts/run_sweep.py --config configs/experiment/overhead.yaml

figures-data:
	.venv/bin/python scripts/make_partition_figures.py

figures:
	.venv/bin/python scripts/make_figures.py

tables:
	.venv/bin/python scripts/make_tables.py

# Phase 10 -- runs the smoke config and checks final metrics against a recorded
# tolerance band (scripts/verify_repro.py).
verify-repro:
	.venv/bin/python scripts/verify_repro.py

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
