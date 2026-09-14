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

Status: **Phase 0 — scaffolding.** No experiments have been run yet.

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

Populated as each phase lands; empty during Phase 0.
