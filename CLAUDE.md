# CLAUDE.md — working notes for FedSwarm

## Package manager

`uv`. Environment lives in `.venv/`. `make setup` creates it and installs the project
editable with dev deps. `flwr[simulation]` (via `ray`) has no Intel-macOS wheel — see
README.md. Install `.[simulation]` only in the Linux training environment (Colab/Kaggle).

## Running one experiment

```bash
.venv/bin/python scripts/run_experiment.py --config configs/experiment/<name>.yaml --seed 0
```

## Where results go

Every run writes exactly one self-describing JSON to `results/` (gitignored). Schema is
defined in Phase 6 of the implementation plan. `results/manifest.jsonl` indexes runs for
sweep resume.

## Flower API

This codebase targets the installed `flwr==1.36.0` **Message API**
(`flwr.app`, `flwr.clientapp.ClientApp`, `flwr.serverapp.{Grid, ServerApp}`,
`flwr.serverapp.strategy.Strategy`). The old `fl.client.NumPyClient` /
`fl.server.strategy.Strategy` / `fl.simulation.start_simulation` API is dead — never write
it. See `docs/FLOWER_API_NOTES.md` for the actual verified signatures; that file is the
ground truth, not this plan or any recollection of older Flower versions.

## Commit conventions

One commit per implementation-plan phase (or per meaningful step within a phase), message
format `phase-N: <summary>`. Tag milestones (`v0.1-data`, `v0.2-fedavg`, `v0.3-fedaco`, …).
Never fabricate a number, dataset property, or API signature — if it can't be verified,
stop and record it in `docs/OPEN_QUESTIONS.md`.

## Determinism

Every run takes a seed (`utils/seed.py`) and embeds full config + git SHA + resolved
library versions (`utils/provenance.py`) in its result JSON.

## Naming

Project was renamed from the working title "FedACO" to "FedSwarm" to avoid an acronym
collision with an unrelated 2025 paper ("FedACo"). See `docs/OPEN_QUESTIONS.md`.
