# 7. Reproducibility

## What is in the artifact

- Every run writes one self-describing JSON containing the **full resolved config, the git SHA,
  and the resolved versions of every relevant library**, plus per-round metrics and the
  strategy's own diagnostics (`src/fedswarm/utils/provenance.py`, `utils/results.py`).
- Run identity is a hash of the whole resolved config plus the seed, so two cells differing in
  any setting cannot collide on one result file.
- `results/manifest.jsonl` indexes every run, which is what makes a sweep resumable across the
  session timeouts these experiments were actually run under.
- The de-duplicated split ships as `data/processed/manifest.csv` with a `pseudo_patient_id` per
  image, so the split is a citable artifact rather than a procedure to re-derive.

## Reproducing the paper's artifacts

```bash
make setup                 # uv venv + editable install
make test                  # 718 tests
make verify-repro          # smoke config lands inside a recorded tolerance band
make verify-phase9         # regenerates every figure and table TWICE, byte-compares
```

`make verify-phase9` is plan §9.3's acceptance criterion: deleting `paper/figures/` and
`paper/tables/` and re-running must restore them byte-for-byte. It passes on 10 artifacts.

To reproduce the experiments themselves (~70 GPU-hours):

```bash
make b-all GPUS=0.1        # gate_fitness, then A1 -- the framing decision
make c-all GPUS=0.1        # main_reduced, r1, r2, a2
make figures tables
```

**`GPUS` is not optional on a GPU box.** Ray hides the GPU from any actor requested with
`num_gpus=0`, so with no fraction set every client trains on CPU while the server keeps the
card — and every logged metric looks normal. `1/num_clients` is the value that lets all clients
share one card. The sweep runner refuses to start rather than run a sweep that would do this.

## Before spending GPU hours

```bash
python scripts/synthetic_heterogeneous_dataset.py --out-dir /tmp/fx --num-images 600 --image-size 32
make preflight-configs FIXTURE=/tmp/fx CONFIGS='configs/experiment/*.yaml'
```

Runs one cell of every declared arm of every sweep at 2 rounds and K=6 against a synthetic
fixture. 155 arms, ~100 minutes of CPU, and it verifies that every strategy, attack, cold-start,
persistence and aggregation path executes with real Flower messages. It verifies nothing about
whether the numbers mean anything.

## Known reproducibility limits

- **`flwr run` exits 0 when the simulation dies.** Every runner here judges a cell by whether a
  result JSON appeared, not by exit status. Anyone extending this code should do the same.
- **The dataset requires credentials we cannot ship.** `data/raw` holds a dataset card and no
  images. The de-duplicated manifest, the leakage report and the split statistics are committed,
  so the audit is checkable without the images; re-running training is not.
- **Tag pushes fail with HTTP 403 in our environment**, so the milestone tags `v0.5-pipeline`
  and `v0.6-pipeline-verified` exist locally but not on the remote. Commit SHAs in
  `docs/EXPERIMENT_LOG.md` are the durable references.
- **Zenodo DOI: not yet minted.** [TODO before submission — plan §10 requires it and the paper
  must cite it.]
