# Open questions / unverifiable claims log

Per the implementation plan's ground rule: if a number, dataset property, or API signature
cannot be verified, it is recorded here rather than guessed.

## Naming collision (§0.3 of the plan) — RESOLVED 2026-09-14

The working title "FedACO" collides with an unrelated 2025 PRCV/Springer paper, "FedACo:
Adaptive Collaboration with Fine-Grained Aggregation for Personalized Federated Learning"
(https://link.springer.com/chapter/10.1007/978-981-95-4987-0_1). Same acronym (case
difference only), different expansion, different method.

**Decision (author, 2026-09-14): renamed the project to "FedSwarm."** Alternate names that
were considered and rejected: ACO-Agg, PherAgg, FedPher. If the author wants to reconsider,
alternatives are still on the table — this is a cheap rename now (before any paper text
exists) and an expensive one later.

## Compute environment

Local machine (this Mac) is **Intel macOS** (`macosx_26_0_x86_64` platform tag observed
during `uv pip install`, 2026-09-14). `flwr[simulation]`'s `ray==2.55.1` dependency ships
wheels only for `manylinux2014_{x86_64,aarch64}`, `macosx_12_0_arm64` (Apple Silicon), and
`win_amd64` — **no Intel-macOS wheel exists**. Verified by direct `uv` resolver failure, not
assumed.

Consequence: `flwr[simulation]` cannot be installed on this machine. Local work (Phase 0
scaffolding, unit tests, non-simulation Flower API introspection) uses plain `flwr==1.36.0`.
Actual FL simulation runs (Phase 3 onward) must happen in the author's Colab/Kaggle
environment (both Linux), where the `simulation` extra installs cleanly. This is a hard
blocker for locally running anything from Phase 3 (`ClientApp`/`ServerApp`/`start()`)
onward — flagged here rather than silently worked around.

Confirmed directly (not just via the resolver): downloaded the official
`@flwrlabs/quickstart-pytorch` reference app and ran `flwr run . --stream` on this
machine — it fails with `Unable to launch 'flower-superlink' for local simulation:
[Errno 2] No such file or directory: 'flower-superlink'`, since that binary ships with the
`simulation` extra. So the plan's §0.2 acceptance criterion ("run the unmodified Flower
quickstart ≥2 rounds locally") cannot be satisfied on this machine at all — full detail in
`docs/FLOWER_API_NOTES.md`. Everything else that criterion was meant to de-risk (the real
`Strategy`/`FedAvg`/`ArrayRecord` API surface) was verified by reading the installed
package and the reference app's source directly.

A secondary, unrelated finding from the same install attempts: `torch` has no Intel-macOS
wheel past `2.2.x`, and `torch==2.2.2` needs `numpy<2` (verified `torch.from_numpy` crashes
under numpy 2.1.3, works under 1.26.4) — pinned in `pyproject.toml`. Re-evaluate this pin
in the Linux training environment, where it doesn't apply.

## Kaggle dataset access

Author confirmed a Kaggle account exists and the dataset can be fetched from there, but
`kaggle.json` API credentials are not yet confirmed configured on the machine that will run
`fedswarm.data.download`. Phase 1 must verify `~/.kaggle/kaggle.json` (or `KAGGLE_USERNAME`
/ `KAGGLE_KEY` env vars) exist before relying on the Kaggle-API download path, and fall back
to the documented manual-zip path (`data/raw/`) otherwise.
