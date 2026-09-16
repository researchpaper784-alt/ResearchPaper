# Flower API notes — verified against the actually-installed package

**Verified:** 2026-09-14, against `flwr==1.36.0` / `flwr-datasets==0.5.0` installed in
`.venv` on this machine. Everything below was captured with `inspect` against the live
install, or by downloading and reading the official `@flwrlabs/quickstart-pytorch`
reference app — nothing here is recalled from training data. If the plan and this file
ever disagree, **this file wins**; re-run the introspection commands below if `flwr` gets
upgraded.

## Installed versions

| Package | Version |
|---|---|
| `flwr` | 1.36.0 |
| `flwr-datasets` | 0.5.0 |
| `torch` | 2.2.2 (see platform note below) |
| `torchvision` | 0.17.2 |
| `numpy` | 1.26.4 (see platform note below) |

## `flwr.serverapp.strategy.Strategy` — real source

Confirmed to match the plan's §0.2 exactly: an ABC with five abstract methods
(`configure_train`, `aggregate_train`, `configure_evaluate`, `aggregate_evaluate`,
`summary`) and one concrete method, `start(...)`, which the plan must not override.

```python
class Strategy(ABC):
    @abstractmethod
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]: ...

    @abstractmethod
    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]: ...

    @abstractmethod
    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]: ...

    @abstractmethod
    def aggregate_evaluate(
        self, server_round: int, replies: Iterable[Message]
    ) -> MetricRecord | None: ...

    @abstractmethod
    def summary(self) -> None: ...

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: ConfigRecord | None = None,
        evaluate_config: ConfigRecord | None = None,
        evaluate_fn: Callable[[int, ArrayRecord], MetricRecord | None] | None = None,
    ) -> Result: ...
```

`start()`'s real body (read in full): initializes empty configs if `None`, calls
`evaluate_fn(0, initial_arrays)` once before round 1 if provided, then per round calls
`grid.send_and_receive(messages=self.configure_train(...), timeout=timeout)`, passes the
replies to `self.aggregate_train(...)`, updates `arrays` from the returned `ArrayRecord`
if not `None`, and repeats for evaluate. **FedACO's `FedACO(Strategy)` must not override
`start()`** — all the ACO logic belongs in `aggregate_train`.

## `FedAvg` — real constructor and `aggregate_train`

```python
FedAvg.__init__(
    self,
    fraction_train: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_train_nodes: int = 2,
    min_evaluate_nodes: int = 2,
    min_available_nodes: int = 2,
    weighted_by_key: str = "num-examples",
    arrayrecord_key: str = "arrays",
    configrecord_key: str = "config",
    train_metrics_aggr_fn: Callable[[list[RecordDict], str], MetricRecord] | None = None,
    evaluate_metrics_aggr_fn: Callable[[list[RecordDict], str], MetricRecord] | None = None,
) -> None
```

Confirms the plan's renames: `fraction_train` (not `fraction_fit`). Weighting key defaults
to `"num-examples"` in the client's returned `MetricRecord` — **FedACO's client replies
must include this key** even though FedACO ignores it for its own weighting (baselines and
the `q_k` heuristic term both need it).

`aggregate_train` body: filters replies via `self._check_and_log_replies(replies,
is_train=True)`, then calls module-level `aggregate_arrayrecords(reply_contents,
weighted_by_key)` and `self.train_metrics_aggr_fn(reply_contents, weighted_by_key)`. For
FedACO, `aggregate_arrayrecords` is *not* reusable as-is (it presumably does the
size-weighted convex combination) — FedACO's `aggregate_train` must extract per-client
`ArrayRecord`s itself, compute deltas against the current global `arrays`, run the colony,
and materialize the result manually (§4.8 of the plan).

## `ArrayRecord` ⟷ `torch.state_dict` — real signatures

```python
ArrayRecord.to_torch_state_dict(self, *, keep_input: bool = True) -> OrderedDict[str, torch.Tensor]
ArrayRecord.from_torch_state_dict(state_dict: dict[str, torch.Tensor], *, keep_input: bool = True) -> ArrayRecord
```

Also present: `from_numpy_ndarrays`, `to_numpy_ndarrays`, `from_array_dict`. `ArrayRecord`
also supports plain construction from a state dict directly, confirmed in the reference
app: `ArrayRecord(model.state_dict())`.

## `flwr_datasets` — real constructor and partitioner signatures

```python
FederatedDataset.__init__(
    self, *,
    dataset: str,
    subset: str | None = None,
    preprocessor: Callable[[DatasetDict], DatasetDict] | dict[str, tuple[str, ...]] | None = None,
    partitioners: dict[str, Partitioner | int],
    shuffle: bool = True,
    seed: int | None = 42,
    **load_dataset_kwargs: Any,
) -> None

IidPartitioner(self, num_partitions: int) -> None

DirichletPartitioner(
    self, num_partitions: int, partition_by: str,
    alpha: int | float | list[float] | np.ndarray,
    min_partition_size: int = 10, self_balancing: bool = False,
    shuffle: bool = True, seed: int | None = 42,
) -> None

PathologicalPartitioner(
    self, num_partitions: int, partition_by: str, num_classes_per_partition: int,
    class_assignment_mode: Literal["random", "deterministic", "first-deterministic"] = "random",
    shuffle: bool = True, seed: int | None = 42,
) -> None
```

Note: our dataset is a local image-folder JPEG dataset (Kaggle Brain Tumor MRI), not an
HF Hub dataset id, and partitioning must happen at the **pseudo-patient** level after
de-duplication (plan §1.2/§1.4), not directly via `FederatedDataset(dataset=...)`'s
HF-download path. Plan: build the manifest ourselves, then feed pseudo-patient-grouped
indices into `IidPartitioner`/`DirichletPartitioner`/`PathologicalPartitioner` by
subclassing/wrapping them against an in-memory HF `Dataset` built from the manifest (or by
implementing an equivalent thin partitioner directly over the manifest, since these classes
partition an existing HF `Dataset`/`DatasetDict`, not arbitrary file lists). Revisit exact
approach at Phase 1.4 — flagging now, not deciding now, since that's out of scope for
Phase 0.

## Official quickstart-pytorch app — confirms the plan's client/server patterns exactly

Downloaded via `flwr new "@flwrlabs/quickstart-pytorch"` (the `flwr new --framework
PyTorch` invocation from older docs is **deprecated** — `flwr new` now takes a registry app
specifier like `@account/app_name`). Reading the generated `client_app.py` and
`server_app.py` verbatim confirms:

- `app = ClientApp()`, `@app.train()`, `@app.evaluate()` handlers, each taking
  `(msg: Message, context: Context)` and returning `Message(content=..., reply_to=msg)`.
- Client reads weights via `msg.content["arrays"].to_torch_state_dict()`, returns
  `RecordDict({"arrays": ArrayRecord(model.state_dict()), "metrics": MetricRecord({...})})`.
  Note real client code reads `context.node_config["partition-id"]` /
  `context.run_config[...]` for data/hyperparameters — **not** `msg.content["config"]`
  for everything; per-round overrides (e.g. FedACO's `client_probe` candidate list) go in
  the train `ConfigRecord` passed to `configure_train`'s messages and are read from
  `msg.content["config"]`, while static run config comes from `context.run_config`.
- `app = ServerApp()`, `@app.main()` taking `(grid: Grid, context: Context)`.
  `strategy.start(grid=grid, initial_arrays=arrays, train_config=ConfigRecord({...}),
  num_rounds=num_rounds, evaluate_fn=global_evaluate)`.
- `evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord`, matching the
  plan.

## ⚠️ Platform blocker — cannot run simulations on this machine

**Verified, not assumed.** This machine is Intel macOS (`macosx_26_0_x86_64` platform tag
from the `uv` resolver). Two real, separate failures were reproduced:

1. `flwr[simulation]` depends on `ray==2.55.1`, which has wheels only for
   `manylinux2014_{x86_64,aarch64}`, `macosx_12_0_arm64` (Apple Silicon only), and
   `win_amd64`. **No Intel-macOS wheel exists.** `uv pip install -e ".[simulation]"` fails
   to resolve.
2. Running the downloaded `quickstart-pytorch` app locally
   (`flwr run . --stream`) fails with:
   `Unable to launch 'flower-superlink' for local simulation: [Errno 2] No such file or
   directory: 'flower-superlink'` — that binary ships with the `simulation` extra we can't
   install.

**Consequence:** the plan's §0.2 acceptance criterion ("the unmodified Flower PyTorch
quickstart completes ≥ 2 rounds locally") **cannot be met on this machine**, full stop —
not a code problem, a platform one. What *was* verified locally instead: every API
signature above, read directly from the installed package and from the real reference
app's source, which is the part of §0.2 that actually de-risks Phases 3–4. The quickstart
run itself (and all of Phase 3 onward — `ClientApp`/`ServerApp`/`Strategy.start()`) must
happen in the author's Colab/Kaggle environment (both Linux, where `ray` has wheels).
Logged in `docs/OPEN_QUESTIONS.md`.

## Phase 3 -- `[tool.flwr]` app config schema, verified via `flwr build`

**Verified 2026-09-16**, two ways: (1) re-ran `flwr new @flwrlabs/quickstart-pytorch`
fresh (same `flwr==1.36.0` install) and read its generated `pyproject.toml` and
`pytorchexample/{client_app,server_app}.py` directly -- confirms `client_app.py`/
`server_app.py`'s structure (this repo's `fl/`) matches the current real pattern
exactly: `context.node_config["partition-id"]`/`["num-partitions"]` for per-client
identity, `context.run_config[...]` for static settings, `msg.content["config"][...]`
for per-round overrides, `Message(content=..., reply_to=msg)` replies. (2) `flwr build`
(packaging only -- does **not** need the `simulation` extra) against this repo's own
`pyproject.toml`, which caught two real errors before Colab ever would have:

1. `fab-format-version = 1` rejects `[project].license = {text = "MIT"}` -- requires
   `{file = "LICENSE"}` referencing a real root-level file.
2. `fab-format-version = 1` rejects an exact `==` pin on the `flwr` dependency itself --
   requires an inclusive `>=` lower bound. Changed to `flwr>=1.36.0,<1.37.0` (still
   resolves to exactly the installed 1.36.0 today).

After both fixes, `flwr build` succeeds and resolves both component paths -- real
confirmation the `[tool.flwr.app.components]` strings are correct, not an assumption.
The reference app's README confirms `flwr run . --stream` (no `--federation` flag, no
`[tool.flwr.federations]` section) uses a default local/CPU federation -- matches this
repo's `make smoke` target, and works on a CPU-only Colab runtime unchanged (see
`fl/app.py`'s module docstring: nothing here requires a GPU).

**2026-09-16, consolidated:** `fl/task.py`, `fl/client_app.py`, `fl/server_app.py`, and
`fl/checkpoint.py` were merged into one file, `fl/app.py` (at the author's request, for
a single file that's easy to read start-to-finish and easy to hand to Colab -- no
behavior change). `[tool.flwr.app.components]` now points both component strings at it:
`fedswarm.fl.app:server_app` / `fedswarm.fl.app:client_app` (two distinctly-named
objects in the one file, re-verified with `flwr build` after the move). `tests/
test_fl_app.py` replaces the four now-deleted test files with the same 23 tests.

**What `flwr build` cannot verify** (needs the `simulation` extra, i.e. Colab/Kaggle):
whether `strategy.start()`'s actual round-by-round `Grid` behavior, `node_config`
population per SuperNode, and message routing match what `fl/app.py` assumes.
`ClientApp.train()`/`.evaluate()` and `ServerApp.main()`'s decorators are confirmed (by
reading their source) to register and return the function *unmodified* -- so
`tests/test_fl_app.py` calls the handlers directly with hand-built `Message`/`Context`/
`ArrayRecord` objects, which exercises every line of this repo's own logic but not
Flower's runtime routing itself. The first real `flwr run .` on Colab is what confirms
that; nothing local can.

## torch/numpy pin note

`torch` has no Intel-macOS wheel past the `2.2.x` line (tried `2.4.1`: no wheel; `2.2.2`:
resolves). `torch==2.2.2` was built against the NumPy 1.x ABI — installing NumPy 2.x
alongside it makes `torch.from_numpy` raise `RuntimeError: Numpy is not available`
(reproduced directly, not a hypothetical). Pinned `numpy>=1.26,<2` in `pyproject.toml` to
fix it; confirmed `torch.from_numpy` round-trips correctly after the pin. This pin is a
property of *this machine's* wheel availability, not of Flower or the method — re-evaluate
when installing in the Linux training environment, where newer `torch` (and NumPy 2.x) are
available and preferred.
