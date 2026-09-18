"""Shared plumbing for the scripts that drive `flwr run` subprocesses.

`run_hparam_search.py` (Phase 5) and `run_sweep.py` (Phase 6) both shell out to
`flwr run`, both render a run config as TOML, both decide whether an existing result file
matches the configuration they are about to run, and both have to explain a failure from
captured output. Each grew its own copy of those helpers and the copies drifted -- which
is the reason this module exists rather than tidiness:

`diagnose` in the sweep filtered log noise with `"arn" not in line.lower()`, a crude
stand-in for "not a warning". It also discards every line containing *learn*, so a
`ValueError` naming a learning rate -- the single likeliest way a tuning cell dies -- was
thrown away and the sweep reported "no diagnostic line found in output" for exactly the
failures it most needed to explain. The search's copy had already been fixed to filter
`Warning`/`warn` explicitly. Two copies meant one fix. One definition means the next fix
lands in both.

Nothing here knows about FedSwarm's models, data or strategies; it is process plumbing.
It lives in the package rather than in `scripts/` so that the scripts and the tests can
import it by name instead of by `sys.path` insertion, and it deliberately imports only the
standard library so that a results-analysis script pays nothing for it.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

__all__ = [
    "diagnose",
    "format_run_config",
    "load_result_files",
    "load_results",
    "run_flwr",
    "same_value",
]


# ======================================================================================
# Rendering a run config
# ======================================================================================


def format_run_config(config: dict) -> str:
    """Render a run config as the TOML fragment `--run-config` expects.

    Booleans must be lowercase: flwr parses the assembled string with tomli, and Python's
    `str(False)` is `"False"`, which is not valid TOML. The whole config string is then
    rejected with a bare "[code: 15]" naming nothing -- so a single boolean override kills
    every cell that carries it. `ablation_all.yaml`'s `no_safety_fallback` variant (the one
    that separates "the colony helped" from "the fallback protected it") is exactly such a
    cell.
    """
    parts = []
    for key, value in sorted(config.items()):
        if isinstance(value, bool):
            parts.append(f"{key}={str(value).lower()}")
        elif isinstance(value, str):
            parts.append(f"{key}='{value}'")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def same_value(left, right) -> bool:
    """Whether two resolved run-config values denote the same setting.

    Floats are compared with a tolerance because a value makes a round trip through TOML
    and JSON between being requested and being read back out of a result file. An exact
    `==` on `0.1` would be fine and on some others would not; a resume check that answers
    "different" for a config it already ran recomputes the cell forever.
    """
    if isinstance(right, float) or isinstance(left, float):
        try:
            return abs(float(left) - float(right)) < 1e-12
        except (TypeError, ValueError):
            return False
    return left == right


# ======================================================================================
# Reading result files
# ======================================================================================

_RESULT_CACHE: dict[tuple[str, int], list[tuple[Path, dict]]] = {}


def load_result_files(
    directory: Path, use_cache: bool = False, recursive: bool = False
) -> list[tuple[Path, dict]]:
    """Every parseable result JSON in `directory`, with its path.

    Unparseable and unreadable files are skipped rather than raising: a run killed
    mid-write leaves a truncated JSON document behind, and one of those must not stop an
    analysis of the 300 that completed.

    `use_cache` parses the directory once instead of once per caller. The sweep calls this
    for all 360 cells at resume and again after each completed run; re-globbing and
    re-parsing each time is O(cells x files), roughly 130,000 file reads before a resumed
    main sweep starts. The cache key includes the file count, so it invalidates as soon as
    a new result lands -- which is what makes it safe to use inside a running sweep.

    `recursive` is how the analysis scripts differ from the runners. A sweep's
    `--output-dir` is flat by construction (the app writes `<output-dir>/<run_id>.json`),
    but `make_tables.py` and friends are pointed at `results/` as a whole, where a main
    sweep and an ablation sweep sit in separate subdirectories. Defaulting to flat keeps
    the resume check from matching a result that belongs to a different sweep.
    """
    directory = Path(directory)
    paths = sorted(directory.rglob("*.json") if recursive else directory.glob("*.json"))
    key = (str(directory), len(paths))
    if use_cache and key in _RESULT_CACHE:
        return _RESULT_CACHE[key]
    loaded: list[tuple[Path, dict]] = []
    for path in paths:
        try:
            loaded.append((path, json.loads(path.read_text())))
        except (json.JSONDecodeError, OSError):
            continue
    if use_cache:
        _RESULT_CACHE.clear()
        _RESULT_CACHE[key] = loaded
    return loaded


def load_results(directory: Path, recursive: bool = False) -> list[dict]:
    """The result payloads in `directory`, without their paths."""
    return [result for _, result in load_result_files(directory, recursive=recursive)]


# ======================================================================================
# Running and explaining a subprocess
# ======================================================================================


def run_flwr(
    run_config: str,
    repo_root: Path,
    stream: bool = False,
    echo_command: bool = False,
) -> tuple[int, str]:
    """One `flwr run`, blocking. Returns (returncode, captured output).

    `--stream` is passed unconditionally, and that is load-bearing rather than cosmetic:
    **a bare `flwr run` is asynchronous.** It submits the run to the SuperLink, prints
    "Successfully started run <id>", and returns immediately with exit 0 while the
    simulation is still starting. A caller that shells out without it checks for the
    result file microseconds after launching, finds nothing, and reports every run as
    failed -- which is what the first end-to-end run of the Phase 5 search did, while the
    orphaned simulations it had launched kept executing and drove the load average to 11.8
    on a 4-core box.

    `flwr run` also exits 0 when the simulation itself dies (a missing image cache
    surfaces as an in-run traceback and "Exit Code: 700" while the outer process still
    returns success), so the returncode cannot decide whether a run worked either. Both
    facts together are why every caller judges success by "did a result file appear", not
    by this function's verdict.

    `FEDSWARM_REPO_ROOT` is exported because `flwr run` executes an *installed copy* of the
    app, whose `__file__` is not in this clone; the app resolves data and cache paths
    against that variable (see `fl/app.py`).

    `stream=True` echoes the run's output live instead of capturing it -- in which case
    there is nothing left to hand to `diagnose`, and the returned output is empty.
    """
    cmd = ["flwr", "run", ".", "--stream", "--run-config", run_config]
    env = {**os.environ, "FEDSWARM_REPO_ROOT": str(repo_root)}
    if echo_command:
        print(f'  $ flwr run . --stream --run-config "{run_config}"', flush=True)
    if stream:
        completed = subprocess.run(cmd, cwd=repo_root, env=env)
        return completed.returncode, ""
    completed = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True, text=True)
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


_DIAGNOSTIC_MARKERS = (
    "Invalid run configuration",
    "Dataset root does not exist",
    "No cache at",
    "FileNotFoundError",
    "Traceback",
    "Exit Code:",
    "ValueError",
    "KeyError",
    "Error",
)


def diagnose(output: str) -> str:
    """The one line from a failed run's output most likely to explain it.

    Markers are tried in priority order and each is scanned across the whole output, so a
    specific cause wins over an incidental match anywhere in the log. Scanning
    line-by-line against a flat marker set instead let Ray's `FutureWarning: ... turn off
    this error message` be reported as the reason a trial failed, purely because it
    appeared first and contains the word "error".

    Warnings are filtered on the words `Warning` and `warn`, not on a substring like
    "arn": that shorter test also matches *learn*, and would silently discard
    `ValueError: learning rate must be positive` -- the likeliest failure of the very
    sweeps that call this.
    """
    interesting = [
        line.strip()
        for line in output.splitlines()
        if "Warning" not in line and "warn" not in line.lower()
    ]
    for marker in _DIAGNOSTIC_MARKERS:
        for line in interesting:
            if marker in line:
                return line[:300]
    return "no diagnostic line found in output"
