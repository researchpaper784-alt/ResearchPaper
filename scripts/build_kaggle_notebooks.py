"""Generate the Day 1 / 2 / 3 Kaggle notebooks from one source.

The three notebooks share their first eight cells (clone, install, dataset, restore, cache,
federation, and the gate-fix guard). Those cells have already needed fixing twice, and three
hand-maintained copies is how a fix lands in Day 1 and not Day 3 -- the drift this repository
keeps producing. So the notebooks are generated, and `tests/test_kaggle_notebooks_built.py`
fails if a committed notebook differs from what this script produces.

Edit the cells HERE, then:

    python scripts/build_kaggle_notebooks.py

Cell ids are deterministic (`d1-07`), not nbformat's random ones, so a rebuild with no source
change is byte-identical.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "notebooks"


def md(text: str) -> tuple[str, str]:
    return ("markdown", text.strip("\n") + "\n")


def code(text: str) -> tuple[str, str]:
    return ("code", text.strip("\n") + "\n")


# ======================================================================================
# Shared cells


PREFLIGHT = code(r'''
# Settings check FIRST, in two seconds -- not after four minutes of installs. The GPU is the one
# thing no code can switch on; it is a notebook setting, and it persists once set.
import shutil
import subprocess

gpu = (subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
       if shutil.which("nvidia-smi") else None)
if gpu is None or gpu.returncode != 0 or "GPU" not in gpu.stdout:
    raise RuntimeError(
        "NO GPU -- a notebook setting, not a code problem. Fix once and it stays set:\n"
        "  right sidebar -> 'Session options' (or top menu 'Settings') -> Accelerator ->\n"
        "  'GPU T4 x2'. Then Run All again.\n"
        "If the Accelerator menu is greyed out, your Kaggle account needs phone verification\n"
        "(kaggle.com -> Settings -> Phone verification)."
    )
print(gpu.stdout.strip())
print("GPU OK")
''')


HOW_TO_RUN = r"""
## How to run this — no uploads, no clicks on data

**Two settings, once** (right sidebar → **Session options**; they stay set for this notebook):

- **Accelerator → GPU T4 x2**
- **Internet → On**

Everything else is automatic: the code is cloned from GitHub and the MRI dataset is attached by
the notebook itself.

**Then, to avoid babysitting it:**

1. Click **Run All** and watch the first few cells (~5 min). You want to see `GPU OK`,
   `INSTALL OK`, and `dataset check: 7200/7200`.
2. Once those pass, **Save Version → Save & Run All (Commit)** and close the tab. The commit
   runs in the background for up to 12 hours and keeps every output. An interactive session can
   die if the browser disconnects; a commit does not.

If anything stops, the error says exactly what to change. The red `ERROR: pip's dependency
resolver…` block during install is **not** one of them — it is harmless, see the install cell.
"""


CLONE = code(r'''
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/researchpaper784-alt/ResearchPaper.git"
REPO_DIR = "/kaggle/working/ResearchPaper"

# BRANCH is not optional. A bare `git clone` takes the DEFAULT branch, `main`, and every config
# and script these notebooks run exists only on the feature branch.
BRANCH = "claude/happy-hamilton-c5jjil"

if not Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "checkout", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "reset", "--hard", f"origin/{BRANCH}"], check=True)

on = subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True, text=True).stdout.strip()
print("branch:", on)
print(subprocess.run(["git", "-C", REPO_DIR, "log", "--oneline", "-3"],
                     capture_output=True, text=True).stdout)
if on != BRANCH:
    raise RuntimeError(f"Checked out {on!r}, not {BRANCH!r}.")

# Re-running this cell in a live kernel pulls new code, but Python keeps any fedswarm module it
# already imported -- so a function added since would be "missing". Drop the cached copies.
import importlib  # noqa: E402
import sys  # noqa: E402

stale = [name for name in sys.modules if name == "fedswarm" or name.startswith("fedswarm.")]
for name in stale:
    del sys.modules[name]
importlib.invalidate_caches()
if stale:
    print(f"dropped {len(stale)} cached fedswarm module(s); the fresh code will be imported")

# Put src/ on THIS KERNEL's path, now, before anything imports fedswarm. `pip install -e .` in the
# next cell registers the package through a .pth file, and Python reads .pth files only when a
# process STARTS -- this kernel is already running, so it would never see the install. Only
# fresh subprocesses (the sweep scripts, `flwr run`) do. Missing this line is what produced
# "ModuleNotFoundError: No module named 'fedswarm'" on Kaggle.
SRC = f"{REPO_DIR}/src"
if SRC not in sys.path:
    sys.path.insert(0, SRC)
''')

INSTALL = code(r'''
%cd /kaggle/working/ResearchPaper

# flwr[simulation] pulls in ray; installed first and on its own. fedswarm is --no-deps because
# its own pins target the dev machine and have no Kaggle CUDA build -- Kaggle's base image
# already has a newer working torch/numpy. pip still enforces fedswarm's requires-python
# (>=3.11), so an older runtime fails HERE, loudly, rather than hours later.
#
# EXPECT A RED BLOCK here reading "ERROR: pip's dependency resolver does not currently take into
# account all the packages that are installed", listing bigframes, google-colab, gradio,
# grpcio-tools and others. It is HARMLESS: those are Kaggle's own preinstalled packages
# disagreeing with each other about protobuf/rich/starlette versions, none of which this
# project imports. The line after the installs is the one that matters.
!pip install -q "flwr[simulation]>=1.36.0,<1.37.0"
!pip install -q --no-deps -e .
!pip install -q omegaconf rich

# Check the imports IN THIS KERNEL -- the process every later cell runs in. An earlier version
# checked with `!python -c "import fedswarm"`, which starts a NEW process; that one reads pip's
# .pth file and passed, while this kernel could not import fedswarm at all.
import importlib  # noqa: E402
import sys  # noqa: E402

importlib.invalidate_caches()
import flwr  # noqa: E402

import fedswarm  # noqa: E402

print("INSTALL OK -- fedswarm from", fedswarm.__file__)
print("flwr", flwr.__version__, "| Python", sys.version.split()[0])
''')

DATASET = code(r'''
import glob
import os
import sys
from pathlib import Path

# The dataset is fetched automatically -- nothing to click. If it is already attached (sidebar,
# or a previous session) that copy is used; otherwise kagglehub attaches the public dataset to
# this session and returns where it mounted.
from fedswarm.data.download import locate_or_fetch_kaggle_dataset  # noqa: E402

print("inputs mounted:", [Path(p).name for p in sorted(glob.glob("/kaggle/input/*"))] or "none yet")
DATA_ROOT = locate_or_fetch_kaggle_dataset()
if DATA_ROOT is None:
    raise RuntimeError(
        "Could not attach masoudnickparvar/brain-tumor-mri-dataset automatically.\n"
        "  Most likely cause: Internet is off. Right sidebar -> 'Session options' -> Internet ->\n"
        "  On, then Run All again. (The clone cell above needs internet too, so if it passed,\n"
        "  this is something else -- attach it by hand once: right sidebar -> '+ Add Input' ->\n"
        "  search 'brain tumor mri dataset' by masoudnickparvar -> (+).)"
    )
DATA_ROOT = str(DATA_ROOT)
print("dataset root:", DATA_ROOT)

# `flwr run` executes an INSTALLED COPY of the app, whose __file__ is not this clone, so the
# app resolves data and cache paths against FEDSWARM_REPO_ROOT rather than its own location.
os.environ["FEDSWARM_DATA_ROOT"] = DATA_ROOT
os.environ["FEDSWARM_REPO_ROOT"] = "/kaggle/working/ResearchPaper"
os.environ["FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION"] = "1"

sys.path.insert(0, "src")
import torch  # noqa: E402

print("CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU visible to PyTorch; these sweeps will not finish on CPU. Right sidebar -> "
        "'Session options' -> Accelerator -> 'GPU T4 x2', then Run All again."
    )
print("CPU cores:", os.cpu_count(), "| Python:", sys.version.split()[0])

import flwr  # noqa: E402
from fedswarm.data.download import find_split_parent  # noqa: E402

print("flwr:", flwr.__version__, "| torch:", torch.__version__)
SPLIT_PARENT = find_split_parent(Path(DATA_ROOT))
print("split parent:", SPLIT_PARENT)

# Verify every image is BYTE-IDENTICAL to the one the committed split was built from. A missing
# file would fail loudly later; a re-versioned Kaggle dataset with the same filenames and
# different images would not -- every pseudo-patient boundary would then describe images you
# are not training on, and no result file would show it. ~165 MB of reads, a few seconds.
import hashlib  # noqa: E402

import pandas as pd  # noqa: E402

_manifest = pd.read_csv("data/processed/manifest.csv", dtype={"sha256": str})
missing, changed = [], []
for rel, digest in zip(_manifest["path"], _manifest["sha256"]):
    f = SPLIT_PARENT / rel
    if not f.exists():
        missing.append(rel)
    elif hashlib.sha256(f.read_bytes()).hexdigest() != digest:
        changed.append(rel)
ok = len(_manifest) - len(missing) - len(changed)
print(f"dataset check: {ok}/{len(_manifest)} images present and byte-identical to the manifest")
if missing or changed:
    raise RuntimeError(
        f"{len(missing)} manifest image(s) missing and {len(changed)} with different content "
        f"(first few: {(missing + changed)[:3]}). The attached dataset is not the version the "
        "split was built from -- check you added masoudnickparvar/brain-tumor-mri-dataset and "
        "not a fork, and that Kaggle has not published a new version of it."
    )
''')

RESTORE = code(r'''
# Self-contained on purpose: a Run-All from the middle must not NameError here.
import glob
import shutil
from pathlib import Path

RESULTS = Path("/kaggle/working/ResearchPaper/results")
RESULTS.mkdir(parents=True, exist_ok=True)

restored = 0
for prior in glob.glob("/kaggle/input/**/results", recursive=True):
    for src in Path(prior).rglob("*.json*"):
        dst = RESULTS / src.relative_to(prior)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
            restored += 1
print(f"restored {restored} file(s) from attached outputs")
for sub in ("gate", "ablation", "main", "robustness"):
    d = RESULTS / "fl" / sub
    print(f"  results/fl/{sub}: {len(list(d.glob('*.json'))) if d.exists() else 0}")
''')

CACHE = code(r'''
import pandas as pd

from fedswarm.data.cache import build_and_save_cache
from fedswarm.data.download import find_split_parent, resolve_root

manifest = pd.read_csv("data/processed/manifest.csv")
print("manifest rows:", len(manifest), "| pseudo-patients:", manifest.pseudo_patient_id.nunique())
cache_path = build_and_save_cache(manifest, find_split_parent(resolve_root(None)), 112)
print("cache:", cache_path)
''')

FEDERATION_MD = md(r'''
## Federation, GPU, and the gate-fix guard — **not optional**

**GPU.** Ray hides the GPU from any actor requested with `num_gpus=0`, so with no fraction set
every client trains on **CPU** while the server keeps the card — and every logged metric looks
normal. `1/num_clients` lets all ten clients share one T4.

**The gate fix.** Day 1's gate chooses a fitness fix and writes it into `pyproject.toml` —
**inside that Kaggle session only**. It is never pushed to GitHub, so every fresh clone,
including a restart of this notebook, starts on the *broken* default. `ensure_gate_fix()`
re-derives the same patch from the gate's result files (restored from Day 1's saved output) and
re-applies it. It is idempotent, so every cell that runs or reads a FedACO sweep calls it first.

It **raises** instead of printing, because a failing `!` line does not stop Run-All — the
notebook would carry straight on into hours of GPU on the broken objective.
''')

FEDERATION = code(r'''
import subprocess
import sys

GPUS_PER_CLIENT = 0.1   # 1/10 clients


def ensure_gate_fix():
    """Apply the gate's fitness fix to pyproject.toml, or halt Run-All."""
    r = subprocess.run(
        [sys.executable, "scripts/apply_gate_fix.py",
         "--from-results", "results/fl/gate", "--apply-verdict"],
        capture_output=True, text=True,
    )
    print(r.stdout[-3000:])
    if r.stderr.strip():
        print(r.stderr[-1500:])
    if r.returncode != 0:
        raise RuntimeError(
            "No usable fitness fix, so no FedACO sweep may run. Either the gate's results are "
            "not here (attach the Day 1 notebook's output: + Add Input -> Your Work), or the "
            "gate found NO arm that closes the corner -- then re-run the gate with a larger "
            "aco-gamma-entropy before anything else."
        )


print("GPUs per ClientApp:", GPUS_PER_CLIENT)
''')


GATE_IF_MISSING = code(r'''
# The gate's results normally arrive with Day 1's attached output. If they are not here, run the
# gate now (~20 min) rather than stopping: its seeds are fixed and it runs deterministic, so it
# should reach Day 1's verdict. ensure_gate_fix() prints the verdict -- compare it with Day 1's.
import subprocess
from pathlib import Path

if not list(Path("results/fl/gate").glob("*.json")):
    print("No gate results attached -- running the gate here first (~20 min).")
    subprocess.run(["git", "checkout", "--", "pyproject.toml"], check=True)
    !python scripts/run_sweep_granular.py --config configs/experiment/gate_fitness.yaml --gpus-per-client {GPUS_PER_CLIENT}

ensure_gate_fix()
''')


def save(next_step: str) -> list[tuple[str, str]]:
    return [
        md(r'''
---
# Before the session ends — SAVE, or you lose everything

Kaggle discards `/kaggle/working` unless the notebook is **committed**: use
**Save Version → Save & Run All (Commit)**, not the quick save. Next session, attach this
notebook's output as an input (**+ Add Input → Your Work**) so the restore cell brings it back
and every sweep resumes per-cell.
'''),
        code(r'''
from pathlib import Path

results = sorted(Path("results").rglob("*.json"))
print(f"{len(results)} result file(s) to save")
for directory in sorted({p.parent for p in results}):
    print(f"  {directory}: {len(list(directory.glob('*.json')))}")
print("\nSave Version -> Save & Run All (Commit). The quick save does NOT keep /kaggle/working.")
print("NEXT: ''' + next_step + r'''")
'''),
    ]


def granular_summary(config: str, body: str, imports: str = "") -> tuple[str, str]:
    """A summary cell that filters results by the sweep's own predicted run_ids.

    `imports` joins the header's import block, so every import stays at the top of the cell.
    """
    return code(r'''
# Filter by the run_ids THIS sweep's runner would produce, via the same code path. Other
# sweeps write to the same directory -- A1 and A2 both use results/fl/ablation -- so filtering
# on config values would fold one into the other. ensure_gate_fix() runs before the ids are
# computed, because run_ids hash pyproject's defaults and the fix changes them.
import json
''' + imports + r'''from pathlib import Path

from fedswarm.sweep import granular_runs, planned_run_ids

ensure_gate_fix()

CONFIG = "''' + config + r'''"
mine = planned_run_ids(granular_runs(CONFIG, Path(".").resolve()), "pyproject.toml")
print(f"{len(mine)} cells belong to {Path(CONFIG).name}")

f1 = {}   # label -> final test macro-F1
for path in Path("results/fl").rglob("*.json"):
    result = json.loads(path.read_text())
    if result.get("run_id") in mine:
        value = (result.get("final") or {}).get("final_test_macro_f1")
        if value is not None:
            f1[mine[result["run_id"]]] = float(value)
print(f"{len(f1)}/{len(mine)} of this sweep's cells have a result\n")


def mean_std(vals):
    m = sum(vals) / len(vals)
    s = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
    return m, s

''' + body)


# ======================================================================================
# Day 1 — the gate, then A2

DAY1 = [
    md(r'''
# FedSwarm — DAY 1: the gate, then A2 (Kaggle GPU)

| step | what | cells | time |
|---|---|---|---|
| 1 | **`gate_fitness`** — which fitness fix closes the degenerate optimum | 8 | ~15 min |
| 2 | **read the verdict and apply it** | — | 1 min |
| 3 | **`a2_reduced`** — does cross-round pheromone memory help at all | 30 | ~6 GPU-h |

**Why A2 first.** FedAWA (CVPR 2025), Adp-FL-PSO and FedPSO already optimise aggregation weights,
and all three are **stateless between rounds**. Cross-round pheromone persistence is the only
structural novelty left, and A2 is the only experiment that tests it.

**Two outcomes that stop you:** step 2 finds no arm that closes the corner (the notebook halts
itself), or step 3 shows `none` ≈ `decayed` ≈ `full` (persistence buys nothing — re-plan before
spending the remaining ~64 GPU-hours). Full context: `HANDOVER.md`.
'''),
    md(HOW_TO_RUN),
    md("## 1. Setup — GPU check, then clone from GitHub"), PREFLIGHT, CLONE, INSTALL, DATASET,
    md("### Restore results from a previous session\n\nSkip on your first run. On a re-run, "
       "attach this notebook's previous output (**+ Add Input → Your Work**) first."),
    RESTORE,
    md("## 2. Build the image cache (~3 min, once per session)"), CACHE,
    FEDERATION_MD, FEDERATION,
    md(r'''
---
# STEP 1 — the gate: which fitness fix closes the degenerate optimum

8 cells, 120 rounds, **~15 minutes.** On the first real GPU run the best **single-client vertex**
outscored the FedAvg reference point in **15 of 15 rounds** (mean margin **+0.6252**). The colony
did not go there only because the search was too short — so the failure arrives as the search
gets *better*, while every health signal reads normal.

| arm | changes | evidence so far |
|---|---|---|
| `default` | nothing — reproduces the failure | corner won 15/15, +0.6252 |
| `gamma_entropy_0.45` | concentration penalty at its closed-form crossing | a *lower* bound; 0.6 broke everything downstream |
| `dispersion_aggregate` | dispersion shape | local fixture: corner won 0/15, −0.5283 |
| `fedavg` | — | baseline for health check 4 |

**Read `corner_margin`, not macro-F1** — 15 rounds is far too short for macro-F1.
'''),
    code(r'''
# The gate must run against the DOCUMENTED defaults, or its `default` arm is not the default.
# If this session already applied a fix, put pyproject back first. Every later cell calls
# ensure_gate_fix(), which re-applies it.
import subprocess
subprocess.run(["git", "checkout", "--", "pyproject.toml"], check=True)

!python scripts/run_sweep_granular.py --config configs/experiment/gate_fitness.yaml --gpus-per-client {GPUS_PER_CLIENT}
'''),
    code("!python scripts/check_fedaco_health.py --results-dir results/fl/gate"),
    md(r'''
---
# STEP 2 — the verdict, and applying it

The rule is **negative in every round, not on average**: the penalty weight and dispersion shape
are set once per run, so a value that clears the mean round leaves the worst rounds degenerate.
The fix goes into **`pyproject.toml`**, the one lever every later sweep inherits — A1 and A2 never
read `configs/strategy/fedaco.yaml`, so patching that file would leave the two sweeps that decide
the paper on the broken default.

**If no arm closed the corner, the next cell halts the notebook.** That is correct: running A2 on
an open corner buys a number that cannot be interpreted.
'''),
    code(r'''
import tomllib

ensure_gate_fix()

with open("pyproject.toml", "rb") as fh:
    live = tomllib.load(fh)["tool"]["flwr"]["app"]["config"]
print("live config every later run will use:")
for key in ("aco-gamma-entropy", "aco-dispersion-reference", "aco-q0", "aco-rho-round"):
    print(f"  {key:28} = {live[key]!r}")
'''),
    md(r'''
---
# STEP 3 — A2: does cross-round pheromone memory do anything?

30 cells, 3,000 rounds, **~6 GPU-h** — one Kaggle session with room. `none`, `decayed`, `full`
across Dirichlet 0.3 and 0.1, 5 seeds. If the session is cut off, re-run the notebook with this
output attached; finished cells are skipped.

5 of these 30 cells are the same resolved config as 5 of Day 2's A1 cells (A2's `decayed` at
Dirichlet 0.3 *is* A1's `aco` there), so Day 2 gets them free.
'''),
    code(r'''
ensure_gate_fix()
!python scripts/run_sweep_granular.py --config configs/experiment/ablation_a2_reduced.yaml --gpus-per-client {GPUS_PER_CLIENT}
'''),
    granular_summary("configs/experiment/ablation_a2_reduced.yaml", r'''
by_arm = defaultdict(list)
for label, value in f1.items():
    arm, partition, _seed = label.split("/")
    by_arm[(arm, partition)].append(value)

if not f1:
    print("Nothing to summarise yet.")
else:
    print(f"{'persistence':12} {'partition':15} {'n':>3} {'mean F1':>9} {'std':>7}")
    for key in sorted(by_arm, key=lambda k: (k[1], k[0])):
        m, s = mean_std(by_arm[key])
        print(f"{key[0]:12} {key[1]:15} {len(by_arm[key]):>3} {m:>9.4f} {s:>7.4f}")
    print("\nIf none / decayed / full sit inside each other's std, cross-round persistence buys")
    print("nothing and the last structural novelty is gone. Stop and re-plan before Day 2.")
''', imports="from collections import defaultdict\n"),
    code("!python scripts/make_tables.py --results-dir results/fl/ablation --out paper/tables"),
    *save("Day 2 -- notebooks/kaggle_day2_a1.ipynb. Attaching THIS notebook's output saves ~20 min."),
]


# ======================================================================================
# Day 2 — A1

DAY2 = [
    md(r'''
# FedSwarm — DAY 2: A1, the go/no-go on the framing (Kaggle GPU)

**Optional, saves ~20 minutes:** attach the **Day 1** notebook's output (right sidebar →
**+ Add Input → Your Work**). It holds the gate's results, from which this notebook re-derives the
fitness fix. Without it, this notebook re-runs the gate itself before A1.

**A1** replaces the colony with random search, coordinate-grid search, PSO and a GA at an
**identical evaluation budget and identical fitness**. 80 cells, ~17 GPU-h — **two sessions**.
Session 2: attach this notebook's own session-1 output too; finished cells are skipped. The 5
cells A1 shares with Day 1's A2 are skipped as well.

**What to expect, written down before it runs:** three local screens found ACO *losing* to every
control (ACO +0.0013 → +0.0066 against random +0.0304, PSO +0.0484, coordinate grid +0.0501; 0 of
3 seeds). The mechanism is known: the desirability signal spans ~0.04 against a level spacing of
0.25, so the greedy branch reproduces the FedAvg point ~70% of the time.

- **ACO loses or ties** → framing A stands as written in `paper/01_INTRODUCTION.md`.
- **ACO significantly beats all four** → delete §1.3 there and paste the alternative framing from
  the end of the same file.
'''),
    md(HOW_TO_RUN),
    md("## 1. Setup — GPU check, then clone from GitHub"), PREFLIGHT, CLONE, INSTALL, DATASET,
    md("### Restore — Day 1's gate results and any earlier A1 session, if attached"),
    RESTORE,
    md("## 2. Build the image cache (~3 min, once per session)"), CACHE,
    FEDERATION_MD, FEDERATION,
    md(r'''
---
# STEP 1 — the fitness fix: from Day 1's results, or by re-running the gate

Halts only if the gate finds no arm that closes the corner -- in which case no FedACO sweep
should run at all.
'''),
    GATE_IF_MISSING,
    md(r'''
---
# STEP 2 — A1: 80 cells, ~17 GPU-h
'''),
    code(r'''
ensure_gate_fix()
!python scripts/run_sweep_granular.py --config configs/experiment/ablation_a1_reduced.yaml --gpus-per-client {GPUS_PER_CLIENT}
'''),
    md(r'''
---
# STEP 3 — did ACO beat the controls?

Paired by seed: the same seed gives every method the same partition and initialisation, so
"ACO beat PSO on seed 3" is a like-for-like comparison. Significance is `make_tables`' job (next
cell); this is the direct reading.
'''),
    granular_summary("configs/experiment/ablation_a1_reduced.yaml", r'''
scores = defaultdict(dict)   # (partition, seed) -> {method: f1}
for label, value in f1.items():
    method, partition, seed = label.split("/")
    scores[(partition, seed)][method] = value

if not f1:
    print("Nothing to summarise yet.")
else:
    controls = ["random", "coordinate_grid", "pso", "ga"]
    for partition in sorted({p for p, _ in scores}):
        seeds = [s for (p, s) in scores if p == partition]
        print(f"=== {partition} ===")
        for method in ["aco", *controls]:
            vals = [scores[(partition, s)][method] for s in seeds if method in scores[(partition, s)]]
            if vals:
                m, s = mean_std(vals)
                print(f"  {method:16} n={len(vals)}  mean {m:.4f}  std {s:.4f}")
        for control in controls:
            pairs = [(scores[(partition, s)]["aco"], scores[(partition, s)][control])
                     for s in seeds
                     if "aco" in scores[(partition, s)] and control in scores[(partition, s)]]
            if pairs:
                wins = sum(a > c for a, c in pairs)
                diff = sum(a - c for a, c in pairs) / len(pairs)
                print(f"  aco vs {control:16} wins {wins}/{len(pairs)} seeds, mean diff {diff:+.4f}")
        print()

    print("Framing A stands unless make_tables (below) shows ACO SIGNIFICANTLY ahead of all four")
    print("controls. Losing to or tying any one of them is enough for framing A.")
''', imports="from collections import defaultdict\n"),
    code("!python scripts/make_tables.py --results-dir results/fl/ablation --out paper/tables"),
    *save("Day 3 -- notebooks/kaggle_day3_main_and_robustness.ipynb."),
]


# ======================================================================================
# Day 3 — the main table and the two robustness sweeps

DAY3 = [
    md(r'''
# FedSwarm — DAY 3: the main table, then R1 and R2 (Kaggle GPU)

**Optional, saves ~20 minutes:** attach the **Day 1** notebook's output (right sidebar →
**+ Add Input → Your Work**) — it holds the gate's results. Without it, this notebook re-runs the
gate itself first.

| step | sweep | cells | GPU-h |
|---|---|---|---|
| 1 | `main_reduced` — 6 strategies × 3 regimes × 8 seeds | 144 | ~30 |
| 2 | `robustness_r1_reduced` — label-flip at 30% + the clean arm | 40 | ~8 |
| 3 | `robustness_r2_reduced` — gaussian and sign-flip at 30% | 40 | ~8 |

**~46 GPU-hours is five or six Kaggle sessions** and more than one account's weekly quota
(~30 GPU-h). Split it across accounts or weeks. Every sweep resumes per cell: re-run with this
notebook's previous output attached and it continues where it stopped.

**R1 must finish before R2's table means anything.** R2 ships with no unattacked arm by design;
its deltas are measured against R1's `clean` arm in the same `results/fl/robustness`.
'''),
    md(HOW_TO_RUN),
    md("## 1. Setup — GPU check, then clone from GitHub"), PREFLIGHT, CLONE, INSTALL, DATASET,
    md("### Restore — Day 1's gate results and any earlier Day 3 session, if attached"),
    RESTORE,
    md("## 2. Build the image cache (~3 min, once per session)"), CACHE,
    FEDERATION_MD, FEDERATION,
    GATE_IF_MISSING,
    md(r'''
---
# STEP 1 — the main table: 144 cells, ~30 GPU-h

Then the IID band check. With almost no heterogeneity to exploit, every strategy should land in a
narrow band on IID; a wide spread means something other than the method is driving the results,
and nothing else in the table can be trusted until it is explained.
'''),
    code(r'''
ensure_gate_fix()
!python scripts/run_sweep.py --config configs/experiment/main_reduced.yaml --gpus-per-client {GPUS_PER_CLIENT}
'''),
    code(r'''
!python scripts/check_iid_band.py --results-dir results/fl/main
!python scripts/make_tables.py --results-dir results/fl/main --out paper/tables
'''),
    md(r'''
---
# STEP 2 and 3 — robustness: R1, then R2

Plan §8: **any configuration where FedACO loses stays in the table** and is stated in the
paper's limitations. Do not quietly drop a losing arm.
'''),
    code(r'''
ensure_gate_fix()
!python scripts/run_sweep_granular.py --config configs/experiment/robustness_r1_reduced.yaml --gpus-per-client {GPUS_PER_CLIENT}
'''),
    code(r'''
ensure_gate_fix()
!python scripts/run_sweep_granular.py --config configs/experiment/robustness_r2_reduced.yaml --gpus-per-client {GPUS_PER_CLIENT}
'''),
    granular_summary("configs/experiment/robustness_r1_reduced.yaml", r'''
clean_labels = [lab for lab in mine.values() if "/clean/" in lab]
clean_done = [lab for lab in clean_labels if lab in f1]
print(f"R1 clean arm: {len(clean_done)}/{len(clean_labels)} cells done")
if len(clean_done) < len(clean_labels):
    print("R2's deltas are measured against this arm. Until it is complete the R2 table below")
    print("will show '—' for some rows -- which is NOT the same as 'no difference'.")
'''),
    code("!python scripts/make_tables.py --results-dir results/fl/robustness --out paper/tables"),
    *save("hand the tables in paper/tables/ to whoever writes paper §5 and the abstract."),
]


NOTEBOOKS = {
    "kaggle_day1_gate_and_a2.ipynb": ("d1", DAY1),
    "kaggle_day2_a1.ipynb": ("d2", DAY2),
    "kaggle_day3_main_and_robustness.ipynb": ("d3", DAY3),
}


def build(prefix: str, cells: list[tuple[str, str]]) -> dict:
    out = []
    for i, (kind, source) in enumerate(cells):
        cell = {"cell_type": kind, "id": f"{prefix}-{i:02d}", "metadata": {},
                "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        out.append(cell)
    return {
        "cells": out,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render(nb: dict) -> str:
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    for name, (prefix, cells) in NOTEBOOKS.items():
        (OUT / name).write_text(render(build(prefix, cells)))
        print(f"wrote notebooks/{name} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
