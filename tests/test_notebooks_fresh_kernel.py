"""Every Kaggle notebook's imports must work in a kernel that cannot see the editable install.

A user hit `ModuleNotFoundError: No module named 'fedswarm'` on Kaggle, one cell after a check
printing INSTALL OK. `pip install -e .` registers fedswarm through a .pth file, which Python
reads only when a process starts; the kernel was already running, so only fresh subprocesses
could import it -- and the INSTALL OK check was one of those.

The static check in test_notebooks.py reasons about this. This one executes it: a subprocess
strips every `src` entry from sys.path and purges fedswarm (asserting fedswarm is then NOT
importable, so the simulation is real), then runs each code cell's top-level imports,
sys.path changes and simple assignments in notebook order. A notebook passes only if every
import succeeds under the condition that failed on Kaggle.

Written after the failure it would have caught; the reconstruction reproduces the user's exact
error against the version they ran.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_NOTEBOOKS = sorted((REPO_ROOT / "notebooks").glob("kaggle_*.ipynb"))

HARNESS = r'''
import ast, importlib, json, os, sys

repo, nb_path = sys.argv[1], sys.argv[2]
sys.path[:] = [p for p in sys.path if not p.rstrip("/").endswith("src") and p != repo]
for m in [m for m in sys.modules if m == "fedswarm" or m.startswith("fedswarm.")]:
    del sys.modules[m]
importlib.invalidate_caches()
try:
    import fedswarm  # noqa: F401
    print("INVALID: fedswarm importable before the notebook ran"); sys.exit(3)
except ModuleNotFoundError:
    pass

os.chdir(repo)
ns = {"__name__": "__main__", "REPO_DIR": repo}
nb = json.load(open(nb_path))
for idx, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
    lines = ["pass" if l.lstrip().startswith(("!", "%")) else l
             for l in "".join(cell["source"]).split("\n")]
    tree = ast.parse("\n".join(lines))
    for stmt in tree.body:
        keep = isinstance(stmt, (ast.Import, ast.ImportFrom))
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            keep = ast.unparse(stmt.value.func).startswith("sys.path.")
        if (isinstance(stmt, ast.If) and "sys.path" in ast.unparse(stmt)
                and all(isinstance(s, ast.Expr) for s in stmt.body)):
            keep = True
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == "SRC"):
            keep = True
        if not keep:
            continue
        try:
            exec(compile(ast.Module(body=[stmt], type_ignores=[]), f"cell{idx}", "exec"), ns)
        except ModuleNotFoundError as exc:
            print(f"FAIL cell {idx} line {stmt.lineno}: {ast.unparse(stmt)[:90]} -> {exc}")
            sys.exit(1)
print("OK")
'''


def run_fresh(nb_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", HARNESS, str(REPO_ROOT), str(nb_path)],
                          capture_output=True, text=True, timeout=180)


@pytest.mark.parametrize("path", KAGGLE_NOTEBOOKS, ids=lambda p: p.name)
def test_every_import_works_in_a_fresh_kernel(path: Path) -> None:
    result = run_fresh(path)
    assert result.returncode == 0 and "OK" in result.stdout, (
        f"{path.name}:\n{result.stdout}\n{result.stderr[-1500:]}"
    )


def test_the_harness_reproduces_the_kaggle_failure(tmp_path) -> None:
    """Against the shape of the notebook the user ran -- fedswarm imported before src is on
    the path -- the harness must fail the way Kaggle did. Otherwise it proves nothing."""
    broken = {"cells": [
        {"cell_type": "code", "source": ["!pip install -e .\n"]},
        {"cell_type": "code", "source": [
            "import sys\n",
            "from fedswarm.data.download import locate_or_fetch_kaggle_dataset\n",
            "sys.path.insert(0, 'src')\n",
        ]},
    ], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    path = tmp_path / "as_run_on_kaggle.ipynb"
    path.write_text(json.dumps(broken))
    result = run_fresh(path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "No module named 'fedswarm'" in result.stdout
