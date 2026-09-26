"""Every notebook's code must agree with the code it calls.

Written after the Day 1 notebook (`notebooks/kaggle_day1_gate_and_a2.ipynb`) shipped with five
bugs, all statically detectable and all of the kind that surfaces only at the end of a
six-GPU-hour Kaggle session:

* an import of a name `fedswarm.sweep` does not export (`load_experiment_config`);
* a call to `run_sweep` missing a required argument (`pyproject_path`);
* a read of `final["test_macro_f1"]` when the result file writes `final_test_macro_f1` --
  which returned None and printed an empty table indistinguishable from "no results yet".

The pytest suite never saw any of it, because nothing executed notebook code. These checks do
not execute it either (it needs a GPU, Kaggle paths and the dataset); they parse it and check
every reference it makes against the source it references. Each source of truth is read from
the code that produces it -- the valid `final` keys are parsed out of `fl/app.py`, script flags
out of each script -- so a rename there fails here instead of on Kaggle.

`test_each_check_catches_the_bug_it_exists_for` feeds a deliberately broken notebook through
every check. A check that cannot fail is not a check, which is this repository's recurring
failure mode.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))
MAGIC = re.compile(r"^(\s*)[!%]")


# --------------------------------------------------------------------------------------
# Extraction


def code_cells(nb: dict) -> list[tuple[int, str, list[str]]]:
    """(cell index, python source with magics replaced by `pass`, the magic lines).

    Magics become `pass` at their own indentation rather than being deleted, so a `!` line
    inside an `if` block leaves the block syntactically whole.
    """
    out = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        lines, magics = [], []
        for line in "".join(cell["source"]).split("\n"):
            m = MAGIC.match(line)
            if m:
                magics.append(line.strip())
                lines.append(f"{m.group(1)}pass")
            else:
                lines.append(line)
        out.append((i, "\n".join(lines), magics))
    return out


def load(path: Path) -> dict:
    return json.loads(path.read_text())


# --------------------------------------------------------------------------------------
# Sources of truth, read from the code that produces them


def final_keys() -> set[str]:
    """Keys of the `final={...}` dict `fl/app.py` writes into every result file."""
    tree = ast.parse((REPO_ROOT / "src/fedswarm/fl/app.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "final" and isinstance(node.value, ast.Dict):
            return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    raise AssertionError("could not find `final={...}` in fl/app.py; this test needs updating")


def script_flags(script: Path) -> set[str]:
    return set(re.findall(r"""["'](--[\w-]+)["']""", script.read_text())) | {"--help"}


# --------------------------------------------------------------------------------------
# Checks. Each returns a list of human-readable problems; empty means clean.


def check_parses(nb: dict) -> list[str]:
    problems = []
    for i, src, _ in code_cells(nb):
        try:
            ast.parse(src)
        except SyntaxError as exc:
            problems.append(f"cell {i}: SyntaxError: {exc.msg} (line {exc.lineno})")
    return problems


def _fedswarm_imports(tree: ast.AST) -> list[tuple[str, str, str]]:
    """(module, name, local alias) for every `from fedswarm... import name`."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("fedswarm"):
            for alias in node.names:
                out.append((node.module, alias.name, alias.asname or alias.name))
    return out


def check_imports(nb: dict) -> list[str]:
    problems = []
    for i, src, _ in code_cells(nb):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for module, name, _ in _fedswarm_imports(tree):
            try:
                mod = importlib.import_module(module)
            except ImportError as exc:
                problems.append(f"cell {i}: cannot import module {module}: {exc}")
                continue
            if not hasattr(mod, name):
                problems.append(f"cell {i}: {module} has no attribute {name!r}")
    return problems


def check_call_signatures(nb: dict) -> list[str]:
    """Bind every call to an imported fedswarm callable against its real signature.

    Placeholder values, so this checks arity and keyword names only -- which is exactly what
    `run_sweep(runs, dry_run=True)` got wrong.
    """
    problems = []
    for i, src, _ in code_cells(nb):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        bound = {}
        for module, name, local in _fedswarm_imports(tree):
            try:
                obj = getattr(importlib.import_module(module), name)
            except (ImportError, AttributeError):
                continue
            if callable(obj):
                bound[local] = (f"{module}.{name}", obj)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id not in bound:
                continue
            if any(isinstance(a, ast.Starred) for a in node.args) or any(
                k.arg is None for k in node.keywords
            ):
                continue  # *args / **kwargs: arity is not statically knowable
            qualname, obj = bound[node.func.id]
            try:
                sig = inspect.signature(obj)
            except (TypeError, ValueError):
                continue
            try:
                sig.bind(*[None] * len(node.args), **{k.arg: None for k in node.keywords})
            except TypeError as exc:
                problems.append(f"cell {i}: {qualname}(...) line {node.lineno}: {exc}")
    return problems


def _is_final_expr(node: ast.AST) -> bool:
    """True for expressions that evaluate to a result file's `final` dict."""
    if isinstance(node, ast.Name) and node.id == "final":
        return True
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        return _is_final_expr(node.values[0])
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get" and node.args
            and isinstance(node.args[0], ast.Constant) and node.args[0].value == "final"):
        return True
    if (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
            and node.slice.value == "final"):
        return True
    return False


def check_final_keys(nb: dict) -> list[str]:
    """A key read from `final` must be one `fl/app.py` writes there."""
    valid, problems = final_keys(), []
    for i, src, _ in code_cells(nb):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            key, receiver = None, None
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get" and node.args
                    and isinstance(node.args[0], ast.Constant)):
                key, receiver = node.args[0].value, node.func.value
            elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                key, receiver = node.slice.value, node.value
            if key is None or not _is_final_expr(receiver) or key == "final":
                continue
            if key not in valid:
                problems.append(
                    f"cell {i}: reads final[{key!r}], but fl/app.py writes only {sorted(valid)}"
                )
    return problems


SCRIPT_CALL = re.compile(r"python\s+(scripts/[\w./-]+\.py)(.*)")


def check_scripts_and_flags(nb: dict) -> list[str]:
    problems = []
    for i, _, magics in code_cells(nb):
        for magic in magics:
            m = SCRIPT_CALL.search(magic)
            if not m:
                continue
            script = REPO_ROOT / m.group(1)
            if not script.exists():
                problems.append(f"cell {i}: no such script {m.group(1)}")
                continue
            known = script_flags(script)
            for flag in re.findall(r"(?<!\S)(--[\w-]+)", m.group(2)):
                if flag not in known:
                    problems.append(f"cell {i}: {m.group(1)} has no flag {flag}")
    return problems


CONFIG_PATH = re.compile(r"configs/[\w./-]+\.ya?ml")


def check_config_paths(nb: dict) -> list[str]:
    problems = []
    for i, src, magics in code_cells(nb):
        for text in [src, *magics]:
            for path in set(CONFIG_PATH.findall(text)):
                if "*" in path or "[" in path:
                    continue
                if not (REPO_ROOT / path).exists():
                    problems.append(f"cell {i}: references missing {path}")
    return problems


# Stdlib modules and the Python version that introduced them. A notebook may use any module
# available at the project's DECLARED floor (`requires-python` in pyproject.toml), because pip
# refuses to install fedswarm below it -- so a runtime older than the floor fails loudly at the
# install cell, not quietly at a later one.
#
# An earlier version of this check hardcoded "Kaggle images have shipped 3.10" and flagged
# `import tomllib`. That was an assumption, not a measurement, and it was wrong in the way
# that matters: the floor is 3.11, and B's notebook installed fedswarm on Kaggle on
# 2026-09-19, which pip would have refused below 3.11. A check that flags code which cannot
# break teaches people to ignore it. The floor is now read, not assumed.
STDLIB_SINCE = {"tomllib": (3, 11), "graphlib": (3, 9), "zoneinfo": (3, 9)}


def python_floor() -> tuple[int, int]:
    text = (REPO_ROOT / "pyproject.toml").read_text()
    m = re.search(r'requires-python\s*=\s*">=\s*(\d+)\.(\d+)', text)
    assert m, "pyproject.toml declares no requires-python lower bound"
    return int(m.group(1)), int(m.group(2))


def check_stdlib_portability(nb: dict, floor: tuple[int, int] | None = None) -> list[str]:
    """Flag a stdlib import newer than the declared floor, unless guarded by a try."""
    floor = floor or python_floor()
    problems = []
    for i, src, _ in code_cells(nb):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        guarded = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                catches = {ast.unparse(h.type) for h in node.handlers if h.type is not None}
                if catches & {"ImportError", "ModuleNotFoundError", "Exception"}:
                    for inner in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                        guarded.add(id(inner))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                since = STDLIB_SINCE.get(name)
                if since and since > floor and id(node) not in guarded:
                    problems.append(
                        f"cell {i}: unguarded `import {name}` needs Python "
                        f"{since[0]}.{since[1]}+, above the declared floor "
                        f"{floor[0]}.{floor[1]}"
                    )
    return problems


CHECKS = [check_parses, check_imports, check_call_signatures, check_final_keys,
          check_scripts_and_flags, check_config_paths, check_stdlib_portability]


# --------------------------------------------------------------------------------------
# Tests


def test_there_are_notebooks_to_check() -> None:
    assert len(NOTEBOOKS) >= 5, [p.name for p in NOTEBOOKS]


@pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook(path: Path, check) -> None:
    problems = check(load(path))
    assert not problems, f"{path.name}:\n  " + "\n  ".join(problems)


def test_final_keys_are_read_from_the_producer() -> None:
    """If this ever returns an empty or tiny set, check_final_keys passes everything."""
    keys = final_keys()
    assert "final_test_macro_f1" in keys
    assert "best_val_macro_f1" in keys
    assert "test_macro_f1" not in keys, "the exact bug this exists for would pass"


def test_each_check_catches_the_bug_it_exists_for() -> None:
    """The five Day 1 bugs, reconstructed. Every check must flag its own."""
    def cell(src: str) -> dict:
        return {"cell_type": "code", "source": src.splitlines(keepends=True)}

    broken = {"cells": [
        cell("x = (\n"),                                                           # parse
        cell("from fedswarm.sweep import load_experiment_config\n"),               # import
        cell("from fedswarm.sweep import run_sweep\nrun_sweep([], dry_run=True)\n"),  # signature
        cell("final = {}\nf1 = final.get('test_macro_f1')\n"                      # final key
             "g = (r.get('final') or {}).get('test_macro_f1')\n"),
        cell("!python scripts/run_sweep_granular.py --config x --no-such-flag\n"),  # flag
        cell("!python scripts/no_such_script.py\n"),                               # script
        cell("CONFIG = 'configs/experiment/does_not_exist.yaml'\n"),               # config
        cell("import tomllib\n"),                                                  # portability
    ]}

    assert check_parses(broken), "check_parses missed a SyntaxError"
    assert any("load_experiment_config" in p for p in check_imports(broken))
    assert any("pyproject_path" in p for p in check_call_signatures(broken)), \
        check_call_signatures(broken)
    final_problems = check_final_keys(broken)
    assert len(final_problems) == 2, final_problems  # both spellings of the receiver
    flag_problems = check_scripts_and_flags(broken)
    assert any("--no-such-flag" in p for p in flag_problems)
    assert any("no_such_script" in p for p in flag_problems)
    assert any("does_not_exist" in p for p in check_config_paths(broken))
    # Against a hypothetical 3.10 floor the check must fire; against the real floor it must
    # not, because tomllib cannot break a runtime pip was willing to install into.
    assert any("tomllib" in p for p in check_stdlib_portability(broken, floor=(3, 10)))
    assert not check_stdlib_portability(broken), "flags tomllib despite a >=3.11 floor"

    guarded = {"cells": [cell("try:\n    import tomllib\nexcept ModuleNotFoundError:\n    pass\n")]}
    assert not check_stdlib_portability(guarded, floor=(3, 10))


def test_a_clean_notebook_passes_every_check() -> None:
    """The converse: the checks must not flag correct code, or they get ignored."""
    good = {"cells": [{"cell_type": "code", "source": [
        "from fedswarm.sweep import run_sweep\n",
        "run_sweep([], pyproject_path='pyproject.toml', dry_run=True)\n",
        "f1 = (r.get('final') or {}).get('final_test_macro_f1')\n",
        "!python scripts/run_sweep_granular.py --config configs/experiment/gate_fitness.yaml\n",
    ]}]}
    for check in CHECKS:
        assert not check(good), f"{check.__name__} flagged correct code: {check(good)}"
