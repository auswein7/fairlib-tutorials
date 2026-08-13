# tutorials/_build_notebooks.py
"""Regenerate the notebooks/ twin of every tutorial script in python/.

The scripts under tutorials/python/ are the source of truth. Each is a
runnable, plain-prose Python file in the percent cell format, and this
script renders it into the matching tutorials/notebooks/ notebook so the
two tracks never drift. The mapping to notebook cells:

  - The leading module docstring is skipped: it is the script track's
    header, and the first '# %% [markdown]' block is the notebook's
    single opening cell, so the notebook does not open with two
    overlapping introductions.
  - A line reading '# %% [markdown]' opens a markdown cell; the
    comment lines that follow it (their leading '# ' stripped) are the
    cell's markdown source.
  - A line reading '# %%' opens a code cell that runs to the next
    marker.
  - A statement-level asyncio.run(...) call is rewritten to a
    top-level await in the notebook, which is how the same flow runs
    under Jupyter's already-running event loop. The scripts stay
    plain-python runnable; the notebooks stay step-through friendly.

The narrative in a '# %% [markdown]' block may use light markdown
(headings, bold, bullet lists, backticked identifiers) because it is
rendered as a notebook cell; keep it legible as raw comment text too.
Code cells and docstrings follow the normal fairlib style rules.

Run from anywhere, inside the repo venv:

    python tutorials/_build_notebooks.py            # write all notebooks
    python tutorials/_build_notebooks.py --check    # verify freshness only
    python tutorials/_build_notebooks.py 01 03      # only these tutorials
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Optional

_HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = _HERE / "python"
NOTEBOOKS_DIR = _HERE / "notebooks"

_MARKER_RE = re.compile(r"^# %%( \[markdown\])?\s*.*$")
_ASYNCIO_RUN_RE = re.compile(r"^(\s*)asyncio\.run\((.+)\)\s*$")


def _docstring_end(lines: List[str]) -> int:
    """Return the index of the first line after the module docstring.

    Only a docstring that opens the file (comments and blank lines may
    precede it) is recognized; anything else returns 0.
    """
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith("#")):
        i += 1
    if i >= len(lines) or not lines[i].lstrip().startswith('"""'):
        return 0
    first = lines[i].lstrip()[3:]
    if first.rstrip().endswith('"""') and len(first.rstrip()) >= 3:
        return i + 1
    i += 1
    while i < len(lines):
        if lines[i].rstrip().endswith('"""'):
            return i + 1
        i += 1
    return 0


def _cell(cell_type: str, source_lines: List[str], index: int) -> dict:
    """Build one notebook cell dict from raw source lines."""
    while source_lines and not source_lines[0].strip():
        source_lines = source_lines[1:]
    while source_lines and not source_lines[-1].strip():
        source_lines = source_lines[:-1]
    source = [line + "\n" for line in source_lines]
    if source:
        source[-1] = source[-1].rstrip("\n")
    cell = {
        "cell_type": cell_type,
        "id": f"cell-{index}",
        "metadata": {},
        "source": source,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def _transform_code_line(line: str) -> str:
    """Rewrite statement-level asyncio.run(...) to top-level await."""
    match = _ASYNCIO_RUN_RE.match(line)
    if match is None:
        return line
    return f"{match.group(1)}await {match.group(2)}"


def build_notebook(py_path: Path) -> dict:
    """Parse one percent-format tutorial script into a notebook dict."""
    lines = py_path.read_text(encoding="utf-8").splitlines()
    cells: List[dict] = []

    # The module docstring is the script track's header only; skipping it
    # here keeps the notebook to a single opening markdown cell.
    start = _docstring_end(lines)

    current_type: Optional[str] = None
    current: List[str] = []

    def flush() -> None:
        nonlocal current
        if current_type is not None and any(line.strip() for line in current):
            cells.append(_cell(current_type, current, len(cells)))
        current = []

    for line in lines[start:]:
        marker = _MARKER_RE.match(line)
        if marker:
            flush()
            current_type = "markdown" if marker.group(1) else "code"
            continue
        if current_type == "markdown":
            stripped = re.sub(r"^# ?", "", line)
            current.append(stripped)
        elif current_type == "code":
            current.append(_transform_code_line(line))
        # Pre-marker leftovers (imports before the first cell) are not
        # expected in tutorial files and are dropped deliberately: the
        # docstring plus the first '# %%' cell must open every file.
    flush()

    # The asyncio.run -> await rewrite can leave the asyncio import
    # unused in the notebook; drop it so the generated notebooks lint
    # as clean as their source scripts.
    code_text = "\n".join(
        "".join(c["source"]) for c in cells if c["cell_type"] == "code"
    )
    if "asyncio." not in code_text:
        for cell in cells:
            if cell["cell_type"] == "code":
                cell["source"] = [
                    line
                    for line in cell["source"]
                    if line.rstrip("\n") != "import asyncio"
                ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(argv: List[str]) -> int:
    check_only = "--check" in argv
    # A bare number selects by two-digit rung: '1' means 01, not 10-12.
    selectors = [
        a.zfill(2) if a.isdigit() else a for a in argv if a != "--check"
    ]
    scripts = sorted(
        p
        for p in SCRIPTS_DIR.glob("[0-9][0-9]_*.py")
        if not selectors or any(p.name.startswith(s) for s in selectors)
    )
    if not scripts:
        print("no tutorial scripts matched")
        return 1
    NOTEBOOKS_DIR.mkdir(exist_ok=True)
    stale = []
    for script in scripts:
        notebook = build_notebook(script)
        rendered = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
        target = NOTEBOOKS_DIR / (script.stem + ".ipynb")
        if check_only:
            if not target.exists() or target.read_text(encoding="utf-8") != rendered:
                stale.append(target.name)
            continue
        target.write_text(rendered, encoding="utf-8")
        print(f"wrote {target.name}: {len(notebook['cells'])} cells")
    if stale:
        print("STALE (rerun _build_notebooks.py): " + ", ".join(stale))
        return 1
    if check_only:
        print(f"all {len(scripts)} notebooks up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
