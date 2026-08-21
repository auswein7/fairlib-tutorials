"""Verify this machine is ready to run the fairlib tutorials.

Run it right after `pip install` and again any time a tutorial fails
to import something:

    python check_setup.py

Every check explains what is wrong and the exact command that fixes it.
No model is loaded; this finishes in a few seconds.
"""

import importlib
import importlib.metadata
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_PY = (3, 11)
MAX_PY = (3, 13)  # fair-llm 0.6.1 pins wheels that do not exist for 3.14+

problems = []


def ok(msg):
    print(f"  [ok]   {msg}")


def fail(msg, fix):
    print(f"  [FAIL] {msg}\n         fix: {fix}")
    problems.append(msg)


print(f"fairlib tutorials setup check - {platform.system()} {platform.machine()}")
print(f"  python {platform.python_version()} at {sys.executable}\n")

# 1. Python version - the single most common cause of a failed install.
v = sys.version_info[:2]
if v < MIN_PY:
    fail(
        f"Python {v[0]}.{v[1]} is too old (need 3.11-3.13)",
        "install Python 3.12 from https://www.python.org/downloads/ and "
        "recreate the venv with it: see README 'Setup'",
    )
elif v > MAX_PY:
    fail(
        f"Python {v[0]}.{v[1]} is too new - fair-llm's pinned wheels (torch, "
        "faiss, onnxruntime) do not exist for it yet",
        "install Python 3.12 alongside it and recreate the venv with "
        "`py -3.12 -m venv .venv` (Windows) or `python3.12 -m venv .venv`",
    )
else:
    ok(f"Python {v[0]}.{v[1]} is supported")

# 2. Running inside a virtual environment, ideally this repo's.
in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
if not in_venv:
    fail(
        "not running inside a virtual environment",
        "activate it first: `source .venv/bin/activate` "
        "(Windows: `.venv\\Scripts\\activate`), then rerun",
    )
else:
    expected = os.path.join(HERE, ".venv")
    if os.path.realpath(sys.prefix) != os.path.realpath(expected):
        print(f"  [warn] venv is {sys.prefix}, not {expected} - fine if intentional")
    ok("virtual environment is active")

# 3. The right fairlib: fair-llm from PyPI, not the unrelated `fairlib` package.
try:
    ver = importlib.metadata.version("fair-llm")
    ok(f"fair-llm {ver} is installed")
except importlib.metadata.PackageNotFoundError:
    fail(
        "fair-llm is not installed in this interpreter",
        'pip install "fair-llm[all]" jupyterlab',
    )
    ver = None

try:
    importlib.metadata.version("fairlib")
    fail(
        "the unrelated PyPI package `fairlib` (a fairness library) is installed "
        "and shadows fair-llm's `fairlib` module",
        'pip uninstall -y fairlib && pip install --force-reinstall --no-deps "fair-llm"',
    )
except importlib.metadata.PackageNotFoundError:
    pass

# 4. The imports the tutorials actually use.
checks = [
    ("dotenv", "python-dotenv", "fair-llm"),
    ("fairlib", "fair-llm", "fair-llm"),
    ("torch", "torch", "fair-llm[local]"),
    ("transformers", "transformers", "fair-llm[local]"),
    ("faiss", "faiss-cpu", "fair-llm[rag]"),
    ("chromadb", "chromadb", "fair-llm[rag]"),
    ("sentence_transformers", "sentence-transformers", "fair-llm[rag]"),
]
for mod, dist, extra in checks:
    try:
        importlib.import_module(mod)
        ok(f"import {mod}")
    except Exception as e:  # ImportError or a broken native lib
        fail(f"import {mod} failed: {type(e).__name__}: {e}", f'pip install "{extra}"')

if ver:
    try:
        from fairlib import Message, HuggingFaceAdapter, SimpleAgent  # noqa: F401

        ok("from fairlib import Message, HuggingFaceAdapter, SimpleAgent")
    except Exception as e:
        fail(f"fairlib top-level imports failed: {e}", 'pip install --upgrade "fair-llm[all]"')

# 5. Jupyter kernel points at this interpreter (notebook track).
try:
    importlib.import_module("jupyterlab")
    ok("jupyterlab is installed in this venv")
except ImportError:
    print("  [warn] jupyterlab not installed - only needed for the notebooks/ track "
          "(pip install jupyterlab)")

# 6. .env present and filled.
env_path = os.path.join(HERE, ".env")
if not os.path.exists(env_path):
    fail(
        ".env not found at repo root",
        "cp env.example .env  (Windows: copy env.example .env), then fill it in",
    )
else:
    from_env = {}
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, val = line.partition("=")
                from_env[k.strip()] = val.strip().strip('"').strip("'")
    if not from_env.get("HF_TOKEN"):
        fail("HF_TOKEN is empty in .env", "paste your Hugging Face token after HF_TOKEN=")
    else:
        ok("HF_TOKEN is set")
    model = from_env.get("FAIR_LLM_DEMO_MODEL")
    if not model:
        fail("FAIR_LLM_DEMO_MODEL is empty in .env", "pick a model row from HARDWARE.md")
    else:
        ok(f"FAIR_LLM_DEMO_MODEL = {model}")

# 7. GPU visibility (informational).
try:
    import torch

    if torch.cuda.is_available():
        ok(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        ok("Apple Metal (MPS) GPU available")
    else:
        print("  [warn] no GPU detected - use a <=1.5B model from HARDWARE.md; runs will be slow")
except Exception:
    pass

print()
if problems:
    print(f"{len(problems)} problem(s) found - fix the items marked [FAIL] and rerun.")
    sys.exit(1)
print("All checks passed. Start with: python python/01_meet_the_model.py")
