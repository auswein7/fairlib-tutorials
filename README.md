# fairlib Tutorials

Twelve runnable tutorials that ramp you from "what is a language
model" to a complete expert agent - one live scenario per rung of the
ladder. Built for the cadet who was just handed a capstone title and a
framework they have never seen.

Every tutorial drives a **real local language model** - never a mock,
never canned strings. You will watch a live model succeed, fail,
recover, and be corrected by the framework, because that experience is
the lesson. Outputs vary from run to run. That is not a bug: the whole
series is about engineering *reliable systems* on top of a
*stochastic component*.

## Setup

**You need Python 3.11, 3.12, or 3.13.** Not older (macOS's built-in
`python3` is 3.9) and not 3.14 yet - fairlib's pinned numeric stack
(torch, faiss, onnxruntime) has no wheels for 3.14, so `pip install`
fails outright and nothing lands. If you are unsure what you have,
run `python3 --version` (Windows: `py --list`). Install 3.12 from
<https://www.python.org/downloads/> if needed; on Windows tick
**"Add python.exe to PATH"** in the installer.

Everything else happens inside a virtual environment, once:

```bash
git clone https://github.com/auswein7/fairlib-tutorials.git
cd fairlib-tutorials

# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell or cmd)
py -3.12 -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install "fair-llm[all]" jupyterlab
cp env.example .env                # Windows: copy env.example .env
python check_setup.py
```

`check_setup.py` verifies the interpreter, the venv, every import the
tutorials use, and your `.env`; each failure prints the exact fix.
Run it again any time an import fails - **before** trying
`pip install` guesses. In particular, do not `pip install fairlib`:
that is an unrelated fairness library on PyPI (and it does not even
build on modern Python). The package is `fair-llm`; the import is
`fairlib`. There is no `fair_llm` module - if a tutorial only runs
after you rewrite its imports, your environment is wrong, not the
tutorial.

Your prompt shows `(.venv)` while the environment is active; open a
new terminal and it is gone until you activate again. Always start
the tutorials from an activated terminal, and launch Jupyter from the
same terminal (`jupyter lab`) so notebooks use this venv's kernel.

Then open `.env` in any editor and fill in the two values - your
Hugging Face token and the model your machine can carry. **Pick the
model with [HARDWARE.md](HARDWARE.md)**: it has measured numbers for
everything from an 8 GB no-GPU laptop to a lab machine, and tells you
which tutorials each size can complete. The first tutorial you run
downloads the model weights; after that you are offline-friendly.

That is the entire setup.

## Where the models come from

The [Hugging Face Hub](https://huggingface.co/models) is the public
registry of open-weight models - the "GitHub of models." A name like
`Qwen/Qwen2.5-1.5B-Instruct` is an address on it: organization, model
family, parameter count (1.5 billion), and variant (`Instruct` means
tuned to follow instructions and chat - what you want here). Your
`HF_TOKEN` identifies you to the Hub when the tutorials download
weights.

Under the hood, fairlib's `HuggingFaceAdapter` uses the `transformers`
library to load those weights and run them **in-process, on your own
hardware** - no server, no API bill, full control, and the same
library you would later use for fine-tuning. It is one of several
interchangeable adapters: `OllamaAdapter` talks to the Ollama app if
you prefer models managed outside Python, and the OpenAI/Anthropic
adapters talk to hosted frontier models. Tutorial 01 makes that seam
concrete; tutorial 11 swaps providers live. Select the model for the
mission.

## Pick your track

Every tutorial exists in two forms, generated from one source so they
never drift. The two folders hold the same twelve tutorials - walk
either one in numbered order:

| Track | Folder | Best for |
|---|---|---|
| Script | `python/` | Running end-to-end, diffing, copying into your project |
| Notebook | `notebooks/` | Stepping through cell by cell, re-running one step, experimenting |

```bash
# script track
python python/01_meet_the_model.py

# notebook track
jupyter lab notebooks/01_meet_the_model.ipynb
```

## Coming from the lessons?

If you just watched the fairlib lessons, you have seen most of this
series run on screen: the confident liar, REEF 41, the runway-wind
tool, "an agent is a loop, not a model." This repo is where you run it
yourself.

- **Full climb (recommended):** start at 01. Tutorials 01-05 put your
  hands on everything Part 1 showed - messages, adapters, prompts,
  structured output, your first agent, tools, memory and RAG - and
  the climb is fast when the ideas are fresh.
- **Straight to the hand-off:** run 06-09 yourself (events,
  reliability, parallelism, a team of agents), then climb rungs 10-12
  above them - security, shipping day, and the full Archivist.

## The ladder

Each rung teaches a bounded set of framework concepts through a
scenario it carries out live, and each ends with a **Capstone
connection** telling you which capstone seams it just unlocked.

| # | Tutorial | You learn |
|---|---|---|
| 01 | Meet the Model | `Message`, adapters, the system role, statelessness, hallucination - the two limits the whole framework exists to fix |
| 02 | Prompts That Hold | `PromptBuilder`, `RoleDefinition`, `FormatInstruction`, `Example`; structured (JSON/Pydantic) output a program can consume |
| 03 | Your First Agent | the ReAct loop: `SimpleAgent`, `SimpleReActPlanner`, `ToolRegistry`, `ToolExecutor`, `WorkingMemory`; Thought -> Action -> Observation |
| 04 | Tools Are Power | your own `AbstractTool` with Pydantic I/O schemas, the typed registry, tool groups, the standard file tools, typed tool errors |
| 05 | Memory That Survives | pinning, summarization events, long-term memory, embeddings, vector stores (FAISS/Chroma), RAG |
| 06 | Watch It Think | the event bus, typed events, structured trace export, token streaming + the final-answer stream filter |
| 07 | When Models Misbehave | `arun(validator=...)`, degraded responses, retries, circuit breaker, hard timeouts, loop guards |
| 08 | Many Hands | multi-action turns, batched dispatch, side-effect-aware scheduling, keyed observations |
| 09 | A Team of Agents | workers-as-tools: `WorkerAgentTool`, `build_worker_manager`, manager/specialist fan-out |
| 10 | Trust Nothing | prompt injection (direct and indirect), `BasicSecurityManager`, lifecycle-hook screening, action verification |
| 11 | Ship Your Capstone | central config, agent export/load, checkpoint/resume, provider swap, load balancing |
| 12 | The Academy Archivist | PDF ingestion with OCR, RAG over a real noisy corpus, deterministic extraction into SQLite, serving a tool over MCP, pinned standing orders - all composed in one expert agent |

Climb in order. Each rung assumes the ones below it and *only* the
ones below it.

## Tutorial 12 needs a little more

The Archivist ingests real PDFs with OCR and serves a tool over MCP,
so it wants three extra Python packages, two system packages, and a
folder of PDFs:

```bash
pip install pdfplumber pdf2image pytesseract "mcp<2"
sudo apt install tesseract-ocr poppler-utils   # macOS: brew install tesseract poppler
```

(pip will warn that fair-llm pins an older pillow than pdfplumber
wants; the newer pillow is fine - the tutorials are validated with it.)

Then drop two or three real PDFs (course catalogs, policy documents,
any scanned material) into `_scratch/12_archivist/corpus/` - the
tutorial's Step 1 walks you through it. OCR is CPU-bound, so this rung
is most comfortable on a lab machine, but it runs at any model size.

## When a run fails

**`ModuleNotFoundError` (dotenv, fairlib, torch, ...)** - your
terminal is not in the venv, or the install never completed. Run
`python check_setup.py`; it names the cause and the fix.

Otherwise check [HARDWARE.md](HARDWARE.md)'s pass/fail matrix for your
model size. The series is calibrated against Qwen2.5-7B-Instruct;
smaller models fumble more - and watching the framework catch a
fumble is the point - but a hard crash in 08 or 09 on a small model
is the known multi-action JSON cliff, not your setup. Run those two
rungs on a bigger model, and rerun anything you like at 7B on a lab
machine.

---

*For maintainers: `python/` is the source of truth, written in percent
cell format. `notebooks/` is generated - never edit a notebook by
hand. After editing a script, run `python _build_notebooks.py` (or
`--check` to verify freshness).*
