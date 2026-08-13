# Tutorial Series Hardware Guide

What it actually takes to run the twelve tutorials, measured - not
guessed. Every number below comes from live runs of the real tutorial
scripts on 2026-08-11 (fair-llm 0.6.x, Qwen2.5 Instruct models). Use this to pick the machine and the model size
before your first session, especially if that machine is your laptop.

## TL;DR

| Your machine | Set FAIR_LLM_DEMO_MODEL to | What you get |
|---|---|---|
| Any laptop, 8 GB RAM, no GPU | `Qwen/Qwen2.5-0.5B-Instruct` | 10 of 12 tutorials, fast; skip 08 and 09 |
| Laptop, 16 GB RAM, no GPU | `Qwen/Qwen2.5-1.5B-Instruct` | 10 of 12 tutorials, ~1-3 min each |
| Laptop with 4-6 GB NVIDIA GPU | `Qwen/Qwen2.5-1.5B-Instruct` | same coverage, several times faster |
| Laptop with 8 GB NVIDIA GPU | `Qwen/Qwen2.5-3B-Instruct` | 10 of 12, best small-model quality |
| GPU with 16 GB+ VRAM (lab machine) | default (`Qwen/Qwen2.5-7B-Instruct`) | all 12 tutorials, the intended experience |

The series defaults to Qwen2.5-7B-Instruct and is calibrated against
it. Smaller models run most rungs - and their extra fallibility makes
the reliability lessons vivid - but each smaller size hard-crashed at
least one rung in our runs (details below). For a first pass on a
laptop, that trade is fine: run the rungs your hardware supports, then
do 08/09 (and a 7B rerun of anything) on a lab machine.

## What a tutorial actually loads

- The chat model (the only heavy component; size is your choice via
  `FAIR_LLM_DEMO_MODEL`).
- From tutorial 05 on: the `all-MiniLM-L6-v2` sentence-transformers
  embedder (88 MB download, negligible RAM).
- Tutorial 12 only: pdfplumber/pdf2image/pytesseract plus the
  tesseract and poppler system packages, a folder of PDFs, and SQLite.
  OCR is CPU-bound and dominates that tutorial's runtime regardless of
  GPU.

## Measured model footprints

Measured through the same `HuggingFaceAdapter` the tutorials use,
default settings, one representative generation. GPU figures from an
RTX A6000; CPU figures with torch pinned to 8 threads to approximate a
modern laptop CPU. Laptop numbers will be in this ballpark, not
identical.

| Model | Disk (download) | GPU VRAM, steady | CPU RAM, steady | GPU speed | CPU speed (8 threads) |
|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 1.0 GB | 1.0 GB | 1.8 GB | 35 tok/s | 15 tok/s |
| Qwen2.5-1.5B-Instruct | 2.9 GB | 3.2 GB | 3.7 GB | 37 tok/s | 8.5 tok/s |
| Qwen2.5-3B-Instruct | 5.8 GB | 6.3 GB | 6.6 GB | 36 tok/s | 5.2 tok/s |
| Qwen2.5-7B-Instruct | 15 GB | 15.3 GB | 14.1 GB | 31 tok/s | 2.7 tok/s |

Add roughly 2 GB of RAM headroom on top of the model for Python,
torch, and (tutorial 05+) the embedder, and keep 10 GB of free disk
beyond the weights for the Python environment (the CUDA build of torch
alone is several GB; a CPU-only torch install is much smaller).

What this means in practice:

- CPU-only is viable at 0.5B and 1.5B. A complete agent tutorial
  (03, the ReAct loop) finished in 47 seconds CPU-only at 1.5B with a
  4.1 GB peak process footprint. Expect single-digit minutes per
  tutorial on a laptop; the model answers in seconds-to-tens-of-seconds
  per turn, which is watchable, not painful.
- CPU-only 7B is not practical: 2.7 tok/s means minutes per model
  turn, and the agent tutorials make many turns.
- 4-bit quantization does not currently rescue small GPUs. The
  adapter's `quantized=True` path settles at 5.9 GB for 7B, but during
  load it transiently allocates the full-precision 14.3 GB on the GPU,
  so it OOMs on the 8 GB cards it would otherwise fit.
- Apple Silicon: `device_map="auto"` routes through MPS on paper, and
  16 GB unified memory covers 1.5B-3B comfortably. Untested by this
  study - treat as promising, not verified.

## Which model sizes complete which tutorials

One full pass of all twelve scripts per size, same seed conditions,
GPU. PASS means exit 0 end to end; the failures are hard crashes, not
quality dips.

| Tutorial | 0.5B | 1.5B | 3B | 7B (default) |
|---|---|---|---|---|
| 01 Meet the Model | PASS | PASS | PASS | PASS |
| 02 Prompts That Hold | PASS | PASS | PASS | PASS |
| 03 Your First Agent | PASS | PASS | PASS | PASS |
| 04 Tools Are Power | PASS | PASS | PASS | PASS |
| 05 Memory That Survives | PASS | PASS | FAIL (a) | PASS |
| 06 Watch It Think | PASS | PASS | PASS | PASS |
| 07 When Models Misbehave | PASS | PASS | PASS | PASS |
| 08 Many Hands | FAIL (b) | FAIL (b) | PASS | PASS |
| 09 A Team of Agents | FAIL (b) | PASS | FAIL (a) | PASS |
| 10 Trust Nothing | PASS | PASS | PASS | PASS |
| 11 Ship Your Capstone | PASS | PASS | PASS | PASS |
| 12 The Academy Archivist | PASS | FAIL (c) | PASS | PASS |

(a) `MaxStepsExceeded` - the model wanders and burns the step budget,
which the tutorials calibrate for 7B behavior.
(b) `PlannerParseError` from `MultiActionReActPlanner` - the model
cannot reliably hold the multi-action JSON contract tutorials 08/09
require. This is the series' hardest formatting ask and the first
thing to break as models shrink.
(c) `PlannerParseError` from `SimpleReActPlanner` late in the long
archivist session - format discipline decays over a long haul.

Two caveats. First, these are single runs of a stochastic system: a
size that passed a rung once can fail it on another run and vice versa
(the 7B default itself has a roughly 1-in-4 recovered-fumble rate on
tutorial 04's parser - recovered, because the framework coaches the
retry). Read the table as
"where the cliff is", not as a guarantee. Second, structural PASS is
not lesson quality: at 0.5B the tutorial completes but the model
sometimes fails the very task the rung is demonstrating (in one 03 run
it answered the math question by echoing the expression instead of
using the calculator's result). 1.5B is the floor at which the lessons
still land; 0.5B is a smoke-test tier.

## Suggested first-run path for a laptop cohort

1. Install with CPU-only torch unless the laptop has an NVIDIA GPU;
   set `FAIR_LLM_DEMO_MODEL=Qwen/Qwen2.5-1.5B-Instruct` (or 0.5B on
   8 GB machines).
2. Run tutorials 01-07, 10, 11 locally. That is the entire
   single-agent curriculum: prompts, agents, tools, memory/RAG,
   events, reliability, security.
3. Do 08, 09, and a full-series 7B pass on a lab GPU machine
   (16 GB+ VRAM). Tutorial 12 runs at any size but wants the OCR
   system packages, so it is also most comfortable on the lab image.
4. Expect and embrace extra model fumbles at small sizes - watching
   the framework catch them is the point of the series. A crash in
   08/09 on a small model is the one failure mode that is not a
   lesson; that is the multi-action JSON cliff, not your setup.

## Methodology

Measurements taken on a 48-core EPYC server with RTX A6000 GPUs
(CUDA 12.9); CPU probes pinned to 8 torch threads to approximate a
laptop. Footprints are `torch.cuda.max_memory_reserved` /
`memory_allocated` for VRAM and peak process RSS for RAM, captured by
a probe driving `HuggingFaceAdapter.ainvoke` with the tutorials'
default settings. Pass/fail comes from running every tutorial script
exactly as shipped (`FAIR_LLM_DEMO_MODEL` was the only override).
Model weights were pre-downloaded; first-run downloads add the disk
column's size at your network speed.
