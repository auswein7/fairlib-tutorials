# python/11_ship_your_capstone.py
"""Tutorial 11 - Ship Your Capstone.

Eleventh rung of the fairlib tutorial series: centralized
configuration, the agent as a JSON artifact, conversation checkpoints,
and provider portability - hand-off day for your capstone. Setup and
how to run: README.md.
"""

# %% [markdown]
# ## Tutorial 11 - Ship Your Capstone
#
# Hand-off day. Your capstone demo is tomorrow, your teammate is
# presenting, and they need to run your agent on their machine - same
# persona, same tools, same behavior - and pick up the session where
# you left it. "It works in my notebook" is not a deliverable. A
# capstone is an application: **reproducible from artifacts, configured
# from one source of truth, and not welded to any single model
# provider.**
#
# What you will learn:
#
# - Where central configuration lives: the chain from `settings.yml`
#   through Pydantic schemas to `from fairlib import settings`.
# - How `save_agent_config` and `load_agent` treat the agent as a JSON
#   artifact - what that config captures, and what it honestly cannot
#   (callables).
# - How `save_state` and `load_state` checkpoint and rewind a
#   conversation.
# - How provider portability lets one agent config run behind any
#   adapter.
#
# It closes with a debrief of rungs 01-11, mapped to your capstone;
# tutorial 12 then composes them into one end-to-end expert agent.
#
# *Requirements: a local HuggingFace model (torch plus transformers; a
# GPU is recommended). Set `FAIR_LLM_DEMO_MODEL` to choose a different
# model.*

# %% [markdown]
# ### Setup
#
# *This cell is plumbing, not part of the lesson: it locates the repo
# folder and loads your `.env` settings. **Just run it** and move on.*

# %%
import asyncio
import json
import os
from pathlib import Path

try:
    # Running as a script: the repo root is this file's parent's parent.
    TUTORIALS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # Running as a notebook: resolve the repo root from the working directory.
    _cwd = os.getcwd()
    if os.path.basename(_cwd) == "notebooks":
        TUTORIALS_DIR = os.path.dirname(_cwd)
    else:
        TUTORIALS_DIR = _cwd

# A .env file at the repo root (copy env.example to .env) provides
# environment variables like FAIR_LLM_DEMO_MODEL and HF_TOKEN before
# any tutorial code reads them.
from dotenv import load_dotenv

load_dotenv(os.path.join(TUTORIALS_DIR, ".env"))

# %%
# fairlib imports run simplest to most complex - the order you meet them.
from fairlib import (
    RoleDefinition,
    FormatInstruction,
    Example,
    settings,
    SafeCalculatorTool,
    ToolRegistry,
    ToolExecutor,
    WorkingMemory,
    HuggingFaceAdapter,
    SimpleReActPlanner,
    SimpleAgent,
)

# The export/load helpers are not exported at the fairlib top level in
# the current PyPI release, so they come from their home module.
from fairlib.modules.agent.factory import save_agent_config, load_agent

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# %% [markdown]
# ### A window into the agent's head
#
# The same helper from tutorial 03. Here it is the proof instrument for
# step 5: after a checkpoint rewind we will open the memory and see
# which turns survived.

# %%
def show_agent_mind(agent: SimpleAgent, max_chars: int = 400) -> None:
    """Print the agent's working memory: every message, in order.

    The banner lines are here on purpose: they fence the dump off from
    whatever prints next, so the agent's mind never bleeds into the
    following output or notebook cell.
    """
    history = agent.memory.get_history()
    print("=" * 72)
    print(f"AGENT MIND - {len(history)} messages in working memory")
    print("=" * 72)
    for msg in history:
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        clipped = text[:max_chars]
        if len(text) > max_chars:
            clipped += f" ... [+{len(text) - max_chars} more chars]"
        print(f"--- {msg.role} " + "-" * (60 - len(msg.role)))
        print(clipped)
    print("=" * 72)
    print("END AGENT MIND")
    print("=" * 72)

# %% [markdown]
# ### Step 1: one source of truth for configuration
#
# Every policy knob you met in this series lives in one chain:
#
# - how many tools may run in parallel, from tutorial 08,
# - whether model calls stream, from tutorial 06,
# - whether the circuit breaker is armed, from tutorial 07.
#
# `fairlib/config/settings.yml` is parsed through Pydantic schemas into
# a single validated `AppSettings` object, importable anywhere as
# `from fairlib import settings`.
#
# The discipline this buys you is that **policy lives in config, not
# scattered through code**. When your teammate needs to know how your
# agent is tuned, they read one file; when they need to change a limit,
# they edit one file and every component sees the same value. A magic
# number buried in your agent-construction code is invisible at
# hand-off; a schema-validated setting is documentation that cannot
# drift. Typos and out-of-range values fail at load time with a named
# field, not at 2 a.m. mid-demo.

# %%
print("tool_dispatch.max_parallel_tools:", settings.tool_dispatch.max_parallel_tools)
print("tool_dispatch.max_actions_per_turn:", settings.tool_dispatch.max_actions_per_turn)
print("streaming.enabled:", settings.streaming.enabled)
print("breaker.enabled:", settings.breaker.enabled)

# %% [markdown]
# ### Step 2: build the agent worth handing off
#
# A compact agent in the shape your capstone will use: one tool, a
# persona set through `RoleDefinition`, an answer-format rule set
# through `FormatInstruction`, the standard planner wiring. One model
# load, as always in this series.

# %%
print(f"Loading {MODEL_NAME} (first run downloads the weights)...")
llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=256)

registry = ToolRegistry()
registry.register_tool(SafeCalculatorTool())

planner = SimpleReActPlanner(llm, registry)
planner.prompt_builder.role_definition = RoleDefinition(
    "You are Falcon Ops, a terse mission-support calculator for an "
    "airfield operations team. Use the safe_calculator tool for every "
    "computation - never compute mentally. Pass only a pure arithmetic "
    "expression in the expression field, for example '12 * 7'."
)
planner.prompt_builder.format_instructions.append(
    FormatInstruction("Answer with the result in one short sentence.")
)

# One worked turn keeps a small local model's tool-call syntax exact
# (tutorial 03's lesson). It is part of the agent's prompt content, so
# it rides along in the exported config later in this tutorial.
planner.prompt_builder.examples.append(
    Example(
        "User: What is 12 * 31?\n"
        "Thought: Arithmetic goes to the calculator. The input is the "
        "bare expression only.\n"
        "Action:\n"
        "tool_name: safe_calculator\n"
        "tool_input: 12 * 31"
    )
)

agent = SimpleAgent(
    llm=llm,
    planner=planner,
    tool_executor=ToolExecutor(registry),
    memory=WorkingMemory(max_size=30),
    max_steps=6,
)

# %% [markdown]
# ### Step 3: export the agent to JSON
#
# `save_agent_config` extracts the agent's full declarative
# configuration - the prompts, tool names, planner type, model info,
# and step limit - and writes it to a JSON file. That file IS the
# hand-off artifact: version-control it, diff it in review, email it to
# your teammate. It is the same schema fair_prompt_optimizer reads and
# writes, so an optimized prompt config flows back in through the same
# door. For prompts alone, the prompt store offers the narrower
# `save_builder` and `load_builder` pair.

# %%
HANDOFF = Path(TUTORIALS_DIR) / "_scratch" / "11_handoff"
HANDOFF.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = HANDOFF / "agent_config.json"

config = save_agent_config(agent, str(CONFIG_PATH))
print(f"saved {CONFIG_PATH}\n")
print(json.dumps(config, indent=2))

# %% [markdown]
# Read the JSON like a manifest. Captured:
#
# - every prompt field, so the persona travels,
# - the tool list, by class name,
# - the planner type,
# - the model's name and generation knobs,
# - and `max_steps`.
#
# Three things are NOT captured, and this is honest by design:
#
# - **The model itself.** The config records what model was used, but
#   the loader takes a live adapter as an argument, because weights are
#   not config.
# - **Callables.** Lifecycle hooks, action verifiers, and validators
#   from tutorials 07 and 10 are Python functions, and JSON cannot
#   carry them. After loading, you re-wire them in code with
#   `SimpleAgent(..., lifecycle_hooks=...)` or per-call on `arun`. Your
#   hand-off is therefore a config file plus a small build script,
#   which is exactly what your capstone repo's entry point should be.
# - **Conversation state.** Memory is runtime state, not
#   configuration; checkpoints handle it, two sections down.
#
# And one limitation that is a current gap, not a design choice: the
# loader's tool factory only knows the built-in `safe_calculator` -
# other tool names in a config are skipped with a log warning, so a
# loaded agent is missing them. That is
# why this export deliberately scopes the demo agent to the
# calculator. For your capstone, treat the config as the record and
# your build script as the authority: register your tools in code
# after loading.

# %% [markdown]
# ### Step 4: load it back, your teammate's first run
#
# `load_agent(path, llm)` rebuilds the whole stack from the file: the
# registry from the tool names, a planner of the recorded type, the
# prompts applied to a fresh builder, and a fresh `WorkingMemory`. We
# pass the same `llm` instance we already loaded, so there is no second
# model load - which is also exactly the swap seam your teammate uses
# to run your config on their hardware.
#
# The live run proves the reloaded agent kept both its capability and
# its persona.

# %%
handoff_agent = load_agent(str(CONFIG_PATH), llm)


async def first_run() -> None:
    answer = await handoff_agent.arun("What is 314 * 27?")
    print("HANDOFF AGENT:", answer)

asyncio.run(first_run())

# %% [markdown]
# ### Step 5: checkpoint and resume
#
# Hand-off is not only wiring, it is state. `agent.save_state(path)`
# writes an `AgentCheckpoint`: a lightweight bookmark holding the
# current history position, not message copies, with optional metadata.
# `agent.load_state(path)` rewinds memory back to that bookmark by
# truncating everything after it.
#
# The proof sequence is three short turns:
#
# 1. Ask a question, then checkpoint.
# 2. Ask a second question, so history grows past the bookmark.
# 3. Rewind, then ask the agent what came first; if the rewind worked,
#    the second exchange is gone from its memory.

# %%
CHECKPOINT_PATH = HANDOFF / "day1_checkpoint.json"


async def checkpoint_demo() -> None:
    first = await handoff_agent.arun("Calculate 19 * 23.")
    print("TURN 1:", first)

    checkpoint = handoff_agent.save_state(CHECKPOINT_PATH, metadata={"label": "after-turn-1"})
    print(f"\ncheckpoint saved at history position {checkpoint.history_length}")

    second = await handoff_agent.arun("Now calculate 1000 / 8.")
    print("\nTURN 2:", second)
    print("history length after turn 2:", len(handoff_agent.memory.history))

    handoff_agent.load_state(CHECKPOINT_PATH)
    print("\nrewound; history length now:", len(handoff_agent.memory.history))

    third = await handoff_agent.arun(
        "What was the most recent calculation I asked you for? Answer from our conversation."
    )
    print("\nTURN 3 (after rewind):", third)

asyncio.run(checkpoint_demo())

# %% [markdown]
# The turn-3 answer should point at the 19 * 23 exchange. The 1000 / 8
# turn happened, but the rewind truncated it out of memory, so from the
# agent's point of view it never did. A small model may phrase this
# loosely; the history lengths printed either side of `load_state` are
# the ground truth - and the memory itself is the proof. Open it: the
# first exchange and turn 3 are there, and the 1000 / 8 exchange is
# nowhere.

# %%
show_agent_mind(handoff_agent)

# %% [markdown]
# Because a checkpoint is a tiny JSON bookmark, you can drop one before
# every risky operation - the same pattern as a database savepoint.
# Full crash recovery with message copies is a separate concern with
# its own session store; the bookmark is for cheap, frequent rewind
# points inside a live process.

# %% [markdown]
# ### Step 6: provider portability
#
# Everything above ran on a local HuggingFace model, because that is
# free and private. Nothing above depends on it. The adapter seam from
# tutorial 01 means the hand-off config runs behind any provider; your
# teammate swaps one line, as the next cell shows.
#
# Cross-provider parity is a framework design bar, not an accident: the
# same agent is expected to behave equivalently across providers, so
# your capstone can develop locally and demo on hosted hardware. The
# `FAIR_LLM_DEMO_MODEL` override you have used all series is this
# principle in miniature; every tutorial stayed hardware-flexible by
# never naming a model in code.

# %%
# Each line below is a drop-in replacement for the HuggingFaceAdapter
# load in step 2 - the rest of this tutorial would run unchanged.
# They are commented out because they need a server or an API key.
#
# llm = OllamaAdapter(model_name="llama3", host="http://localhost:11434")
# llm = OpenAIAdapter(model_name="gpt-4o")            # OPENAI_API_KEY from env
# llm = AnthropicAdapter(model_name="claude-sonnet-4-5")  # ANTHROPIC_API_KEY from env
# llm = LoadBalancerAdapter("10.0.0.5", "qwen25-7b")  # fan out across a vLLM pool
#
# handoff_agent = load_agent(str(CONFIG_PATH), llm)   # same config, new provider
print("Provider swap is one line; the agent config does not change.")

# %% [markdown]
# ### Debrief: what this means for your capstone
#
# What shipping meant in this tutorial: policy in one config file, the
# agent as a version-controlled JSON artifact plus a small build script
# that re-wires the callables, conversation state bookmarked to disk,
# and the model provider reduced to an argument. That is the difference
# between a notebook that worked once and an application a teammate
# runs tomorrow.
#
# The debrief so far. Rungs 01-11, and for each one the fairlib
# surface it taught and the capstone seam it unlocked:
#
# - **01, Meet the Model:** `Message`, adapters, statelessness, and
#   hallucination. Its seam is the adapter, which every project uses.
# - **02, Prompts That Hold:** `PromptBuilder`, `RoleDefinition`, and
#   structured output. Its seam serves the bio-regenerative life
#   support and space situational awareness capstones.
# - **03, Your First Agent:** `SimpleAgent`, `SimpleReActPlanner`, and
#   the ReAct loop. Its seam is the starting skeleton in the capstone
#   template every project builds from.
# - **04, Tools Are Power:** `AbstractTool`, the typed registry, tool
#   groups, and file tools. Its seam serves Neural Shields, the
#   agent-based SAST capstone, the ICS capstone, and the
#   offensive-security capstone.
# - **05, Memory That Survives:** pinning, summarization, vector
#   stores, and RAG. Its seam serves Falcon Telescope AI and space
#   situational awareness.
# - **06, Watch It Think:** the event bus, typed events, traces, and
#   streaming. Its seam serves the security-automation capstone and the
#   avatar language tutor.
# - **07, When Models Misbehave:** validators, retries, the circuit
#   breaker, and timeouts. Its seam serves the ICS and
#   offensive-security capstones.
# - **08, Many Hands:** multi-action turns and side-effect-aware
#   dispatch. Its seam serves Neural Shields.
# - **09, A Team of Agents:** workers-as-tools and manager fan-out. Its
#   seam serves the multi-domain AI proof of concept.
# - **10, Trust Nothing:** the security manager, lifecycle hooks, and
#   action verification. Its seam serves the enterprise GenAI security
#   capstone, and everything that executes.
# - **11, Ship Your Capstone:** config, export and load, checkpoints,
#   and the provider swap. Its seam is hand-off day, which every
#   project reaches.
#
# **Capstone connection.** For every project, this rung IS hand-off
# day. The capstone template your team is given includes a worked
# example of bringing an adapter fairlib does not ship, and its house
# rule applies to everything you saw here: **extend from the outside,
# never edit fairlib**. Your repo holds your tools, your hooks, your
# config files; the framework stays pristine underneath.
#
# **Next:** tutorial 12, The Academy Archivist - one agent composing
# everything this series taught over a real scanned-document corpus:
# OCR ingestion, RAG, structured extraction, a tool served over MCP,
# and a pinned standing order. It is the capstone shape end to end,
# and the last rung before your semester. And when the framework
# fights you - an interface that will not stretch, a seam that is not
# there - that friction is not your failure. It is a finding. Write it
# in your project's FINDINGS.md; your instructors want it, and it is
# how fairlib gets better. The series will grow with the framework -
# sandboxed execution and evaluation harnesses next; see "Where the
# series goes next" in the series README for the current ladder.
