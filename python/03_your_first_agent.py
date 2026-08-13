# python/03_your_first_agent.py
"""Tutorial 03 - Your First Agent.

Third rung of the fairlib tutorial series: assembling SimpleAgent from
its five components and watching the ReAct loop reason, act, and
observe. Setup and how to run: README.md.
"""

# %% [markdown]
# ## Tutorial 03 - Your First Agent
#
# In tutorial 01 the model claimed `48193 * 90271` was a number it
# invented. You could fix that by hand: prompt the model to ask for a
# calculation, parse its reply, run the math in Python, paste the result
# back, ask again - and you would have reinvented, badly, the loop this
# framework exists to run well.
#
# That loop has a name, **ReAct**: the model *reasons* about what to do,
# *acts* by requesting a tool, *observes* the result, and repeats until
# it can answer. Your job is not to write that loop. Your job is to
# assemble the components and let fairlib run it.
#
# What you will learn:
#
# - The five-part anatomy of an agent: model, planner, tools, executor,
#   memory.
# - How to build a `SimpleAgent` and drive it with `await agent.arun(...)`.
# - How to read the agent's mind by following Thought -> Action ->
#   Observation in memory.
# - How conversation continuity carries across `arun` calls.
# - The guardrails you get for free: `max_steps` and typed input errors.
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
import os

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
    InvalidAgentInputError,
    SafeCalculatorTool,
    ToolRegistry,
    ToolExecutor,
    WorkingMemory,
    HuggingFaceAdapter,
    SimpleReActPlanner,
    SimpleAgent,
)

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# One model, loaded once - the same instance serves every call below.
print(f"Loading {MODEL_NAME}...")
llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=512)

# %% [markdown]
# ### A window into the agent's head
#
# One helper before we start, and it will reappear near the top of every
# tutorial that runs an agent: `show_agent_mind` prints the agent's full
# working memory - every thought, tool call, observation, and answer, in
# order. Whenever something important happens inside the loop, we will
# call it and look at the brain directly.

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
# ### Step 1: the anatomy of an agent
#
# An agent is five components, each swappable, each with one
# responsibility. "Swappable" is meant literally: every one of these is
# an interface (an abstract base class in `fairlib.core.interfaces`), and
# fairlib ships a default implementation of each. If a default does not
# fit your capstone - a planner with a different format, a memory that
# persists to disk, a tool catalog of your own - you implement that one
# interface and plug it in; nothing else has to change. You will see this
# theme in every tutorial. Here are the five:
#
# - The **brain** is the model - the `HuggingFaceAdapter` from tutorial
#   01 - which produces text.
# - The **mind** is the planner, `SimpleReActPlanner`, which turns the
#   history into the next decision.
# - The **toolbelt** is the `ToolRegistry`, which holds the tools the
#   agent may use.
# - The **hands** are the `ToolExecutor`, which actually runs the tool
#   the mind chose.
# - The **memory** is `WorkingMemory`, which carries the conversation
#   and the observations.
#
# The planner deserves a paragraph, because it is the bridge straight
# back to tutorial 02. Every planner owns a `PromptBuilder` - the very
# class you used in tutorial 02 - as a member you reach through the
# planner instance: `planner.prompt_builder`. You have two ways to set
# its contents:
#
# - **Provide your own** builder when you construct the planner:
#   `SimpleReActPlanner(llm, tools, prompt_builder=my_builder)`.
# - **Use the default** the planner makes for you, and reach in to
#   customize it - which is what we do below:
#   `planner.prompt_builder.role_definition = ...`.
#
# Either way, the role, format rules, and examples are the same typed
# parts from tutorial 02. On top of them the planner adds two things you
# do not write: the parser-coupled ReAct format rules, and a tool catalog
# generated from each tool's typed schema (the schema-to-prompt bridge
# you built by hand in tutorial 02, now automatic for every registered
# tool). It then parses the model's reply into a typed decision: a tool
# call or a final answer.
#
# The next cell builds each component as its own named object, then
# hands the finished parts to `SimpleAgent`. Read it top to bottom and
# you are reading the anatomy list again, in code.

# %%
# The toolbelt: what the agent is allowed to do.
tools = ToolRegistry()
tools.register_tool(SafeCalculatorTool())

# The mind: wraps the model with ReAct rules plus the tool catalog. We
# pass no prompt_builder, so the planner creates a default one; we then
# reach through the planner instance - planner.prompt_builder - to set
# the same typed parts (RoleDefinition, FormatInstruction, Example) you
# met in tutorial 02.
planner = SimpleReActPlanner(llm, tools)
planner.prompt_builder.role_definition = RoleDefinition(
    "You are a precise operations assistant. Use your tools for any "
    "arithmetic instead of computing in your head, and follow the "
    "strict formatting rules that follow."
)
planner.prompt_builder.format_instructions.append(
    FormatInstruction(
        "Final answers are one or two plain sentences stating the "
        "result, with no working shown."
    )
)

# Tutorial 02's Example machinery, reapplied: one worked turn showing
# the exact tool-call syntax. Small local models follow a format far
# more reliably after seeing it done once - note tool_input carries the
# BARE expression, not a field name.
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

# The hands: runs whichever tool the mind chose.
executor = ToolExecutor(tools)

# The memory: carries the conversation and the observations.
memory = WorkingMemory()

# The agent: the five parts, assembled. fairlib runs the loop.
agent = SimpleAgent(
    llm=llm,
    planner=planner,
    tool_executor=executor,
    memory=memory,
    max_steps=6,
)

# %% [markdown]
# ### Read the assembled prompt: yours versus the framework's
#
# This is the payoff of tutorial 02's closing note. The planner's system
# prompt is **assembled, not hand-written**, and you should read the
# printout below as two layers stacked together:
#
# - **What you defined** (from your `prompt_builder`): the role text you
#   wrote, your format instruction, and the worked `Example` - the same
#   typed parts from tutorial 02, rendered into labeled sections.
# - **What fairlib injected** (from the planner): the ReAct format rules
#   that tell the model exactly how to emit a `tool_name`/`tool_input`
#   action, and the **tool catalog**, a description of `safe_calculator`
#   generated automatically from its typed schema. You never wrote that
#   section; the planner did, from the tool you registered.
#
# Scan the whole thing and label each section in your head as *yours* or
# *the framework's*. That split - your content wrapped in the planner's
# scaffolding - is the entire idea of a planner, and it is why you never
# hand-write an agent's prompt.

# %%
# The full assembled prompt. Find your role/example (yours) and the ReAct
# format block plus the tool catalog (injected by the planner).
print(planner.render_system_prompt())

# %% [markdown]
# ### Step 2: the redemption run
#
# Same multiplication the raw model flubbed in tutorial 01. This time
# the model does not have to *know* the answer; it has to know to *ask
# the calculator*. That is a much easier problem, and exactly the kind a
# small local model handles reliably.

# %%
a, b = 48193, 90271

async def redemption() -> None:
    answer = await agent.arun(f"What is {a} * {b}?")
    print("AGENT SAYS: ", answer)
    print("PYTHON SAYS:", a * b)

asyncio.run(redemption())

# %% [markdown]
# ### Step 3: read the agent's mind
#
# The whole exchange is in memory, including the parts you did not see:
# the planner's committed turns record what the model thought, which
# tool it called, and what it observed back. This is the ReAct loop, in
# the agent's own words - our first real use of `show_agent_mind`.

# %%
show_agent_mind(agent)

# %% [markdown]
# ### Step 4: memory makes it a conversation
#
# Call `arun` again on the same agent and the new request lands on top
# of the existing history. "That result" means something because the
# memory carried it. In tutorial 01 you replayed history by hand; this
# is that, automated.

# %%
async def follow_up() -> None:
    answer = await agent.arun("Divide that result by 7 and round to two decimals.")
    print("AGENT SAYS: ", answer)
    print("PYTHON SAYS:", round(a * b / 7, 2))

asyncio.run(follow_up())

# %% [markdown]
# Open the mind once more and you can see continuity as data: the memory
# now holds *both* exchanges in order - the original multiplication and
# then the follow-up, whose "that result" only resolved because the
# earlier turns were still there for the model to read. This is the
# hand-rolled history replay of tutorial 01, now kept for you.

# %%
show_agent_mind(agent)

# %% [markdown]
# ### Step 5: the guardrails you already have
#
# Two are worth knowing about on day one:
#
# - **`max_steps`** (we set it to 6): a model that cannot decide gets
#   that many loop iterations, then the run ends with a typed
#   `MaxStepsExceeded` instead of spinning your GPU forever.
# - **Input validation:** empty input is rejected with a typed error,
#   not passed to the model as a confusing blank turn.

# %%
async def blank_input() -> None:
    try:
        await agent.arun("   ")
    except InvalidAgentInputError as exc:
        print(f"{type(exc).__name__}: {exc}")

asyncio.run(blank_input())

# %% [markdown]
# ### Debrief: what this means for your capstone
#
# An agent is five swappable parts around one loop. You configured the
# components; fairlib ran the loop - the prompt assembly, the parsing,
# the tool dispatch, and the bookkeeping. The model's job shrank from
# *knowing the answer* to *choosing the right tool*, and that shrinkage
# is the single biggest reliability win in agentic design.
#
# Every tutorial from here upgrades one part of this anatomy:
#
# - better tools in 04,
# - better memory in 05,
# - visibility into the loop in 06,
# - armor around it in 07 and 10,
# - and more hands and minds in 08 and 09.
#
# **Capstone connection.** Open the capstone template your team is
# given: its `build_agent()` is this tutorial's wiring, line for line.
# Every capstone - Neural Shields' threat responder, the avatar tutor,
# the telescope duty officer - starts as exactly this five-part assembly
# with different parts plugged in.
#
# **Next:** tutorial `04_tools_are_power` is where you write your own
# typed tools - your capstone's primary way of teaching the agent to
# touch your problem.
