# python/04_tools_are_power.py
"""Tutorial 04 - Tools Are Power.

Fourth rung of the fairlib tutorial series: writing your own typed
tools, generated tool catalogs, typed registry lookups, least-privilege
tool groups, and typed tool failures. Setup and how to run:
README.md.
"""

# %% [markdown]
# ## Tutorial 04 - Tools Are Power
#
# The situation: you are building an **Airfield Ops Assistant**, an
# agent that answers a duty officer's questions about a small airfield.
# Some questions need math the model cannot be trusted to do (tutorial
# 01, limit 2). Some need facts that live in files the model has never
# seen. Both are solved the same way: give the agent a tool.
#
# A tool is the framework's *effector* - the unit that lets an agent act
# on the world instead of guessing about it. Your capstone will live or
# die by the tools you write, so this tutorial covers the full craft:
#
# - The anatomy of a custom `AbstractTool`: typed input, typed output,
#   and a declared side effect.
# - The tool catalog the model reads, **generated from your schema**
#   rather than hand-written.
# - Typed registry lookups - `get` and `get_by_name` - for your own
#   application code.
# - The standard file tools, granted through the **least-privilege
#   group** pattern.
# - What happens when a tool call fails validation: a typed failure the
#   contract catches.
#
# The one-sentence version: declare the contract once, and the framework
# enforces it everywhere.
#
# *Requirements: a local HuggingFace model (torch plus transformers; a
# GPU is recommended). The first run downloads the weights. Set
# `FAIR_LLM_DEMO_MODEL` to choose a different model.*

# %% [markdown]
# ### Setup
#
# *This cell is plumbing, not part of the lesson: it locates the repo
# folder and loads your `.env` settings. **Just run it** and move on.*

# %%
import asyncio
import logging
import math
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
from pydantic import BaseModel, Field

# fairlib imports run simplest to most complex - the order you meet them.
from fairlib import (
    RoleDefinition,
    FormatInstruction,
    Example,
    SideEffect,
    ToolOutput,
    TextResult,
    AbstractTool,
    ToolNotFoundError,
    ToolInputValidationError,
    ReadFileTool,
    ListDirTool,
    GrepTool,
    ToolRegistry,
    ToolExecutor,
    WorkingMemory,
    HuggingFaceAdapter,
    SimpleReActPlanner,
    SimpleAgent,
)

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# A small local model occasionally fumbles a tool call mid-loop (names a
# tool that does not exist, or guesses a filename before listing). The
# agent recovers - it hands the mistake back to the model as an
# observation and the next turn corrects course - but fairlib also logs
# each recovered fumble at ERROR level, which clutters this teaching
# notebook. We raise the agent logger's level so those recovered-error
# lines stay quiet; you still see any fumble-and-recovery in the agent's
# own memory via show_agent_mind, and tutorial 06 shows how to observe
# them properly on the event bus. This hides nothing that was not already
# recovered from.
logging.getLogger("fairlib.modules.agent.simple_agent").setLevel(logging.CRITICAL)

# %% [markdown]
# ### A window into the agent's head
#
# The same helper as tutorial 03, and it earns its keep here: when an
# agent answers from files, the loop underneath the answer is the whole
# lesson, and `show_agent_mind` is how we watch it.

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
# ### Step 1: load the model
#
# One model, loaded once, powers every agent in this tutorial. As
# always, nothing below binds to the provider - swap the adapter and
# everything else stands.

# %%
print(f"Loading {MODEL_NAME} (first run downloads the weights)...")
llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=512)

# %% [markdown]
# ### Step 2: anatomy of a tool
#
# A pilot asking "what's my crosswind?" needs trigonometry, and a
# language model predicts text, it does not compute (tutorial 01,
# limit 2). So we build the computation as a tool. Every fairlib tool
# declares four things:
#
# - **`input_schema`** - a Pydantic model, and it is the contract:
#   field names, types, and constraints. The framework validates every
#   call the model makes against it before your code runs.
# - **`output_schema`** - the typed result; `TextResult` is the shared
#   schema for tools whose result is plain text.
# - **`side_effect`** - what the tool does to the world: `READ_ONLY`
#   tools are safe to run in parallel and cache, while `MUTATING` and
#   `EXTERNAL` tools are dispatched sequentially.
# - **`acall`** - the logic. It receives an already validated instance
#   of your input schema and returns an instance of your output schema,
#   with no string parsing and no string returns.
#
# Note what the tool body is: **pure, deterministic math**. Given the
# same input it returns the same answer every time - exactly the
# property the model lacks, and the reason the tool exists.

# %%
class WindComponentInput(BaseModel):
    """Runway heading and wind, for a headwind/crosswind breakdown."""

    # ge and le in the Field(...) lines are Pydantic constraint keywords:
    # "greater than or equal" and "less than or equal". They are validated
    # bounds - a call outside them is rejected before the tool runs.
    runway_heading_deg: int = Field(
        ge=0,
        le=360,
        description="Runway magnetic heading in degrees (0-360), e.g. 350 for runway 35.",
    )
    wind_dir_deg: int = Field(
        ge=0,
        le=360,
        description="Direction the wind is blowing FROM, in degrees (0-360).",
    )
    wind_speed_kt: float = Field(
        ge=0,
        description="Wind speed in knots; zero or greater.",
    )


class RunwayWindComponentTool(AbstractTool):
    """Resolve a reported wind into components along and across a runway."""

    name = "runway_wind_components"
    description = (
        "Compute the headwind and crosswind components for a runway heading "
        "and a reported wind. Use this for any wind-component question; do "
        "not estimate the trigonometry yourself."
    )
    input_schema = WindComponentInput
    output_schema = TextResult
    side_effect = SideEffect.READ_ONLY

    async def acall(self, tool_input: WindComponentInput) -> ToolOutput:
        angle = math.radians(tool_input.wind_dir_deg - tool_input.runway_heading_deg)
        headwind = tool_input.wind_speed_kt * math.cos(angle)
        crosswind = tool_input.wind_speed_kt * math.sin(angle)
        along = "headwind" if headwind >= 0 else "tailwind"
        across = "from the right" if crosswind >= 0 else "from the left"
        return TextResult(
            result=(
                f"Runway heading {tool_input.runway_heading_deg:03d}, wind "
                f"{tool_input.wind_dir_deg:03d} at {tool_input.wind_speed_kt:g} kt: "
                f"{abs(headwind):.1f} kt {along}, "
                f"{abs(crosswind):.1f} kt crosswind {across}."
            )
        )


# %% [markdown]
# Before any agent touches it, call the tool directly - it is just an
# async method with a typed argument. Runway 35 (heading 350), wind 020
# at 18 knots: the 30-degree offset should give about 15.6 knots of
# headwind and 9.0 knots of crosswind.

# %%
async def try_tool_directly() -> None:
    wind_tool = RunwayWindComponentTool()
    output = await wind_tool.acall(
        WindComponentInput(runway_heading_deg=350, wind_dir_deg=20, wind_speed_kt=18)
    )
    print(output.render())


asyncio.run(try_tool_directly())

# %% [markdown]
# ### Step 3: the catalog is generated, not written
#
# How does the model learn that this tool exists and what fields it
# takes? **You never write that prompt text.** The planner renders a
# tool catalog directly from each registered tool's schema: every field
# name, type, requiredness, and `Field` description you declared above.
#
# This matters more than it looks. A hand-written tool description and
# the real schema will drift apart as your capstone evolves, and a
# drifted description means the model calls tools that no longer exist
# or passes fields that were renamed. Generated from one source, they
# cannot disagree.
#
# Note the `FormatInstruction` in the cell: the role says who the agent
# *is*, and the format instruction says how its final answers should
# *read*. Keeping them separate keeps each one small, and every section
# of the rendered prompt ends up with content in it.
#
# The `render_system_prompt` method shows the complete system prompt the
# planner will send: the role you set, your output-format rule, the
# mandatory response-format rules the parser depends on, and the
# schema-derived catalog. Find your field descriptions in it.

# %%
registry = ToolRegistry()
registry.register_tool(RunwayWindComponentTool())

planner = SimpleReActPlanner(llm, registry)
planner.prompt_builder.role_definition = RoleDefinition(
    "You are an airfield operations assistant serving the duty officer. "
    "You answer questions precisely and use your tools for any "
    "computation or lookup instead of estimating. Take exactly one action "
    "per turn: one tool_name and one tool_input, then wait for the "
    "observation. Never write a second tool_name in the same reply."
)
planner.prompt_builder.format_instructions.append(
    FormatInstruction(
        "Final answers are one or two plain-language sentences the duty "
        "officer can read over the radio: state the numbers with their "
        "units, rounded to one decimal place. No working, no jargon."
    )
)

# Tutorial 03's lesson, now with a MULTI-FIELD tool: the worked example
# shows the model that tool_input for a schema'd tool is a JSON object
# with the schema's field names - and that final_answer input is plain
# text, never JSON. Small local models copy what they are shown.
planner.prompt_builder.examples.append(
    Example(
        "User: Wind check for runway 08, wind 120 at 10 knots.\n"
        "Thought: A wind-component question goes to the tool. Its input "
        "is a JSON object using the schema's field names.\n"
        "Action:\n"
        "tool_name: runway_wind_components\n"
        'tool_input: {"runway_heading_deg": 80, "wind_dir_deg": 120, '
        '"wind_speed_kt": 10.0}\n'
        "(after the observation arrives)\n"
        "Thought: I have the components; time to answer in plain words.\n"
        "Action:\n"
        "tool_name: final_answer\n"
        "tool_input: Runway 08 has a 7.7 knot headwind and a 6.4 knot "
        "crosswind from the right."
    )
)

print(planner.render_system_prompt())

# %% [markdown]
# ### Step 4: the agent uses your tool
#
# Same wiring as tutorial 03: planner, executor, memory, agent. The
# only new part is that the tool in the registry is yours. Watch the
# agent route the wind question through `runway_wind_components`
# instead of guessing at the trigonometry.

# %%
agent = SimpleAgent(
    llm=llm,
    planner=planner,
    tool_executor=ToolExecutor(registry),
    memory=WorkingMemory(),
    max_steps=6,
)


async def ask_wind_question() -> None:
    answer = await agent.arun(
        "We are landing on runway 35, heading 350. Tower reports wind 020 "
        "at 18 knots. What are the headwind and crosswind components?"
    )
    print("\nAGENT:", answer)


asyncio.run(ask_wind_question())

# %% [markdown]
# The one-line answer above is only the surface. Open the agent's mind
# and read the loop that produced it - this is the first of several mind
# dumps in this tutorial, and there are four specific things worth
# finding in this one:
#
# 1. A **Thought** where the model decides the question needs the tool
#    rather than a guess - the reliability win from tutorial 03, now on
#    your own tool.
# 2. The **Action**, whose `tool_input` is a JSON object using your
#    schema's exact field names (`runway_heading_deg`, `wind_dir_deg`,
#    `wind_speed_kt`). The model learned those names from the generated
#    catalog, not from you writing them into the prompt.
# 3. The **Observation** - the deterministic string your `acall` returned.
#    Every number in the final answer traces back to it, not to the model.
# 4. The **final answer**, phrased in the plain radio style your
#    `FormatInstruction` demanded.

# %%
show_agent_mind(agent)

# %% [markdown]
# ### Step 5: typed lookups for your own code
#
# The model dispatches tools by name, but your application code around
# the agent often needs a tool handle too - to pre-warm a cache, to
# call it directly in a test, or to wire it somewhere else. The
# registry gives you typed lookups so that access is checkable:
#
# - `registry.get(ToolClass)` resolves by class, and the result is
#   typed as that class, so a rename or a missing registration surfaces
#   immediately instead of as a runtime miss deep in a run.
# - `registry.get_by_name("name")` is the same name-keyed lookup the
#   planner uses.
# - Both raise a typed `ToolNotFoundError` when nothing matches - a
#   failure you can catch by type rather than a silent `None`.

# %%
by_type = registry.get(RunwayWindComponentTool)
print(f"get(RunwayWindComponentTool) -> {by_type.name} ({type(by_type).__name__})")

by_name = registry.get_by_name("runway_wind_components")
print(f"get_by_name(...) -> the same instance: {by_name is by_type}")

try:
    registry.get_by_name("fuel_planner")
except ToolNotFoundError as exc:
    print(f"Typed miss -> {type(exc).__name__}: {exc}")

# %% [markdown]
# ### Step 6: standard file tools and least privilege
#
# Not every tool is one you write. fairlib ships a standard file-tool
# library - `ReadFileTool`, `ListDirTool`, `GrepTool`, and `GlobTool` -
# and each instance is **rooted**: it is confined at construction to one
# directory and rejects any path that escapes it, symlinks included. The
# agent gets exactly the slice of the filesystem you granted and nothing
# else.
#
# First, some airfield records for the tools to be rooted at. In your
# capstone this directory is your telemetry drop, your scan target, or
# your document store; here it is three small ops files.

# %%
AIRFIELD_DIR = os.path.join(TUTORIALS_DIR, "_scratch", "04_airfield")
os.makedirs(AIRFIELD_DIR, exist_ok=True)

airfield_files = {
    "runway_status.txt": (
        "RWY 17/35: OPEN. Surface dry, braking action good.\n"
        "RWY 08/26: CLOSED. The runway is closed for pavement repair, "
        "estimated reopening Friday.\n"
    ),
    "notams.txt": (
        "NOTAM A0412: RWY 08/26 closed for pavement repair effective "
        "immediately.\n"
        "NOTAM A0415: Bird activity reported north of the field at dusk.\n"
        "NOTAM A0417: Fuel pit 2 out of service; use pit 1.\n"
    ),
    "fuel_log.txt": (
        "0800 Truck 1 dispensed 450 gal JP-8.\n"
        "1130 Truck 2 dispensed 300 gal JP-8.\n"
        "1400 Fuel pit 2 flagged out of service.\n"
    ),
}
for filename, content in airfield_files.items():
    with open(
        os.path.join(AIRFIELD_DIR, filename), "w", encoding="utf-8"
    ) as handle:
        handle.write(content)
print(f"Wrote {len(airfield_files)} airfield records to {AIRFIELD_DIR}")

# %% [markdown]
# Now the pattern your capstone should copy: **tool groups**. A group is
# a named set of tools inside one registry, and `get_group` returns a
# registry view holding only that group, which you hand straight to an
# agent as its entire toolset.
#
# This is least privilege as one line of wiring. The records-clerk agent
# below gets the read-only file tools and *nothing else*: no wind tool,
# and - because we registered no write tools at all - no way to modify
# the records it reads. Its role teaches it the records workflow -
# `list_dir` to see what exists, `grep` to find which files mention a
# phrase, `read_file` for the full contents - plus one piece of search
# doctrine worth stealing for your capstone: never claim the records do
# not mention something unless a search for it came back empty.

# %%
registry.register_group(
    "readonly_ops",
    [
        ReadFileTool(AIRFIELD_DIR),
        ListDirTool(AIRFIELD_DIR),
        GrepTool(AIRFIELD_DIR),
    ],
)

ops_registry = registry.get_group("readonly_ops")
print("Full registry:", sorted(registry.get_all_tools()))
print("Clerk's view: ", sorted(ops_registry.get_all_tools()))

clerk_planner = SimpleReActPlanner(llm, ops_registry)
clerk_planner.prompt_builder.role_definition = RoleDefinition(
    "You are an airfield records clerk. Answer questions strictly from "
    "the airfield record files, never from memory. Work the records: "
    "list the files to see what exists, grep to find which files mention "
    "a phrase, and read a file for its full contents.\n"
    "CRITICAL SEARCH RULE: every single grep call MUST include "
    "ignore_case: true. The records mix capitalization (for example the "
    "files write 'Fuel pit 2' with a capital F), so a case-sensitive "
    "grep will miss real matches and make you wrongly report that "
    "nothing exists. Never omit ignore_case, and never claim the records "
    "do not mention something unless a grep with ignore_case: true "
    "returned zero matches.\n"
    "ONE ACTION PER TURN: emit exactly one tool_name and one tool_input, "
    "then stop and wait for its observation before choosing the next "
    "action. Even when a task lists several steps, do them one turn at a "
    "time. Never write a second tool_name in the same reply."
)
clerk_planner.prompt_builder.format_instructions.append(
    FormatInstruction(
        "Final answers are one or two sentences that cite the record "
        "file they came from, e.g. 'per runway_status.txt'. The "
        "tool_input of final_answer is plain prose - never JSON, never "
        "a file path."
    )
)

# The clerk's worked turn demonstrates the reliable workflow: list the
# directory, grep for the phrase, read the file that matched. tool_input
# for each tool is a JSON object with that tool's fields - except
# final_answer, which takes plain text.
clerk_planner.prompt_builder.examples.append(
    Example(
        "User: Do any records mention bird activity?\n"
        "Thought: List the record files first to see what is available.\n"
        "Action:\n"
        "tool_name: list_dir\n"
        'tool_input: {"path": ""}\n'
        "(the observation lists the record files)\n"
        "Thought: Search all file contents for the phrase, with "
        "ignore_case true so different capitalization still matches.\n"
        "Action:\n"
        "tool_name: grep\n"
        'tool_input: {"pattern": "bird activity", "ignore_case": true}\n'
        "(the observation shows one match, in notams.txt)\n"
        "Thought: Read the matching file for the full context.\n"
        "Action:\n"
        "tool_name: read_file\n"
        'tool_input: {"path": "notams.txt"}\n'
        "(the observation returns the file's lines)\n"
        "Thought: Answer in plain prose, citing the file.\n"
        "Action:\n"
        "tool_name: final_answer\n"
        "tool_input: Yes - NOTAM A0415 reports bird activity north of "
        "the field at dusk, per notams.txt."
    )
)

records_clerk = SimpleAgent(
    llm=llm,
    planner=clerk_planner,
    tool_executor=ToolExecutor(ops_registry),
    memory=WorkingMemory(),
    # The fuel-pit tasking below needs about six clean steps (list, grep,
    # read two or three files, answer); the budget leaves room for the
    # occasional recovered fumble on top of that.
    max_steps=12,
)


async def ask_records_question() -> None:
    answer = await records_clerk.arun(
        "Are any runways closed right now, and if so, why?"
    )
    print("\nRECORDS CLERK:", answer)


asyncio.run(ask_records_question())

# %% [markdown]
# ### Step 7: the loop under the answer
#
# The clerk's answer is the surface; the loop underneath is the lesson.
# `show_agent_mind` again, like tutorial 03 - watch which file the clerk
# chose, what it observed there, and how the citation in its answer
# traces straight back to the file contents. Nothing in the answer came
# from the model's imagination; every claim has an observation behind
# it.

# %%
show_agent_mind(records_clerk)

# %% [markdown]
# One more tasking, written to pull on **all three tools in the group**.
# A new duty officer wants an orientation plus a specific issue chased
# down, so the tasking walks the professional workflow: inventory the
# records (`list_dir`), find where the issue is mentioned (`grep`),
# read the details (`read_file`). The second half of the question is
# what makes searching alone insufficient: the fuel-dispensing lines
# never contain the phrase "fuel pit 2", so the clerk cannot brief the
# full picture without opening the log.
#
# One honest expectation to set: **the planner is stochastic, so the
# route varies run to run.** Sometimes the clerk greps and then reads;
# sometimes it reads directly and skips the grep; between this tasking
# and the previous one you will see all three tools earn their keep.
# What never varies is the contract around every route: typed calls,
# validated inputs, rooted paths. Rerun the cell and watch it choose
# differently.
#
# You may also see the small model briefly fumble on a run - it names a
# tool the registry does not have, or guesses a filename before listing.
# Those are not raised: the executor hands the mistake back to the model
# as a typed observation and the agent recovers on its next turn. That
# self-correction is the resilience tutorial 07 is about, happening early.
# We quieted the framework's recovered-error logs in the setup cell so
# they do not clutter the output, but you can still spot any
# fumble-and-recovery in the mind dumps below. The clerk's role tells it
# to take one action per turn and to list before it reads, which keeps
# these fumbles rare to begin with.
#
# There is a trap hidden in it, on purpose: the records write "Fuel pit
# 2" with a capital F, and the duty officer asks about "fuel pit 2". A
# case-sensitive search would come back empty and the clerk would
# wrongly report no records exist - which is why its role orders every
# grep to run with `ignore_case` true. Search doctrine is part of the
# prompt, not something you hope the model figures out.

# %%
async def ask_fuel_pit_question() -> None:
    answer = await records_clerk.arun(
        "I just took over the desk. First list the record files we keep. "
        "Then search their contents to find every file that mentions "
        "fuel pit 2, and read those files in full so you can brief me: "
        "what is the fuel pit 2 situation, and how much fuel was "
        "dispensed earlier in the day before the pit went down?"
    )
    print("\nRECORDS CLERK:", answer)


asyncio.run(ask_fuel_pit_question())

# The clerk's answer printed above. Below is its full mind for this
# tasking - the AGENT MIND banner marks where the answer ends and the
# memory dump begins, so the two never run together. The history now
# holds both taskings; scroll past the first one and check which route
# the clerk took this time, and that every claim in its briefing traces
# to an observation.
print("\n(the agent's full working memory follows)\n")
show_agent_mind(records_clerk)

# %% [markdown]
# ### Step 8: when a tool call fails - typed rejection
#
# What if the model calls your tool with garbage - a word where a
# number belongs, a heading of 999? The executor validates every call
# against your `input_schema` **before the tool runs**, and a mismatch
# is a typed `ToolInputValidationError`, not a stack trace from inside
# your math.
#
# The error carries a `schema_hint`: a rendered description of the
# expected fields. Inside the agent loop this hint is echoed back to the
# model as the observation, so the model sees exactly what shape was
# expected and self-corrects on its next step. You saw the contract;
# here is the contract enforcing itself.

# %%
async def show_typed_rejection() -> None:
    executor = ToolExecutor(registry)
    bad_call = {"runway_heading_deg": 350, "wind_dir_deg": 20, "wind_speed_kt": "gusty"}
    print(f"Calling runway_wind_components with {bad_call}")
    try:
        await executor.aexecute("runway_wind_components", bad_call)
    except ToolInputValidationError as exc:
        print(f"\nRejected before the tool ran -> {type(exc).__name__}")
        print(f"Tool: {exc.tool_name}")
        print("schema_hint (what the agent loop echoes back to the model):")
        print(exc.schema_hint)


asyncio.run(show_typed_rejection())

# %% [markdown]
# ### Debrief: what this means for your capstone
#
# You are now a toolsmith, and the craft comes down to four habits:
#
# - **One schema buys three guarantees.** The Pydantic `input_schema`
#   you declare generates the model's documentation, validates every
#   call before dispatch, and turns bad calls into typed,
#   self-correcting failures. You never write prompt text for a tool
#   and never parse a tool argument by hand.
# - **Declare the side effect.** `READ_ONLY` versus `MUTATING` versus
#   `EXTERNAL` is not paperwork; it is how the framework decides what is
#   safe to run in parallel and what must be a barrier.
# - **Grant rather than trust.** Rooted file tools and the
#   `register_group` / `get_group` pair let you hand each agent exactly
#   the capability set its job needs. Build the habit now: every tool
#   you register is attack surface.
# - **Use typed lookups everywhere.** `get` with a tool class,
#   `get_by_name`, and `ToolNotFoundError` give your application code
#   the same contract discipline the agent has.
#
# **Capstone connection.** Whatever your project touches, it reaches it
# through tools, and the patterns here carry over directly:
#
# - A project that audits or searches a body of files - a codebase, a
#   log store, a document set - gets there with the shipped file-tool
#   library rooted at the target and left read-only: an agent that can
#   look but not change.
# - A project that reads sensors and can also act on them models the
#   readers as `READ_ONLY` tools and the actions as `MUTATING` ones, so
#   the framework runs the safe reads in parallel and treats the actions
#   with the caution they deserve.
# - A project that points a tool at real equipment or a live service
#   leans on the input-schema contract as the line between a validated,
#   controlled action and an unvalidated string reaching something that
#   matters.
#
# The through-line for every one of them: each capability your agent has
# is a typed tool you granted on purpose, and each one is attack surface.
#
# **Next:** tutorial `05_memory_that_survives` covers what happens when
# the conversation outgrows the context window, and how an agent knows
# things that were never said in the conversation.
