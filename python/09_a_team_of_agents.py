# python/09_a_team_of_agents.py
"""Tutorial 09 - A Team of Agents.

Ninth rung of the fairlib tutorial series: multi-agent orchestration by
composition - specialist agents wrapped as typed worker tools, and a
manager that delegates to them over the shared event bus. Setup and how
to run: README.md.
"""

# %% [markdown]
# ## Tutorial 09 - A Team of Agents
#
# Your mission has two halves that want different skills. Finding facts
# in an archive rewards careful searching and verbatim reporting;
# computing statistics rewards discipline about arithmetic. You could
# hand one agent the file tools, the calculator, and a long prompt
# explaining when to use which - or you could build two small
# specialists that are each easy to trust, and a manager that splits
# the work between them.
#
# fairlib's answer to multi-agent orchestration is deliberately boring:
# **a worker agent is just a tool**. That one idea is this whole
# tutorial. The scenario is a multi-domain ops cell: a manager
# coordinates a records specialist (file search tools over an
# intel-memo archive) and a quantitative analyst (a calculator),
# splitting a mission neither specialist could finish alone and
# synthesizing their answers into one report.
#
# What you will learn:
#
# - When a team beats a generalist, and when it does not.
# - That specialists are just tutorial-03 agents with a persona and a
#   narrow toolset.
# - How `WorkerAgentTool` wraps any agent as a typed tool with a
#   subtask input contract.
# - How `build_worker_manager` wires a manager as a plain `SimpleAgent`
#   over worker tools in one call.
# - How to watch delegations live on the shared event bus.
#
# This rung leans directly on tutorial 08: delegation rides the same
# multi-action batch dispatch, so `READ_ONLY` workers fan out
# concurrently and everything is observable.
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
    PromptBuilder,
    AgentEventBus,
    AgentStepEvent,
    ToolCallPreEvent,
    ToolCallPostEvent,
    SideEffect,
    SafeCalculatorTool,
    ReadFileTool,
    ListDirTool,
    GrepTool,
    ToolRegistry,
    ToolExecutor,
    WorkingMemory,
    HuggingFaceAdapter,
    SimpleReActPlanner,
    SimpleAgent,
    WorkerAgentTool,
    build_worker_manager,
)

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# %% [markdown]
# ### A window into the agent's head
#
# The same helper from tutorial 03. In a team it earns a new use: after
# the mission runs, we will open the **manager's** memory and read the
# delegation transcript from its point of view.

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
# ### Step 1: why teams, and when not
#
# A specialist works from a short, focused prompt: one role, a handful
# of tools, no distractions. That narrowness is the reliability win:
#
# - A model choosing among 3 tools with a persona that matches them
#   makes fewer bad calls than the same model choosing among 20.
# - Each worker plans in its own clean context instead of one context
#   carrying every subproblem's clutter.
# - The manager only reasons about who should do what - it never sees
#   the file paths or the arithmetic.
#
# Be honest about the cost, though: every delegation is a full agent
# run, with its own model calls, its own steps, its own failure modes.
# If one agent with three tools can do the mission, build that - a
# single agent is simpler, cheaper, and easier to debug. Reach for a
# team when subproblems genuinely want different toolsets, different
# personas, or isolated contexts. **Your capstone is a mission, not a
# showcase.**
#
# One model load serves the whole cell - the manager and both workers
# all share the same underlying `llm` instance. Why is that safe, when
# they are three logically separate agents? Because of tutorial 01's very
# first lesson: the model is **stateless**. A `HuggingFaceAdapter` holds
# only the weights; it remembers nothing between calls, and every
# `invoke` is answered purely from the message list handed to it that
# turn. The per-agent state - who each agent is and what it has seen -
# does not live in the model; it lives in each agent's own planner and
# memory. So one set of weights (the expensive thing to put on the GPU)
# can back any number of agents at once without them bleeding into each
# other. You load the model once and reuse the instance everywhere; a
# second `HuggingFaceAdapter` would just be a duplicate copy of the same
# weights eating more VRAM for nothing.

# %%
print(f"Loading {MODEL_NAME} (first run downloads the weights)...")
llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=512)

# %% [markdown]
# ### Step 2: build the specialists, which you already know how to do
#
# Each worker is literally the tutorial-03 pattern: a
# `SimpleReActPlanner`, its own small `ToolRegistry`, a `ToolExecutor`,
# and a `WorkingMemory`, wrapped in a `SimpleAgent`. Two additions make
# it a good team member:
#
# - **A persona:** a `PromptBuilder` whose `RoleDefinition` states the
#   worker's role and standards, paired with a `FormatInstruction` for
#   how its answers must read - tutorial 02's machinery, pointed at a
#   specialist.
# - **`stateless=True`:** a worker clears its memory before every run,
#   so each delegation is planned fresh instead of against the
#   accumulated history of earlier delegations.
#
# First, the records specialist needs something to search. We write a
# small intel-memo archive to a scratch corpus. The corpus is staged,
# but the agents that will search, read, and reason over it are real
# models.

# %%
OPS_DIR = os.path.join(TUTORIALS_DIR, "_scratch", "09_ops_cell")
os.makedirs(OPS_DIR, exist_ok=True)

_MEMOS = {
    "memo_early_week.txt": (
        "INTEL MEMO - checkpoint KILO, early week summary\n"
        "Monday night: 12 vehicles crossed the checkpoint between 2100 and 0500.\n"
        "Tuesday night: 9 vehicles crossed the checkpoint; routine agricultural "
        "traffic.\n"
    ),
    "memo_midweek.txt": (
        "INTEL MEMO - checkpoint KILO, midweek summary\n"
        "Wednesday night: 11 vehicles crossed the checkpoint; no anomalies.\n"
        "Thursday night: 48 vehicles crossed the checkpoint in convoy groups, "
        "headlights off, uniform spacing.\n"
    ),
    "memo_end_of_week.txt": (
        "INTEL MEMO - checkpoint KILO, end of week\n"
        "Friday night: 10 vehicles crossed the checkpoint; routine traffic "
        "resumed.\n"
    ),
}
for filename, text in _MEMOS.items():
    with open(os.path.join(OPS_DIR, filename), "w", encoding="utf-8") as memo:
        memo.write(text)
print(f"Wrote {len(_MEMOS)} memos to {OPS_DIR}")

# %%
def build_specialist(
    role_text: str, format_text: str, tools: list, example: Example
) -> SimpleAgent:
    """The tutorial-03 wiring around the shared model, plus a persona,
    an answer-format rule, and one worked turn keeping the tool-call
    syntax exact."""
    registry = ToolRegistry()
    for tool in tools:
        registry.register_tool(tool)
    prompts = PromptBuilder()
    prompts.role_definition = RoleDefinition(role_text)
    prompts.format_instructions.append(FormatInstruction(format_text))
    prompts.examples.append(example)
    planner = SimpleReActPlanner(llm, registry, prompts)
    # stateless=True: the worker plans every delegation from a clean memory.
    return SimpleAgent(
        llm, planner, ToolExecutor(registry), WorkingMemory(), stateless=True
    )


records = build_specialist(
    "You are the records specialist of an operations cell. The intel memo "
    "archive is your file root. Use grep to locate relevant lines across the "
    "archive and read_file to open a memo. Keep grep patterns to one simple "
    "keyword. If a search returns no matches, never conclude the information "
    "is absent: the archive is small, so list the memos and read them "
    "directly before answering. Report facts exactly as recorded - never "
    "estimate or invent numbers.",
    # The one-line rule is load-bearing, not style: the action parser
    # reads a single line, so a multi-line answer would get truncated.
    "Give your final answer as ONE single line; when reporting several "
    "facts, separate them with semicolons on that line.",
    [ReadFileTool(OPS_DIR), ListDirTool(OPS_DIR), GrepTool(OPS_DIR)],
    Example(
        "User: Which memos mention convoy movement?\n"
        "Thought: Search the archive; a multi-field tool takes a JSON "
        "object with the schema's fields.\n"
        "Action:\n"
        "tool_name: grep\n"
        'tool_input: {"pattern": "convoy"}\n'
        "(after the observation arrives)\n"
        "Thought: The matches name the memos; answer in plain text.\n"
        "Action:\n"
        "tool_name: final_answer\n"
        "tool_input: Convoy movement appears in memo_midweek.txt."
    ),
)

analyst = build_specialist(
    "You are a careful quantitative analyst. Use the calculator for every "
    "computation - never do arithmetic in your head. If a request does not "
    "include the actual numbers to compute on, do not invent any: answer "
    "that you need the numbers.",
    "State the numeric results plainly in your final answer.",
    [SafeCalculatorTool()],
    Example(
        "User: What is 12 * 31?\n"
        "Thought: Arithmetic goes to the calculator. The input is the "
        "bare expression only.\n"
        "Action:\n"
        "tool_name: safe_calculator\n"
        "tool_input: 12 * 31"
    ),
)

# %% [markdown]
# ### Step 3: a worker is a tool
#
# Here is the composition trick. `WorkerAgentTool` takes a **full agent**
# - one of the complete `SimpleAgent` specialists you built in step 2,
# ReAct loop and all - and adapts it into an ordinary typed tool. Note
# what its first constructor argument is not: not a model, not a prompt,
# not a function. It is an entire working agent, wrapped so a manager can
# call it like any other tool:
#
# - Its input schema is `WorkerSubtaskInput`, whose one required field,
#   `subtask`, carries the complete, self-contained task for the
#   worker.
# - Its output is the worker's final answer.
# - Its description is what the manager's model reads in the rendered
#   tool catalog to decide who gets which job.
#
# That means the manager plans delegations exactly like tool calls -
# same planner machinery, same typed validation, same events, same
# batch scheduling. There is nothing new to learn, and that is the
# point: **fairlib has one dispatch path, and delegation rides it**.
#
# The `side_effect` declaration from tutorial 08 applies unchanged. The
# conservative default for a worker is `EXTERNAL`, since it reaches
# outward through its own model calls, which makes every delegation a
# sequential barrier. Declaring a worker `READ_ONLY` is your assertion
# that it mutates nothing - both of ours only read files or compute -
# and it is what lets independent delegations in one turn fan out
# concurrently. Two guardrails come free:
#
# - Delegations to the same worker instance serialize on a per-worker
#   lock, since an agent's memory is not reentrant.
# - A delegation chain that reaches the same worker tool again (a
#   manager wired as its own worker) fails with a typed
#   `ToolInvocationError` instead of recursing forever.

# %%
# The first argument to WorkerAgentTool is a FULL AGENT: `records` and
# `analyst` below are the complete SimpleAgent specialists from step 2,
# each with its own planner, tools, memory, and ReAct loop. We are not
# passing a model or a function here - we are wrapping whole agents.
worker_tools = [
    WorkerAgentTool(
        records,  # <- a full SimpleAgent from step 2, not a model
        name="records",
        description=(
            "Delegate an archive-lookup subtask, phrased as a complete "
            "question, to the records specialist, who can search and read "
            "the cell's intel memos."
        ),
        # READ_ONLY is an assertion: this worker only reads files, so
        # independent delegations to it may fan out concurrently.
        side_effect=SideEffect.READ_ONLY,
    ),
    WorkerAgentTool(
        analyst,
        name="analyst",
        description=(
            "Delegate a math subtask, phrased as a complete question with "
            "all needed numbers included, to the quantitative analyst, who "
            "computes with a safe calculator."
        ),
        side_effect=SideEffect.READ_ONLY,
    ),
]

for tool in worker_tools:
    fields = ", ".join(tool.input_schema.model_fields)
    print(f"{tool.name}: side_effect={tool.side_effect.value}, input fields: {fields}")

# %% [markdown]
# ### Step 4: assemble the cell
#
# `build_worker_manager` is a fully implemented fairlib function - you do
# not write it, you call it. You hand it the parts (the shared model, the
# list of worker tools, and optionally a `prompt_builder` and an event
# bus) and it hands back a ready-to-run manager. Under the hood it is
# pure wiring, and by now you can name every part it assembles for you: a
# `MultiActionReActPlanner` from tutorial 08 over a fresh registry
# holding the worker tools, a `ToolExecutor`, a `WorkingMemory`, and one
# shared event bus, returned as a plain `SimpleAgent`. The manager is not
# a special orchestrator class; it is
# the same agent loop you have run since tutorial 03, whose tools
# happen to contain agents. It is itself a `BaseAgent`, so a bigger
# mission could wrap it in a `WorkerAgentTool` - composition all the
# way up.
#
# Because delegations are tool calls, tutorial 06's events are our
# window: `ToolCallPreEvent` fires as each delegation starts and
# `ToolCallPostEvent` as it returns, on the shared bus, with
# `AgentStepEvent` marking the manager's own turns.

# %%
bus = AgentEventBus()


def subtask_of(tool_input: object) -> str:
    """Render the delegation's subtask however the planner phrased it."""
    if isinstance(tool_input, dict) and "subtask" in tool_input:
        return str(tool_input["subtask"])
    return str(tool_input)


def on_step(event: AgentStepEvent) -> None:
    print(f"\n[manager] step {event.step + 1} of {event.max_steps}")


def on_delegate(event: ToolCallPreEvent) -> None:
    print(f"  [delegate] {event.tool_name}: {subtask_of(event.tool_input)[:140]}")


def on_return(event: ToolCallPostEvent) -> None:
    print(
        f"  [returned] {event.tool_name} (ok={event.succeeded}): "
        f"{event.observation[:140]}"
    )


bus.subscribe(AgentStepEvent, on_step)
bus.subscribe(ToolCallPreEvent, on_delegate)
bus.subscribe(ToolCallPostEvent, on_return)

manager_prompts = PromptBuilder()
manager_prompts.role_definition = RoleDefinition(
    "You are the duty officer of a multi-domain operations cell. Break the "
    "mission into subtasks and delegate each one to the right specialist as "
    "a complete, self-contained question. Never delegate a computation "
    "before you hold its inputs: when a subtask needs an earlier subtask's "
    "result, wait for that observation and write the actual values into the "
    "next delegation. Batch several delegations in one turn only when they "
    "are fully independent."
)
manager_prompts.format_instructions.append(
    FormatInstruction(
        "Synthesize the specialists' answers into one final report."
    )
)

# The worked-example discipline from tutorials 03 and 04, applied to
# the manager's JSON turn format - and to DEPENDENCY ORDER: the analyst
# is delegated only after the records observation supplies real
# numbers, and final_answer always stands alone in its own turn.
manager_prompts.examples.append(
    Example(
        'Turn 1: {"thought": "The counts must come from records before '
        'any math; delegate that alone and wait.", "actions": '
        '[{"tool_name": "records", "tool_input": "List the nightly '
        'supply-run counts recorded in the memos."}]}\n'
        "Turn 2, after the observation reports counts 12, 9, and 11: "
        '{"thought": "Now the analyst gets the real numbers.", '
        '"actions": [{"tool_name": "analyst", "tool_input": "Compute '
        'the average of 12, 9, and 11."}]}\n'
        "Turn 3, after the observation reports 10.67: "
        '{"thought": "I have everything; report.", "actions": '
        '[{"tool_name": "final_answer", "tool_input": "Nightly supply '
        'runs averaged 10.7."}]}'
    )
)

# build_worker_manager is the fairlib library function - we only pass
# parameters: the shared model, the worker tools, our prompt builder, and
# the event bus. It returns a ready SimpleAgent; we implement none of it.
manager = build_worker_manager(
    llm, worker_tools, prompt_builder=manager_prompts, events=bus
)

# %% [markdown]
# ### Step 5: the mission, live
#
# One mission forces both specialists: the nightly counts live in the
# memo archive, which is the records specialist's job, and the average
# and the flagging threshold are arithmetic, which is the analyst's.
# Note this mission is staged - the analyst cannot compute until
# records reports the counts - so expect the manager to delegate in
# sequence across its turns. That is the dispatcher doing its job, not
# a limitation: when a manager sees genuinely independent subtasks, it
# can delegate several in one turn and the `READ_ONLY` declarations let
# them run concurrently, exactly as tutorial 08's telemetry checks did.
#
# As ever with a live model, runs vary: a manager may phrase subtasks
# oddly, retry a delegation, or take an extra turn. Watch the
# delegation prints to see how it actually decomposed the mission.

# %%
async def run_mission() -> None:
    mission = (
        "From the intel memos, find how many vehicles crossed the checkpoint "
        "on each night, Monday through Friday, exactly as the memos label "
        "the nights. Then have the "
        "analyst compute the nightly average of those counts. Finally, in "
        "your own report, state each night's count, the average, and call "
        "out any night whose count is more than twice the average."
    )
    print(f"Mission: {mission}")
    answer = await manager.arun(mission)
    print("\n=== DUTY OFFICER'S REPORT ===")
    print(answer)

asyncio.run(run_mission())

# %% [markdown]
# ### Step 6: read the manager's mind
#
# The event stream showed the delegations as they happened; the
# manager's memory shows the mission from its own seat. Look at what is
# there - subtasks out, one-line specialist answers back in as
# observations - and at what is not: no file paths, no grep patterns,
# no raw arithmetic. The specialists kept their clutter to themselves.

# %%
show_agent_mind(manager)

# %% [markdown]
# ### Debrief: what this means for your capstone
#
# One mental model holds all the way up:
#
# - **Delegation is tool calling.** A worker is a typed tool with a
#   subtask contract, and the manager is a plain `SimpleAgent`, so
#   every mechanism you learned for tools - schemas, events,
#   scheduling, typed failures - applies to teams unchanged. There is
#   no second dispatch path to learn.
# - **`READ_ONLY` workers fan out.** Independent delegations in one
#   turn run concurrently, by the same side-effect scheduling as any
#   other batch.
# - **The sharp edges are handled.** Same-worker delegations serialize
#   on a per-worker lock, a delegation cycle raises a typed
#   `ToolInvocationError`, and a failed worker becomes a failure
#   observation the manager can adapt to, never a crashed mission.
#
# **Capstone connection.** For the multi-domain AI proof of concept,
# this is the natural home for fairlib's multi-agent orchestration: one
# manager per domain-spanning mission, one specialist per domain. For
# Neural Shields, a detection specialist with telemetry tools plus a
# response specialist with containment tools mirrors this tutorial's
# split exactly. More broadly, any capstone where subproblems want
# different toolsets is a worker boundary waiting to be drawn, and the
# committee-of-agents demos in the demos directory show the same
# pattern grading essays and code with a panel of judge workers.
#
# **Next:** tutorial `10_trust_nothing` - your agents now read files
# and act on the world, which makes them targets. It covers prompt
# injection, security managers, and screening every action before it
# runs.
