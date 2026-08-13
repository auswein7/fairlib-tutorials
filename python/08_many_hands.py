# python/08_many_hands.py
"""Tutorial 08 - Many Hands.

Eighth rung of the fairlib tutorial series: multi-action turns with
side-effect-aware batch dispatch, where read-only checks fan out in
parallel and mutating actions run alone as barriers. Setup and how to
run: README.md.
"""

# %% [markdown]
# ## Tutorial 08 - Many Hands
#
# Your agent from tutorials 03 and 04 is a careful, one-step-at-a-time
# worker. Ask it to triage a suspicious host and it checks the process
# list, waits, checks the ports, waits, checks the auth log, waits -
# three round trips through the model for three lookups that never
# depended on each other. And when it finally acts, quarantining the
# host, you want the opposite guarantee: that the dangerous action runs
# alone, after every observation is in, never racing a concurrent read
# or, worse, another mutation.
#
# fairlib resolves both needs with one mechanism, and this tutorial
# walks it end to end. The scenario is a threat triage sweep: a defense
# agent inspects a suspicious host with three independent read-only
# telemetry checks, weighs the evidence, and, if the host looks
# compromised, quarantines it with a mutating action that must never
# overlap anything else.
#
# What you will learn:
#
# - The `MultiActionReActPlanner`, which keeps the same ReAct contract
#   but lets the model emit several actions in one turn.
# - The `SideEffect` attribute every tool declares, and why it is a
#   **scheduling contract**, not documentation.
# - How the executor schedules a batch - parallel where safe, serial
#   where it matters - with results always in call order.
# - The `ToolBatchScheduledEvent` that lets you watch the scheduling
#   decision live.
# - `max_actions_per_turn`, the safety ceiling on a single turn.
#
# This rung assumes the ladder below it: the ReAct loop in tutorial 03,
# custom tools with typed schemas in tutorial 04, the event bus in
# tutorial 06, and the typed failure taxonomy in tutorial 07.
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
    ToolInvocationError,
    ToolRegistry,
    ToolExecutor,
    WorkingMemory,
    AgentEventBus,
    ToolBatchScheduledEvent,
    ToolCallPostEvent,
    HuggingFaceAdapter,
    MultiActionReActPlanner,
    SimpleAgent,
)

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# %% [markdown]
# ### A window into the agent's head
#
# The same helper from tutorial 03. New in this tutorial: a single
# turn's memory can now carry several observations, and the helper is
# how we will inspect that.

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
# ### Step 1: from one hand to many
#
# Tutorial 03's `SimpleReActPlanner` produces one Thought and one
# Action per turn. `MultiActionReActPlanner` keeps the same contract -
# think, act, observe - but changes the response format: the model
# emits a single JSON object with a `thought` string and an `actions`
# array, each element naming a `tool_name` and its `tool_input`. One
# element is an ordinary single-call turn. Several elements are the
# model's own judgment that the calls are independent and can run at
# the same time; if one call's input depends on another's result, the
# format instructions tell it to issue just that one and wait for the
# Observation. Either way the agent loop receives a `ToolCallBatch`,
# where a single-action turn is simply a batch of one, and dispatches
# it down one path. To finish, the model emits a single `final_answer`
# action, exactly as before.
#
# The planner drops into the same seat the single-action planner
# occupied - same constructor shape, same agent wiring. Load the model
# once; everything in this tutorial shares it.

# %%
print(f"Loading {MODEL_NAME} (first run downloads the weights)...")
llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=512)

# %% [markdown]
# ### Step 2: the toolset declares its side effects
#
# Every `AbstractTool` declares one `side_effect` class attribute, the
# most restrictive that applies:
#
# - `SideEffect.READ_ONLY` observes and changes nothing; it is the only
#   class safe to run in parallel.
# - `SideEffect.MUTATING` changes local state, such as files,
#   databases, or a host's network access.
# - `SideEffect.EXTERNAL` reaches outward to a remote or
#   non-deterministic service.
#
# This is a **scheduling contract, not documentation**. The executor
# reads the declarations to build each turn's schedule: contiguous
# READ_ONLY calls form one parallel group, bounded by
# `max_parallel_tools` (default 8 from central config), while every
# MUTATING or EXTERNAL call runs alone, in order, as a sequential
# barrier - everything before it has finished before it starts, and
# everything after it waits. Results always come back in the original
# call order regardless of how they were scheduled. A call the
# executor cannot classify - an unregistered name or a missing
# declaration - is scheduled as MUTATING, the conservative choice, so
# nothing unknown ever runs concurrently.
#
# Our triage toolset is three READ_ONLY telemetry checks and one
# MUTATING containment action. The telemetry tools answer from tiny
# canned lookup tables, a stand-in for a real telemetry backend, so the
# tutorial is self-contained. The tools may be deterministic
# simulations, but the model deciding what to call and what to conclude
# is always real. A short sleep in each check simulates query latency,
# which is what makes the parallel schedule visible as real overlap.

# %%
# The scratch directory this run writes into, and the file the mutating
# containment tool will append to.
TRIAGE_DIR = os.path.join(TUTORIALS_DIR, "_scratch", "08_triage")
QUARANTINE_LOG = os.path.join(TRIAGE_DIR, "quarantine.log")
os.makedirs(TRIAGE_DIR, exist_ok=True)

# Canned telemetry standing in for a real backend. Each table maps a host
# to what a telemetry query would return. bravo-7 is staged as an obvious
# compromise; alpha-2 is a healthy control you can ask about on a re-run.
_PROCESS_TABLE = {
    "bravo-7": (
        "47 processes. NOTABLE: /tmp/.cache/kworkerd running as root, "
        "unsigned, not on the baseline manifest, spawning an outbound "
        "beacon child every 60 seconds."
    ),
    "alpha-2": "31 processes, all present on the baseline manifest.",
}
_PORT_TABLE = {
    "bravo-7": (
        "Listening: 22 (sshd), 443 (nginx), 4444 (UNKNOWN service, "
        "bound by /tmp/.cache/kworkerd)."
    ),
    "alpha-2": "Listening: 22 (sshd), 443 (nginx).",
}
_AUTH_TABLE = {
    "bravo-7": (
        "312 failed root logins from 203.0.113.66 over 90 minutes, then "
        "one SUCCESSFUL root login at 02:17 with no authorized change "
        "ticket on file."
    ),
    "alpha-2": "Routine logins by two known administrators during duty hours.",
}

# A tiny shared counter so the run can report whether the checks actually
# overlapped in time. Each check bumps "now" on entry and drops it on
# exit, and we remember the high-water mark in "max". If the reads truly
# ran in parallel, "max" ends the run greater than 1.
_in_flight = {"now": 0, "max": 0}

# %% [markdown]
# Now the read-only telemetry tool. Look closely at its `__init__`: it
# takes a `name`, a `description`, and a `table`. The *behavior* (look the
# host up in a table, after a short simulated delay) is written once, but
# the identity is supplied per instance. That is the trick step 3 uses to
# turn this single class into three differently-named tools - one class,
# many tools. And `side_effect = READ_ONLY` is the scheduling contract:
# these are the calls the executor is allowed to fan out in parallel.

# %%
class HostInput(BaseModel):
    """The host a telemetry check inspects."""

    host: str = Field(description="Host identifier, e.g. 'bravo-7'.")


class TelemetryLookupTool(AbstractTool):
    """A read-only telemetry query answered from a canned lookup table.

    One class, many tools: the name, description, and lookup table are
    supplied per instance through __init__, so each instance we build is
    a distinct registered tool with its own identity.
    """

    input_schema = HostInput
    output_schema = TextResult
    side_effect = SideEffect.READ_ONLY  # safe to fan out in parallel

    def __init__(self, name: str, description: str, table: dict) -> None:
        # These three lines are what make one class into many tools: the
        # framework identifies a tool by self.name, so every instance we
        # build with a different name is, to the agent, a different tool.
        self.name = name
        self.description = description
        self._table = table

    async def acall(self, tool_input: HostInput) -> ToolOutput:
        _in_flight["now"] += 1
        _in_flight["max"] = max(_in_flight["max"], _in_flight["now"])
        try:
            await asyncio.sleep(0.2)  # simulate query latency so overlap is visible
            host = tool_input.host.strip().lower()
            report = self._table.get(
                host, f"No telemetry on record for host '{host}'."
            )
            return TextResult(result=report)
        finally:
            _in_flight["now"] -= 1

# %% [markdown]
# And the mutating containment tool. Unlike the reads, this one changes
# the world - it appends to the quarantine log - so it declares
# `side_effect = MUTATING`, which tells the executor to run it alone, as a
# barrier, never overlapping anything. Its input schema requires a
# `reason`, because a containment action with no recorded justification is
# an audit finding waiting to happen.

# %%
class QuarantineInput(BaseModel):
    """The host to isolate, and the evidence justifying it."""

    host: str = Field(description="Host identifier to isolate, e.g. 'bravo-7'.")
    reason: str = Field(
        description=(
            "One line of evidence justifying the quarantine. Required: "
            "a containment action with no recorded justification is an "
            "audit finding."
        ),
    )


class QuarantineHostTool(AbstractTool):
    """A mutating containment action: isolate a host and log the decision."""

    name = "quarantine_host"
    description = (
        "Isolate a compromised host from the network and record the reason "
        "in the quarantine log. Use only when the evidence justifies it."
    )
    input_schema = QuarantineInput
    output_schema = TextResult
    side_effect = SideEffect.MUTATING  # must run alone, as a barrier

    async def acall(self, tool_input: QuarantineInput) -> ToolOutput:
        try:
            with open(QUARANTINE_LOG, "a", encoding="utf-8") as log:
                # One log line per action: a model-supplied reason with a
                # newline in it must not be able to forge extra entries.
                reason = " ".join(tool_input.reason.split())
                log.write(f"QUARANTINED {tool_input.host}: {reason}\n")
        except OSError as exc:
            raise ToolInvocationError(
                f"Could not write the quarantine log: {exc}",
                tool_name=self.name,
                raw_output=str(exc),
            ) from exc
        return TextResult(
            result=(
                f"Host {tool_input.host} is now isolated from the network. "
                f"Decision logged."
            )
        )

# %% [markdown]
# ### Step 3: watch the scheduler work
#
# The scheduling decision is not something you infer from timing - the
# executor announces it. Before any call in a batch runs, a bus-wired
# executor emits one `ToolBatchScheduledEvent` describing the groups it
# built: each `ScheduledGroup` carries the side-effect class that
# formed it, whether its members run in parallel, and the tool names in
# original call order. Per-call outcomes then ride the familiar
# `ToolCallPreEvent` and `ToolCallPostEvent` pairs from tutorial 06.
#
# Wire it all together: the registry, one shared bus, the executor, the
# multi-action planner, and a plain `SimpleAgent`. Nothing about the
# agent itself changes - the planner produces batches, the executor
# schedules them, and the agent loop stays the loop you already know.
#
# One thing to expect in the next cell, so it does not confuse you: we
# call `TelemetryLookupTool(...)` **three times**. That is not a mistake
# and it is not three copies of the code - it is the payoff of the "one
# class, many tools" design from step 2. We wrote the lookup behavior once
# and now stamp out three configured instances - `check_process_list`,
# `check_open_ports`, `check_auth_log` - each with its own name,
# description, and table. To the agent they are three distinct tools; to
# you they are one class you maintain in one place. Write a tool family
# once, register it as many named tools as your capstone needs.

# %%
registry = ToolRegistry()
# Three instances of ONE class - each is a separate named tool the agent
# can call, all sharing the single TelemetryLookupTool implementation.
registry.register_tool(
    TelemetryLookupTool(
        "check_process_list",
        "List the processes running on a host, flagged against its baseline manifest.",
        _PROCESS_TABLE,
    )
)
registry.register_tool(
    TelemetryLookupTool(
        "check_open_ports",
        "List the network ports a host is currently listening on.",
        _PORT_TABLE,
    )
)
registry.register_tool(
    TelemetryLookupTool(
        "check_auth_log",
        "Summarize recent authentication activity on a host.",
        _AUTH_TABLE,
    )
)
registry.register_tool(QuarantineHostTool())

bus = AgentEventBus()


def on_schedule(event: ToolBatchScheduledEvent) -> None:
    print(
        f"\n[scheduler] {event.batch_size} call(s) this turn, "
        f"max parallel {event.max_parallel_tools}:"
    )
    for i, group in enumerate(event.groups, start=1):
        mode = "PARALLEL" if group.parallel else "sequential"
        names = ", ".join(group.tool_names)
        print(f"  group {i}: {mode:10} [{group.side_effect.value}] {names}")


def on_done(event: ToolCallPostEvent) -> None:
    print(f"  [done] {event.tool_name} (ok={event.succeeded})")


bus.subscribe(ToolBatchScheduledEvent, on_schedule)
bus.subscribe(ToolCallPostEvent, on_done)

executor = ToolExecutor(registry, events=bus)
planner = MultiActionReActPlanner(llm, registry)
planner.prompt_builder.role_definition = RoleDefinition(
    "You are a host-defense triage agent. Inspect a host with the "
    "read-only telemetry tools, weigh the evidence, and quarantine the "
    "host only if the evidence shows a compromise. Run independent checks "
    "together in one turn, and issue final_answer alone once you decide."
)
# How final reports should read lives in a FormatInstruction (tutorial
# 04's split), so the Output Format section has real content beyond the
# mandatory JSON rules.
planner.prompt_builder.format_instructions.append(
    FormatInstruction(
        "The final_answer action's tool_input is a short plain-language "
        "triage report: what the evidence showed and what action, if any, "
        "was taken."
    )
)
# A worked multi-action turn keeps a small local model's JSON format exact
# (the lesson from tutorials 03/04, now for the batch format). It shows
# two things the model gets wrong without it: independent checks batched
# in one turn's actions array, and final_answer issued ALONE in its own
# later turn, never in the same actions array as tool calls. The example
# uses the healthy control host so it never leaks the bravo-7 conclusion.
planner.prompt_builder.examples.append(
    Example(
        'Turn 1: {"thought": "The three checks are independent, so I run '
        'them together in one batch.", "actions": ['
        '{"tool_name": "check_process_list", "tool_input": {"host": "alpha-2"}}, '
        '{"tool_name": "check_open_ports", "tool_input": {"host": "alpha-2"}}, '
        '{"tool_name": "check_auth_log", "tool_input": {"host": "alpha-2"}}]}\n'
        "Turn 2, after the three observations arrive: "
        '{"thought": "The evidence is clean, so no containment is needed; '
        'I report and stop.", "actions": [{"tool_name": "final_answer", '
        '"tool_input": "alpha-2 shows no signs of compromise; no action taken."}]}'
    )
)
agent = SimpleAgent(llm, planner, executor, WorkingMemory(), events=bus)

# Start each run with a clean quarantine log.
if os.path.exists(QUARANTINE_LOG):
    os.remove(QUARANTINE_LOG)

# %% [markdown]
# Now the live run. The canned telemetry makes bravo-7 obviously
# compromised, so a reasonable model gathers the evidence and then
# quarantines. Watch the schedule prints: the three telemetry checks
# should land in one parallel read_only group, and `quarantine_host`,
# if and when the model chooses it, runs in its own sequential mutating
# group, guaranteed to start only after every outstanding read has
# finished. **The model decides what to call; the framework decides how
# it is safe to run.**
#
# Outputs vary run to run. A smaller model may issue one call per turn,
# and then every turn is a batch of one and every group is sequential.
# That is not a failure, just a less decisive planner; re-run, or point
# `FAIR_LLM_DEMO_MODEL` at a stronger instruct model, to see the
# parallel batch.

# %%
async def run_triage() -> None:
    mission = (
        "Triage host bravo-7: check its process list, its open ports, and "
        "its auth log. The three checks are independent of each other. "
        "Then decide whether the host needs quarantine; if it does, "
        "quarantine it with a short reason."
    )
    print(f"Mission: {mission}")
    answer = await agent.arun(mission)
    print(f"\nFinal report: {answer}")
    print(f"\nPeak concurrent telemetry checks this run: {_in_flight['max']}")

asyncio.run(run_triage())

# %% [markdown]
# How does a many-handed turn land in memory? `show_agent_mind` shows
# it: each check's result arrives as its own observation, in the
# original call order, so the model reads the batch back exactly as if
# it had made the calls one by one.

# %%
show_agent_mind(agent)

# %% [markdown]
# The mutating action, unlike the reads, left a mark on the world. This
# is exactly why MUTATING actions get barrier treatment - they are the
# calls you cannot take back:

# %%
print("quarantine.log:")
if os.path.exists(QUARANTINE_LOG):
    with open(QUARANTINE_LOG, encoding="utf-8") as log:
        content = log.read()
    print(content if content.strip() else "  (empty)")
else:
    print("  (no log file - the model chose not to quarantine; re-run the mission)")

# %% [markdown]
# ### Step 4: failures stay in formation
#
# What if one telemetry backend had blown up mid-batch? Nothing
# explodes. A failed call never aborts its siblings and never raises
# across the batch boundary: the executor captures the typed failure -
# the same `ToolInvocationError` taxonomy tutorial 07 taught - into
# that call's result, in its original position, marked
# `succeeded=False`. The model then sees a failure observation for that
# one call alongside the successful observations from the rest of the
# batch, in call order, and adapts on its next turn: retry the check,
# route around it, or decide with the evidence it has. Parallel
# dispatch never turns one bad tool into a lost turn.

# %% [markdown]
# ### Step 5: the safety ceiling
#
# One model turn can now trigger many actions, so the framework caps
# it. `max_actions_per_turn` is a constructor knob on both
# `MultiActionReActPlanner` and `SimpleAgent`, defaulting to
# `tool_dispatch.max_actions_per_turn`, which is 16 in central config.
# The planner states the ceiling to the model, the parser enforces it,
# and the agent enforces it again at the dispatch boundary - an
# oversized turn is rejected and re-prompted, never silently
# dispatched.

# %% [markdown]
# ### Debrief: what this means for your capstone
#
# Three properties held throughout:
#
# - **Parallel where safe.** Independent READ_ONLY calls in one turn
#   ran concurrently, so three 0.2s telemetry queries cost one turn
#   about 0.2s of tool time rather than 0.6s, and one model round trip
#   rather than three.
# - **Serial where it matters.** The quarantine ran as a barrier, with
#   no read in flight when it started and nothing overlapping it - and
#   you did not schedule that; the tool's one-line `side_effect`
#   declaration did.
# - **Observable always.** `ToolBatchScheduledEvent` announced the plan
#   before it ran, and the pre and post pairs reported every outcome.
#
# The deep point is that the scheduling is a framework guarantee
# derived from tool self-declarations. Write your capstone's tools,
# declare each one's side effect honestly, and your tool set gets the
# same guarantee for free - no scheduler code in your project, ever.
#
# **Capstone connection.** For Neural Shields, this scenario is your
# seam: inspect telemetry with read-only tools, act to contain with
# mutating ones, and the dispatcher keeps containment from racing
# collection. For the agent-based SAST capstone, a scan turn fans out
# many read-only file inspections at once, and the standard file tools
# from tutorial 04 already declare READ_ONLY. For the next-generation
# cyber offense capstone, an EXTERNAL action such as firing at a live
# target must never launch twice in a race; declare it EXTERNAL and
# every such call is a barrier by construction.
#
# **Next:** tutorial `09_a_team_of_agents` turns the batch loop you
# just learned into a delegation engine - workers are just tools, and a
# manager agent fans a mission out to a team of specialists.
