# python/06_watch_it_think.py
"""Tutorial 06 - Watch It Think.

Sixth rung of the fairlib tutorial series: instrumenting an agent like
a flight-data recorder with typed events, structured run traces, and
live token streaming. Setup and how to run: README.md.
"""

# %% [markdown]
# ## Tutorial 06 - Watch It Think
#
# Your agent from tutorial 03 works. You ask it a question, the GPU
# spins for a while, and an answer comes back. But what happened in
# between? How many steps did it take? Which tools did it call, with
# what inputs, and did they succeed? If the run went wrong, what
# exactly went wrong, and when?
#
# This tutorial is a mission debrief exercise: we take that same
# calculator agent and wire it up like a flight-data recorder, so every
# run can be watched live and replayed afterward. The difference this
# makes is the difference between **a black box and a system you can
# debug, trust, and demo**. When your capstone agent does something
# surprising in front of an evaluator, "it printed the wrong answer" is
# a dead end; "here is the trace, step 2 called the tool with the wrong
# input" is an engineering conversation.
#
# What you will learn:
#
# - The `AgentEventBus` is the agent's observability seam: consumers
#   subscribe to typed events; they never poll framework internals.
# - `AgentStepEvent`, `ToolCallPreEvent`, and `ToolCallPostEvent` are
#   the loop made visible - one typed object per thing that happens.
# - `arun_with_trace` and `AgentRunTrace` give you a structured,
#   saveable record of one run, grouped by step.
# - Token streaming delivers `ModelStreamChunkEvent` deltas as the
#   model generates, plus `KVFinalAnswerStreamFilter` to isolate the
#   clean final-answer text for a chat surface.
#
# Nothing about the agent's behavior changes in this tutorial. We only
# change how much of it we can see.
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
    SafeCalculatorTool,
    ToolRegistry,
    ToolExecutor,
    WorkingMemory,
    AgentEventBus,
    AgentStepEvent,
    ToolCallPreEvent,
    ToolCallPostEvent,
    StreamSource,
    ModelStreamStartEvent,
    ModelStreamChunkEvent,
    ModelStreamEndEvent,
    KVFinalAnswerStreamFilter,
    HuggingFaceAdapter,
    SimpleReActPlanner,
    SimpleAgent,
)

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# %% [markdown]
# ### A window into the agent's head
#
# The same helper from tutorial 03. This tutorial's machinery is the
# event bus, not the memory - but keeping the helper nearby lets us
# compare the two views of a run when it matters.

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
# ### Step 1: assemble the agent, with one new part
#
# This is the exact wiring from tutorial 03, plus two additions. First,
# we construct an `AgentEventBus` ourselves and pass it to the agent as
# `events=bus`. Every `SimpleAgent` carries a bus even if you do not
# pass one; building it ourselves just keeps a handle to subscribe on.
# The agent binds the same bus into its planner, its model adapter, and
# its tool executor, so everything that happens in the loop emits to
# one place. Second, the adapter is constructed with `stream=True`.
# That is a capability flag: it lets this adapter stream tokens when
# asked to (step 5), and changes nothing until then. Loading a model is
# the expensive step, so we load once, up front, with every capability
# this tutorial needs.

# %%
print(f"Loading {MODEL_NAME} (first run downloads the weights)...")
llm = HuggingFaceAdapter(MODEL_NAME, stream=True, max_new_tokens=512)

tools = ToolRegistry()
tools.register_tool(SafeCalculatorTool())
executor = ToolExecutor(tools)

planner = SimpleReActPlanner(llm, tools)
planner.prompt_builder.role_definition = RoleDefinition(
    "You are a precise mission-support assistant. Use the "
    "safe_calculator tool for any arithmetic."
)
# Output style lives in a FormatInstruction (tutorial 04's split), so
# the Output Format section of the rendered prompt has real content.
planner.prompt_builder.format_instructions.append(
    FormatInstruction("Answer concisely.")
)

# One worked turn keeps a small local model's tool-call syntax exact
# (tutorial 03's lesson); reused for the streaming agent later.
CALC_EXAMPLE = Example(
    "User: What is 12 * 31?\n"
    "Thought: Arithmetic goes to the calculator. The input is the "
    "bare expression only.\n"
    "Action:\n"
    "tool_name: safe_calculator\n"
    "tool_input: 12 * 31"
)
planner.prompt_builder.examples.append(CALC_EXAMPLE)

bus = AgentEventBus()
agent = SimpleAgent(
    llm=llm,
    planner=planner,
    tool_executor=executor,
    memory=WorkingMemory(),
    max_steps=6,
    events=bus,
)
print("Agent assembled. The bus is our window into it.")

# %% [markdown]
# ### Step 2: live telemetry - subscribe to the loop
#
# A subscription is a plain callback. Calling `bus.subscribe(EventType,
# fn)` registers `fn` to be called with every event of exactly that
# type. The events are frozen dataclasses - typed and immutable
# payloads - so a consumer reads fields, never parses log strings.
#
# We subscribe three callbacks that each print one compact telemetry
# line, then fly a question that needs several steps. Each line begins
# with the name of the event that triggered it - `AgentStepEvent`,
# `ToolCallPreEvent`, or `ToolCallPostEvent` - so you can read the loop
# as a sequence of typed events: step markers, each tool call going out,
# each observation coming back.

# %%
# Each callback leads with type(event).__name__ - the name of the event
# that triggered it - so you can watch the mapping from event type to
# handler directly in the output.
def on_step(event: AgentStepEvent) -> None:
    print(
        f"  [{type(event).__name__}] step {event.step} of max {event.max_steps} "
        f"(history: {event.history_length} messages)"
    )


def on_tool_pre(event: ToolCallPreEvent) -> None:
    print(
        f"  [{type(event).__name__}] step {event.step} -> "
        f"tool={event.tool_name} input={event.tool_input!r}"
    )


def on_tool_post(event: ToolCallPostEvent) -> None:
    outcome = "ok" if event.succeeded else f"FAILED: {event.error}"
    print(
        f"  [{type(event).__name__}] step {event.step} <- "
        f"tool={event.tool_name} {outcome}, observation={event.observation[:60]!r}"
    )


telemetry_handles = [
    bus.subscribe(AgentStepEvent, on_step),
    bus.subscribe(ToolCallPreEvent, on_tool_pre),
    bus.subscribe(ToolCallPostEvent, on_tool_post),
]


async def first_mission() -> None:
    question = "Compute 34 * 61 and 87 * 23, then report which product is larger."
    print(f"You: {question}")
    answer = await agent.arun(question)
    print(f"Agent: {answer}")


asyncio.run(first_mission())

# %% [markdown]
# Two views of one run: the telemetry lines above were the loop *live*,
# and the working memory below is the *committed record* the loop left
# behind. Same mission, one view per audience - the events are for
# monitors, the memory is for the model.

# %%
show_agent_mind(agent)

# %% [markdown]
# ### Step 3: exact types, many subscribers, and handle hygiene
#
# Dispatch is by exact type: subscribing to `AgentStepEvent` gets you
# `AgentStepEvent` and nothing else; there is no subclass fan-in to
# reason about.
#
# Every subscribe call returns a `SubscriptionHandle`, and
# `bus.unsubscribe(handle)` removes exactly that callback. Keep your
# handles and unsubscribe when a consumer is done; a monitor should be
# a bolt-on, never a leak.
#
# The bus itself is just an object with an `emit` method. The framework
# calls it from inside the loop, but nothing stops us from emitting a
# hand-built event to see the mechanics. Note that both subscribers
# fire on the first emit: the bus is **multi-subscriber by design**, so
# your dashboard and your audit log never fight over the same signal.

# %%
def drill_watcher(event: ToolCallPreEvent) -> None:
    print(f"  [drill] saw a {type(event).__name__} for tool={event.tool_name}")


drill_handle = bus.subscribe(ToolCallPreEvent, drill_watcher)

probe = ToolCallPreEvent(step=0, tool_name="drill", tool_input="synthetic", call_index=0)
print("Emitting a hand-built event (two subscribers are registered):")
bus.emit(probe)

bus.unsubscribe(drill_handle)
print("After unsubscribing the drill watcher (only step 2's line remains):")
bus.emit(probe)

# %% [markdown]
# ### Step 4: the flight recorder - structured trace export
#
# Live telemetry is for watching. For accountability you want the whole
# run as one structured object you can store, diff, and replay.
# `arun_with_trace` behaves exactly like `arun`, but records every
# framework event for the duration of the run and returns an
# `AgentRunTrace`: the input, the output, the run status, the ordered
# event list, and the same events grouped by the step that caused them.
#
# The design principle behind it, straight from the framework's
# blueprint, is that **a failed run must be reproducible from its trace
# alone**. When your capstone misbehaves at 2 a.m. before demo day, the
# trace is what you attach to the bug report.
#
# Under the hood this is just a `TraceRecorder` subscribed to the same
# bus for the span of the run, a pattern you can also drive yourself to
# trace any custom span.

# %%
async def debrief_mission() -> None:
    question = "What is 18 + 27?"
    print(f"You: {question}")
    trace = await agent.arun_with_trace(
        question,
        trace_metadata={"mission": "tutorial-06-debrief"},
    )
    print(f"Agent: {trace.output}")


asyncio.run(debrief_mission())

# %% [markdown]
# The finished trace is also stored on `agent.last_trace`, so we can
# inspect it in a fresh cell. `to_dict()` gives plain JSON-ready data;
# `save(path)` writes it to disk.

# %%
trace = agent.last_trace
summary = trace.to_dict()

print("Run status:  ", summary["status"])
print("Run input:   ", summary["input"])
print("Run metadata:", summary["metadata"])
print("Event types: ", [event["event_type"] for event in summary["events"]])
print("Events grouped by causal step:")
for step_record in summary["steps"]:
    names = [event["event_type"] for event in step_record["events"]]
    print(f"  step {step_record['step']}: {names}")

# A trace always carries at least the step events of the run - assert
# structure, never exact model text.
assert len(summary["events"]) > 0

trace_path = os.path.join(TUTORIALS_DIR, "_scratch", "06_debrief", "trace.json")
saved = trace.save(trace_path)
print(f"\nTrace saved to {saved}")

# %% [markdown]
# ### Step 5: token streaming - watch the words form
#
# Everything so far observed the loop at the granularity of steps and
# tool calls. Streaming goes one level deeper: the model's own tokens,
# as typed events, while generation is still running. This is what a
# responsive chat UI, a live dashboard, or a speaking avatar consumes;
# nobody wants to stare at a frozen screen for twenty seconds.
#
# Three flags opt in, and we already set the first at load time:
#
# - the **adapter** was built with `stream=True` - the capability;
# - the **planner** gets `stream=True` so its model calls stream;
# - the **agent** gets `stream=True` so its own calls stream too.
#
# The model is not reloaded; the same `llm` instance serves both agents.
#
# We build this up one cell at a time, on purpose, so setup and execution
# stay separate: first the streaming agent, then two consumers of the
# same stream (a raw debugging feed and a filtered chat surface), and
# finally the mission run in its own cell. Each of the next cells explains
# its own part as you reach it, so you can run them one at a time and see
# what each adds before anything streams.

# %%
# Setup, part 1: detach the step 2 telemetry (handle hygiene, for real
# this time) and build a streaming agent. The model is not reloaded - the
# same llm instance serves both agents.
for handle in telemetry_handles:
    bus.unsubscribe(handle)

stream_planner = SimpleReActPlanner(llm, tools, stream=True)
stream_planner.prompt_builder.role_definition = RoleDefinition(
    "You are a precise mission-support assistant. Use the "
    "safe_calculator tool for any arithmetic."
)
stream_planner.prompt_builder.format_instructions.append(
    FormatInstruction("Answer concisely.")
)
stream_planner.prompt_builder.examples.append(CALC_EXAMPLE)
stream_agent = SimpleAgent(
    llm=llm,
    planner=stream_planner,
    tool_executor=executor,
    memory=WorkingMemory(),
    max_steps=6,
    events=bus,
    stream=True,
)
print("Streaming agent assembled. Next, wire the two consumers.")

# %% [markdown]
# Setup, part 2: **consumer one, the raw feed.** It prints every
# `ModelStreamChunkEvent` delta as it arrives - the whole completion,
# Thought and Action scaffolding included - the way a debugging surface
# watches generation. Run this cell to register it; nothing streams yet.

# %%
def raw_on_start(event: ModelStreamStartEvent) -> None:
    print(f"\n--- stream {event.stream_id} ({event.source.value}, step={event.step}) ---")
    print("raw feed: ", end="", flush=True)


def raw_on_chunk(event: ModelStreamChunkEvent) -> None:
    print(event.text, end="", flush=True)


def raw_on_end(event: ModelStreamEndEvent) -> None:
    print(
        f"\n--- stream {event.stream_id} ended: {event.finish_reason.value}, "
        f"{event.chunk_count} chunks, {event.total_chars} chars ---"
    )


bus.subscribe(ModelStreamStartEvent, raw_on_start)
bus.subscribe(ModelStreamChunkEvent, raw_on_chunk)
bus.subscribe(ModelStreamEndEvent, raw_on_end)
print("Consumer 1 (raw feed) subscribed.")

# %% [markdown]
# Setup, part 3: **consumer two, the chat surface.** It routes the same
# deltas through a `KVFinalAnswerStreamFilter` to keep only the clean
# final-answer text. One filter instance serves exactly one stream, so we
# key filters by `stream_id`. Route by `event.source`: PLANNER streams
# carry the planner wire format and get filtered; a VALIDATOR_REWRITE
# stream (you will meet validators in tutorial 07) is already plain answer
# text and passes through unfiltered. Chunk callbacks run on the hot path,
# so they only print or append and return.

# %%
filters: dict[int, KVFinalAnswerStreamFilter] = {}
answer_parts: list[str] = []


def chat_on_start(event: ModelStreamStartEvent) -> None:
    if event.source is StreamSource.PLANNER:
        filters[event.stream_id] = KVFinalAnswerStreamFilter()


def chat_on_chunk(event: ModelStreamChunkEvent) -> None:
    if event.source is StreamSource.PLANNER:
        text = filters[event.stream_id].feed(event.text)
    else:
        text = event.text
    if text:
        answer_parts.append(text)


def chat_on_end(event: ModelStreamEndEvent) -> None:
    if event.source is StreamSource.PLANNER:
        tail = filters.pop(event.stream_id).finish()
        if tail:
            answer_parts.append(tail)


bus.subscribe(ModelStreamStartEvent, chat_on_start)
bus.subscribe(ModelStreamChunkEvent, chat_on_chunk)
bus.subscribe(ModelStreamEndEvent, chat_on_end)
print("Consumer 2 (chat surface) subscribed.")

# %% [markdown]
# With the streaming agent built and both consumers listening, run the
# mission in its own cell and watch the tokens form. The raw feed prints
# the whole completion live; the filtered feed collects only the final
# answer.

# %%
async def streamed_mission() -> None:
    question = "What is 6 times 7? Use the calculator, then answer in one sentence."
    print(f"You: {question}")
    answer = await stream_agent.arun(question)
    print(f"\nFiltered final-answer feed: {''.join(answer_parts)!r}")
    print(f"Returned final answer:      {answer!r}")


asyncio.run(streamed_mission())

# %% [markdown]
# The raw torrent and the filtered feed came from the same model call:
# streaming is a consumption mode, not a second run, and the returned
# answer is identical to what a non-streaming run would have produced.
#
# One deliberate design note: chunk events are not recorded in
# structured traces. A trace already carries the committed messages
# once; duplicating every token would bloat it with thousands of
# records of the same, potentially sensitive, text. Streams appear in
# traces as start/end accounting only.

# %% [markdown]
# ### Debrief: what this means for your capstone
#
# Three layers of visibility, all built on one seam:
#
# - **Events are the integration seam.** UIs, dashboards, audit logs,
#   TTS avatars, SOC alert enrichment - they all subscribe; they do not
#   poll internals, diff memory lengths, or parse log text. The
#   framework emits typed objects; your capstone code reads fields.
# - **Traces are accountability.** `arun_with_trace` turns any run into
#   a saveable artifact, and a failed run must be reproducible from its
#   trace alone - build your capstone to keep that sentence true.
# - **Streaming is experience.** Token deltas as events, with a filter
#   to separate what the machinery said from what the user should see.
#
# **Capstone connection.** The security automation and detection
# capstone is a pipeline that must be fully observable to be trusted,
# and its seam is the event bus: every detection step, every tool call,
# every degraded response lands on subscriptions its dashboard and
# audit log consume. And the Avatar Language Immersion capstone needs
# exactly what you built in step 5: streaming tokens driving a speaking
# avatar, with the final-answer filter deciding what gets voiced.
#
# **Next:** tutorial `07_when_models_misbehave` - models ramble,
# repeat, time out, and providers fail, and every one of those becomes
# a typed, catchable, observable condition.
