# python/07_when_models_misbehave.py
"""Tutorial 07 - When Models Misbehave.

Seventh rung of the fairlib tutorial series: turning every model and
provider failure mode into a typed, catchable, observable condition
with validators, spin detection, degraded responses, circuit breakers,
and loop guards. Setup and how to run: README.md.
"""

# %% [markdown]
# ## Tutorial 07 - When Models Misbehave
#
# Your agent demos beautifully on your desk. Now picture demo day: the
# model decides to answer in three paragraphs when your parser needs
# one line; it rewrites the same wrong answer twice; the GPU box
# someone else is also using times out mid-call. None of that is
# hypothetical - it is Tuesday. **A capstone that only works when
# everything goes right does not work.**
#
# The framework's position is no silent fallbacks and no string
# matching. Every failure mode you will meet in this tutorial surfaces
# as a typed object you can catch by `isinstance` and a typed event you
# can subscribe to; the observability seam from tutorial 06 carries the
# bad news too. This tutorial is a field hardening exercise: we take an
# agent into degraded conditions on purpose and watch each defense
# engage.
#
# What you will learn:
#
# - `arun(validator=...)` enforces output contracts with coached
#   retries, using `Verdict.approve()` and `Verdict.reject(feedback)`,
#   and raises the typed `ValidatorRejectedError` when the budget runs
#   out.
# - `ResponseRepeatEvent` detects a model spinning in place.
# - `DegradedResponse` turns provider failure into a typed signal with
#   machine-readable recovery policy, including hard timeouts.
# - `CircuitBreakerRegistry` fails fast instead of hammering a dead
#   provider, then probes and recovers.
# - The loop guards emit `LoopGuardTrippedEvent` when the agent looks
#   stuck.
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
from pydantic import BaseModel

# fairlib imports run simplest to most complex - the order you meet them.
from fairlib import (
    Message,
    RoleDefinition,
    FormatInstruction,
    Example,
    Verdict,
    ValidatorRejectedError,
    ResponseRepeatEvent,
    DegradedResponse,
    LoopGuardTrippedEvent,
    MaxStepsExceeded,
    SideEffect,
    ToolOutput,
    TextResult,
    AbstractTool,
    ToolInvocationError,
    SafeCalculatorTool,
    ToolRegistry,
    ToolExecutor,
    WorkingMemory,
    HuggingFaceAdapter,
    SimpleReActPlanner,
    SimpleAgent,
)

# The circuit-breaker types are not exported at the fairlib top level in
# the current PyPI release, so they come from their home module.
from fairlib.modules.mal.circuit_breaker import BreakerState, CircuitBreakerRegistry

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# %% [markdown]
# ### A window into the agent's head
#
# The same helper from tutorial 03. In a tutorial about failure, the
# working memory is where you check what a defense actually left
# behind.

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
# ### Step 1: assemble the agent for hardening
#
# The tutorial 03 wiring again. Two constructor arguments are new -
# `repeat_signature_threshold` and `consecutive_error_threshold`, the
# loop guards. We set them here and explain them in step 6; they cost
# nothing until something goes wrong, which is the whole point of this
# tutorial.

# %%
print(f"Loading {MODEL_NAME} (first run downloads the weights)...")
llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=256)

tools = ToolRegistry()
tools.register_tool(SafeCalculatorTool())

planner = SimpleReActPlanner(llm, tools)
planner.prompt_builder.role_definition = RoleDefinition(
    "You are a mission-support assistant. Use the safe_calculator tool "
    "for any arithmetic."
)
# Output style lives in a FormatInstruction (tutorial 04's split), so
# the Output Format section of the rendered prompt has real content.
planner.prompt_builder.format_instructions.append(
    FormatInstruction("Keep your final answer short and direct.")
)

# One worked turn keeps a small local model's tool-call syntax exact
# (tutorial 03's lesson).
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
    tool_executor=ToolExecutor(tools),
    memory=WorkingMemory(),
    max_steps=6,
    repeat_signature_threshold=3,
    consecutive_error_threshold=3,
)
print("Agent assembled.")

# %% [markdown]
# ### Step 2: output contracts - the validator
#
# An agent can do the work correctly and still phrase the answer in a
# way your application cannot use. You do not want to re-run the whole
# reasoning loop over a wording problem, and you do not want retry
# loops scattered through your own code.
#
# `arun(validator=...)` is the framework primitive: after the ReAct
# cycle produces a final answer, your async validator judges the text
# and returns a `Verdict`. Approved answers are released. Rejected
# answers trigger a direct LLM rewrite with your feedback attached; the
# tool work and reasoning are kept, only the wrap-up changes.
#
# Our house rule, in radio-brevity style: every reply must end with the
# word OUT, meaning transmission complete. The role definition does not
# mention it, so the first draft will almost certainly fail; watch the
# rejection feedback coach the retry. We print every verdict from
# inside the validator so the retries are visible.

# %%
async def radio_discipline(response: str) -> Verdict:
    words = response.strip().rstrip(".!").upper().split()
    if words and words[-1] == "OUT":
        print(f"  [validator] APPROVED: {response.strip()[:60]!r}")
        return Verdict.approve()
    print(f"  [validator] REJECTED: {response.strip()[:60]!r}")
    return Verdict.reject(
        "End your entire reply with the single word OUT, radio brevity "
        "for transmission complete. Example: 'The result is 42. OUT'"
    )


async def contract_mission() -> None:
    question = "What is 13 * 17?"
    print(f"You: {question}")
    try:
        answer = await agent.arun(question, validator=radio_discipline, max_retries=2)
        print(f"Agent (released): {answer}")
    except ValidatorRejectedError as exc:
        # The typed failure carries everything a deliberate fallback
        # needs; the framework never substitutes a placeholder answer.
        print(
            f"No conforming answer after {exc.attempt_count} attempts; "
            f"last feedback: {exc.last_feedback[:60]!r}"
        )


asyncio.run(contract_mission())

# %% [markdown]
# Either way this cell ends in a decided state: a released answer that
# passed the contract, or a typed `ValidatorRejectedError` whose fields
# (`last_response`, `last_feedback`, `attempt_count`) let your fallback
# be informed instead of generic. A malformed reply never leaks
# through.
#
# `show_agent_mind` backs up the "only the wrap-up changes" claim: the
# calculator call and its observation are still in memory - the rewrite
# spent no tool work, only rephrasing.

# %%
show_agent_mind(agent)

# %% [markdown]
# ### Step 3: detecting a spinning model - `ResponseRepeatEvent`
#
# A subtler failure: the model burns its retry budget producing
# near-identical rejected answers. The framework compares consecutive
# retry responses and, when their similarity crosses the
# `similarity_threshold` you pass to `arun` (default 0.95), fires a
# `ResponseRepeatEvent` on the agent's bus. Detection only; the loop is
# not aborted - you decide what the signal means.
#
# To see it, we harden the conditions on purpose: a validator that
# rejects everything with feedback too vague to act on. Cornered like
# this, the model stops giving new answers and starts rehashing the same
# explanation over and over - exactly the spin the detector exists to
# catch.
#
# Two settings make the demonstration reliable on a small local model.
# First, we allow several retries (`max_retries=3`), because this model's
# very first rewrite jumps from a terse "42" to a verbose explanation -
# it is the *later* rewrites that rehash each other. Second, we pass
# `similarity_threshold=0.8` rather than the 0.95 default: cornered, this
# model repeats itself at roughly 80 percent similarity (it reshuffles a
# few words each time) rather than character-for-character, and 0.8
# reliably catches that rehash while still being a high bar. Both are
# knobs you would tune to your own model in a real deployment.

# %%
def notice_spin(event: ResponseRepeatEvent) -> None:
    print(
        f"  [spin detector] attempt {event.attempt} is {event.similarity:.0%} "
        f"similar to the previous answer (threshold {event.threshold:.0%})"
    )


spin_handle = agent.events.subscribe(ResponseRepeatEvent, notice_spin)


async def impossible_standard(response: str) -> Verdict:
    print(f"  [validator] REJECTED: {response.strip()[:60]!r}")
    return Verdict.reject("Not good enough. Try again.")


async def spin_mission() -> None:
    question = "What is 19 + 23?"
    print(f"You: {question}")
    try:
        await agent.arun(
            question,
            validator=impossible_standard,
            max_retries=3,
            similarity_threshold=0.8,
        )
    except ValidatorRejectedError as exc:
        print(
            f"Exhausted as expected after {exc.attempt_count} attempts - "
            "the typed error is the contract working, not breaking."
        )


asyncio.run(spin_mission())
agent.events.unsubscribe(spin_handle)

# %% [markdown]
# ### Step 4: typed degradation - `DegradedResponse`
#
# Now the failures your code did not cause: the provider itself. When
# an adapter call fails, fairlib raises `DegradedResponse`, a typed
# signal carrying what went wrong (its `Kind`) and what to do about it
# (machine-readable policy fields). Your recovery logic branches on
# typed values; it never parses error prose:
#
# - `should_compress` set: shrink the context, then retry.
# - `retryable` set: wait `retry_after` seconds, then resend unchanged.
# - Neither set: escalate; retrying cannot help.
#
# Adapters build the signal with `DegradedResponse.classify(exc)` for a
# raw provider error, or `DegradedResponse.for_kind(...)` when they
# already know the kind. No outage is needed to see the anatomy; we
# classify a couple of synthetic provider errors, the same way the real
# adapters do.

# %%
# These two classes are STAND-INS for real errors. When you call a hosted
# provider (OpenAI, Anthropic, an Ollama server), its SDK raises its own
# exception types on failure - a throttling error, an oversized-prompt
# error, and so on. We do not want to trigger a real outage just to see
# those, so we define two fake ones that carry the same tell-tale fields
# a real SDK error would (an HTTP status_code, and for throttling a
# retry_after hint). They are ordinary Exception subclasses; the only
# thing that matters is the data attached to them.
class RelayRateLimitError(Exception):
    """Stand-in for a provider SDK throttling error (HTTP 429: too many
    requests). Real 429s usually tell you how long to wait, so this one
    carries a retry_after."""

    status_code = 429
    retry_after = 2.0


class RelayContextError(Exception):
    """Stand-in for a provider rejecting an oversized prompt (HTTP 400):
    the conversation grew past the model's context window."""

    status_code = 400


# DegradedResponse.classify is what the real adapters call: hand it a raw
# provider exception and it reads those fields and returns a typed
# DegradedResponse whose policy fields (kind, retryable, should_compress,
# retry_after) tell your code what to do - without your code ever parsing
# an error message. Watch each raw error map to a different recovery
# policy: the 429 becomes retryable with a wait, the 400 becomes a
# compress-then-retry.
synthetic_failures = [
    RelayRateLimitError("slow down"),
    RelayContextError("maximum context length is 8192 tokens"),
]
for raw_error in synthetic_failures:
    signal = DegradedResponse.classify(raw_error, provider="field-relay")
    print(
        f"{type(raw_error).__name__:20s} -> kind={signal.kind.value:15s} "
        f"retryable={signal.retryable} should_compress={signal.should_compress} "
        f"retry_after={signal.retry_after}"
    )

timeout_signal = DegradedResponse.for_kind(
    DegradedResponse.Kind.TIMEOUT, "no reply within the deadline", provider="field-relay"
)
print(
    f"{'(adapter deadline)':20s} -> kind={timeout_signal.kind.value:15s} "
    f"retryable={timeout_signal.retryable} "
    f"should_compress={timeout_signal.should_compress}"
)

# %% [markdown]
# The timeout story. Every adapter accepts `timeout=` (seconds) at
# construction, as in `HuggingFaceAdapter(MODEL_NAME, timeout=30.0)`.
# When a call blows the deadline, the adapter stops waiting and raises
# `DegradedResponse` with `kind=TIMEOUT` and `retryable=True`, the third
# row above. A hung provider costs you the deadline, never the evening.
#
# We will not stall the GPU for real - that is a poor trade - so the next
# cell shows the calling shape two ways with the SAME recovery function.
# First it wraps a genuine, healthy call in `except DegradedResponse`: the
# call succeeds, so the guard costs nothing and the except body never
# runs. Then it hands that same recovery logic a `DegradedResponse` we
# raise on purpose, so you can watch the except branch actually fire and
# choose a recovery. That second block is the one that answers "does this
# except ever catch anything?" - yes, and here it is doing it.

# %%
def recover_from(exc: DegradedResponse) -> None:
    """Decide what to do about a degraded call, branching only on the
    typed policy fields - never on the error text."""
    if exc.should_compress:
        print(f"  recovery: degraded ({exc.kind.value}) -> compress the context, then retry")
    elif exc.retryable:
        wait = exc.retry_after or "a moment"
        print(f"  recovery: degraded ({exc.kind.value}) -> wait {wait} and retry")
    else:
        print(f"  recovery: degraded ({exc.kind.value}) -> not recoverable, escalate")


# 1. Healthy path: a real call that succeeds. The guard costs nothing and
# the except body does not run.
print("Healthy call (except body should NOT run):")
try:
    reply = llm.invoke([Message(role="user", content="Radio check. Answer in five words or fewer.")])
    print("  reply:", reply.content)
except DegradedResponse as exc:
    recover_from(exc)

# 2. Degraded path, made real: we raise the exact object a timed-out
# adapter would raise and let the same handler catch it, so the recovery
# branch genuinely executes this time.
print("\nSimulated timeout (except body SHOULD run):")
try:
    raise DegradedResponse.for_kind(
        DegradedResponse.Kind.TIMEOUT,
        "no reply within the deadline",
        provider="field-relay",
    )
except DegradedResponse as exc:
    recover_from(exc)

# %% [markdown]
# ### Step 5: the circuit breaker
#
# Retrying is fine when a provider hiccups. When it is down, retrying
# is how you turn one outage into a stalled agent and a hammered
# provider. The circuit breaker is what stands between your agent and a
# dead endpoint, and it moves through three states:
#
# - **CLOSED** - healthy: calls flow, and failures are counted in a
#   rolling window.
# - **OPEN** - too many failures: calls fail instantly with
#   `kind=circuit_open` until the cooldown elapses.
# - **HALF_OPEN** - cooldown over: exactly one probe call is let
#   through, and its outcome decides between CLOSED and back to OPEN.
#
# The `CircuitBreakerRegistry` keeps one breaker per provider name. No
# model is needed to watch the cycle; we drive it with a deliberately
# failing async callable, then a healthy one, and print every state
# transition.

# %%
def report_transition(
    provider: str, old_state: BreakerState, new_state: BreakerState, failure_count: int
) -> None:
    print(
        f"  [breaker] {provider}: {old_state.value} -> {new_state.value} "
        f"(failures in window: {failure_count})"
    )


breakers = CircuitBreakerRegistry(
    enabled=True,
    failure_threshold=2,
    window_seconds=60.0,
    cooldown_seconds=1.0,
    on_state_change=report_transition,
)


async def dead_provider() -> str:
    raise DegradedResponse.for_kind(
        DegradedResponse.Kind.SERVER_ERROR, "ground station down", provider="ground-station"
    )


async def healthy_provider() -> str:
    return "ack"


async def breaker_drill() -> None:
    for attempt in range(1, 5):
        try:
            await breakers.acall("ground-station", dead_provider)
        except DegradedResponse as exc:
            state = breakers.state_of("ground-station").value
            print(f"attempt {attempt}: kind={exc.kind.value} (state now {state})")

    print("cooling down...")
    await asyncio.sleep(1.1)

    result = await breakers.acall("ground-station", healthy_provider)
    print(f"probe call returned {result!r}")
    assert breakers.state_of("ground-station") is BreakerState.CLOSED
    print("state:", breakers.state_of("ground-station").value)


asyncio.run(breaker_drill())

# %% [markdown]
# Read the transcript back: two real failures trip the breaker OPEN,
# attempts 3 and 4 fail in microseconds with `kind=circuit_open`
# (nothing was actually called - that is the protection), the cooldown
# elapses, the HALF_OPEN probe succeeds, and the breaker CLOSES. Full
# cycle, no provider harmed.

# %% [markdown]
# ### Step 6: loop guards - the agent watching itself
#
# The last defense lives inside the agent loop. Two counters run on every
# `SimpleAgent`, set at construction (step 1 configured them on the main
# agent):
#
# - `repeat_signature_threshold` counts how many consecutive steps the
#   planner emitted the same (tool_name, tool_input) pair.
# - `consecutive_error_threshold` counts how many consecutive times tool
#   dispatch failed.
#
# Crossing either fires a `LoopGuardTrippedEvent`. Detection only, like
# every signal in this tutorial: the loop is not aborted, `max_steps`
# remains the hard backstop, but your subscriber hears the early warning
# before the ceiling hits.
#
# To watch a guard actually trip, we stage a stuck agent on purpose. Its
# one tool, `sensor_read`, always fails, and its role tells it to keep
# retrying the same call rather than give up or invent a value - so it
# loops on a failing action, exactly the situation the guards exist to
# catch. We set both thresholds to 2, so the alarm fires after only two
# repeats, and you should see both guards trip (the repeated identical
# call and the repeated failure), followed by `max_steps` as the hard
# backstop that finally ends the run. On a healthy agent - every other
# run in this series - these same guards stay silent, which is the sound
# of progress; you wire the alarm anyway, because the run that needs it
# will not announce itself in advance.

# %%
class StuckSensorInput(BaseModel):
    """The telemetry channel to read."""

    channel: str


class BrokenSensorTool(AbstractTool):
    """A tool that always fails - a stand-in for a dead backend, so we can
    watch the loop guards react to an agent stuck retrying it."""

    name = "sensor_read"
    description = "Read a telemetry sensor channel and return its value."
    input_schema = StuckSensorInput
    output_schema = TextResult
    side_effect = SideEffect.READ_ONLY

    async def acall(self, tool_input: StuckSensorInput) -> ToolOutput:
        raise ToolInvocationError("sensor bus offline", tool_name=self.name)


def notice_loop_guard(event: LoopGuardTrippedEvent) -> None:
    print(
        f"  [guard] {event.guard_type.value} tripped at step {event.step} "
        f"({event.count} >= {event.threshold}) - the agent may be stuck"
    )


stuck_tools = ToolRegistry()
stuck_tools.register_tool(BrokenSensorTool())
stuck_planner = SimpleReActPlanner(llm, stuck_tools)
stuck_planner.prompt_builder.role_definition = RoleDefinition(
    "You are a telemetry assistant. To read a value you MUST call the "
    "sensor_read tool. If the call fails, immediately try the exact same "
    "sensor_read call again - never invent a value and never answer from "
    "your own knowledge, because only the tool can give the reading."
)
stuck_planner.prompt_builder.format_instructions.append(
    FormatInstruction("Report the sensor value once you have it.")
)
stuck_planner.prompt_builder.examples.append(
    Example(
        "User: Read channel A.\n"
        "Thought: Only the sensor tool can read it.\n"
        "Action:\n"
        "tool_name: sensor_read\n"
        'tool_input: {"channel": "A"}'
    )
)

# Thresholds of 2 so the guard fires after two repeats, before max_steps.
stuck_agent = SimpleAgent(
    llm=llm,
    planner=stuck_planner,
    tool_executor=ToolExecutor(stuck_tools),
    memory=WorkingMemory(),
    max_steps=6,
    repeat_signature_threshold=2,
    consecutive_error_threshold=2,
)
stuck_agent.events.subscribe(LoopGuardTrippedEvent, notice_loop_guard)


async def stuck_mission() -> None:
    question = "Read telemetry channel THRUST-1 and report its value."
    print(f"You: {question}")
    try:
        answer = await stuck_agent.arun(question)
        # If the model gives up and answers anyway, the guards still
        # warned above; the run simply ended before max_steps.
        print(f"Agent: {answer}")
    except MaxStepsExceeded:
        print(
            "\nThe guards warned early (the [guard] lines above), and "
            "max_steps was the hard backstop that finally ended the stuck "
            "run - detection plus a ceiling, working together."
        )


asyncio.run(stuck_mission())

# %% [markdown]
# Open the stuck agent's mind to see what "stuck" looks like as data: the
# same `sensor_read` call and the same failure observation, over and over.
# That visible repetition is exactly the signature the loop guards counted.

# %%
show_agent_mind(stuck_agent)

# %% [markdown]
# ### Debrief: what this means for your capstone
#
# Each failure mode maps to a typed surface and a handler:
#
# - **Correct but unusable** (wrong format, policy violation): the
#   surface is `Verdict.reject(feedback)`, which drives a coached retry
#   and raises `ValidatorRejectedError` on exhaustion. Reject with
#   actionable feedback; catch the typed error for a deliberate
#   fallback.
# - **Near-identical rewrites** burning the retry budget: the surface
#   is `ResponseRepeatEvent`. Flag the session and stop spending budget
#   on a spinning model.
# - **Provider timeout, throttle, overflow, or error**: the surface is
#   `DegradedResponse` with a `kind` and policy fields. Branch on
#   `should_compress`, `retryable`, and `retry_after`; escalate the
#   rest.
# - **Provider down, every call failing**: the surface is
#   `CircuitBreakerRegistry`, which returns an instant
#   `kind=circuit_open` and exposes `BreakerState`. Fail fast, let the
#   cooldown probe recover, and route to a backup provider meanwhile.
# - **The agent repeating itself, or tools failing back to back**: the
#   surface is `LoopGuardTrippedEvent`. Alert or escalate before
#   `max_steps` ends the run.
#
# Every one of those is `isinstance`-catchable and event-observable.
# Nothing in this tutorial matched a string, guessed from a log line,
# or swallowed a failure silently, and nothing in your capstone should
# either.
#
# **Capstone connection.** The ICS exploitation and next-generation
# cyber offense capstones run agents over real networks against real
# targets, flaky by nature; their seam is exactly typed tools plus this
# resilience-and-timeout stack. And every project on the board will one
# day demo live in front of brass: the difference between "it crashed"
# and "it degraded, said so, and recovered" is this tutorial.
#
# **Next:** tutorial `08_many_hands` - one planner turn, many tool
# calls, batched dispatch and side-effect-aware scheduling.
