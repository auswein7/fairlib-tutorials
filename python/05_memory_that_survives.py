# python/05_memory_that_survives.py
"""Tutorial 05 - Memory That Survives.

Fifth rung of the fairlib tutorial series: short-term memory that
summarizes instead of forgetting, pinned messages that survive
compression, and long-term knowledge grounded through RAG. Setup and
how to run: README.md.
"""

# %% [markdown]
# ## Tutorial 05 - Memory That Survives
#
# The situation: you are the duty officer's assistant at the Falcon
# Telescope. An operational shift is a long conversation of log entries,
# equipment notes, and visitor traffic, and buried in it is one fact
# that must never be lost: the dome crane is red-tagged, so the dome
# must not rotate. Meanwhile the observatory has an ops manual the
# assistant is expected to know cold, even though nobody recites it in
# conversation.
#
# Those are the two halves of agent memory, and they have different
# machinery:
#
# - **Short-term memory** is conversation continuity: what was said in
#   this session. The enemy is the finite context window.
# - **Long-term memory** is knowledge grounding: what is true regardless
#   of the session. The enemy is the model answering from its training
#   data instead of your documents.
#
# What you will learn:
#
# - What `WorkingMemory`, the ring buffer, silently loses.
# - How `SummarizingMemory` compresses old turns with the LLM instead of
#   dropping them.
# - How a `Message` with `importance="pinned"` survives compression
#   verbatim - the home for mission-critical constraints, and something
#   only `SummarizingMemory` honors.
# - How embeddings, `FaissVectorStore`, `LongTermMemory`, and
#   `KnowledgeBaseQueryTool` make retrieval a tool call the agent
#   chooses to make.
#
# *Requirements: a local HuggingFace model (torch plus transformers; a
# GPU is recommended) plus the sentence-transformers package, which
# downloads a small embedding model on first use. Set
# `FAIR_LLM_DEMO_MODEL` to choose a different model.*

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
    Message,
    RoleDefinition,
    FormatInstruction,
    Example,
    ToolRegistry,
    ToolExecutor,
    WorkingMemory,
    SummarizingMemory,
    SentenceTransformerEmbedder,
    FaissVectorStore,
    LongTermMemory,
    SimpleRetriever,
    KnowledgeBaseQueryTool,
    HuggingFaceAdapter,
    SimpleReActPlanner,
    SimpleAgent,
)

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# %% [markdown]
# ### A window into the agent's head
#
# The same helper from tutorial 03: it prints an agent's full working
# memory, every message in order. Memory is this tutorial's whole
# subject, so we will be looking inside heads a lot.

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
# ### Step 1: the problem - the context window always wins
#
# This is a universal problem, not a telescope one: it hits any agent
# that runs long enough. A customer-support bot over a day of tickets, a
# coding agent working through a large repo, a mission assistant on a
# six-hour shift - all of them share the same trap. Every call to a chat
# model resends the whole history (tutorial 01), and the model can only
# hold so much. So every long-running agent eventually faces the same
# decision: something old must go.
#
# The naive answer - drop the oldest messages - fails in the worst
# possible way, because the age of a message says nothing about its
# importance. Some fact stated early in a long session is old news by the
# end, and yet it can be the one thing the agent must never forget. We
# will make that concrete in a few steps with a safety constraint (a
# piece of equipment tagged out of service) that has to survive a long,
# chatty shift - but hold onto the general shape first.
#
# A memory system therefore needs two distinct moves:
#
# - **Compress** the ordinary middle of the conversation instead of
#   deleting it: keep the gist, spend fewer tokens.
# - **Pin** what must survive compression verbatim, because a summary
#   of a safety constraint is not a safety constraint.
#
# fairlib ships both. But first, look at what the naive strategy loses.

# %% [markdown]
# ### Step 2: `WorkingMemory`, the ring buffer
#
# A **ring buffer** is a fixed-size container: it holds at most N items,
# and once it is full, adding a new item overwrites the oldest one to
# make room. Picture N chairs in a circle - when every chair is taken,
# the next arrival sits in the oldest occupant's chair and that occupant
# is gone. The size never grows; the newest data always displaces the
# oldest. It is a common, efficient pattern for "keep only the most
# recent N of something," and it is exactly the wrong policy when an old
# item mattered more than a new one.
#
# `WorkingMemory` with a `max_size` of N is that buffer, and it is the
# simple memory from tutorial 03: a list of `Message` objects, trimmed
# when it grows past that size. The trim keeps slot 0 (by convention the
# system prompt) plus the most recent messages, and everything between
# simply falls off - overwritten and unrecoverable. No model is needed
# to see it happen: add nine log entries to a six-slot memory and count
# the survivors.

# %%
ring = WorkingMemory(max_size=6)
for i in range(1, 10):
    ring.add_message(Message(role="user", content=f"log entry {i}"))

survivors = [m.content for m in ring.get_history()]
print(f"Added 9 messages; memory holds {len(survivors)}:")
for content in survivors:
    print(" ", content)
print("\nEntries 2 through 4 are gone - no summary, no trace, no event.")

# %% [markdown]
# If "log entry 3" had been "the crane is red-tagged", the agent would
# now violate it with total confidence. That is the failure mode the
# rest of this tutorial exists to prevent.

# %% [markdown]
# ### Step 3: load the model
#
# One chat model, loaded once, powers both halves of this tutorial: it
# drives the agents AND writes the history summaries in the next step.
# Why can a single instance do both jobs? For the same reason tutorial 01
# gave: the model is stateless. Summarizing a conversation is not a
# special mode - it is just another `invoke` call with a different prompt
# ("condense these messages into a summary"). Because the model keeps
# nothing between calls, the very same weights can answer the duty officer
# one moment and compress old history the next, with no interference. You
# do not need, and would not want, a second model just to summarize.
#
# Keep one relationship in mind as you read the next step: **pinning and
# summarizing memory go together.** Pinning is a feature of
# `SummarizingMemory` specifically. The plain `WorkingMemory` ring buffer
# from step 2 trims purely by position and never looks at a message's
# importance, so marking a message pinned there would do nothing at all.
# Pinning only means something once a memory has to choose what to
# compress versus keep - which is exactly the choice `SummarizingMemory`
# makes.

# %%
print(f"Loading {MODEL_NAME} (first run downloads the weights)...")
llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=512)

# %% [markdown]
# ### Step 4: `SummarizingMemory` - compress the middle, pin what matters
#
# `SummarizingMemory` holds the history until it exceeds
# `max_history_length`, then uses the LLM to fold the middle of the
# conversation into a summary, keeping the first message and the most
# recent ones verbatim. Two features make it trustworthy rather than
# just lossy:
#
# - **Pinning.** A `Message` with `importance="pinned"` is excluded from
#   summarization and reinserted verbatim, at full fidelity, at its
#   original position. Read this part carefully, because cadets get it
#   backwards: **pinning is your decision, made in your code - the agent
#   never pins anything.** You, the application author, decide what counts
#   as mission-critical and set `importance="pinned"` on that message at
#   the callsite (you will see exactly that below). The model is never
#   asked "is this important?"; you answered that when you built the
#   message. Defining that criterion is real design work - it is the
#   difference between a constraint that survives word for word and one
#   that gets blurred into a summary.
# - **Observability.** Every compression also emits a
#   `MemorySummarizedEvent` carrying what was dropped, what was kept, and
#   the summary text, so an application can observe compression instead of
#   inferring it. Watching those events live is the event bus's job, and
#   the event bus is tutorial 06's entire subject - so we leave it there.
#   In this tutorial we confirm compression the direct way: by opening the
#   memory afterward and reading what it holds.
#
# We keep `max_history_length` deliberately tiny, at six messages, so a
# compression happens inside one short shift. Production values are much
# larger; the mechanics are identical.

# %%
# No event bus here on purpose - that is tutorial 06's topic. Compression
# still happens inside SummarizingMemory; we confirm it later by reading
# the memory directly rather than by subscribing to its event.
memory = SummarizingMemory(
    llm=llm,
    max_history_length=6,
    messages_to_keep_at_end=3,
)

# The planner requires a registry; the duty log needs no tools, so an
# empty one is exactly right - the agent still ends turns cleanly via
# the built-in final_answer sentinel.
duty_registry = ToolRegistry()
duty_planner = SimpleReActPlanner(llm, duty_registry)
duty_planner.prompt_builder.role_definition = RoleDefinition(
    "You are the duty officer's assistant at the Falcon Telescope. "
    "Treat any PRIORITY message as a standing constraint and enforce "
    "it in every answer you give."
)
# The role says who the assistant is; how answers should read lives in
# a FormatInstruction (tutorial 04's split), so the Output Format
# section of the rendered prompt has real content.
duty_planner.prompt_builder.format_instructions.append(
    FormatInstruction("Acknowledge routine log entries in one short sentence.")
)

duty_agent = SimpleAgent(
    llm=llm,
    planner=duty_planner,
    tool_executor=ToolExecutor(duty_registry),
    memory=memory,
    max_steps=4,
)

# %% [markdown]
# Now the shift. Turn two is the mission-critical fact, and here the
# pinning rule gets applied at a callsite: this fact is **critical to
# the mission**, so it is sent as a pinned `Message`. `arun` accepts a
# full `Message`, so the caller decides at the callsite what must
# survive. The surrounding turns are ordinary log traffic, free to be
# summarized away.


# %%
async def run_the_shift() -> None:
    shift_turns = [
        "Evening shift start. Log: dome closed, sky clear, humidity 42 percent.",
        Message(
            role="user",
            content=(
                "PRIORITY: The dome crane is red-tagged for maintenance. Do "
                "not schedule or approve any dome rotation until maintenance "
                "clears the tag."
            ),
            # WE decide this is critical - the dome cannot move - so WE set
            # it pinned here at the callsite. The agent never makes this
            # call; defining the pin criterion is the author's job, and a
            # pinned message can never be summarized away.
            importance="pinned",
        ),
        "Log: spectrograph calibration complete, dark frames archived.",
        "Log: cadet astronomy club toured the control room at 2030, departed 2100.",
    ]
    for turn in shift_turns:
        text = turn.content if isinstance(turn, Message) else turn
        is_pinned = isinstance(turn, Message) and turn.importance == "pinned"
        tag = " (pinned)" if is_pinned else ""
        print(f"DUTY OFFICER{tag}: {text}")
        response = await duty_agent.arun(turn)
        print(f"ASSISTANT: {str(response)[:200]}\n")


asyncio.run(run_the_shift())

# %% [markdown]
# The history now sits at the limit, so the next turn pushes it over.
# Somewhere around here `SummarizingMemory` quietly folds the older,
# routine turns into a summary. There is no live announcement now that we
# have set the event bus aside for tutorial 06 - the step after this one
# opens the memory and shows you the result. What matters is the answer
# itself: it hinges on a constraint from several turns ago that just
# survived compression. This is the money shot of short-term memory:
# **mundane traffic became a summary, and the red-tag priority did not.**


# %%
async def ask_the_question_that_matters() -> None:
    question = "Can we rotate the dome tonight to track a target in the west?"
    print(f"DUTY OFFICER: {question}")
    answer = await duty_agent.arun(question)
    print(f"\nASSISTANT: {answer}")
    # A trailing rule so this answer does not run visually into the next
    # section's text when the notebook renders the cells back to back.
    print("\n" + "-" * 72)


asyncio.run(ask_the_question_that_matters())

# %% [markdown]
# Look inside the memory to confirm what compression did: a summary
# message where the ordinary turns used to be, and the PRIORITY
# constraint still present word for word. This is a hand-rolled variant
# of `show_agent_mind` because we want one extra column here: each
# message's `importance` flag, so you can see the pinned message stand
# out from the rest.
#
# One detail matters: `SummarizingMemory` is an *async* memory, and
# `aget_history` - the same call the agent loop makes - is its supported
# read surface. That is the view that honors pinning, so it is the view
# we inspect.


# %%
async def inspect_duty_memory() -> None:
    # aget_history is the pinning-honoring read path on SummarizingMemory.
    history = await memory.aget_history()
    print(f"Memory holds {len(history)} messages:")
    for i, m in enumerate(history):
        marker = "[pinned]" if m.importance == "pinned" else "        "
        preview = m.content[:90].replace("\n", " ")
        print(f"  {i:2d}. {marker} {m.role:>9}: {preview}")


asyncio.run(inspect_duty_memory())

# %% [markdown]
# ### Step 5: long-term memory - knowledge the conversation never contained
#
# The second half of the problem: the duty officer asks about the wind
# limit for opening the dome. Nobody said it this shift. It lives in
# the ops manual, and the model's training data is exactly the wrong
# place to get it from (tutorial 01, limit 2: it will happily invent a
# plausible number).
#
# The fix is RAG - Retrieval-Augmented Generation. If you want the
# authoritative treatment, RAG was introduced by Lewis et al. (2020),
# "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks":
# https://arxiv.org/abs/2005.11401 . The idea is four moves:
#
# - **Embed** each document into a vector that captures its meaning,
#   using `SentenceTransformerEmbedder`, a small local model.
# - **Store** the vectors; `FaissVectorStore` is used here, and it does
#   real similarity math and persists its index to disk.
# - **Retrieve** the most relevant passages for a query with
#   `SimpleRetriever`.
# - **Ground** the agent by exposing retrieval as a tool, so it looks
#   facts up instead of hallucinating them.
#
# First, the corpus. Four short ops-manual sections, ingested through
# `LongTermMemory`, which wraps each string into a `Document` and has
# the store embed and index it.

# %%
ops_manual = [
    (
        "Dome operations: The dome may be opened only when the sustained "
        "surface wind is 25 knots or less. Close the dome immediately if "
        "gusts exceed 35 knots or any precipitation is detected.",
        {"section": "dome_operations"},
    ),
    (
        "Weather minima for observing: relative humidity must be below 85 "
        "percent at the dome slit. Suspend observing when cloud cover "
        "exceeds fifty percent or the dew point margin falls under 2 "
        "degrees Celsius.",
        {"section": "weather_minima"},
    ),
    (
        "Instrument startup checklist: power on the mount controller, home "
        "both axes, cool the CCD camera to minus 20 Celsius, and take five "
        "bias frames before the first science exposure.",
        {"section": "instrument_startup"},
    ),
    (
        "Visitor policy: visiting groups must be escorted in the control "
        "room, and the observing floor is closed to visitors while the "
        "telescope is slewing.",
        {"section": "visitor_policy"},
    ),
]

print("Loading the embedding model (first run downloads it)...")
embedder = SentenceTransformerEmbedder()

vector_store = FaissVectorStore(
    embedder,
    index_dir=os.path.join(TUTORIALS_DIR, "_scratch", "05_telescope", "vector_store"),
)
ops_library = LongTermMemory(vector_store)
# One add_document call per section is fine for a six-section manual,
# but note that each call embeds and persists the index to disk. For a
# real corpus, build Document objects and hand them to the store's
# add_documents in one batch, the way tutorial 12 ingests its archive.
for text, metadata in ops_manual:
    ops_library.add_document(text, metadata)

print(f"Indexed {vector_store.ntotal} manual sections (persisted to disk).")

# %% [markdown]
# Retrieval works with no chat model in the loop - it is vector math.
# Ask the store for the passages nearest to a wind question and the
# dome-operations section should surface first, because its meaning is
# closest, even though the query shares few exact words with it.

# %%
retriever = SimpleRetriever(vector_store)
hits = retriever.retrieve("wind limit for opening the dome", top_k=2)
for rank, doc in enumerate(hits, start=1):
    print(f"{rank}. [{doc.metadata.get('section')}] {doc.page_content[:100]}...")

# %% [markdown]
# Now make retrieval something the agent can decide to do.
# `KnowledgeBaseQueryTool` wraps the retriever as a standard read-only
# tool (tutorial 04: typed schema, generated catalog entry) named
# `course_knowledge_query`. To the agent it is just another tool; the
# knowledge base behind it is invisible.

# %%
knowledge_registry = ToolRegistry()
knowledge_registry.register_tool(KnowledgeBaseQueryTool(retriever, top_k=2))

librarian_planner = SimpleReActPlanner(llm, knowledge_registry)
librarian_planner.prompt_builder.role_definition = RoleDefinition(
    "You are the Falcon Telescope duty assistant. For any question about "
    "procedures, limits, or policy you MUST look up the answer with the "
    "course_knowledge_query tool. Never answer procedure questions from "
    "your own knowledge."
)
# Output style again lives in the FormatInstruction, not the role.
librarian_planner.prompt_builder.format_instructions.append(
    FormatInstruction("Quote the specific figure you find in your final answer.")
)

# One worked turn keeps a small local model's tool-call syntax exact
# (tutorial 03's lesson): the query tool takes the bare search text.
librarian_planner.prompt_builder.examples.append(
    Example(
        "User: What is the frost limit for pier work?\n"
        "Thought: A procedure question goes to the knowledge base; the "
        "input is the bare search text.\n"
        "Action:\n"
        "tool_name: course_knowledge_query\n"
        "tool_input: frost limit pier work\n"
        "(after the observation arrives)\n"
        "Thought: The retrieved passage gives the figure; quote it.\n"
        "Action:\n"
        "tool_name: final_answer\n"
        "tool_input: The manual sets the frost limit for pier work at "
        "30 inches."
    )
)

librarian = SimpleAgent(
    llm=llm,
    planner=librarian_planner,
    tool_executor=ToolExecutor(knowledge_registry),
    memory=WorkingMemory(),
    max_steps=6,
)


async def ask_the_manual() -> None:
    question = "What is the maximum sustained wind for opening the dome?"
    print(f"DUTY OFFICER: {question}")
    answer = await librarian.arun(question)
    print(f"\nASSISTANT: {answer}")


asyncio.run(ask_the_manual())

# %% [markdown]
# The retrieval was a tool call the agent chose to make, and
# `show_agent_mind` proves it: the agent's thought, the
# `course_knowledge_query` call, the retrieved manual text arriving as
# an observation, and a final answer that cites the 25-knot limit
# because it read it, not because it guessed it.

# %%
show_agent_mind(librarian)

# %% [markdown]
# ### Debrief: what this means for your capstone
#
# Two memories, two jobs, both behind swappable interfaces:
#
# - **Short-term memory is conversation continuity.** A ring buffer
#   forgets blindly. `SummarizingMemory` compresses the ordinary middle,
#   and `importance="pinned"` - a flag you set, never the agent - carries
#   the non-negotiables through verbatim. Every compression also emits a
#   `MemorySummarizedEvent` your application can subscribe to (the event
#   bus, tutorial 06) so nothing is guessed. Pin your constraints: never
#   let a safety rule's survival depend on its age.
# - **Long-term memory is knowledge grounding.** Embed your corpus,
#   store the vectors, and expose retrieval as a tool. The agent then
#   cites your documents instead of hallucinating around them, and
#   because the store persists, the knowledge outlives every session.
# - **Both are seams, not choices.** Every memory here is an
#   implementation of one shared memory interface, and the vector stores
#   are pluggable too - FAISS here, Chroma ships as well. If neither
#   fits your capstone, you write your own: implement the memory
#   interface with a custom summarizer, a database-backed store, or a
#   pinning policy of your own design, and the agent loop never knows the
#   difference. Nothing in this tutorial is a fixed menu; each part is a
#   slot you are free to fill yourself.
#
# **Capstone connection.** Two shapes recur across projects, whatever the
# domain:
#
# - Any assistant that must answer from a fixed body of knowledge - an
#   instrument manual, a data catalog, a policy binder, a corpus of past
#   reports - is the RAG pipeline you just built, only with a bigger
#   corpus. Embed it once, expose retrieval as a tool, and the agent
#   cites your documents instead of hallucinating around them.
# - Any assistant that runs a long session over a stream of incoming
#   information - telemetry, tickets, sensor logs, tracking data - is the
#   `SummarizingMemory`-plus-pinning pattern: the critical alerts are the
#   pinned messages you set, and the routine chatter is what gets
#   compressed away.
#
# Most real projects need both at once, exactly as this tutorial paired
# them.
#
# **Next:** tutorial `06_watch_it_think` introduces the event bus - the
# observability layer we deliberately set aside here - as its main
# character: typed events, structured traces, and streaming an agent's
# reasoning as it happens, including the `MemorySummarizedEvent` this
# tutorial's memory was already emitting under the hood.
