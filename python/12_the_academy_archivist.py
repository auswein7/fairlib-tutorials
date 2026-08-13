# python/12_the_academy_archivist.py
"""Tutorial 12 - The Academy Archivist.

Twelfth rung of the fairlib tutorial series: ingesting real scanned
documents with OCR, retrieval over a noisy corpus, structured
extraction into a queryable database, serving a tool over MCP, and one
agent composing all of it with a pinned standing order. Setup and how
to run: README.md.
"""

# %% [markdown]
# ## Tutorial 12 - The Academy Archivist
#
# The situation: the Academy's curriculum archives span more than sixty
# years of course catalogs. The oldest are typewriter-era scans - some
# pages have no digital text at all, only ink. The newest are clean
# born-digital PDFs. Somewhere in that pile is the answer to every
# "when did we teach X?" question anyone at the Academy has ever asked,
# and you are building the archivist who can answer them: a subject
# matter expert that reads the real documents, cites its sources, and
# refuses to guess.
#
# This is the composition rung. Every piece here appeared alone on an
# earlier rung - tools on 04, memory and RAG on 05, events on 06 -
# and this tutorial wires them into one working expert, the same shape
# your capstone will take.
#
# What you will learn:
#
# - How `DocumentProcessor` turns real PDFs into `Document` chunks,
#   and how its OCR fallback quietly rescues pages with no text layer.
# - What retrieval over a REAL corpus feels like - including what
#   sixty-year-old OCR noise does to your chunks, and why retrieval
#   works anyway.
# - How deterministic extraction builds a structured database from the
#   same documents, so prose questions and structured questions get
#   answered from the same source of truth.
# - How to serve a tool from a separate process over MCP, and why the
#   agent cannot tell - and should not care - where a tool lives.
# - How a pinned standing order shapes every answer the archivist
#   gives, without you repeating it once.
#
# *Requirements: a local HuggingFace model (torch plus transformers; a
# GPU is recommended) plus the sentence-transformers package, which
# downloads a small embedding model on first use. PDF ingestion needs
# pdfplumber, pdf2image, and pytesseract on the Python side and the
# tesseract and poppler system packages. The MCP step needs the mcp
# package. Set `FAIR_LLM_DEMO_MODEL` to choose a different model.*

# %% [markdown]
# ### Setup
#
# *This cell is plumbing, not part of the lesson: it locates the repo
# folder and loads your `.env` settings. **Just run it** and move on.*

# %%
import asyncio
import os
import sys

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
import re
import sqlite3

from fairlib import (
    Message,
    RoleDefinition,
    Example,
    SentenceTransformerEmbedder,
    FaissVectorStore,
    SimpleRetriever,
    KnowledgeBaseQueryTool,
    ToolRegistry,
    ToolExecutor,
    MCPServerConfig,
    AgentEventBus,
    ToolCallPostEvent,
    SummarizingMemory,
    HuggingFaceAdapter,
    SimpleReActPlanner,
    SimpleAgent,
)

# These two are not exported at the fairlib top level in the current
# PyPI release, so they come from their home modules.
from fairlib.utils.document_processor import DocumentProcessor
from fairlib.modules.mcp import create_mcp_enhanced_registry

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# %% [markdown]
# ### Step 1 - The archives
#
# The corpus is a folder of real PDFs. By default the tutorial looks in
# `_scratch/12_archivist/corpus/` - drop any course catalogs,
# policy documents, or handbooks there (two or three files is plenty;
# one old scan plus one modern PDF makes the OCR lesson visible). Set
# `FAIR_LLM_ARCHIVE_DIR` to point somewhere else, like a full archive
# folder.
#
# This rung was built and tested against the Academy's real curriculum
# archives: the AY1959-60 catalog (a typewriter-era scan whose pages
# are part OCR text layer, part raw ink) and the AY2024-25 fall
# supplement (born digital, clean text).

# %%
ARCHIVE_DIR = os.environ.get(
    "FAIR_LLM_ARCHIVE_DIR",
    os.path.join(TUTORIALS_DIR, "_scratch", "12_archivist", "corpus"),
)
SCRATCH_DIR = os.path.join(TUTORIALS_DIR, "_scratch", "12_archivist")
os.makedirs(ARCHIVE_DIR, exist_ok=True)

pdf_names = sorted(
    name for name in os.listdir(ARCHIVE_DIR) if name.lower().endswith(".pdf")
)
if not pdf_names:
    print(
        f"No PDFs found in {ARCHIVE_DIR}.\n"
        "Drop one or two PDF documents there (course catalogs, policy\n"
        "documents, any real scanned material) and run this cell again,\n"
        "or set FAIR_LLM_ARCHIVE_DIR to a folder that has some."
    )
    # Stop here with the guidance above as the last visible output; the
    # nonzero exit keeps script runners honest about the no-op.
    raise SystemExit(1)
print(f"Archive folder: {ARCHIVE_DIR}")
for name in pdf_names:
    print(f"  {name}")

# %% [markdown]
# ### Step 2 - Reading paper: ingestion with an OCR fallback
#
# `DocumentProcessor.process_file` is fairlib's one-call path from a
# PDF to a list of `Document` chunks with page-level metadata. Under
# the hood it extracts each page's text layer - and when a page has no
# text layer at all (a pure image, common in old scans), it renders the
# page and runs OCR on it instead.
#
# `get_stats()` afterwards tells you exactly what happened: how many
# pages needed OCR and how many succeeded. In a real pipeline that
# number is the first thing you check - a document that silently
# produced zero chunks is a document your expert silently does not
# know.

# %%
processor = DocumentProcessor(
    {
        "files_directory": ARCHIVE_DIR,
        "max_chunk_chars": 1500,
        "enable_ocr": True,
        "ocr_dpi": 200,
    }
)

archive_chunks: list = []
for name in pdf_names:
    docs = processor.process_file(os.path.join(ARCHIVE_DIR, name))
    print(f"{name}: {len(docs)} chunks")
    archive_chunks.extend(docs)

stats = processor.get_stats()
print(
    f"\nIngestion: {stats['files_processed']} files, "
    f"{stats['total_chunks']} chunks, "
    f"{stats['ocr_attempts']} pages OCRed "
    f"({stats['ocr_successes']} succeeded), "
    f"{stats['extraction_errors']} errors"
)

# %% [markdown]
# Look at what real archives give you. A chunk from a modern PDF reads
# like a course catalog; a chunk from a 1959 scan can read like static.
# That noise is not a bug in your pipeline - it IS the data. The
# embedding model shrugs most of it off because enough clean text
# survives per chunk, but you should always look at a few chunks raw
# before you trust a corpus.

# %%
if archive_chunks:
    oldest = archive_chunks[0]
    newest = archive_chunks[-1]
    print(f"[{oldest.metadata.get('source', '?')}]")
    print(oldest.page_content[:220].replace("\n", " "))
    print()
    print(f"[{newest.metadata.get('source', '?')}]")
    print(newest.page_content[:220].replace("\n", " "))

# %% [markdown]
# ### Step 3 - The library: retrieval over the corpus
#
# The wiring is exactly tutorial 05's, at corpus scale: an embedder, a
# FAISS store that persists itself to disk, a retriever over the store,
# and the retriever wrapped as a tool the agent can call. On a re-run
# the store loads from disk instead of re-embedding, but only after
# checking that the persisted chunk count still matches what the
# corpus yields - an index built from an older corpus would silently
# not know your new files. Delete the `vector_store` folder under
# `_scratch/12_archivist/` to force a clean rebuild.

# %%
embedder = SentenceTransformerEmbedder()
vector_store = FaissVectorStore(
    embedder, index_dir=os.path.join(SCRATCH_DIR, "vector_store")
)
if vector_store.load() and vector_store.ntotal == len(archive_chunks):
    print(f"Loaded persisted index: {vector_store.ntotal} chunks")
else:
    # Either no persisted index, or the corpus changed since it was
    # built (chunk counts disagree) - serving the stale index would
    # silently pretend the changed files do not exist, so rebuild.
    if vector_store.ntotal > 0:
        print(
            f"Persisted index has {vector_store.ntotal} chunks but the "
            f"corpus now yields {len(archive_chunks)}; rebuilding."
        )
        vector_store.clear()
    print(f"Embedding {len(archive_chunks)} chunks (first run takes a minute)...")
    vector_store.add_documents(archive_chunks)
    print(f"Indexed {vector_store.ntotal} chunks")

retriever = SimpleRetriever(vector_store)

# %% [markdown]
# Before any agent touches it, prove retrieval works the same way you
# would prove a detector works: directly, deterministically, no model
# in the loop. Ask the store a question and look at what comes back and
# WHERE it came from - the metadata carries file and page, which is
# what lets the archivist cite sources instead of waving at "the
# archives".

# %%
for probe in ["Shakespeare literature course", "public health course"]:
    hits = retriever.retrieve(probe, top_k=2)
    print(f"query: {probe}")
    for hit in hits:
        source = hit.metadata.get("source", "?")
        preview = hit.page_content[:90].replace("\n", " ")
        print(f"  [{source}] {preview}")
    print()

# %% [markdown]
# ### Step 4 - The card catalog: structured extraction into SQLite
#
# Retrieval answers prose questions. It is bad at counting. "How many
# special-topics courses ran in 2024, by department?" wants a database,
# not a similarity search.
#
# So build one - from the same archives. The extraction below is
# deterministic code, not a model: a regex over each document's full
# text that matches the catalog's own course-entry format
# (`Dept 123. Title.`), stamped with the catalog year from the
# filename. Run it twice, get the same database twice.
#
# Look at the rows before you trust them. On the real archives this
# regex pulls clean rows out of the 2024 supplement, pulls
# typewriter-era rows out of 1959 complete with OCR scars
# ("Literatu-re"), and occasionally mis-parses a header line into a
# course title. Extraction quality follows scan quality - the database
# inherits the corpus's honesty problems, and a capstone that extracts
# structure from real documents must audit a sample by hand.

# %%
COURSE_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z]{1,14}(?:\s[A-Za-z]{2,10})?)\s+(\d{3})\.\s+([A-Z][^.]{3,80})\."
)
DB_PATH = os.path.join(SCRATCH_DIR, "courses.db")

connection = sqlite3.connect(DB_PATH)
connection.execute("DROP TABLE IF EXISTS courses")
connection.execute(
    "CREATE TABLE courses ("
    "department TEXT, number TEXT, title TEXT, catalog_year TEXT)"
)
for name in pdf_names:
    catalog_year = name.split()[0].removesuffix(".pdf")
    text = processor.read_file_text(os.path.join(ARCHIVE_DIR, name))
    rows = [
        (dept.strip(), number, title.strip(), catalog_year)
        for dept, number, title in COURSE_PATTERN.findall(text)
    ]
    connection.executemany("INSERT INTO courses VALUES (?, ?, ?, ?)", rows)
    print(f"{catalog_year}: {len(rows)} course entries extracted")
connection.commit()

sample = connection.execute(
    "SELECT department, number, title, catalog_year FROM courses LIMIT 4"
).fetchall()
for row in sample:
    print("  ", row)
connection.close()

# %% [markdown]
# ### Step 5 - Serving the catalog over MCP
#
# The course database could be a local tool - tutorial 04 taught you
# how. Instead, this rung serves it from a SEPARATE PROCESS over MCP,
# because that is the shape real capstones grow into: the data source
# lives where the data lives, and agents connect to it.
#
# `python/_archivist_mcp_server.py` is a small stdio MCP
# server that opens the SQLite file read-only and exposes two tools,
# `course_search` and `catalog_summary`. It is written against the MCP
# SDK directly, not fairlib's MCPServer, on purpose: it plays the
# third-party server your capstone does not control. Both tools run
# parameterized queries - no text from the model is ever pasted into
# SQL. The agent
# side needs only a config saying how to launch it;
# `create_mcp_enhanced_registry` starts the server, discovers its
# tools, and merges them with the local registry under an `mcp_`
# prefix.
#
# One rule of engagement, learned the hard way: MCP connection
# failures degrade QUIETLY - the registry simply comes back without
# the remote tools. So after assembly, COUNT what you got. An expert
# missing half its tools should never be discovered mid-conversation.

# %%
local_registry = ToolRegistry()
local_registry.register_tool(KnowledgeBaseQueryTool(retriever, top_k=3))

MCP_SERVER_SCRIPT = os.path.join(
    TUTORIALS_DIR, "python", "_archivist_mcp_server.py"
)
mcp_configs = [
    MCPServerConfig(
        name="archives",
        transport="stdio",
        command=sys.executable,
        args=[MCP_SERVER_SCRIPT, DB_PATH],
        timeout=30,
        enabled=True,
    )
]
print("Local tools:", list(local_registry.get_all_tools().keys()))
print("MCP server:", os.path.basename(MCP_SERVER_SCRIPT))

# %% [markdown]
# ### Step 6 - The archivist takes the desk
#
# Now the composition. One driver assembles everything that has to
# share an event loop - the MCP connection lives inside the loop that
# created it, so registry assembly, agent construction, and the
# conversation all happen inside one async function, the same way
# tutorial 05 ran a whole duty shift in one call.
#
# Four pieces to notice as you read it:
#
# - The role tells the archivist HOW to work these archives: names are
#   abbreviated, old pages carry OCR errors, prose questions go to
#   retrieval, counting questions go to the database. An expert is not
#   just tools - it is tools plus judgment about which one the
#   question deserves, and that judgment lives in the role.
# - The memory is `SummarizingMemory`, and the first message the
#   archivist receives is PINNED: a standing order to cite catalog year
#   and source file in every answer drawn from the archives. Tutorial
#   05 taught the contract; here it does real work - the order
#   survives any amount of summarization, and you never repeat it.
# - The agent and the executor share one event bus, and a subscriber
#   prints which tool produced each observation. When an answer names
#   1959 course offerings, the trail shows whether it came from
#   retrieval (`course_knowledge_query`) or the database
#   (`mcp_archives_course_search`) - the record of truth is the event
#   stream, not the prose.
# - The tool inventory is printed and CHECKED after assembly. If the
#   MCP server failed to start, this run says so immediately.

# %%
print(f"Loading {MODEL_NAME} (first run downloads the weights)...")
llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=512)

# %%
STANDING_ORDER = (
    "Standing order for this desk: you are the Academy Archivist. "
    "Every claim you draw from the archives must name the catalog year "
    "and the source file it came from. If the archives do not contain "
    "the answer, say so plainly instead of guessing."
)

QUESTIONS = [
    "Search the archive text: what does the 1959-60 catalog say the "
    "English department taught? Cite the source you draw from.",
    "Using the course database, how many 495-numbered special topics "
    "courses are in the 2024-25 catalog, and which departments offer "
    "them?",
    "Was Shakespeare taught at the Academy in 1959, and does anything "
    "like it appear in the 2024-25 catalog? Compare what you find.",
]


async def staff_the_archivist_desk() -> None:
    registry = await create_mcp_enhanced_registry(
        local_registry, mcp_configs=mcp_configs, tool_prefix="mcp"
    )
    tool_names = sorted(registry.get_all_tools().keys())
    print("Tool inventory:", ", ".join(tool_names))
    mcp_tools = [name for name in tool_names if name.startswith("mcp_")]
    if not mcp_tools:
        print(
            "WARNING: no MCP tools discovered - the archives server did "
            "not come up. Structured catalog questions will fail; fix "
            "this before trusting any answer below."
        )

    bus = AgentEventBus()
    bus.subscribe(
        ToolCallPostEvent,
        lambda event: print(
            f"    [tool] {event.tool_name} -> "
            f"{'ok' if event.succeeded else 'FAILED'}"
        ),
    )

    executor = ToolExecutor(registry, events=bus)
    memory = SummarizingMemory(
        llm=llm, max_history_length=8, messages_to_keep_at_end=3
    )
    planner = SimpleReActPlanner(llm, registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        "You are the Academy Archivist, the subject matter expert on "
        "the Academy's course history. The archives are real scanned "
        "documents: department names are abbreviated (Engl, Beh Sci, "
        "Pol Sci) and old pages carry OCR errors, so search with short "
        "keywords and try a variant before concluding something is "
        "absent. For prose questions about what the catalogs say, "
        "search the archive text with course_knowledge_query. For "
        "counting or filtering specific courses, use the course "
        "database tools - but remember the database holds only what a "
        "regex could extract, so when it comes back empty, check the "
        "archive text before concluding absence. When comparing eras, "
        "check both."
    )
    # The database tools take a JSON object, not bare text - and a
    # small model will happily write key: value lines unless a worked
    # example shows it the shape. One example covering both tool
    # families is the cheapest reliability fix on this rung.
    planner.prompt_builder.examples.append(
        Example(
            "User: Did the Academy teach astronautics in 1959?\n"
            "Thought: A specific course lookup goes to the database; "
            "its tool_input is a JSON object.\n"
            "Action:\n"
            "tool_name: mcp_archives_course_search\n"
            'tool_input: {"keyword": "astronautics", '
            '"catalog_year": "AY1959-60"}\n'
            "(after the observation arrives)\n"
            "Thought: The database matched a row; for the prose "
            "description I search the archive text, whose input is "
            "bare search text.\n"
            "Action:\n"
            "tool_name: course_knowledge_query\n"
            "tool_input: astronautics course description\n"
            "(after the observation arrives)\n"
            "Thought: I can answer, citing year and source file.\n"
            "Action:\n"
            "tool_name: final_answer\n"
            "tool_input: Yes - the AY1959-60 catalog (AY1959-60.pdf) "
            "lists an astronautics course; the description covers "
            "orbital mechanics fundamentals."
        )
    )
    archivist = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=memory,
        events=bus,
        max_steps=8,
    )

    opening = Message(role="user", content=STANDING_ORDER, importance="pinned")
    acknowledgement = await archivist.arun(opening)
    print(f"\nArchivist: {acknowledgement}")
    print("-" * 72)

    for question in QUESTIONS:
        print(f"\nYou: {question}")
        answer = await archivist.arun(question)
        print(f"\nArchivist: {answer}")
        print("-" * 72)


asyncio.run(staff_the_archivist_desk())

# %% [markdown]
# ### Debrief: what this means for your capstone
#
# Read the transcript against the tool trail. The prose question rode
# `course_knowledge_query` into the vector store; the counting question
# rode `mcp_archives_course_search` into SQLite; the comparison
# question should have used both. Same archives, two complementary
# stores, one agent - and every answer stamped with a catalog year
# because a single pinned message said so once.
#
# What survived contact with real data is worth remembering, too. The
# 1959 scan needed OCR on pages that had no text at all; its chunks
# carry sixty years of typewriter noise; the extraction regex
# mis-parses the occasional header and skips ranged course numbers
# entirely. And one scar earned its own lesson: the 1959 catalog spells
# a course title "Literatu-re", so a database keyword search for
# "literature" honestly returns nothing - the row is there, the
# keyword is not. That is why the role tells the archivist the
# database is a lossy extraction and the archive text is the ground
# truth. None of this was hidden from you - the stats said how many
# pages were rescued, the raw chunks showed the noise, the sample rows
# showed the mis-parse. That is the posture to keep: measure the mess,
# show the mess, build anyway.
#
# **Capstone connection.** This rung IS the capstone shape. Your
# project will ingest some real, imperfect corpus; ground prose answers
# in retrieval and structured answers in extraction; reach at least one
# capability that lives outside your process; and carry standing
# constraints that must survive a long session. You have now seen every
# seam: `DocumentProcessor` for ingestion, the embedder-store-retriever
# stack for grounding, an `AbstractTool` or an MCP server for
# capabilities, pinning for what must not be forgotten, and the event
# bus as the record of what actually happened. Compose them for your
# domain, and audit the seams the way this tutorial did - directly,
# before the model ever gets involved.
#
# **Next:** your semester. The ladder in README.md ends here;
# the remaining rungs - sandboxed execution and evaluation harnesses -
# arrive as the framework grows them.
