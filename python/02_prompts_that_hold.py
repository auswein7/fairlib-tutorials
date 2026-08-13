# python/02_prompts_that_hold.py
"""Tutorial 02 - Prompts That Hold.

Second rung of the fairlib tutorial series: composing system prompts
from typed parts, and getting model output structured enough for real
code to consume. Setup and how to run: README.md.
"""

# %% [markdown]
# ## Tutorial 02 - Prompts That Hold
#
# Your capstone needs the model to do a job, the same way, every time -
# not chat. Job-shaped prompting has two halves:
#
# - **Composing the instructions** so that role, rules, and examples do
#   not drift apart as your project grows. That is what `PromptBuilder`
#   and its typed parts are for.
# - **Getting output a program can use.** Prose is for humans; your code
#   needs fields, and a guarantee that the fields are there. That is
#   Pydantic-validated JSON wrapped in a retry loop.
#
# By the end you will have used `PromptBuilder`, `RoleDefinition`,
# `FormatInstruction`, and `Example` to assemble a system prompt;
# derived a JSON schema from a Pydantic model; and run the
# validate-and-retry pattern that closes the loop - watching a live
# model fail and get corrected along the way.
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
import json
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
from pydantic import BaseModel, Field, ValidationError

# fairlib imports run simplest to most complex - the order you meet them.
from fairlib import (
    Message,
    RoleDefinition,
    FormatInstruction,
    Example,
    PromptBuilder,
    HuggingFaceAdapter,
)

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# One model, loaded once - the same instance serves every call below.
print(f"Loading {MODEL_NAME}...")
llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=384)

# %% [markdown]
# ### Step 1: the job - a briefing formatter
#
# The scenario: aircrew radio in messy, free-form status reports, and
# Ops needs each one as a clean record - callsign, aircraft, souls on
# board, fuel, and whether an emergency is declared. Here is tonight's
# traffic.

# %%
RADIO_REPORT = (
    "uh center, this is REEF 41, single F-16, we're showing twenty-five "
    "hundred pounds of fuel, call it three and a half hours remaining, "
    "three souls on board - correction, two souls on board - negative "
    "emergency at this time, requesting flight following to Bravo sector."
)
print(RADIO_REPORT)

# %% [markdown]
# ### Step 2: the first instinct - just ask
#
# Ask for a summary and the model does fine - for a human reader.
#
# **Run the next cell two or three times.** The wording shifts, the
# ordering shifts, even which details make the cut shifts. Now read the
# output as a programmer: which characters are the callsign? Where does
# the fuel number start? There is no slice, split, or regex you could
# write tonight that still works tomorrow, because the text is different
# on every run. It is programmatically impossible to reliably extract
# fields from non-deterministic prose - and that impossibility is the
# problem this tutorial exists to solve.

# %%
casual = llm.invoke(
    [Message(role="user", content=f"Summarize this radio report: {RADIO_REPORT}")]
)
print(casual.content)

# %% [markdown]
# ### Step 3: compose the prompt from typed parts
#
# fairlib treats a system prompt as a small document assembled from
# typed items, each with one job:
#
# - A `RoleDefinition` says who the model is.
# - A `FormatInstruction` states a rule its output must follow.
# - An `Example` is a worked demonstration - the few-shot learning the
#   model imitates.
#
# `PromptBuilder` renders them into labeled sections. Because the parts
# are objects, your code can inspect, swap, and reuse them, and in
# tutorial 11 you will save and reload them as files. This same builder
# is what fairlib's agents use internally, so everything you learn here
# transfers directly.
#
# Read the printout below carefully: every line of it is something *you*
# defined - your role, your rule, your example, and nothing else. That
# matters because in tutorial 03 you will hand this same builder to an
# agent's planner, print the full prompt again, and see fairlib wrap your
# text with extra sections it injects (the tool catalog, the ReAct
# format). Knowing what is yours here makes it obvious what the framework
# adds there.

# %%
builder = PromptBuilder()
builder.role_definition = RoleDefinition(
    "You are an operations desk formatter. You convert messy aircrew "
    "radio reports into precise structured records. You never invent "
    "information that is not in the report."
)
builder.format_instructions.append(
    FormatInstruction(
        "Reply with ONLY a JSON object matching the schema you are "
        "given. No commentary, no markdown fences. If the crew corrects "
        "themselves mid-report, record the corrected value."
    )
)
builder.examples.append(
    Example(
        'Report: "center, HAWK 22, single F-15, eight thousand pounds of '
        'fuel, about five hours remaining, two souls on board, no '
        'emergency"\n'
        'Record: {"callsign": "HAWK 22", "aircraft_type": "F-15", '
        '"souls_on_board": 2, "fuel_pounds": 8000.0, '
        '"crew_estimated_fuel_hours": 5.0, "emergency": false}'
    )
)

print(builder.build_system_prompt_string())

# %% [markdown]
# ### Step 4: define the record as a Pydantic model
#
# The schema is code, not prose. Pydantic gives us both halves of the
# contract from one class:
#
# - a **JSON Schema** to show the model, and
# - a **validator** to check what comes back.
#
# If you have not met Pydantic: it is a library for defining data
# shapes as Python classes and validating raw data against them, and
# fairlib uses it everywhere.

# %%
class FlightRecord(BaseModel):
    callsign: str = Field(description="The flight's callsign, e.g. REEF 41")
    aircraft_type: str = Field(description="Airframe, e.g. F-16")
    souls_on_board: int = Field(description="Total persons aboard the flight")
    fuel_pounds: float = Field(description="Pounds of fuel remaining, as reported")
    crew_estimated_fuel_hours: float = Field(
        description="Hours of fuel remaining, as ESTIMATED by the crew"
    )
    emergency: bool = Field(description="True only if the crew declared an emergency")


print(json.dumps(FlightRecord.model_json_schema(), indent=2))

# %% [markdown]
# ### Step 5: extract, validate, retry - in the open
#
# This loop is the load-bearing pattern of the tutorial. No helper
# functions, no hidden plumbing: every string it sends is built right
# here in the cell, so you can read exactly what the model sees. Each
# attempt:
#
# - **Sends** the composed system prompt, the schema, and the report.
# - **Validates** the reply against `FlightRecord` - code checks the
#   contract, not eyeballs.
# - **On failure, retries with evidence:** the bad reply stays in the
#   conversation and the validation error becomes the next user message,
#   so the model sees exactly what to fix.
#
# Watch the attempt printouts. Small local models regularly fail on the
# first try - a markdown fence around the JSON, a string where a number
# belongs, an invented field. That is not the tutorial misbehaving;
# **models fail routinely, and code like this is how you engineer around
# it.** The error message is the retry prompt: you are not hoping the
# model behaves, you are checking, and correcting it with evidence.

# %%
schema_text = json.dumps(FlightRecord.model_json_schema(), indent=2)
system_text = builder.build_system_prompt_string()
request_text = (
    f"JSON schema for the record:\n{schema_text}\n\n"
    f'Report: "{RADIO_REPORT}"\nRecord:'
)

messages = [
    Message(role="system", content=system_text),
    Message(role="user", content=request_text),
]

record = None
for attempt in range(1, 4):
    raw = llm.invoke(messages).content.strip()
    print(f"--- attempt {attempt} raw reply " + "-" * 30)
    print(raw)
    try:
        record = FlightRecord.model_validate_json(raw)
        print(f"--> attempt {attempt}: VALID")
        break
    except ValidationError as exc:
        print(f"--> attempt {attempt}: invalid - sending the error back\n")
        # The failed reply and the validator's complaint join the
        # conversation, so the next attempt sees its own mistake.
        messages.append(Message(role="assistant", content=raw))
        messages.append(
            Message(
                role="user",
                content=(
                    "That was not valid. Fix these problems and reply "
                    f"with ONLY the corrected JSON object:\n{exc}"
                ),
            )
        )

if record is None:
    raise ValueError("no valid record after 3 attempts - rerun the cell")

# %% [markdown]
# ### Step 6: the reward - typed fields catch a life-safety error
#
# `record` is not text anymore. It is a validated Python object whose
# fields your program can trust - and that is what lets your code do
# something the crew could not do in their heads under stress: check
# their math.
#
# The crew *estimated* three and a half hours of fuel. But endurance is
# not a guess; it is fuel quantity divided by burn rate. An F-16 at
# cruise burns on the order of 2,000 pounds per hour, and the crew
# reported 2,500 pounds remaining. Run the numbers on the typed fields
# and the real endurance is about **1.25 hours** - a fraction of what
# they think they have. That is below the reserve needed to divert
# safely, so this flight has a fuel emergency the crew has *not*
# declared.
#
# This is the whole point of the tutorial in one cell: the model turned
# messy radio prose into trustworthy fields, and then ordinary Python -
# not the model - did the safety-critical arithmetic and caught the
# discrepancy. You would never trust a hallucination-prone model to do
# this math (tutorial 01); you trust your own code, running on fields the
# model was forced to extract cleanly.

# %%
# Known airframe performance - this lives in YOUR code, not the model.
BURN_RATE_LB_PER_HOUR = 2000.0
# Minimum endurance to reach a divert field with legal reserves.
RESERVE_HOURS = 1.5

print(record)
print()

# The math the crew did in their heads - now done on trustworthy fields.
actual_endurance_hours = record.fuel_pounds / BURN_RATE_LB_PER_HOUR
print(f"Crew estimated:   {record.crew_estimated_fuel_hours:.2f} hours")
print(f"Actual endurance: {actual_endurance_hours:.2f} hours "
      f"({record.fuel_pounds:.0f} lb / {BURN_RATE_LB_PER_HOUR:.0f} lb/hr)")
print(f"Discrepancy:      {record.crew_estimated_fuel_hours - actual_endurance_hours:.2f} "
      "hours the crew THINKS it has but does not")
print()

# Your code, not the model, makes the safety call.
emergency_by_the_numbers = actual_endurance_hours < RESERVE_HOURS
if emergency_by_the_numbers and not record.emergency:
    print("VERDICT: FUEL EMERGENCY - actual endurance is below the "
          f"{RESERVE_HOURS:.2f} hour reserve, but the crew declared none.")
    print("A human missed this. Your code did not.")
elif record.emergency:
    print("VERDICT: crew already declared an emergency.")
else:
    print("VERDICT: endurance is within reserve; no emergency.")

# %% [markdown]
# ### Debrief: what this means for your capstone
#
# - **Prompts are composed from typed parts,** not concatenated strings:
#   role, rules, and examples live in objects your code can manage.
# - **Structured output is a contract:** the schema is shown to the
#   model, a validator enforces it, and error-driven retry closes the
#   loop.
# - **The model extracts; your code decides.** Step 6's fuel-emergency
#   catch came from ordinary Python doing safety-critical math on typed
#   fields - never from trusting the model to reason about numbers.
# - Note what the model did with "three souls - correction, two souls":
#   the composed rules told it how to handle a mid-report correction.
#
# Keep the shape of that loop in mind - schema in the prompt, validation
# on the reply, evidence-driven retry on failure - because it is exactly
# how fairlib's agents call tools. In tutorial 03 the framework starts
# running that loop for you.
#
# **Capstone connection.** The bio-regenerative life-support capstone's
# fairlib layer is exactly this: telemetry and mission text in,
# validated structured analysis out. The space-situational-awareness
# project turns tracking chatter into records the same way, and every
# capstone that shows an LLM to a database, a dashboard, or another
# program crosses this bridge.
#
# **Next:** tutorial `03_your_first_agent` is where the model stops just
# describing and starts doing - the ReAct loop, and the calculator
# redemption for tutorial 01's arithmetic disaster.
