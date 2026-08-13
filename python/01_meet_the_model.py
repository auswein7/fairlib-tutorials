# python/01_meet_the_model.py
"""Tutorial 01 - Meet the Model.

First rung of the fairlib tutorial series: a live local language model
behind the provider-agnostic adapter, and the two limits every capstone
must design around. Setup and how to run: README.md.
"""

# %% [markdown]
# ## Tutorial 01 - Meet the Model
#
# You just got your capstone assignment. Maybe it is Neural Shields,
# maybe the Avatar Language Immersion System, maybe Falcon Telescope AI.
# Every one of them reduces to the same first question: **how do you get
# useful, reliable work out of a language model?**
#
# This tutorial starts at the bottom of that stack. No agents, no tools,
# no memory - just you and the model. Everything fairlib gives you later
# is scaffolding around the component you are about to meet.
#
# What you will learn:
#
# - What a chat model actually is: a next-token predictor wrapped in a
#   conversation format.
# - How a `Message` carries one turn of that conversation.
# - How the `HuggingFaceAdapter` is one of several interchangeable model
#   adapters.
# - How the *system role* steers the model's behavior.
# - The two limits the rest of the series exists to fix: the model is
#   **stateless** (it remembers nothing between calls), and it is
#   **confidently wrong** (it predicts text rather than computing
#   answers).
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
from fairlib import Message, HuggingFaceAdapter

MODEL_NAME = os.environ.get("FAIR_LLM_DEMO_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# %% [markdown]
# ### Step 1: load the model
#
# The `HuggingFaceAdapter` downloads the weights on the first run and
# loads a chat model onto your GPU. The class itself is not the point;
# the seam it implements is. Every adapter in fairlib - `OllamaAdapter`,
# `OpenAIAdapter`, and `AnthropicAdapter` among them - satisfies the same
# `AbstractChatModel` interface, so nothing you build in this series
# depends on which model sits underneath. That is what lets a capstone
# develop against a free local model today and move to a hosted one by
# changing a single line. The `max_new_tokens` argument caps how much
# text one call may generate.

# %%
print(f"Loading {MODEL_NAME} (first run downloads the weights)...")
llm = HuggingFaceAdapter(MODEL_NAME, max_new_tokens=256)
print("Capabilities:", llm.get_model_capabilities())

# %% [markdown]
# ### Step 2: first contact
#
# A conversation is a list of `Message` objects, each with a role and
# content. You send the whole list in one call; the model reads all of
# it and returns exactly one new assistant message. That single
# request-and-reply is what we will call a **turn**: one call to
# `llm.invoke(...)` in, one assistant `Message` out.
#
# Why a *list* and not a single message? Because the model is stateless
# (step 4 proves it): it keeps nothing between turns, so the list you
# pass is the model's entire world for that one turn. If you want the
# model to weigh several messages at once - a system instruction plus a
# question, or an entire back-and-forth so far - they all have to ride in
# the single list you send. A list with three messages is still one
# turn; it just tells the model "here are three messages, now give me one
# reply."
#
# Note that we are calling `invoke` on `llm` - the **same model
# instance** loaded in step 1. One model, loaded once, will serve every
# call in this tutorial.

# %%
# The same llm instance from step 1 - we never load a second model.
reply = llm.invoke(
    [Message(role="user", content="In two sentences, why do aircraft take off into the wind?")]
)
print(reply.content)

# %% [markdown]
# ### Step 3: the system role sets the character
#
# Every `Message` carries a **role**, and a chat model treats the three
# roles differently:
#
# - **system**: instructions from the *operator* - you, the engineer who
#   built the application. The model treats these as standing orders that
#   outrank anything the user says: who it is, what it may do, how it
#   should answer.
# - **user**: whatever the *end user* (the cadet at the keyboard, the
#   pilot on the radio, the person in the chat box) typed. The model
#   treats this as a request to satisfy, not an order that can rewrite
#   its standing rules.
# - **assistant**: the model's own replies. You will meet this role in
#   step 4, when we replay a past answer back to the model.
#
# That operator-versus-user split is the whole reason a system prompt
# works. It is a privileged channel where you, the operator, set the
# model's character and rules *before* the user ever says a word - so the
# same user question can land very differently depending on the system
# message above it. The next cell proves it: it sends the *same* user
# question twice under two different system prompts, and the register of
# the answer changes with it.
#
# Notice also what the cell does **not** do: it never mentions step 2.
# This is still the same `llm` instance, but it has already forgotten
# the question you just asked it - each call starts from only the
# messages in the list you send. Step 4 pins that down.
#
# In fairlib you will rarely write these prompts by hand: tutorial 02
# introduces the prompt engineering system that composes them for you
# and keeps them consistent as your project grows.

# %%
question = Message(role="user", content="Why do aircraft take off into the wind?")

# Same model instance, new call: nothing from step 2 carries over unless
# it is inside this message list.
formal = llm.invoke(
    [
        Message(
            role="system",
            content=(
                "You are a terse military briefing officer. Answer in at most "
                "two short sentences, no pleasantries."
            ),
        ),
        question,
    ]
)
print("BRIEFING OFFICER:", formal.content)

friendly = llm.invoke(
    [
        Message(
            role="system",
            content=(
                "You are a patient flight instructor explaining things to a "
                "brand new student pilot, with a simple analogy."
            ),
        ),
        question,
    ]
)
print("\nFLIGHT INSTRUCTOR:", friendly.content)

# %% [markdown]
# ### Step 4: the first limit - no memory
#
# The model is a pure function: a list of messages goes in, one new
# message comes out, and nothing is remembered afterward. Every call
# starts from a blank slate - even on the same `llm` instance you have
# been using since step 1. If the model should know something from an
# earlier call, you must send it again yourself.
#
# The next cell demonstrates this in three moves:
#
# 1. Tell the model your callsign; it acknowledges. (turn one)
# 2. Ask for the callsign in a **fresh call** - the model has no idea.
#    (turn two: a brand new list that does not contain turn one)
# 3. Ask again, but this time **replay the earlier turns** inside the
#    message list - now it answers. (turn three)
#
# Watch move 3 closely, because this is where cadets most often trip. It
# is still a single turn - one `invoke` call - but the list we pass holds
# *three* messages: the original intro, the model's own acknowledgement
# (an `assistant` message), and the new question. That bundled-up
# transcript is what we mean by **history**. History is not something the
# model stores between calls; it is something *you* re-send inside the one
# list, every turn. So "giving the model history" and "sending several
# messages in a single turn" are the same act - there is no hidden
# memory, only the list you choose to build.
#
# Assembling and protecting that replayed history is a real engineering
# problem: what should survive when a conversation grows past the
# context window? The fairlib memory system owns that question, and
# tutorial 05 is devoted to it.

# %%
intro = Message(role="user", content="My callsign is Viper-2. Acknowledge in three words.")
ack = llm.invoke([intro])
print("ACK:", ack.content)

# A fresh call on the same llm instance: it just acknowledged Viper-2,
# and it has already forgotten - no state survives between calls.
amnesia = llm.invoke([Message(role="user", content="What is my callsign?")])
print("\nWITHOUT HISTORY:", amnesia.content)

# Same question, but we replay the conversation so far - now it knows.
recall = llm.invoke(
    [
        intro,
        Message(role="assistant", content=ack.content),
        Message(role="user", content="What is my callsign?"),
    ]
)
print("\nWITH HISTORY:", recall.content)

# %% [markdown]
# ### Step 5: the second limit - confidently wrong
#
# The model predicts plausible text; it does not compute, and it cannot
# look anything up. It has no calculator, no clock, and no internet
# connection - only the patterns frozen into its weights the day training
# stopped. Ask it for something outside those patterns and it still
# answers, in the very same confident tone it uses when it is right.
#
# There are two flavors of this, and the next cell shows both:
#
# - **It cannot compute.** Ask for arithmetic that never appeared in its
#   training data and it produces a number-shaped guess. Run the cell a
#   few times and the answer may even change between runs.
# - **It cannot reach live data.** Ask for right-now weather or the
#   current price of Bitcoin and it has no path to the outside world.
#   It will either refuse or - worse for you - invent a specific,
#   plausible-looking figure that is pure fabrication.
#
# This is the single most important thing to internalize before your
# capstone: **fluency is not accuracy**. A security capstone cannot act
# on a hallucinated CVE number, and a telescope capstone cannot log a
# hallucinated magnitude.
#
# The fix is not a bigger model. It is giving the model a way to *act and
# observe*: a calculator it can call, a weather API it can query, a
# database it can read. That capability is called a **tool**, and letting
# the model reach for tools is what turns it into an agent - the subject
# of tutorials 03 and 04.

# %%
# Flavor one: arithmetic it was never trained on. Fluent, and wrong.
a, b = 48193, 90271
guess = llm.invoke(
    [Message(role="user", content=f"What is {a} * {b}? Reply with the number only.")]
)
print("MODEL SAYS: ", guess.content.strip())
print("PYTHON SAYS:", a * b)

# Flavor two: live data it has no way to reach. The model has no internet
# connection, yet it rarely admits that - watch it either refuse or
# invent a confident, specific figure it cannot possibly know.
for live_question in (
    "What is the temperature in Colorado Springs right now?",
    "What is the current price of Bitcoin in US dollars?",
):
    live = llm.invoke([Message(role="user", content=live_question)])
    print(f"\nQ: {live_question}\nMODEL SAYS: {live.content.strip()}")

# %% [markdown]
# ### Debrief: what this means for your capstone
#
# You have seen two limits, live, on your own hardware:
#
# - **Stateless.** Anything the model should know must be assembled and
#   sent on every call. Done by hand that is drudgery, and done wrong it
#   silently loses information - which is why fairlib gives you a memory
#   system in tutorial 05.
# - **Confidently wrong.** For anything that must be correct - math,
#   file contents, live data, running code - the model must be given the
#   ability to *act and observe* instead of guess. That action and
#   observation loop is what turns a model into an agent, and it is the
#   subject of tutorials 03 and 04.
#
# Carry one design win forward as well: because you talked to the model
# through an adapter interface, nothing above the model is bound to a
# provider. That seam is the backbone of the whole framework.
#
# **Capstone connection.** The capstone template your team will be given
# builds against this same adapter seam, including a worked example of
# bringing a provider fairlib does not ship. Whatever model your project
# ends up needing, what you build in the next tutorials will not have to
# change.
#
# **Next:** tutorial `02_prompts_that_hold` covers the prompt
# engineering system and how to make a model's output structured enough
# for real code to consume.
