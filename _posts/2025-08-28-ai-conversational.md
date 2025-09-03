---
layout: default
title: "My Experience in Building a Conversational AI Morning Planner"
date: 2025-08-25 14:37:33
categories: [ai]
tags: [chat,bot,nlp]
thumbnail: /assets/images/ai_conversational.webp
featured: true
---
I wanted to build something lightweight yet useful—a conversational morning planner that runs locally and feels personal. The idea was simple:

*   A **Planner tab** where I could fill in some details and get a neat morning plan.
    
*   A **Chat tab** where I could type natural language like _“run 20 minutes this evening”_ and the bot would update my plan instantly.
    

And to make it smarter, I added a compact embedding model that picks a **relevant focus tip** for my goals. No API keys, no heavy infrastructure.

<iframe
	src="https://rahulbhattacharya-rahuls-morning-planner.hf.space"
style="width:100%;height:820px;border:0;border-radius:12px;overflow:hidden"></iframe>

Demo Goals
----------

*   **Planner tab**: fill a simple form → get a structured morning plan.
    
*   **Chat tab**: type natural phrases (like _“walk fast only”_) → the bot updates activity, intensity, duration, and time.
    
*   A small AI embedding layer → suggests the most relevant focus tip.
    

Requirements
------------

I created a requirements.txt with the basics:

```python
gradio>=4.0.0
torch
sentence-transformers
```
When building a lightweight conversational planner, I wanted to make setup and deployment as simple as possible. That’s why I created a requirements.txt file to lock in the essential dependencies. This file tells any environment—whether it’s my laptop, a cloud runtime, or Hugging Face Spaces—exactly which Python libraries are needed to run the app.

I kept the list short and purposeful. The first dependency, Gradio (>=4.0.0), is the framework that powers the user interface. It makes it easy to design the Planner and Chat tabs with textboxes, dropdowns, and buttons, while handling state and chat history without much boilerplate code. Next is Torch, which provides the deep learning backend required by modern models. Even though the planner is lightweight, Torch is the foundation that allows the embedding model to run locally. Finally, I included sentence-transformers, the library that gives me access to compact, high-quality embedding models like all-MiniLM-L6-v2. This is what makes the app “smart,” enabling it to interpret natural language and suggest the most relevant focus tip.

By limiting requirements to just these essentials, I ensured the app is easy to install, reproducible, and deployable without unnecessary overhead.

Folder Layout
-------------

Just two files were enough:

```python
.
├── app.py
└── requirements.txt
```
To keep things simple, I structured the entire project with just two files: app.py and requirements.txt. This minimal setup was intentional. The app.py file contains all of the application logic—imports, the embedding model, helper functions, the planner generator, the conversational parser, and finally the Gradio UI. In other words, it’s the single script that defines how both the Planner and Chat tabs work, from parsing natural language to rendering the interface.

The second file, requirements.txt, is equally important. It lists the exact Python dependencies the app needs (gradio, torch, and sentence-transformers). When anyone installs or deploys the project, this file ensures the same libraries and versions are pulled in automatically, making the build consistent across environments.

This minimal layout also makes deployment to Hugging Face Spaces painless. Spaces only needs to see an app.py file to know what to execute, and a requirements.txt file to know what to install. Once uploaded, I just had to set app.py as the app file in the Space settings, and everything worked out of the box. By avoiding extra files and folders, the project stays lightweight, portable, and easy to share.

How It Works (High-Level)
-------------------------

1.  **Data & Model**
    
    *   I defined a set of short, practical **focus tips**.
        
    *   Loaded a small embedding model: sentence-transformers/all-MiniLM-L6-v2.
        
    *   Precomputed embeddings for tips so matching was instant.
        
2.  **Planner tab**
    
    *   Input fields: name, timezone, energy, wake time, top priorities, note.
        
    *   If AI toggle was on, embeddings picked the best tip.
        
    *   Output was a Markdown plan.
        
3.  **Chat tab**
    
    *   Used regex + synonyms for parsing, with embeddings as fallback.
        
    *   Maintained state for activity, intensity, duration, time of day, energy, notes.
        
    *   Responded with friendly acknowledgments and follow-ups.
        

The Code, Piece by Piece
------------------------

### 1) Imports, Constants, and the Embedding Model

```python
import os, re, random, textwrap
from datetime import datetime, date
from zoneinfo import ZoneInfo
import gradio as gr
from sentence_transformers import SentenceTransformer, util

FOCUS_TIPS = [
"Work in 25-minute Focus sprints (Pomodoro), then 5-minute breaks.",
"Silence notifications for 60 minutes and batch-check messages afterward.",
"Write your top 3 outcomes for the day before opening email.",
"If it takes <2 minutes, do it now—otherwise schedule it.",
"Tackle the hardest task first while energy is highest.",
"Stand up, stretch, sip water every hour to keep energy up.",
"Use a single-tab rule while doing deep work.",
]

ENERGY_HINTS = {
"Low": "Keep the first block light. Add a short walk + 5 deep breaths before starting.",
"Medium": "Start with a quick win task to build momentum, then move to your #1 priority.",
"High": "Go straight to your hardest task for 60–90 minutes—protect this time!",
}

COMMON_TIMEZONES = [
"America/Chicago","America/New_York","America/Los_Angeles",
"Europe/London","Europe/Berlin","Europe/Paris",
"Asia/Kolkata","Asia/Singapore","Asia/Tokyo","Australia/Sydney"
]
```
The first part of the code sets up the foundation for the planner by importing essential libraries and defining key constants. The imports are deliberately lightweight: os, re, random, and textwrap handle system tasks, regex parsing, randomness, and neat formatting. From the standard library, datetime and zoneinfo ensure the app can display time correctly in different user-selected time zones. On top of that, I use Gradio for the user interface and sentence-transformers for embedding-based intelligence.

Next, I defined **FOCUS_TIPS**, a curated list of short, practical reminders that users might find helpful while planning their day—things like Pomodoro breaks, single-tasking, or protecting deep work time. These tips form the “knowledge base” that the AI embedding model will later match against the user’s stated goals.

The **ENERGY_HINTS** dictionary provides context-sensitive suggestions depending on how energized the user feels (Low, Medium, High). This makes the plan more personalized, since productivity advice should adapt to the user’s state.

Finally, COMMON_TIMEZONES is a small set of popular time zones. This ensures users can localize their plan no matter where they are, without scrolling through an overwhelming list of every possible timezone. Together, these constants act as the backbone of the planner’s personalization.

```python
_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
_TIP_EMB = _MODEL.encode(FOCUS_TIPS, normalize_embeddings=True)
```
At this stage, I load the embedding model and prepare it for fast, repeated use. I chose sentence-transformers/all-MiniLM-L6-v2, a compact but powerful embedding model that strikes a good balance between accuracy and performance. At around 80–90 MB, it’s small enough to run locally or in free-tier environments like Hugging Face Spaces, while still providing high-quality semantic understanding.

Once the model is loaded, I immediately precompute embeddings for the list of **FOCUS_TIPS**. Embeddings are vector representations that capture the meaning of each tip. By computing these vectors just once at startup, I avoid recalculating them every time the user interacts with the app. This drastically improves responsiveness—when a user types in their goals, the app only needs to embed that single query and compare it to the pre-stored tip embeddings, rather than embedding all tips again.

The normalization step (normalize_embeddings=True) ensures that cosine similarity scores are meaningful and consistent across comparisons. With everything preprocessed, the system can instantly suggest the most relevant tip that matches the user’s context or goals. This setup turns a static list of tips into a smart, AI-driven recommendation engine with virtually no runtime overhead.

### 2) Minimal AI Helper

```python
def _ai_pick_best(items_emb, query_text):
if not query_text or not query_text.strip():
return None
q_emb = _MODEL.encode(query_text, normalize_embeddings=True)
scores = util.cos_sim(q_emb, items_emb)[0]
return int(scores.argmax())
```
The function _ai_pick_best is a compact utility that powers the “intelligence” in the planner. Its job is simple: given a list of precomputed embeddings (such as the FOCUS_TIPS) and a user’s input, it figures out which option best matches the meaning of the query.

The function begins with a quick safeguard: if the input text is empty or only whitespace, it immediately returns None. This avoids unnecessary computation and prevents errors when the user hasn’t typed anything meaningful.

If there is valid input, the function encodes it into an embedding using the same model we loaded earlier (_MODEL). By normalizing the embeddings, we ensure that all comparisons are consistent. Next, it computes cosine similarity between the query embedding and the list of stored embeddings. Cosine similarity is a measure of how close two vectors are in direction, which translates to how similar two pieces of text are in meaning.

Finally, the function returns the index of the highest-scoring match using argmax(). This index corresponds to whichever tip, activity, or energy level is most semantically aligned with the user’s request.

In just a few lines, this helper makes the system context-aware, allowing static lists to feel interactive and personalized.
### 3) Planner: Generate a Morning Plan

```python
def generate_plan(name, tz, energy, wake_time, primary_goal, secondary_goal, extra_note, use_ai):
    seed = int(date.today().strftime("%Y%m%d"))
    random.seed(seed)

    intent = " ".join([s for s in [primary_goal or "", secondary_goal or "", extra_note or ""] if s]).strip()

    if use_ai and intent:
        t_idx = _ai_pick_best(_TIP_EMB, intent)
        tip = FOCUS_TIPS[t_idx]
        ai_badge = " (AI-personalized)"
    else:
        tip = random.choice(FOCUS_TIPS)
        ai_badge = ""

    now_local = datetime.now(ZoneInfo(tz))
    time_str = now_local.strftime("%I:%M %p").lstrip("0")

    blocks = []
    if wake_time:
        blocks.append(f"☀️ **{wake_time}** — Wake, hydrate (250 ml), 5 deep breaths, quick stretch")
    blocks.append("📓 **Morning Prime (10 min)** — Write top 3 outcomes & clear desk")
    blocks.append(f"🎯 **Deep Work (60–90 min)** — Focus on: **{primary_goal or 'your #1 task'}**")
    if secondary_goal:
        blocks.append(f"🔁 **Block 2 (45–60 min)** — Next: **{secondary_goal}**")
    blocks.append("🍎 **Micro-break (5 min)** — Stand up, sip water, quick walk")
    if extra_note:
        blocks.append(f"📝 **Note** — {extra_note}")

    plan_md = "\n".join([f"- {b}" for b in blocks])
    energy_md = ENERGY_HINTS.get(energy, "")
    greeting = f"Good morning{f', {name}' if name else ''}! It’s **{time_str}** in **{tz}**."

    body = f"""
**Today’s vibe:** {energy} energy{ai_badge}  
**Focus tip:** {tip}
### Your Morning Plan
{plan_md}
**Pro tip:** {energy_md}
"""
    return textwrap.dedent(greeting + "\n\n" + body).strip()
```
The generate_plan function is the heart of the structured planner tab. It takes in user inputs—such as name, timezone, energy level, wake time, and top priorities—and returns a clean, Markdown-formatted morning plan.

The function starts by setting a random seed based on today’s date, ensuring that any “random” selections (like fallback tips) stay consistent for that day, but vary across different days. It then constructs an intent string by combining the user’s goals and notes. If AI is enabled and there’s meaningful input, the system uses _ai_pick_best to select the most relevant focus tip. Otherwise, it randomly picks a tip from the list.

Next, the function localizes the time using the user’s chosen timezone, giving the plan a personal touch with a timestamped greeting. It then builds a list of activity blocks—wake-up routine, journaling, deep work, secondary goals, breaks, and optional notes. Each block is formatted with friendly icons and Markdown for easy readability.

The output also integrates energy-based guidance from ENERGY_HINTS, so the advice feels adapted to the user’s current state. Finally, everything is combined into a Markdown string that Gradio can render beautifully in the UI.

This makes the planner both structured and personalized, while still lightweight.
### 4) Conversational Parsing (Chat)

```python
INTENTS = [
"run","walk","yoga","strength","cycle","meditate",
"stretch","read","write","code","study","rest"
]

_INTENT_EMB = _MODEL.encode(INTENTS, normalize_embeddings=True)

_ENERGY_WORDS = ["low","medium","high"]

_ENERGY_EMB = _MODEL.encode(_ENERGY_WORDS, normalize_embeddings=True)
ACTIVITY_SYNS = {
"run": ["run","jog","jogging","tempo"],
"walk": ["walk","brisk walk","stroll"],
"yoga": ["yoga","sun salutation"],
"strength": ["strength","weights","lift","lifting","resistance"],
"cycle": ["cycle","bike","biking","cycling"],
"meditate": ["meditate","meditation","mindfulness"],
"stretch": ["stretch","mobility"],
"read": ["read","reading"],
"write": ["write","writing","journal"],
"code": ["code","coding","program"],
"study": ["study","learn"],
"rest": ["rest","recover","recovery","off"]
}

WORD2ACT = {w: act for act, words in ACTIVITY_SYNS.items() for w in words}
```
This section gives the chat its “understanding” of everyday phrases by combining simple lists with embeddings. I start by defining a compact set of INTENTS (run, walk, yoga, strength, etc.). These are the canonical activities the bot can act on. I then precompute INTENT embeddings (_INTENT_EMB) so that later, a user’s free-form message can be compared semantically to these anchors using cosine similarity. The same approach is used for energy words (low/medium/high), enabling lightweight sentiment-like interpretation without a heavy NLU stack.

To make the parser robust to real phrasing, I add an ACTIVITY_SYNS dictionary that maps each canonical intent to common synonyms (“jog” → run, “bike” → cycle, “journal” → write, etc.). From that, WORD2ACT is built to provide an O(1) lookup from any seen word to its canonical activity. At runtime, the bot tries exact word matches first (fast and precise), and only then falls back to embeddings (flexible and semantic). This two-step strategy keeps responses snappy while handling typos, style differences, and varied vocab.

Overall, these small, precomputed structures let the chat feel smart and forgiving—able to interpret “tempo jog,” “brisk walk,” or “coding session”—without heavyweight NLP pipelines or cloud dependencies.
### Extraction Helpers

```python
def _extract_numbers_minutes(text):

m = re.search(r'(\d+)\s*(?:min|mins|minute|minutes)\b', text, re.I)
if m: return max(5, min(180, int(m.group(1))))
m = re.search(r'\bfor\s+(\d{1,3})\b', text, re.I)
if m: return max(5, min(180, int(m.group(1))))
m = re.search(r'\b(\d{1,3})\b', text)
if m: return max(5, min(180, int(m.group(1))))

return None
def _extract_time_of_day(text):

t = text.lower()

if "morning" in t: return "morning"
if "noon" in t or "midday" in t: return "midday"
if "afternoon" in t: return "afternoon"
if "evening" in t or "tonight" in t or "night" in t: return "evening"
if "today" in t: return None

return None
def _choose_intensity(text):

t = text.lower()

if any(w in t for w in ["slow","light","easy","casual"]): return "easy"
if any(w in t for w in ["fast","brisk","hard","intense","tempo","interval"]): return "hard"
if any(w in t for w in ["moderate","steady"]): return "moderate"

return None
```
To make the chatbot conversational, it needs to interpret free-form text and pull out structured details like duration, time of day, or workout intensity. That’s where these helper functions come in. They use lightweight regex patterns and keyword checks to translate natural language into parameters the planner can use.

The first function, _extract_numbers_minutes, looks for numbers that represent duration. It supports multiple formats: explicit mentions like “20 minutes,” shorthand like “20 min,” or even phrases like “for 45.” If none of those patterns match, it falls back to catching a standalone number. To keep results realistic, it clamps values between 5 and 180 minutes.

The second, _extract_time_of_day, detects when the activity should happen. It searches for keywords such as “morning,” “afternoon,” or “tonight,” and maps them to a simple label. If the user says “today” without being specific, it leaves the field unset so the bot can ask a clarifying question later.

Finally, _choose_intensity interprets words that describe pace or effort. “Slow” or “easy” is mapped to easy, while “fast,” “tempo,” or “intense” is mapped to hard, with “moderate” or “steady” falling in between.

Together, these helpers allow the bot to extract meaning from casual language without complex NLP pipelines.
### Activity Matching

```python
def _match_activity(text):

t = text.lower()

for w in WORD2ACT:
if re.search(rf'\b{re.escape(w)}\b', t):
return WORD2ACT[w]

idx = _ai_pick_best(_INTENT_EMB, t)
return INTENTS[idx] if idx is not None else None
```
The _match_activity function is responsible for figuring out which activity the user is talking about, even if they phrase it in different ways. It works in two stages: exact word matching first, embeddings second.

When a message comes in, the text is converted to lowercase for consistent comparisons. The function then loops through every known synonym in WORD2ACT (built earlier from the activity synonyms). If it finds an exact word match—for example, the user types “jogging”—the function quickly maps it to its canonical activity, “run.” This ensures fast and reliable recognition when the input matches one of the defined keywords.

If no direct match is found, the function falls back to semantic similarity using embeddings. It encodes the user’s message, compares it against the precomputed intent embeddings (_INTENT_EMB), and selects the intent with the highest similarity score. This second step makes the system more flexible, since it can still recognize activities from phrasing that isn’t explicitly listed in the dictionary, such as “tempo session” being understood as “run.”

By blending precise keyword detection with embedding-based reasoning, _match_activity makes the chatbot both accurate and forgiving, handling everything from clean inputs to fuzzy, casual language.
### Friendly Acknowledgments & Follow-Ups

```python
def _make_ack(s):
bits = []
if s.get("activity"): bits.append(f"{s['activity']}")
if s.get("intensity"): bits.append(f"{s['intensity']}")
if s.get("duration"): bits.append(f"{s['duration']} min")
if s.get("when"): bits.append(f"in the {s['when']}")
if s.get("energy"): bits.append(f"energy {s['energy']}")
return " • ".join(bits) if bits else "noted"

def _next_question(s):
if not s.get("duration"): return "How many minutes are you thinking? ⏱️"
if not s.get("intensity"): return "Easy, moderate, or hard pace? ⚖️"
if not s.get("when"): return "When would you like to do it—morning, afternoon, or evening? 🕒"
return None
```
These two helper functions make the chatbot feel conversational and supportive instead of robotic. They generate natural feedback after parsing the user’s message, confirming what was understood and guiding the conversation toward missing details.

The first function, _make_ack, assembles a short acknowledgment string based on the current state dictionary. If the user has already specified an activity, intensity, duration, time of day, or energy level, those elements are appended to a list and joined with “•” separators. For example, if the state contains {"activity": "run", "duration": 20, "when": "evening"}, the function would return “run • 20 min • in the evening”. If nothing meaningful was extracted, it defaults to simply returning “noted,” keeping the flow graceful.

The second function, _next_question, ensures the conversation doesn’t stall. It checks for missing fields in order: duration, intensity, then time of day. If one is absent, it returns a friendly follow-up prompt like “How many minutes are you thinking? ⏱️” or “Easy, moderate, or hard pace? ⚖️”. If all the key details are present, it returns None, signaling that no further clarification is needed.

Together, these helpers make the chatbot interactive, encouraging users to naturally fill in gaps while feeling understood.
### State Updater

```python
def _update_state_from_msg(state, msg):
    s = dict(state)
    t = msg.strip()

    if re.search(r'\b(reset|start over|clear)\b', t, re.I):
        return {
            "activity": None, "intensity": None, "duration": None,
            "when": None, "energy": None, "notes": ""
        }, "All set to start fresh! What do you feel like doing today? 🙂"

    act = _match_activity(t)
    if act: s["activity"] = act

    inten = _choose_intensity(t)
    if inten: s["intensity"] = inten

    dur = _extract_numbers_minutes(t)
    if dur: s["duration"] = dur

    when = _extract_time_of_day(t)
    if when: s["when"] = when

    if re.search(r'\blow\b', t): s["energy"] = "Low"
    elif re.search(r'\bmedium\b', t): s["energy"] = "Medium"
    elif re.search(r'\bhigh\b', t): s["energy"] = "High"

    if re.search(r'\bonly\b', t) and act:
        s["intensity"] = s.get("intensity") or "moderate"

    used = set()
    if act: used |= set(ACTIVITY_SYNS.get(act, [])) | {act}
    if inten: used.add(inten)
    if dur: used.add(str(dur))
    if when: used.add(when)
    if s.get("energy"): used.add(s["energy"].lower())

    leftover_words = [w for w in re.findall(r"\b\w+\b", t.lower()) if w not in used]
    leftover = " ".join(leftover_words).strip()
    if leftover:
        s["notes"] = (s.get("notes", "") + " " + leftover).strip()[:500]

    ack = _make_ack(s)
    ask = _next_question(s)

    opener = random.choice([
        "Got it! 🙌",
        "Nice choice! 💪",
        "Sounds good! ✅",
        "On it! 🚀",
    ])
    text_ack = f"{opener} I captured: {ack}."
    if ask: text_ack += f" {ask}"

    return s, text_ack
```
The _update_state_from_msg function is the engine that keeps the chatbot’s memory and conversation flow consistent. Every time the user sends a new message, this function updates the state dictionary (which stores activity, intensity, duration, time of day, energy, and notes) and generates a natural reply.

It starts by checking for reset commands like “reset” or “start over”. If found, it clears the state and returns a friendly fresh-start prompt. Otherwise, it systematically extracts details: _match_activity detects the activity, _choose_intensity sets pace, _extract_numbers_minutes parses duration, and _extract_time_of_day anchors when. Simple regex checks also detect energy levels (“low,” “medium,” “high”).

A neat touch is the handling of the word “only.” If the user says “walk only,” the bot assumes a default “moderate” intensity to avoid ambiguity.

To avoid discarding extra words, the function builds a notes field by collecting leftover text not already matched to an activity, duration, or intensity. This preserves context like “with a friend” or “near the park.”

Finally, the bot crafts a warm acknowledgment using _make_ack and _next_question, adding a random opener like “On it! 🚀” to keep things lively.

This makes the planner feel adaptive, memory-driven, and conversational.
### Suggestion Builder

```python
def _suggest_plan_from_state(state):
    activity = state.get("activity") or "walk"
    intensity = state.get("intensity") or "moderate"
    duration = state.get("duration") or 30
    when = state.get("when") or "morning"

    tip_idx = _ai_pick_best(_TIP_EMB, f"{activity} {intensity} {duration}min {when}")
    tip = FOCUS_TIPS[tip_idx] if tip_idx is not None else random.choice(FOCUS_TIPS)

    plan = [
        f"🕒 **{when.title()}** — {activity.title()} ({intensity}) for **{duration} min**",
    ]
    if state.get("notes"):
        plan.append(f"📝 Note — {state['notes'][:200]}")

    return "\n".join([f"- {p}" for p in plan]) + f"\n\n**Tip:** {tip}"
```
The _suggest_plan_from_state function turns the current conversation state into a clear, actionable plan that feels both structured and personal. It works by filling in defaults when the user hasn’t specified enough details, ensuring the bot always has something useful to say. For example, if no activity is chosen, it defaults to walk; if intensity is missing, it assumes moderate; if duration isn’t set, it defaults to 30 minutes; and if time of day isn’t provided, it defaults to morning.

Once these values are established, the function constructs a query string like “run hard 20min evening” and uses _ai_pick_best to select the most relevant tip from FOCUS_TIPS. This keeps the advice aligned with the user’s context rather than being a generic suggestion. If the AI can’t make a confident match, it falls back to a random tip, so the output never feels empty.

The plan itself is formatted in Markdown with emojis for a friendly touch. For example:

“🕒 Evening — Run (hard) for 20 min”

If the user has left extra notes in the conversation, those are included as well. Finally, the plan concludes with a highlighted Tip, tying together structure and motivation.
### Chat Core

```python
def chat_core(message, state, tz):
    if state is None or not isinstance(state, dict):
        state = {"activity": None, "intensity": None, "duration": None, "when": None, "energy": None, "notes": ""}

    first_time = not any(state.values())
    new_state, ack = _update_state_from_msg(state, message)
    suggestion = _suggest_plan_from_state(new_state)

    now_local = datetime.now(ZoneInfo(tz))
    tstr = now_local.strftime("%I:%M %p").lstrip("0")

    if first_time:
        pre = random.choice([
            "Hey! I’m your planning buddy. Tell me what you feel like doing and I’ll shape a quick plan. 🙂",
            "Hi there! Share a goal like “run 20 minutes in the evening” and I’ll sort it out. 🧭",
        ])
        reply = f"{pre}\n\n{ack}\n\n_Current time in **{tz}**: **{tstr}**_\n\n{suggestion}"
    else:
        reply = f"{ack}\n\n_Current time in **{tz}**: **{tstr}**_\n\n{suggestion}"

    return reply, new_state
```
The chat_core function is the central loop that powers the conversational experience. Every time the user sends a message, this function takes the text, updates the internal state, and produces a friendly, context-aware reply.

It begins by checking whether a valid state exists; if not, it initializes one with empty values for activity, intensity, duration, time, energy, and notes. It then flags whether this is the first interaction (first_time) by checking if the state has any stored values. This distinction helps the bot decide whether to include a warm introduction or continue an ongoing flow.

Next, the function calls _update_state_from_msg to extract structured details (activity, minutes, energy, etc.) and receive a natural acknowledgment. It then passes the updated state to _suggest_plan_from_state, which generates a formatted mini-plan with an AI-picked tip.

To personalize the reply further, the function includes the current local time based on the user’s selected timezone. This reinforces the planner’s “right now” relevance.

Finally, the response is assembled: if it’s the first interaction, a random friendly opener introduces the bot as a planning buddy; otherwise, it just confirms the update and shows the plan. The function then returns both the reply text and the updated state, keeping the conversation flowing naturally.
### 5) Gradio UI Wiring

```python
with gr.Blocks(title="Good Morning AI Bot", css="#app {max-width: 900px; margin: auto;} footer {visibility: hidden;}") as demo:
    gr.Markdown("# 🌅 Good Morning AI Bot\nPlan your morning or just tell me what you feel like doing.")
    with gr.Tabs():
        with gr.Tab("Planner"):
            with gr.Row():
                with gr.Column():
                    name = gr.Textbox(label="Your name (optional)", placeholder="Rahul")
                    tz = gr.Dropdown(choices=COMMON_TIMEZONES, value="America/Chicago", label="Your timezone")
                    energy = gr.Radio(choices=["Low","Medium","High"], value="Medium", label="How’s your energy?")
                    wake_time = gr.Textbox(label="Wake-up time (optional)", placeholder="6:30 AM")
                with gr.Column():
                    primary = gr.Textbox(label="#1 priority today", placeholder="Finish blog post draft")
                    secondary = gr.Textbox(label="#2 priority (optional)", placeholder="30-min coding practice")
                    note = gr.Textbox(label="Note (optional)", placeholder="Gym at 6 PM, call mom")
                    use_ai = gr.Checkbox(value=True, label="AI personalize quote & tip")
                    generate_btn = gr.Button("Generate Morning Plan", variant="primary")
            out = gr.Markdown()
            generate_btn.click(
                fn=generate_plan,
                inputs=[name, tz, energy, wake_time, primary, secondary, note, use_ai],
                outputs=out
            )

        with gr.Tab("Chat"):
            chat_tz = gr.Dropdown(choices=COMMON_TIMEZONES, value="America/Chicago", label="Your timezone")
            state = gr.State({"activity": None, "intensity": None, "duration": None, "when": None, "energy": None, "notes": ""})
            chatbot = gr.Chatbot(height=420)
            msg = gr.Textbox(placeholder="e.g., I want to run 20 minutes this evening", label="Message")
            send = gr.Button("Send", variant="primary")
            clear = gr.Button("Reset")

            def _ui_chat(user_msg, history, cur_state, tz_):
                history = history or []
                reply, new_state = chat_core(user_msg, cur_state, tz_)
                history = history + [(user_msg, reply)]
                return history, new_state

            send.click(_ui_chat, inputs=[msg, chatbot, state, chat_tz], outputs=[chatbot, state])
            msg.submit(_ui_chat, inputs=[msg, chatbot, state, chat_tz], outputs=[chatbot, state])
            clear.click(lambda: ([], {"activity": None, "intensity": None, "duration": None, "when": None, "energy": None, "notes": ""}), outputs=[chatbot, state])
```
The Gradio UI is where all the backend logic connects to an interactive interface that anyone can use without touching code. I wrapped the app in a gr.Blocks container, which provides a flexible layout system and allows custom CSS styling (in this case, limiting max width and hiding the footer for a cleaner look).

Inside, I defined two tabs: Planner and Chat. The Planner tab organizes inputs into two columns. On the left, the user sets personal context—name, timezone, energy level, and wake-up time. On the right, they enter priorities and notes, plus a toggle to enable AI personalization. A “Generate Morning Plan” button triggers the generate_plan function, and the structured Markdown output is displayed immediately below.

The Chat tab creates a more conversational flow. It includes a timezone dropdown, a persistent state object to track progress, a chat window (gr.Chatbot) for back-and-forth messages, and a textbox for typing inputs. The Send button and pressing Enter both call _ui_chat, which routes the message through chat_core, updates the state, and appends the reply to the history. A Reset button clears everything so users can start fresh.

This wiring ties the intelligence of the planner to an approachable, user-friendly interface—ideal for both structured planning and natural language interaction.
### 6) Launching

```python
if os.environ.get("SPACE_ID"):

demo.launch()

else:

if name == "main":

demo.launch()
```
The final piece of the app is making sure it runs smoothly both locally and when deployed on Hugging Face Spaces. That’s what the launch block handles.

First, the code checks whether the environment variable SPACE_ID is set. This variable is automatically present when the app is running inside Hugging Face Spaces. If it exists, the app simply calls demo.launch(), which starts the Gradio interface inside the hosted Space. Users can then interact with the planner directly from their browser, without installing anything.

If SPACE_ID is not found, it means the app is being run locally on a personal machine. In that case, the familiar Python entry-point check if __name__ == "__main__": is used. This ensures that the app only launches when executed directly, and not when imported as a module. Calling demo.launch() here starts a local Gradio server, usually on http://127.0.0.1:7860/, allowing the user to test the planner in their own browser.

This dual setup keeps the project flexible: the exact same code works whether you’re testing locally during development or deploying live for others. No extra configuration or branching scripts are needed.

Example Chats
-------------

**You:** I want to run today **Bot:** Got it! I captured: run. How many minutes are you thinking?

**You:** make it 20 minutes, evening, fast only **Bot:** On it! I captured: run • hard • 20 min • in the evening.

**You:** walk only **Bot:** Sounds good! I captured: walk • hard • 20 min • in the evening.

**You:** reset **Bot:** All set to start fresh! What do you feel like doing today? 🙂

To show how the planner feels in practice, I included a few sample conversations. These highlight how the chatbot listens, extracts details, and gently asks for missing information.

In the first example, the user says: “I want to run today.” The bot recognizes the activity (run) but notices that duration and intensity are missing. Instead of making assumptions, it politely follows up with: “How many minutes are you thinking? ⏱️” This demonstrates the incremental way the bot builds a complete plan.

Next, the user refines their request: “make it 20 minutes, evening, fast only.” The bot now captures four details at once—activity (run), duration (20 min), time of day (evening), and intensity (hard). It responds with a confident acknowledgment: “On it! I captured: run • hard • 20 min • in the evening.”

Another example shows shorthand phrasing: “walk only.” Even though vague, the bot infers a default duration and intensity, keeping the flow moving without confusion.

Finally, the user types “reset.” The bot clears all state and replies warmly: “All set to start fresh! What do you feel like doing today? 🙂” This makes it easy to restart planning anytime.

These conversations showcase the balance of structure, flexibility, and friendliness that makes the tool engaging.

Closing Thoughts
------------

*   Make the chat the default if you prefer a fully conversational planner.
    
*   Add custom synonyms to extend activities.
    
*   Adapt Planner blocks to match your daily rhythm.
    
*   The model is ~80–90 MB and downloads once, so subsequent runs are fast.

If you deploy on **Hugging Face Spaces**, set the app file to app.py in Space settings.
I wrapped up the project with a few practical takeaways that make the planner easier to adapt and extend. First, if you prefer a more natural, back-and-forth interaction, you can simply make the Chat tab the default view. This gives the experience of a personal planning assistant that understands free-form instructions, rather than relying on form inputs.

Second, the design is flexible enough for customization. You can add new synonyms in the activity dictionary so the bot recognizes your own wording—whether that’s “swim,” “hike,” or “dance.” Similarly, the Planner blocks can be adjusted to match your personal morning flow: swap in meditation instead of journaling, remove sections you don’t use, or add new ones like breakfast or commuting prep.

On the performance side, the embedding model is compact (~80–90 MB). It downloads only once, and after that it loads instantly, keeping the tool responsive on repeat runs. This makes it practical for everyday use, even on lightweight setups.

Finally, deployment is straightforward. On Hugging Face Spaces, just ensure the app file is set to app.py in the Space settings. With that, your conversational planner is live and shareable with anyone through a browser link.

Full source (app.py)
--------------------

> This is the exact, complete code discussed above.

```python
import os
import re
import random
import textwrap
from datetime import datetime, date
from zoneinfo import ZoneInfo

import gradio as gr
from sentence_transformers import SentenceTransformer, util

FOCUS_TIPS = [
    "Work in 25-minute Focus sprints (Pomodoro), then 5-minute breaks.",
    "Silence notifications for 60 minutes and batch-check messages afterward.",
    "Write your top 3 outcomes for the day before opening email.",
    "If it takes <2 minutes, do it now—otherwise schedule it.",
    "Tackle the hardest task first while energy is highest.",
    "Stand up, stretch, sip water every hour to keep energy up.",
    "Use a single-tab rule while doing deep work.",
]

ENERGY_HINTS = {
    "Low": "Keep the first block light. Add a short walk + 5 deep breaths before starting.",
    "Medium": "Start with a quick win task to build momentum, then move to your #1 priority.",
    "High": "Go straight to your hardest task for 60–90 minutes—protect this time!",
}

COMMON_TIMEZONES = [
    "America/Chicago", "America/New_York", "America/Los_Angeles",
    "Europe/London", "Europe/Berlin", "Europe/Paris",
    "Asia/Kolkata", "Asia/Singapore", "Asia/Tokyo", "Australia/Sydney"
]

_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
_TIP_EMB = _MODEL.encode(FOCUS_TIPS, normalize_embeddings=True)

def _ai_pick_best(items_emb, query_text):
    if not query_text or not query_text.strip():
        return None
    q_emb = _MODEL.encode(query_text, normalize_embeddings=True)
    scores = util.cos_sim(q_emb, items_emb)[0]
    return int(scores.argmax())

def generate_plan(name, tz, energy, wake_time, primary_goal, secondary_goal, extra_note, use_ai):
    seed = int(date.today().strftime("%Y%m%d"))
    random.seed(seed)

    intent = " ".join([s for s in [primary_goal or "", secondary_goal or "", extra_note or ""] if s]).strip()

    if use_ai and intent:
        t_idx = _ai_pick_best(_TIP_EMB, intent)
        tip = FOCUS_TIPS[t_idx]
        ai_badge = " (AI-personalized)"
    else:
        tip = random.choice(FOCUS_TIPS)
        ai_badge = ""

    now_local = datetime.now(ZoneInfo(tz))
    time_str = now_local.strftime("%I:%M %p").lstrip("0")

    blocks = []
    if wake_time:
        blocks.append(f"☀️ **{wake_time}** — Wake, hydrate (250 ml), 5 deep breaths, quick stretch")
    blocks.append("📓 **Morning Prime (10 min)** — Write top 3 outcomes & clear desk")
    blocks.append(f"🎯 **Deep Work (60–90 min)** — Focus on: **{primary_goal or 'your #1 task'}**")
    if secondary_goal:
        blocks.append(f"🔁 **Block 2 (45–60 min)** — Next: **{secondary_goal}**")
    blocks.append("🍎 **Micro-break (5 min)** — Stand up, sip water, quick walk")
    if extra_note:
        blocks.append(f"📝 **Note** — {extra_note}")

    plan_md = "\n".join([f"- {b}" for b in blocks])
    energy_md = ENERGY_HINTS.get(energy, "")
    greeting = f"Good morning{f', {name}' if name else ''}! It’s **{time_str}** in **{tz}**."

    body = f"""
**Today’s vibe:** {energy} energy{ai_badge}  
**Focus tip:** {tip}

### Your Morning Plan
{plan_md}

**Pro tip:** {energy_md}
"""
    return textwrap.dedent(greeting + "\n\n" + body).strip()

INTENTS = [
    "run", "walk", "yoga", "strength", "cycle", "meditate",
    "stretch", "read", "write", "code", "study", "rest"
]
_INTENT_EMB = _MODEL.encode(INTENTS, normalize_embeddings=True)
_ENERGY_WORDS = ["low", "medium", "high"]
_ENERGY_EMB = _MODEL.encode(_ENERGY_WORDS, normalize_embeddings=True)

ACTIVITY_SYNS = {
    "run": ["run", "jog", "jogging", "tempo"],
    "walk": ["walk", "brisk walk", "stroll"],
    "yoga": ["yoga", "sun salutation"],
    "strength": ["strength", "weights", "lift", "lifting", "resistance"],
    "cycle": ["cycle", "bike", "biking", "cycling"],
    "meditate": ["meditate", "meditation", "mindfulness"],
    "stretch": ["stretch", "mobility"],
    "read": ["read", "reading"],
    "write": ["write", "writing", "journal"],
    "code": ["code", "coding", "program"],
    "study": ["study", "learn"],
    "rest": ["rest", "recover", "recovery", "off"]
}
WORD2ACT = {w: act for act, words in ACTIVITY_SYNS.items() for w in words}

def _extract_numbers_minutes(text):
    m = re.search(r'(\d+)\s*(?:min|mins|minute|minutes)\b', text, re.I)
    if m:
        v = int(m.group(1))
        return max(5, min(180, v))
    m = re.search(r'\bfor\s+(\d{1,3})\b', text, re.I)
    if m:
        v = int(m.group(1))
        return max(5, min(180, v))
    m = re.search(r'\b(\d{1,3})\b', text)
    if m:
        v = int(m.group(1))
        return max(5, min(180, v))
    return None

def _extract_time_of_day(text):
    t = text.lower()
    if "morning" in t: return "morning"
    if "noon" in t or "midday" in t: return "midday"
    if "afternoon" in t: return "afternoon"
    if "evening" in t or "tonight" in t or "night" in t: return "evening"
    if "today" in t: return None
    return None

def _choose_intensity(text):
    t = text.lower()
    if any(w in t for w in ["slow", "light", "easy", "casual"]): return "easy"
    if any(w in t for w in ["fast", "brisk", "hard", "intense", "tempo", "interval"]): return "hard"
    if any(w in t for w in ["moderate", "steady"]): return "moderate"
    return None

def _match_activity(text):
    t = text.lower()
    for w in WORD2ACT:
        if re.search(rf'\b{re.escape(w)}\b', t):
            return WORD2ACT[w]
    idx = _ai_pick_best(_INTENT_EMB, t)
    return INTENTS[idx] if idx is not None else None

def _make_ack(s):
    bits = []
    if s.get("activity"): bits.append(f"{s['activity']}")
    if s.get("intensity"): bits.append(f"{s['intensity']}")
    if s.get("duration"): bits.append(f"{s['duration']} min")
    if s.get("when"): bits.append(f"in the {s['when']}")
    if s.get("energy"): bits.append(f"energy {s['energy']}")
    return " • ".join(bits) if bits else "noted"

def _next_question(s):
    if not s.get("duration"): return "How many minutes are you thinking? ⏱️"
    if not s.get("intensity"): return "Easy, moderate, or hard pace? ⚖️"
    if not s.get("when"): return "When would you like to do it—morning, afternoon, or evening? 🕒"
    return None

def _update_state_from_msg(state, msg):
    s = dict(state)
    t = msg.strip()

    if re.search(r'\b(reset|start over|clear)\b', t, re.I):
        return {
            "activity": None, "intensity": None, "duration": None,
            "when": None, "energy": None, "notes": ""
        }, "All set to start fresh! What do you feel like doing today? 🙂"

    act = _match_activity(t)
    if act: s["activity"] = act

    inten = _choose_intensity(t)
    if inten: s["intensity"] = inten

    dur = _extract_numbers_minutes(t)
    if dur: s["duration"] = dur

    when = _extract_time_of_day(t)
    if when: s["when"] = when

    if re.search(r'\blow\b', t): s["energy"] = "Low"
    elif re.search(r'\bmedium\b', t): s["energy"] = "Medium"
    elif re.search(r'\bhigh\b', t): s["energy"] = "High"

    if re.search(r'\bonly\b', t) and act:
        s["intensity"] = s.get("intensity") or "moderate"

    used = set()
    if act: used |= set(ACTIVITY_SYNS.get(act, [])) | {act}
    if inten: used.add(inten)
    if dur: used.add(str(dur))
    if when: used.add(when)
    if s.get("energy"): used.add(s["energy"].lower())

    leftover_words = [w for w in re.findall(r"\b\w+\b", t.lower()) if w not in used]
    leftover = " ".join(leftover_words).strip()
    if leftover:
        s["notes"] = (s.get("notes", "") + " " + leftover).strip()[:500]

    ack = _make_ack(s)
    ask = _next_question(s)

    opener = random.choice([
        "Got it! 🙌",
        "Nice choice! 💪",
        "Sounds good! ✅",
        "On it! 🚀",
    ])
    text_ack = f"{opener} I captured: {ack}."
    if ask: text_ack += f" {ask}"

    return s, text_ack

def _suggest_plan_from_state(state):
    activity = state.get("activity") or "walk"
    intensity = state.get("intensity") or "moderate"
    duration = state.get("duration") or 30
    when = state.get("when") or "morning"

    tip_idx = _ai_pick_best(_TIP_EMB, f"{activity} {intensity} {duration}min {when}")
    tip = FOCUS_TIPS[tip_idx] if tip_idx is not None else random.choice(FOCUS_TIPS)

    plan = [
        f"🕒 **{when.title()}** — {activity.title()} ({intensity}) for **{duration} min**",
    ]
    if state.get("notes"):
        plan.append(f"📝 Note — {state['notes'][:200]}")

    return "\n".join([f"- {p}" for p in plan]) + f"\n\n**Tip:** {tip}"

def chat_core(message, state, tz):
    if state is None or not isinstance(state, dict):
        state = {"activity": None, "intensity": None, "duration": None, "when": None, "energy": None, "notes": ""}

    first_time = not any(state.values())
    new_state, ack = _update_state_from_msg(state, message)
    suggestion = _suggest_plan_from_state(new_state)

    now_local = datetime.now(ZoneInfo(tz))
    tstr = now_local.strftime("%I:%M %p").lstrip("0")

    if first_time:
        pre = random.choice([
            "Hey! I’m your planning buddy. Tell me what you feel like doing and I’ll shape a quick plan. 🙂",
            "Hi there! Share a goal like “run 20 minutes in the evening” and I’ll sort it out. 🧭",
        ])
        reply = f"{pre}\n\n{ack}\n\n_Current time in **{tz}**: **{tstr}**_\n\n{suggestion}"
    else:
        reply = f"{ack}\n\n_Current time in **{tz}**: **{tstr}**_\n\n{suggestion}"

    return reply, new_state

with gr.Blocks(title="Good Morning AI Bot", css="#app {max-width: 900px; margin: auto;} footer {visibility: hidden;}") as demo:
    gr.Markdown("# 🌅 Good Morning AI Bot\nPlan your morning or just tell me what you feel like doing.")
    with gr.Tabs():
        with gr.Tab("Planner"):
            with gr.Row():
                with gr.Column():
                    name = gr.Textbox(label="Your name (optional)", placeholder="Rahul")
                    tz = gr.Dropdown(choices=COMMON_TIMEZONES, value="America/Chicago", label="Your timezone")
                    energy = gr.Radio(choices=["Low","Medium","High"], value="Medium", label="How’s your energy?")
                    wake_time = gr.Textbox(label="Wake-up time (optional)", placeholder="6:30 AM")
                with gr.Column():
                    primary = gr.Textbox(label="#1 priority today", placeholder="Finish blog post draft")
                    secondary = gr.Textbox(label="#2 priority (optional)", placeholder="30-min coding practice")
                    note = gr.Textbox(label="Note (optional)", placeholder="Gym at 6 PM, call mom")
                    use_ai = gr.Checkbox(value=True, label="AI personalize quote & tip")
                    generate_btn = gr.Button("Generate Morning Plan", variant="primary")
            out = gr.Markdown()
            generate_btn.click(
                fn=generate_plan,
                inputs=[name, tz, energy, wake_time, primary, secondary, note, use_ai],
                outputs=out
            )

        with gr.Tab("Chat"):
            chat_tz = gr.Dropdown(choices=COMMON_TIMEZONES, value="America/Chicago", label="Your timezone")
            state = gr.State({"activity": None, "intensity": None, "duration": None, "when": None, "energy": None, "notes": ""})
            chatbot = gr.Chatbot(height=420)
            msg = gr.Textbox(placeholder="e.g., I want to run 20 minutes this evening", label="Message")
            send = gr.Button("Send", variant="primary")
            clear = gr.Button("Reset")

            def _ui_chat(user_msg, history, cur_state, tz_):
                history = history or []
                reply, new_state = chat_core(user_msg, cur_state, tz_)
                history = history + [(user_msg, reply)]
                return history, new_state

            send.click(_ui_chat, inputs=[msg, chatbot, state, chat_tz], outputs=[chatbot, state])
            msg.submit(_ui_chat, inputs=[msg, chatbot, state, chat_tz], outputs=[chatbot, state])
            clear.click(lambda: ([], {"activity": None, "intensity": None, "duration": None, "when": None, "energy": None, "notes": ""}), outputs=[chatbot, state])

if os.environ.get("SPACE_ID"):
    demo.launch()
else:
    if __name__ == "__main__":
        demo.launch()
```

Closing tips
------------

*   **Make the chat your default**: hide the Planner tab if you want a pure chat experience.
    
*   **Customize synonyms**: add your own activities and words to ACTIVITY\_SYNS.
    
*   **Change the blocks** in the Planner to match your daily routine (or remove any line you don’t want).
    
*   **Speed**: the embedding model is small (~80–90 MB). On first run it downloads once; subsequent launches are faster.
