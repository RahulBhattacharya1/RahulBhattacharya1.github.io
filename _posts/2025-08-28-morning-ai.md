---
layout: default
title: "Building a Conversational Morning Planner in Gradio (with a Sprinkle of AI)"
date: 2025-08-28
categories: [intro]
tags: [getting-started]
thumbnail: /assets/images/ai.webp
---
Here you go — a clean, copy-pasteable blog post (Markdown) you can publish on your GitHub Pages. It explains the **entire** `app.py`: every function, the logic behind the planner and the chat, and how the (small) AI piece works. I’ve included runnable code snippets and the **full code** at the end for readers who want to try it locally or in a Hugging Face Space.

---

# Building a Conversational Morning Planner in Gradio (with a Sprinkle of AI)

> A two-tab Gradio app: a structured **Planner** and a **Chat** that understands everyday language like *“run 20 minutes this evening”* and updates the plan on the fly. We use a compact transformer model to pick a relevant **focus tip**—no API keys required.

## Demo goals

* **Planner** tab: fill a simple form → get a morning plan.
* **Chat** tab: type natural phrases (e.g., *“walk fast only”*) → the bot updates activity, intensity, duration, and time of day.
* A small AI layer (embeddings) selects the **most relevant focus tip** for your goals.

---

## Requirements

Create a `requirements.txt`:

```txt
gradio>=4.0.0
torch
sentence-transformers
```

> Optional: `runtime.txt` with `python-3.11` (or `python-3.10`) for reproducible builds.

---

## Folder layout

```
.
├── app.py
└── requirements.txt
```

If you deploy on **Hugging Face Spaces**, set the app file to `app.py` in Space settings.

---

## How it works (high-level)

1. **Data & Model**

   * We define **focus tips** (short, practical advice).
   * Load a small embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
   * Pre-compute embeddings for tips (fast semantic matching).

2. **Planner tab**

   * Form inputs: name, timezone, energy, wake time, #1/#2 priorities, note.
   * If **AI personalize** is on, we embed the user’s goals/notes and pick the **most relevant tip**.
   * Output is a neat Markdown plan.

3. **Chat tab**

   * Uses a mix of **deterministic parsing** (regex + synonyms) and **light embeddings fallback** to interpret messages.
   * Maintains state: `activity`, `intensity`, `duration`, `when`, `energy`, `notes`.
   * Gives friendly acknowledgments and asks **one focused follow-up** if something’s missing.

---

## The code, piece by piece

### 1) Imports, constants, and the embedding model

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

# Load a compact CPU-friendly embedding model and precompute tips
_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
_TIP_EMB = _MODEL.encode(FOCUS_TIPS, normalize_embeddings=True)
```

### 2) Minimal AI helper

We’ll pick the tip whose embedding is **closest** to the user’s goal text.

```python
def _ai_pick_best(items_emb, query_text):
    if not query_text or not query_text.strip():
        return None
    q_emb = _MODEL.encode(query_text, normalize_embeddings=True)
    scores = util.cos_sim(q_emb, items_emb)[0]
    return int(scores.argmax())
```

### 3) Planner: generate a morning plan

* Builds a plan from form inputs.
* If AI toggle is on, picks a **relevant focus tip** using embeddings.

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

> *Note:* You asked to remove quotes and three generic checklist lines from the chat plan; they’re gone there. This **Planner** keeps its own short blocks (including “Wake, hydrate…”). You can delete or edit that line if you prefer.

### 4) Conversational parsing (Chat)

We’ll parse **activity**, **intensity**, **duration**, **time of day**, **energy** from free text.

```python
INTENTS = [
    "run","walk","yoga","strength","cycle","meditate",
    "stretch","read","write","code","study","rest"
]
_INTENT_EMB = _MODEL.encode(INTENTS, normalize_embeddings=True)
_ENERGY_WORDS = ["low","medium","high"]
_ENERGY_EMB = _MODEL.encode(_ENERGY_WORDS, normalize_embeddings=True)

# Synonyms ensure "jog/jogging" counts as "run", "brisk" maps to "walk" etc.
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

#### Extraction helpers

* **Duration** accepts “20”, “20 min”, “for 20”, etc.
* **When** buckets to morning/noon/afternoon/evening.
* **Intensity** maps words like “brisk/fast” → `hard`, “slow/easy” → `easy`.

```python
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
    if any(w in t for w in ["slow","light","easy","casual"]): return "easy"
    if any(w in t for w in ["fast","brisk","hard","intense","tempo","interval"]): return "hard"
    if any(w in t for w in ["moderate","steady"]): return "moderate"
    return None
```

#### Activity matching

Prefer exact synonyms; if none found, fall back to embeddings across canonical intents.

```python
def _match_activity(text):
    t = text.lower()
    for w in WORD2ACT:
        if re.search(rf'\b{re.escape(w)}\b', t):
            return WORD2ACT[w]
    idx = _ai_pick_best(_INTENT_EMB, t)
    return INTENTS[idx] if idx is not None else None
```

#### Building friendly acknowledgments and follow-ups

Small helpers for chat replies.

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

#### The core state updater

* Recognizes **reset**.
* Extracts **activity, intensity, duration, when, energy**.
* If you write “only” (e.g., *“walk fast only”*), it **overrides** activity and ensures sensible defaults.
* Adds **only the leftover words** to notes (not your whole sentence).

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

    opener = random.choice(["Got it! 🙌","Nice choice! 💪","Sounds good! ✅","On it! 🚀"])
    text_ack = f"{opener} I captured: {ack}."
    if ask: text_ack += f" {ask}"

    return s, text_ack
```

#### Turning state into a concise plan (for Chat)

No quotes; no generic checklist lines—just the headline + your notes + one focus tip.

```python
def _suggest_plan_from_state(state):
    activity = state.get("activity") or "walk"
    intensity = state.get("intensity") or "moderate"
    duration = state.get("duration") or 30
    when = state.get("when") or "morning"

    tip_idx = _ai_pick_best(_TIP_EMB, f"{activity} {intensity} {duration}min {when}")
    tip = FOCUS_TIPS[tip_idx] if tip_idx is not None else random.choice(FOCUS_TIPS)

    plan = [f"🕒 **{when.title()}** — {activity.title()} ({intensity}) for **{duration} min**"]
    if state.get("notes"):
        plan.append(f"📝 Note — {state['notes'][:200]}")

    return "\n".join([f"- {p}" for p in plan]) + f"\n\n**Tip:** {tip}"
```

#### Chat turn function

Composes the final reply, including a friendly opener when it’s the **first** message.

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

### 5) Gradio UI wiring

Two tabs: **Planner** and **Chat**. Each has its inputs and event handlers.

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
                    use_ai = gr.Checkbox(value=True, label="AI personalize focus tip")
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

### 6) Launching on Spaces and locally

On Hugging Face Spaces, the file is **imported**, so we explicitly launch. Locally, we also support `python app.py`.

```python
if os.environ.get("SPACE_ID"):
    demo.launch()
else:
    if __name__ == "__main__":
        demo.launch()
```

---

## Example chats

**You:** “I want to run today”
**Bot:** “Got it! I captured: run. How many minutes are you thinking? ⏱️ …”

**You:** “make it 20 minutes, evening, fast only”
**Bot:** “On it! I captured: run • hard • 20 min • in the evening. …”

**You:** “walk only”
**Bot:** “Sounds good! I captured: walk • hard • 20 min • in the evening. …”

**You:** “reset”
**Bot:** “All set to start fresh! What do you feel like doing today? 🙂”

---

## Full source (`app.py`)

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
                    use_ai = gr.Checkbox(value=True, label="AI personalize focus tip")
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

---

## Closing tips

* **Make the chat your default**: hide the Planner tab if you want a pure chat experience.
* **Customize synonyms**: add your own activities and words to `ACTIVITY_SYNS`.
* **Change the blocks** in the Planner to match your daily routine (or remove any line you don’t want).
* **Speed**: the embedding model is small (\~80–90 MB). On first run it downloads once; subsequent launches are faster.

If you want a version where the chat also parses **exact clock times** (e.g., “at 7:30 am”), I can add a tiny time parser next.
