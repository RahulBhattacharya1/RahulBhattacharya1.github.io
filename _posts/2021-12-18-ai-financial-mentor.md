---
layout: default
title: "Building my own AI Financial Guide"
date: 2021-12-18 14:25:12
categories: [ai]
tags: [python,streamlit,openai]
thumbnail: /assets/images/financial.webp
demo_link: https://rahuls-ai-financial-mentor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_financial_mentor
featured: true
---

I built a financial mentor app in Streamlit. It makes short and clear suggestions. It can run with OpenAI or in an offline mode. The offline mode is helpful during demos. I wrote the code to be simple and predictable.

# What this app does at a glance
The app takes a goal, a horizon, and a risk level. It also takes an income range. It calls OpenAI and asks for JSON advice, or falls back to a rule list. It keeps track of usage and cost so I stay within a budget. It shows nice two column cards with clean text.

# Requirements and quick start
I keep dependencies small. I use Streamlit for the UI. I use the OpenAI SDK for the API call. Here is my exact `requirements.txt`.
```python
streamlit>=1.35.0
openai>=1.3.5
```

I also include the short README so you can see the intent. I mirror the same steps here. It is short but it covers the basics.
```python
# AI Financial Mentor

A Streamlit app that generates concise, actionable personal-finance guidance.  

## Features
- **Two modes:** OpenAI or **Offline (rule-based)** for demos without an API key
- **Guardrails:** cooldown, daily limit, shared hourly cap, and **daily cost budget**
- **Remote config:** pulls `budget.py` from a GitHub raw URL (override with `BUDGET_URL`)
- **Clean UI:** two-column advice cards with categories and details

---

## Quickstart

### 1) Clone & install
```bash
python -m venv .venv && source .venv/bin/activate   # (Windows) .venv\Scripts\activate
pip install -r requirements.txt
```

# Project structure
The project has one main file named `app.py`. It holds the UI, the offline logic, and the OpenAI call. It also holds the rate limit and budget code. I keep things in clear sections so the flow is easy to read.

## Header and imports
This block sets the stage. I declare the script name and goal. I import Python stdlib modules for time, json, and dates. I import Streamlit for the UI. I try to import the OpenAI client but fail soft if the package is missing. That keeps local demos safe.
```python
# app.py — Streamlit AI Financial Mentor (UI-only, OpenAI via st.secrets)
# Mirrors your AI Trip Planner template: rate limits, sidebar, budget, offline fallback.

import os
import time
import json
import random
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple

import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None
```

## App configuration
I set the Streamlit page title and layout. I prefer the wide layout because the cards fit better. This is a one liner but it matters for the look. It keeps the app polished from the start.
```python
# ======================= App Config =======================
st.set_page_config(page_title="AI Financial Mentor", layout="wide")
```

## Rate limiting and budget guardrails
I group the rate code in one section. I define a remote URL for budget settings. I also define local defaults to use as a fallback. The code can fetch a small Python file at runtime and read constants from it. That lets me tune limits without redeploy.
```python
# ======================= Rate Limiting / Budget =======================
import types, urllib.request

BUDGET_URL = os.getenv(
    "BUDGET_URL",
    "https://raw.githubusercontent.com/RahulBhattacharya1/shared_config/main/budget.py",
)

_BUDGET_DEFAULTS = {
    "COOLDOWN_SECONDS": 30,
    "DAILY_LIMIT": 40,
    "HOURLY_SHARED_CAP": 250,
    "DAILY_BUDGET": 1.00,
    "EST_COST_PER_GEN": 1.00,
    "VERSION": "fallback-local",
}

def _fetch_remote_budget(url: str) -> dict:
    mod = types.ModuleType("budget_remote")
    with urllib.request.urlopen(url, timeout=5) as r:
        code = r.read().decode("utf-8")
    exec(compile(code, "budget_remote", "exec"), mod.__dict__)
    return {k: getattr(mod, k, _BUDGET_DEFAULTS[k]) for k in _BUDGET_DEFAULTS}

def get_budget(ttl_seconds: int = 300) -> dict:
    now = time.time()
    cache = st.session_state.get("_budget_cache")
    ts = st.session_state.get("_budget_cache_ts", 0)
    if cache and (now - ts) < ttl_seconds:
        return cache
    try:
        cfg = _fetch_remote_budget(BUDGET_URL)
    except Exception:
        cfg = _BUDGET_DEFAULTS.copy()
    cfg["DAILY_BUDGET"] = float(os.getenv("DAILY_BUDGET", cfg["DAILY_BUDGET"]))
    cfg["EST_COST_PER_GEN"] = float(os.getenv("EST_COST_PER_GEN", cfg["EST_COST_PER_GEN"]))
    st.session_state["_budget_cache"] = cfg
    st.session_state["_budget_cache_ts"] = now
    return cfg

_cfg = get_budget()
COOLDOWN_SECONDS  = int(_cfg["COOLDOWN_SECONDS"])
DAILY_LIMIT       = int(_cfg["DAILY_LIMIT"])
HOURLY_SHARED_CAP = int(_cfg["HOURLY_SHARED_CAP"])
DAILY_BUDGET      = float(_cfg["DAILY_BUDGET"])
EST_COST_PER_GEN  = float(_cfg["EST_COST_PER_GEN"])
CONFIG_VERSION    = str(_cfg.get("VERSION", "unknown"))

def _hour_bucket(now=None):
    now = now or dt.datetime.utcnow()
    return now.strftime("%Y-%m-%d-%H")

@st.cache_resource
def _shared_hourly_counters():
    return {}

def init_rate_limit_state():
    ss = st.session_state
    today = dt.date.today().isoformat()
    if "rl_date" not in ss or ss["rl_date"] != today:
        ss["rl_date"] = today
        ss["rl_calls_today"] = 0
        ss["rl_last_ts"] = 0.0
    ss.setdefault("rl_last_ts", 0.0)
    ss.setdefault("rl_calls_today", 0)

def can_call_now():
    init_rate_limit_state()
    ss = st.session_state
    now = time.time()
    remaining = int(max(0, ss["rl_last_ts"] + COOLDOWN_SECONDS - now))
    if remaining > 0:
        return (False, f"Wait {remaining}s before the next generation.", remaining)
    est_spend = ss["rl_calls_today"] * EST_COST_PER_GEN
    if est_spend >= DAILY_BUDGET:
        return (False, f"Daily cost limit reached (${DAILY_BUDGET:.2f}).", 0)
    if ss["rl_calls_today"] >= DAILY_LIMIT:
        return (False, f"Daily limit reached ({DAILY_LIMIT} generations).", 0)
    if HOURLY_SHARED_CAP > 0:
        bucket = _hour_bucket()
        counters = _shared_hourly_counters()
        if counters.get(bucket, 0) >= HOURLY_SHARED_CAP:
            return (False, "Hourly capacity reached.", 0)
    return (True, "", 0)

def record_successful_call():
    ss = st.session_state
    ss["rl_last_ts"] = time.time()
    ss["rl_calls_today"] += 1
    if HOURLY_SHARED_CAP > 0:
        bucket = _hour_bucket()
        counters = _shared_hourly_counters()
        counters[bucket] = counters.get(bucket, 0) + 1
```

### _fetch_remote_budget()
This helper fetches a remote Python module with budget constants. I load the text with `urllib.request`. I execute it inside a new module object. Then I read a fixed set of keys from it. If a key is missing I use the local default. This keeps remote config safe and bounded.
```python
def _fetch_remote_budget(url: str) -> dict:
    mod = types.ModuleType("budget_remote")
    with urllib.request.urlopen(url, timeout=5) as r:
        code = r.read().decode("utf-8")
    exec(compile(code, "budget_remote", "exec"), mod.__dict__)
    return {k: getattr(mod, k, _BUDGET_DEFAULTS[k]) for k in _BUDGET_DEFAULTS}
```

### get_budget()
This function wraps configuration lookup with a small cache. I store the object and a timestamp in `st.session_state`. If the cache is fresh I reuse it. I also allow env vars to override two values. That gives me a good mix of dynamic control and speed.
```python
def get_budget(ttl_seconds: int = 300) -> dict:
    now = time.time()
    cache = st.session_state.get("_budget_cache")
    ts = st.session_state.get("_budget_cache_ts", 0)
    if cache and (now - ts) < ttl_seconds:
        return cache
    try:
        cfg = _fetch_remote_budget(BUDGET_URL)
    except Exception:
        cfg = _BUDGET_DEFAULTS.copy()
    cfg["DAILY_BUDGET"] = float(os.getenv("DAILY_BUDGET", cfg["DAILY_BUDGET"]))
    cfg["EST_COST_PER_GEN"] = float(os.getenv("EST_COST_PER_GEN", cfg["EST_COST_PER_GEN"]))
    st.session_state["_budget_cache"] = cfg
    st.session_state["_budget_cache_ts"] = now
    return cfg
```

### _hour_bucket()
This helper returns an hour bucket key. I format the UTC time as `YYYY-MM-DD-HH`. I use this key to track usage in a shared dict. This makes it simple to enforce an hourly cap.
```python
def _hour_bucket(now=None):
    now = now or dt.datetime.utcnow()
    return now.strftime("%Y-%m-%d-%H")

@st.cache_resource
```

### _shared_hourly_counters()
This function returns a global counters dict. I decorate it with `@st.cache_resource` to keep one dict across runs. Streamlit will reuse it within the same server process. I index this dict by the hour bucket key. It lets me implement a soft shared cap.
```python
def _shared_hourly_counters():
    return {}
```

### init_rate_limit_state()
This function prepares per-user state. I keep the date, the last call time, and the number of calls today. I reset the counters when the day changes. I call this from the main checks to avoid missing values. This keeps the rest of the code clean.
```python
def init_rate_limit_state():
    ss = st.session_state
    today = dt.date.today().isoformat()
    if "rl_date" not in ss or ss["rl_date"] != today:
        ss["rl_date"] = today
        ss["rl_calls_today"] = 0
        ss["rl_last_ts"] = 0.0
    ss.setdefault("rl_last_ts", 0.0)
    ss.setdefault("rl_calls_today", 0)
```

### can_call_now()
This function enforces the guardrails. I check the cooldown window first and return the wait time if needed. I compute a rough spend using an estimated cost per generation. I check against the daily budget and the daily call limit. I also read and compare the shared hourly counter. The return value holds a boolean, a reason, and a wait number.
```python
def can_call_now():
    init_rate_limit_state()
    ss = st.session_state
    now = time.time()
    remaining = int(max(0, ss["rl_last_ts"] + COOLDOWN_SECONDS - now))
    if remaining > 0:
        return (False, f"Wait {remaining}s before the next generation.", remaining)
    est_spend = ss["rl_calls_today"] * EST_COST_PER_GEN
    if est_spend >= DAILY_BUDGET:
        return (False, f"Daily cost limit reached (${DAILY_BUDGET:.2f}).", 0)
    if ss["rl_calls_today"] >= DAILY_LIMIT:
        return (False, f"Daily limit reached ({DAILY_LIMIT} generations).", 0)
    if HOURLY_SHARED_CAP > 0:
        bucket = _hour_bucket()
        counters = _shared_hourly_counters()
        if counters.get(bucket, 0) >= HOURLY_SHARED_CAP:
            return (False, "Hourly capacity reached.", 0)
    return (True, "", 0)
```

### record_successful_call()
This function updates all counters after a good call. I set the last timestamp and increase the daily calls. If I use the shared cap I also bump the hourly bucket. I keep the logic tiny so it is easy to trust.
```python
def record_successful_call():
    ss = st.session_state
    ss["rl_last_ts"] = time.time()
    ss["rl_calls_today"] += 1
    if HOURLY_SHARED_CAP > 0:
        bucket = _hour_bucket()
        counters = _shared_hourly_counters()
        counters[bucket] = counters.get(bucket, 0) + 1
```

## Data model
I keep advice items as a small dataclass. It has a title, a category, and a detail. The class is simple and typed. It makes my code clear downstream.
```python
# ======================= Data Models =======================
@dataclass
class Advice:
    title: str
    category: str
    detail: str
```

## UI helpers
I render brand colored H2 headers with HTML. I also render a simple card with a title, a subtitle line, and links. The functions use `unsafe_allow_html=True` which is fine here. It helps me control spacing and tone without custom CSS.
```python
# ======================= UI Helpers =======================
def brand_h2(text: str, color: str):
    st.markdown(f"<h2 style='margin:.25rem 0 .75rem 0; color:{color}'>{text}</h2>", unsafe_allow_html=True)

def section_card(title: str, subtitle_html: str = "", links: List[Tuple[str, str]] = None):
    links = links or []
    items = " · ".join(f'<a href="{href}" target="_blank">{label}</a>' for label, href in links)
    st.markdown(
        f"""
<div style="border:1px solid #e5e7eb; padding:.75rem 1rem; border-radius:10px; margin-bottom:.75rem;">
  <div style="font-weight:600">{title}</div>
  {f'<div style="font-size:.95rem; margin:.2rem 0;">{subtitle_html}</div>' if subtitle_html else ''}
  {f'<div style="font-size:.9rem; opacity:.85;">{items}</div>' if items else ''}
</div>
        """,
        unsafe_allow_html=True
    )
```

## Offline generator
I define a small list of classic advice. I shuffle it with a seeded RNG. I pick five items. The seed depends on the user inputs so regen gives variety. This block keeps the demo useful even without API access.
```python
# ======================= Offline Generator =======================
OFFLINE_ADVICE = [
    ("Build an emergency fund", "Savings", "Set aside at least 3–6 months of living expenses."),
    ("Diversify investments", "Investing", "Avoid putting all money into one asset class."),
    ("Track monthly expenses", "Budgeting", "Categorize spending to find areas for savings."),
    ("Contribute to retirement", "Retirement", "Regularly invest in a 401(k) or IRA."),
    ("Pay high-interest debt first", "Debt", "Focus on credit cards and loans with high APR."),
    ("Automate savings", "Savings", "Set up recurring transfers to savings accounts."),
    ("Review insurance coverage", "Risk Management", "Ensure health, auto, and home are covered."),
]

def offline_plan(goal: str, horizon: str, risk: str, seed: int):
    rng = random.Random(seed + len(goal) + len(horizon) + len(risk))
    shuffled = OFFLINE_ADVICE.copy()
    rng.shuffle(shuffled)
    return shuffled[:5]
```

## OpenAI call
This function builds a strict JSON-only prompt. I pass a short system role and a short user role. I set `temperature` and `max_tokens` from the UI. I parse the JSON and coerce each item into my dataclass. I also trim any code fences if the model adds them. The function raises clear errors when the API key or package is missing.
```python
# ======================= OpenAI Call =======================
def call_openai(model: str, goal: str, horizon: str, risk: str, income: str, temp: float, max_tokens: int):
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in Streamlit Secrets.")
    if OpenAI is None:
        raise RuntimeError("openai package not available.")

    client = OpenAI(api_key=api_key)

    sys = (
        "You are a precise financial planning assistant.\n"
        "Return JSON only with one key: 'advice' "
        "(list of objects with keys: title, category, detail).\n"
        "No prose, no markdown fences."
    )
    usr = (
        f"Goal: {goal}\n"
        f"Horizon: {horizon}\n"
        f"Risk tolerance: {risk}\n"
        f"Monthly income: {income}\n"
        "Provide 5–10 actionable financial advice items."
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=float(temp),
        max_tokens=int(max_tokens),
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}]
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1].strip()
    data = json.loads(text)
    raw = data.get("advice", [])
    return [Advice(a.get("title",""), a.get("category",""), a.get("detail","")) for a in raw][:10]
```

## Inputs and sidebar
I build the main UI with Streamlit APIs. I show a provider toggle, a model list, and sliders for creativity and tokens. I show counters for daily calls, hourly capacity, and budget at a glance. I collect the goal, horizon, risk, and income. I use three buttons to trigger generation, regeneration, or clearing the page.
```python
# ======================= Inputs & Sidebar =======================
st.title("AI Financial Mentor")

with st.sidebar:
    st.subheader("Generator")
    provider = st.selectbox("Provider", ["OpenAI", "Offline (rule-based)"])
    model = st.selectbox("Model (OpenAI)", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])
    brand = "#0F62FE"
    temp = st.slider("Creativity (OpenAI)", 0.0, 1.0, 0.4, 0.05)
    max_tokens = st.slider("Max tokens (OpenAI)", 256, 2048, 800, 32)

    init_rate_limit_state()
    ss = st.session_state
    st.markdown("**Usage limits**")
    st.write(f"<span style='font-size:0.9rem'>Today: {ss['rl_calls_today']} / {DAILY_LIMIT} generations</span>", unsafe_allow_html=True)
    if HOURLY_SHARED_CAP > 0:
        counters = _shared_hourly_counters()
        used = counters.get(_hour_bucket(), 0)
        st.write(f"<span style='font-size:0.9rem'>Hour capacity: {used} / {HOURLY_SHARED_CAP}</span>", unsafe_allow_html=True)
    est_spend = ss['rl_calls_today'] * EST_COST_PER_GEN
    st.markdown(
        f"<span style='font-size:0.9rem'>Budget: &#36;{est_spend:.2f} / &#36;{DAILY_BUDGET:.2f}</span><br/>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<span style='font-size:0.8rem; opacity:0.8'>Version: {CONFIG_VERSION}</span>",
        unsafe_allow_html=True
    )

colA, colB = st.columns([1.3, 1])
with colA:
    goal = st.text_input("Financial Goal", placeholder="e.g., Buy a house, Save for retirement")
    horizon = st.selectbox("Time Horizon", ["Short-term (0-2y)", "Medium (3-7y)", "Long-term (8y+)"])
    risk = st.selectbox("Risk Tolerance", ["Low", "Moderate", "High"])
with colB:
    income = st.selectbox("Monthly Income Range", ["<$2k", "$2k-$5k", "$5k-$10k", "$10k+"])

col1, col2, col3 = st.columns([1,1,1])
allowed, reason, _ = can_call_now()
with col1:
    gen = st.button("Generate Advice", type="primary", disabled=(not goal.strip()) or (not allowed))
with col2:
    regen = st.button("Regenerate Suggestions")
with col3:
    clear = st.button("Clear")

if "seed" not in st.session_state:
    st.session_state.seed = 42
if clear:
    st.session_state.pop("results", None)
```

## Orchestrator and button wiring
I wrap the main decision in a `generate()` function. It either calls offline or OpenAI and returns a list plus a mode string. The button code checks the guard first. On success it stores the results in `session_state` and records the call. I also change the seed on regen to vary the offline list.
```python
# ======================= Orchestrator =======================
def generate():
    if provider == "Offline (rule-based)":
        return offline_plan(goal.strip(), horizon, risk, st.session_state.seed), "offline"
    else:
        try:
            return call_openai(model, goal.strip(), horizon, risk, income, temp, max_tokens), "openai"
        except Exception as e:
            st.error(f"OpenAI error: {e}. Falling back to Offline mode.")
            return offline_plan(goal.strip(), horizon, risk, st.session_state.seed), "offline-fallback"

if (gen or regen) and goal.strip():
    allowed, reason, _ = can_call_now()
    if not allowed:
        st.warning(reason)
    else:
        results, mode = generate()
        st.session_state.results = (results, mode)
        record_successful_call()
        if regen:
            st.session_state.seed += 7
```

## Results rendering
I render a caption with the selected mode. I echo the goal and the user inputs for context. I then show a header and a grid of cards. I place items in two columns. The card shows the title and a short description. If there are no results I display a friendly message.
```python
# ======================= Results =======================
if "results" in st.session_state:
    results, mode = st.session_state.results
    st.caption(f"Mode: {mode}")
    brand_h2(f"Financial Plan: {goal} · {horizon}", brand)
    st.write(f"Risk: {risk} | Income: {income}")
    brand_h2("Recommended Actions", brand)
    if not results:
        st.info("No advice found.")
    else:
        cols = st.columns(2)
        for i, a in enumerate(results):
            with cols[i % 2]:
                subtitle = f"Category: {a.category}<br/>{a.detail}"
                section_card(a.title, subtitle_html=subtitle)
else:
    st.info("Enter a financial goal and click Generate Advice.")
```

# Notes on testing and safety
I test the rate checks with short cooldowns and a tiny budget. I hit the buttons a few times and confirm the messages. I also disconnect the API key to see the offline fallback. This gives me confidence that the app fails safe.
I load test the UI by sending random goals. I do not store user data. I only keep per session counters. I avoid logs with personal content. The JSON shape is small and consistent. The app stays easy to reason about.

# Local run and deployment
I install the requirements in a fresh virtualenv. I run `streamlit run app.py`. I set the `OPENAI_API_KEY` in `.streamlit/secrets.toml` for the API path. I can also set `BUDGET_URL`, `DAILY_BUDGET`, or `EST_COST_PER_GEN` as env vars. For cloud, I push this repo to a Streamlit Community Cloud app. The remote config lets me tune without redeploy.

# Closing thoughts
I kept the code small and direct. Each helper does one job. The checks are explicit and easy to test. The UI is compact and readable. You can extend this with storage or auth if you need. I like how the offline mode keeps demos smooth.
