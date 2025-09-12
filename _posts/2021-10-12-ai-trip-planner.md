---
layout: default
title: "My Exciting AI Trip Planner"
date: 2021-10-12 20:28:21
categories: [ai]
tags: [python,streamlit,openai]
thumbnail: /assets/images/ai_rock_paper_scissor.webp
demo_link: https://rahuls-ai-trip-planner.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_trip_planner
featured: true
---

I built a Streamlit app that can plan trips with attractions and hikes. It works in two modes: offline and AI-powered using OpenAI. It also enforces usage limits to keep costs safe.

## Prelude (imports and setup)


```python
# app.py — Streamlit AI Trip Planner (UI-only, OpenAI via st.secrets)
# Mirrors your sample's patterns: st.secrets["OPENAI_API_KEY"], rate limits, sidebar add-ons.

import os
import time
import json
import random
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Tuple

import streamlit as st

# Optional: only needed if provider == "OpenAI"
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
```


**Explanation**


This section groups related functionality for the app. It helps keep the overall structure organized and understandable.



## App Config


```python
st.set_page_config(page_title="AI Trip Planner", layout="wide")
```



Then the code configures the Streamlit page. It sets the page title and the layout to wide. This ensures the interface has space for sidebar and results. It makes the app look polished from the start.


**Explanation**


This block implements rate limiting and budget enforcement. It fetches remote budget values with safe defaults and caches them. Helpers then enforce cooldowns, daily call limits, and hourly capacity. This section prevents excessive use of the app and protects costs.



## Runtime budget/limits loader (auto-updates from GitHub)


```python
import os, time, types, urllib.request
import streamlit as st

# Raw URL of your budget.py in the shared repo (override via env if needed)
BUDGET_URL = os.getenv(
    "BUDGET_URL",
    "https://raw.githubusercontent.com/RahulBhattacharya1/shared_config/main/budget.py",
)

# Safe defaults if the fetch fails
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
    cfg = {}
    for k in _BUDGET_DEFAULTS.keys():
        cfg[k] = getattr(mod, k, _BUDGET_DEFAULTS[k])
    return cfg

def get_budget(ttl_seconds: int = 300) -> dict:
    """Fetch and cache remote budget in session state with a TTL."""
    now = time.time()
    cache = st.session_state.get("_budget_cache")
    ts = st.session_state.get("_budget_cache_ts", 0)

    if cache and (now - ts) < ttl_seconds:
        return cache

    try:
        cfg = _fetch_remote_budget(BUDGET_URL)
    except Exception:
        cfg = _BUDGET_DEFAULTS.copy()

    # Allow env overrides if you want per-deploy tuning
    cfg["DAILY_BUDGET"] = float(os.getenv("DAILY_BUDGET", cfg["DAILY_BUDGET"]))
    cfg["EST_COST_PER_GEN"] = float(os.getenv("EST_COST_PER_GEN", cfg["EST_COST_PER_GEN"]))

    st.session_state["_budget_cache"] = cfg
    st.session_state["_budget_cache_ts"] = now
    return cfg

# Load once (respects TTL); you can expose a "Refresh config" button to clear cache
_cfg = get_budget(ttl_seconds=300)

COOLDOWN_SECONDS  = int(_cfg["COOLDOWN_SECONDS"])
DAILY_LIMIT       = int(_cfg["DAILY_LIMIT"])
HOURLY_SHARED_CAP = int(_cfg["HOURLY_SHARED_CAP"])
DAILY_BUDGET      = float(_cfg["DAILY_BUDGET"])
EST_COST_PER_GEN  = float(_cfg["EST_COST_PER_GEN"])
CONFIG_VERSION    = str(_cfg.get("VERSION", "unknown"))
```


**Explanation**


This section groups related functionality for the app. It helps keep the overall structure organized and understandable.



## End runtime loader


```python
def _hour_bucket(now=None):
    now = now or dt.datetime.utcnow()
    return now.strftime("%Y-%m-%d-%H")

@st.cache_resource
def _shared_hourly_counters():
    # In-memory dict shared by all sessions in this Streamlit process
    # key: "YYYY-MM-DD-HH", value: int count
    return {}

def init_rate_limit_state():
    ss = st.session_state
    today = dt.date.today().isoformat()
    if "rl_date" not in ss or ss["rl_date"] != today:
        ss["rl_date"] = today
        ss["rl_calls_today"] = 0
        ss["rl_last_ts"] = 0.0
    if "rl_last_ts" not in ss:
        ss["rl_last_ts"] = 0.0
    if "rl_calls_today" not in ss:
        ss["rl_calls_today"] = 0

def can_call_now():
    init_rate_limit_state()
    ss = st.session_state
    now = time.time()

    # Cooldown guard
    remaining = int(max(0, ss["rl_last_ts"] + COOLDOWN_SECONDS - now))
    if remaining > 0:
        return (False, f"Please wait {remaining}s before the next generation.", remaining)
```


**Explanation**


This section groups related functionality for the app. It helps keep the overall structure organized and understandable.



## NEW: Daily budget guardrail


```python
    # Uses shared values loaded at runtime: DAILY_BUDGET and EST_COST_PER_GEN
    est_spend = ss["rl_calls_today"] * EST_COST_PER_GEN
    if est_spend >= DAILY_BUDGET:
        return (False, f"Daily cost limit reached (${DAILY_BUDGET:.2f}). Try again tomorrow.", 0)

    # Per-session daily cap (still keeps your old guard)
    if ss["rl_calls_today"] >= DAILY_LIMIT:
        return (False, f"Daily limit reached ({DAILY_LIMIT} generations). Try again tomorrow.", 0)

    # Optional shared hourly cap
    if HOURLY_SHARED_CAP > 0:
        bucket = _hour_bucket()
        counters = _shared_hourly_counters()
        used = counters.get(bucket, 0)
        if used >= HOURLY_SHARED_CAP:
            return (False, "Hourly capacity reached. Please try later.", 0)

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


**Explanation**


This section groups related functionality for the app. It helps keep the overall structure organized and understandable.



## Data Models


```python
@dataclass
class Hike:
    name: str
    difficulty: str
    distance: str
    type: str
```


**Explanation**


This section defines the Hike dataclass. It stores the details of a trail such as name, difficulty, distance, and type. This keeps data organized and ensures consistent handling. It simplifies rendering and parsing across the app.



## UI Helpers


```python
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


**Explanation**


This block adds functions to render styled UI components. One prints colored headers and the other displays cards with links. They make results easier to read and visually consistent. They also keep presentation details separate from logic.



## Offline Generator


```python
OFFLINE_ATTRACTIONS = [
    "Scenic Overlook", "Historic Downtown", "Local Artisan Market", "Regional History Museum",
    "Riverwalk Promenade", "Botanical Garden", "Wildlife Viewing Area", "Cultural Heritage Center",
    "Iconic Bridge", "Lakeside Boardwalk", "Panoramic Viewpoint", "Visitor Center Exhibits"
]

OFFLINE_HIKES = [
    ("Nature Loop", "Easy", "1.5 mi", "Loop"),
    ("Waterfall Trail", "Moderate", "3.2 mi", "Out & Back"),
    ("Summit Ridge", "Hard", "6.0 mi", "Out & Back"),
    ("Canyon Path", "Moderate", "4.1 mi", "Loop"),
    ("Lakeshore Walk", "Easy", "2.3 mi", "Loop"),
    ("Wildflower Route", "Easy", "1.8 mi", "Out & Back"),
    ("Overlook Climb", "Hard", "5.5 mi", "Loop")
]

def offline_trip_plan(destination: str, days: int, difficulty: str, seed: int):
    rng = random.Random(seed + len(destination) + days)
    # Attractions
    base = [f"{destination} {name}" for name in OFFLINE_ATTRACTIONS]
    rng.shuffle(base)
    top_attractions = base[: min(7, max(3, days + 1))]

    # Hikes with difficulty filtering
    hikes = []
    for name, diff, dist, rtype in OFFLINE_HIKES:
        if difficulty != "Any" and diff != difficulty:
            continue
        hikes.append(
            Hike(
                name=f"{destination} {name}",
                difficulty=diff,
                distance=dist,
                type=rtype
            )
        )
    if not hikes:
        for name, diff, dist, rtype in OFFLINE_HIKES:
            hikes.append(Hike(name=f"{destination} {name}", difficulty=diff, distance=dist, type=rtype))
    rng.shuffle(hikes)
    top_hikes = hikes[: min(7, max(3, days + 1))]
    return top_attractions, top_hikes
```


**Explanation**


This block creates offline results without calling OpenAI. It uses predefined attractions and hikes, shuffles them, and filters by difficulty. This ensures the app works even without an API key. It also supports quick demos without cost.



## OpenAI Call (st.secrets)


```python
def call_openai(
    model: str,
    destination: str,
    days: int,
    season: str,
    travel_style: str,
    budget: str,
    interests: List[str],
    difficulty: str,
    kid: bool,
    elder: bool,
    wheelchair: bool,
    temperature: float,
    max_tokens: int
):
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in Streamlit Secrets.")
    if OpenAI is None:
        raise RuntimeError("openai package not available. Add openai to requirements.txt.")

    client = OpenAI(api_key=api_key)

    constraints = []
    if kid: constraints.append("kid-friendly")
    if elder: constraints.append("elder-friendly")
    if wheelchair: constraints.append("wheelchair accessible")

    sys = (
        "You are a precise trip-planning assistant.\n"
        "Return JSON only with two keys: 'attractions' (list of strings) and 'hikes' "
        "(list of objects with keys: name, difficulty, distance, type).\n"
        "No extra keys, no prose, no markdown fences."
    )
    usr = (
        f"Destination: {destination}\n"
        f"Days: {days}\n"
        f"Season: {season}\n"
        f"Pace: {travel_style}\n"
        f"Budget: {budget}\n"
        f"Interests: {', '.join(interests) if interests else 'Any'}\n"
        f"Hiking difficulty preference: {difficulty}\n"
        f"Accessibility constraints: {', '.join(constraints) if constraints else 'None'}\n"
        "Aim for 5–10 attractions and 5–10 hikes when possible. "
        "Each hike must include difficulty, distance with units, and route type."
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": usr}
        ]
    )
    text = resp.choices[0].message.content.strip()

    # Strip accidental code fences
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1].strip()

    data = json.loads(text)
    attractions = [str(x) for x in data.get("attractions", [])][:10]
    hikes_raw = data.get("hikes", [])
    hikes: List[Hike] = []
    for h in hikes_raw:
        if isinstance(h, dict):
            hikes.append(
                Hike(
                    name=str(h.get("name", "")),
                    difficulty=str(h.get("difficulty", "")),
                    distance=str(h.get("distance", "")),
                    type=str(h.get("type", ""))
                )
            )
    hikes = hikes[:10]
    return attractions, hikes
```


**Explanation**


This section handles the OpenAI API call. It builds structured prompts and requests JSON output with attractions and hikes. The response is parsed into Python objects. This is the part that generates dynamic AI-powered plans.



## Inputs & Sidebar


```python
st.title("AI Trip Planner")

with st.sidebar:
    st.subheader("Generator")
    provider = st.selectbox("Provider", ["OpenAI", "Offline (rule-based)"])
    model = st.selectbox("Model (OpenAI)", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])
    brand = "#0F62FE"
    temp = st.slider("Creativity (OpenAI)", 0.0, 1.0, 0.4, 0.05)
    max_tokens = st.slider("Max tokens (OpenAI)", 256, 4096, 1200, 32)

    # Usage panel (rate limits)
    init_rate_limit_state()
    ss = st.session_state
    st.markdown("**Usage limits**")
    st.write(f"<span style='font-size:0.9rem'>Today: {ss['rl_calls_today']} / {DAILY_LIMIT} generations</span>", unsafe_allow_html=True)
    if HOURLY_SHARED_CAP > 0:
        counters = _shared_hourly_counters()
        used = counters.get(_hour_bucket(), 0)
        st.write(f"<span style='font-size:0.9rem'>Hour capacity: {used} / {HOURLY_SHARED_CAP}</span>", unsafe_allow_html=True)
    remaining = int(max(0, ss["rl_last_ts"] + COOLDOWN_SECONDS - time.time()))
    if remaining > 0:
        st.progress(min(1.0, (COOLDOWN_SECONDS - remaining) / COOLDOWN_SECONDS))
        st.caption(f"Cooldown: {remaining}s")
    est_spend = ss['rl_calls_today'] * EST_COST_PER_GEN
    st.markdown(
        f"<span style='font-size:0.9rem'>Budget: &#36;{est_spend:.2f} / &#36;{DAILY_BUDGET:.2f}</span><br/>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<span style='font-size:0.8rem; opacity:0.8'>Version: {CONFIG_VERSION}</span>",
        unsafe_allow_html=True
    )
    
    # Optional: show a warning if we’re on fallback defaults (remote fetch failed)
    if CONFIG_VERSION == "fallback-local":
        st.warning("Using fallback defaults — couldn’t fetch remote budget.py")

colA, colB = st.columns([1.3, 1])
with colA:
    destination = st.text_input(
        "Destination (city, region, or park)",
        placeholder="e.g., Gatlinburg, TN or Rocky Mountain National Park"
    )
    days = st.slider("Number of days", 1, 14, 5)
    season = st.selectbox("Season", ["Any", "Spring", "Summer", "Fall", "Winter"])
    travel_style = st.select_slider("Pace", options=["Relaxed", "Balanced", "Packed"], value="Balanced")
    budget = st.selectbox("Budget", ["Any", "$", "$$", "$$$"])

with colB:
    interests = st.multiselect(
        "Interests (optional)",
        [
            "Nature", "Scenic Drives", "Waterfalls", "Wildlife",
            "Museums", "Local Food", "Photography", "Historic Sites",
            "Lakes", "Caves", "Coastal Views", "Sunrise/Sunset"
        ],
        default=["Nature", "Scenic Drives", "Photography"]
    )
    difficulty = st.selectbox("Hiking difficulty preference", ["Any", "Easy", "Moderate", "Hard"])
    need_kid_friendly = st.checkbox("Kid-friendly options")
    need_elder_friendly = st.checkbox("Elder-friendly options")
    need_wheelchair = st.checkbox("Wheelchair accessible options")

col1, col2, col3 = st.columns([1, 1, 1])
allowed, reason, _wait = can_call_now()
with col1:
    gen = st.button("Generate Plan", type="primary", disabled=(not destination.strip()) or (not allowed))
with col2:
    regen = st.button("Regenerate Suggestions")
with col3:
    clear = st.button("Clear")

if "seed" not in st.session_state:
    st.session_state.seed = 42
if clear:
    st.session_state.pop("results", None)
```


**Explanation**


This section builds the sidebar where users enter their preferences. It includes destination, number of days, travel pace, budget, and accessibility filters. It also displays live usage, cooldowns, and budget consumption. The sidebar is the main control panel of the app.



## Orchestrator


```python
def generate():
    if provider == "Offline (rule-based)":
        return offline_trip_plan(destination.strip(), days, difficulty, st.session_state.seed), "offline"
    else:
        try:
            attractions, hikes = call_openai(
                model=model,
                destination=destination.strip(),
                days=days,
                season=season,
                travel_style=travel_style,
                budget=budget,
                interests=interests,
                difficulty=difficulty,
                kid=need_kid_friendly,
                elder=need_elder_friendly,
                wheelchair=need_wheelchair,
                temperature=temp,
                max_tokens=max_tokens
            )
            return (attractions, hikes), "openai"
        except Exception as e:
            st.error(f"OpenAI error: {e}. Falling back to Offline mode.")
            return offline_trip_plan(destination.strip(), days, difficulty, st.session_state.seed), "offline-fallback"
```


**Explanation**


This block defines the generate function. It decides whether to use the offline generator or OpenAI. It also catches errors and falls back safely. This keeps the experience smooth and predictable.



## Actions


```python
if (gen or regen) and destination.strip():
    # Double-check RL just before the call
    allowed, reason, _ = can_call_now()
    if not allowed:
        st.warning(reason)
    else:
        (attractions, hikes), mode = generate()
        st.session_state.results = (attractions, hikes, mode)
        record_successful_call()
        if regen:
            st.session_state.seed += 7
```


**Explanation**


This block handles button clicks for generate, regenerate, and clear. It re-checks rate limits before calling generate. It updates session state and usage counters. This ensures consistent behavior after each interaction.



## Results


```python
if "results" in st.session_state:
    attractions, hikes, mode = st.session_state.results
    st.caption(f"Mode: {mode}")

    brand_h2(f"Trip Plan: {destination} · {days} days", brand)
    st.write(
        f"Pace: {travel_style} | Budget: {budget} | Season: {season} | "
        f"Interests: {', '.join(interests) if interests else 'Any'} | "
        f"Hike difficulty: {difficulty}"
    )

    brand_h2("Top Attractions", brand)
    if not attractions:
        st.info("No attractions found.")
    else:
        cols = st.columns(2)
        for i, a in enumerate(attractions):
            with cols[i % 2]:
                links = [
                    ("View on Maps", f"https://www.google.com/maps/search/{a.replace(' ', '+')}"),
                    ("Web Search", f"https://www.google.com/search?q={a.replace(' ', '+')}")
                ]
                section_card(a, links=links)

    brand_h2("Top Hiking Trails", brand)
    if not hikes:
        st.info("No hiking trails found.")
    else:
        cols = st.columns(2)
        for i, h in enumerate(hikes):
            with cols[i % 2]:
                title = h.name
                subtitle = f"Difficulty: {h.difficulty} · Distance: {h.distance} · Route: {h.type}"
                links = [
                    ("Trailhead Maps", f"https://www.google.com/maps/search/{title.replace(' ', '+')}"),
                    ("Trail Info", f"https://www.google.com/search?q={title.replace(' ', '+')}")
                ]
                section_card(title, subtitle_html=subtitle, links=links)

else:
    st.info("Enter a destination and click Generate Plan.")
```


**Explanation**


This block renders the generated results. It shows attractions and hikes in cards with links to Google Maps and search. It also shows messages if no results are available. This is where the final output becomes visible to the user.



# Function Explanations


Now I expand on each function, showing the code and then describing its purpose.



## `_fetch_remote_budget()`


```python
def _fetch_remote_budget(url: str) -> dict:
    mod = types.ModuleType("budget_remote")
    with urllib.request.urlopen(url, timeout=5) as r:
        code = r.read().decode("utf-8")
    exec(compile(code, "budget_remote", "exec"), mod.__dict__)
    cfg = {}
    for k in _BUDGET_DEFAULTS.keys():
        cfg[k] = getattr(mod, k, _BUDGET_DEFAULTS[k])
    return cfg
```


**Explanation**


The `_fetch_remote_budget` function performs a specific task in the app. It helps keep the main workflow organized and modular.



## `get_budget()`


```python
def get_budget(ttl_seconds: int = 300) -> dict:
    """Fetch and cache remote budget in session state with a TTL."""
    now = time.time()
    cache = st.session_state.get("_budget_cache")
    ts = st.session_state.get("_budget_cache_ts", 0)

    if cache and (now - ts) < ttl_seconds:
        return cache

    try:
        cfg = _fetch_remote_budget(BUDGET_URL)
    except Exception:
        cfg = _BUDGET_DEFAULTS.copy()

    # Allow env overrides if you want per-deploy tuning
    cfg["DAILY_BUDGET"] = float(os.getenv("DAILY_BUDGET", cfg["DAILY_BUDGET"]))
    cfg["EST_COST_PER_GEN"] = float(os.getenv("EST_COST_PER_GEN", cfg["EST_COST_PER_GEN"]))

    st.session_state["_budget_cache"] = cfg
    st.session_state["_budget_cache_ts"] = now
    return cfg
```


**Explanation**


The `get_budget` function performs a specific task in the app. It helps keep the main workflow organized and modular.



## `_hour_bucket()`


```python
def _hour_bucket(now=None):
    now = now or dt.datetime.utcnow()
    return now.strftime("%Y-%m-%d-%H")
```


**Explanation**


The `_hour_bucket` function performs a specific task in the app. It helps keep the main workflow organized and modular.



## `_shared_hourly_counters()`


```python
@st.cache_resource
def _shared_hourly_counters():
    # In-memory dict shared by all sessions in this Streamlit process
    # key: "YYYY-MM-DD-HH", value: int count
    return {}
```


**Explanation**


The `_shared_hourly_counters` function performs a specific task in the app. It helps keep the main workflow organized and modular.



## `init_rate_limit_state()`


```python
def init_rate_limit_state():
    ss = st.session_state
    today = dt.date.today().isoformat()
    if "rl_date" not in ss or ss["rl_date"] != today:
        ss["rl_date"] = today
        ss["rl_calls_today"] = 0
        ss["rl_last_ts"] = 0.0
    if "rl_last_ts" not in ss:
        ss["rl_last_ts"] = 0.0
    if "rl_calls_today" not in ss:
        ss["rl_calls_today"] = 0
```


**Explanation**


The `init_rate_limit_state` function performs a specific task in the app. It helps keep the main workflow organized and modular.



## `can_call_now()`


```python
def can_call_now():
    init_rate_limit_state()
    ss = st.session_state
    now = time.time()

    # Cooldown guard
    remaining = int(max(0, ss["rl_last_ts"] + COOLDOWN_SECONDS - now))
    if remaining > 0:
        return (False, f"Please wait {remaining}s before the next generation.", remaining)

    # === NEW: Daily budget guardrail ===
    # Uses shared values loaded at runtime: DAILY_BUDGET and EST_COST_PER_GEN
    est_spend = ss["rl_calls_today"] * EST_COST_PER_GEN
    if est_spend >= DAILY_BUDGET:
        return (False, f"Daily cost limit reached (${DAILY_BUDGET:.2f}). Try again tomorrow.", 0)

    # Per-session daily cap (still keeps your old guard)
    if ss["rl_calls_today"] >= DAILY_LIMIT:
        return (False, f"Daily limit reached ({DAILY_LIMIT} generations). Try again tomorrow.", 0)

    # Optional shared hourly cap
    if HOURLY_SHARED_CAP > 0:
        bucket = _hour_bucket()
        counters = _shared_hourly_counters()
        used = counters.get(bucket, 0)
        if used >= HOURLY_SHARED_CAP:
            return (False, "Hourly capacity reached. Please try later.", 0)

    return (True, "", 0)
```


**Explanation**


This function enforces all usage limits. It checks cooldowns, daily budgets, daily call counts, and hourly caps. It returns a flag and message telling if a call is allowed. This central function keeps usage safe and fair.



## `record_successful_call()`


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


**Explanation**


The `record_successful_call` function performs a specific task in the app. It helps keep the main workflow organized and modular.



## `brand_h2()`


```python
def brand_h2(text: str, color: str):
    st.markdown(f"<h2 style='margin:.25rem 0 .75rem 0; color:{color}'>{text}</h2>", unsafe_allow_html=True)
```


**Explanation**


The `brand_h2` function performs a specific task in the app. It helps keep the main workflow organized and modular.



## `section_card()`


```python
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


**Explanation**


The `section_card` function performs a specific task in the app. It helps keep the main workflow organized and modular.



## `offline_trip_plan()`


```python
def offline_trip_plan(destination: str, days: int, difficulty: str, seed: int):
    rng = random.Random(seed + len(destination) + days)
    # Attractions
    base = [f"{destination} {name}" for name in OFFLINE_ATTRACTIONS]
    rng.shuffle(base)
    top_attractions = base[: min(7, max(3, days + 1))]

    # Hikes with difficulty filtering
    hikes = []
    for name, diff, dist, rtype in OFFLINE_HIKES:
        if difficulty != "Any" and diff != difficulty:
            continue
        hikes.append(
            Hike(
                name=f"{destination} {name}",
                difficulty=diff,
                distance=dist,
                type=rtype
            )
        )
    if not hikes:
        for name, diff, dist, rtype in OFFLINE_HIKES:
            hikes.append(Hike(name=f"{destination} {name}", difficulty=diff, distance=dist, type=rtype))
    rng.shuffle(hikes)
    top_hikes = hikes[: min(7, max(3, days + 1))]
    return top_attractions, top_hikes
```


**Explanation**


This function builds attractions and hikes offline without using OpenAI. It shuffles lists and filters hikes based on difficulty. It ensures the app can still generate results in offline mode. This makes the app reliable even without internet or an API key.



## `call_openai()`


```python
def call_openai(
    model: str,
    destination: str,
    days: int,
    season: str,
    travel_style: str,
    budget: str,
    interests: List[str],
    difficulty: str,
    kid: bool,
    elder: bool,
    wheelchair: bool,
    temperature: float,
    max_tokens: int
):
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in Streamlit Secrets.")
    if OpenAI is None:
        raise RuntimeError("openai package not available. Add openai to requirements.txt.")

    client = OpenAI(api_key=api_key)

    constraints = []
    if kid: constraints.append("kid-friendly")
    if elder: constraints.append("elder-friendly")
    if wheelchair: constraints.append("wheelchair accessible")

    sys = (
        "You are a precise trip-planning assistant.\n"
        "Return JSON only with two keys: 'attractions' (list of strings) and 'hikes' "
        "(list of objects with keys: name, difficulty, distance, type).\n"
        "No extra keys, no prose, no markdown fences."
    )
    usr = (
        f"Destination: {destination}\n"
        f"Days: {days}\n"
        f"Season: {season}\n"
        f"Pace: {travel_style}\n"
        f"Budget: {budget}\n"
        f"Interests: {', '.join(interests) if interests else 'Any'}\n"
        f"Hiking difficulty preference: {difficulty}\n"
        f"Accessibility constraints: {', '.join(constraints) if constraints else 'None'}\n"
        "Aim for 5–10 attractions and 5–10 hikes when possible. "
        "Each hike must include difficulty, distance with units, and route type."
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": usr}
        ]
    )
    text = resp.choices[0].message.content.strip()

    # Strip accidental code fences
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1].strip()

    data = json.loads(text)
    attractions = [str(x) for x in data.get("attractions", [])][:10]
    hikes_raw = data.get("hikes", [])
    hikes: List[Hike] = []
    for h in hikes_raw:
        if isinstance(h, dict):
            hikes.append(
                Hike(
                    name=str(h.get("name", "")),
                    difficulty=str(h.get("difficulty", "")),
                    distance=str(h.get("distance", "")),
                    type=str(h.get("type", ""))
                )
            )
    hikes = hikes[:10]
    return attractions, hikes
```


**Explanation**


This function makes a call to the OpenAI API. It builds prompts with user inputs and requests strict JSON output. It parses the JSON into attractions and Hike objects. This enables the AI-powered generation mode of the app.



## `generate()`


```python
def generate():
    if provider == "Offline (rule-based)":
        return offline_trip_plan(destination.strip(), days, difficulty, st.session_state.seed), "offline"
    else:
        try:
            attractions, hikes = call_openai(
                model=model,
                destination=destination.strip(),
                days=days,
                season=season,
                travel_style=travel_style,
                budget=budget,
                interests=interests,
                difficulty=difficulty,
                kid=need_kid_friendly,
                elder=need_elder_friendly,
                wheelchair=need_wheelchair,
                temperature=temp,
                max_tokens=max_tokens
            )
            return (attractions, hikes), "openai"
        except Exception as e:
            st.error(f"OpenAI error: {e}. Falling back to Offline mode.")
            return offline_trip_plan(destination.strip(), days, difficulty, st.session_state.seed), "offline-fallback"
```


**Explanation**


This function orchestrates plan generation. It calls the offline generator or the OpenAI function. It catches errors and falls back gracefully. This makes generation robust under all conditions.



# Classes




## `Hike`


```python
class Hike:
    name: str
    difficulty: str
    distance: str
    type: str
```


**Explanation**


The Hike dataclass stores trail details such as name, difficulty, distance, and type. It keeps data structured, avoids mistakes, and makes rendering results consistent across the app.



## Conclusion

This shows how all pieces connect to make a full trip planner. The design keeps the app reliable and extendable while making usage safe and predictable.
