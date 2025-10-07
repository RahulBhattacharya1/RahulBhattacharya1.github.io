---
layout: default
title: "Making my AI Journalist"
date: 2021-11-14 12:45:32
categories: [ai]
tags: [python,streamlit,openai]
thumbnail: /assets/images/journalist.webp
thumbnail_mobile: /assets/images/journalist_sq.webp
demo_link: https://rahuls-ai-journalist.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_journalist
---

I wanted a simple tool that drafts news style article ideas quickly. I also wanted guardrails around costs and fair usage for shared use. This post walks through the exact code that I shipped today. I write in first person and keep the explanations practical and clear.


## What This App Does

The app offers two generation modes that feel predictable and fast. OpenAI mode calls a chat model and expects clean JSON in response. Offline mode shuffles a fixed catalog so I always have a fallback. Both modes render tidy cards with headline, category, and a short summary.


## Setup And Requirements

I kept the environment light so setup stays easy for contributors. The app uses Streamlit and the official OpenAI Python package only. I pin a modern Streamlit and a stable OpenAI client in requirements. You can install the file and run Streamlit directly during development.


## App Configuration Block

I start by importing standard modules and optional OpenAI support. The Streamlit page configuration sets a wide layout and a page title. I keep this block very small to reduce startup time and confusion. Small configuration changes make a big difference to perceived polish.

```python
# ======================= App Config =======================
st.set_page_config(page_title="AI Journalist", layout="wide")
```

## Remote Budget Guardrails

I load cost and rate settings from a small remote Python file. A fallback dictionary keeps the app safe if the remote is unavailable. I also allow environment overrides for the two cost related fields. This design lets me tune spend controls without redeploying the app.

```python
# ======================= Runtime Budget =======================
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

### Helper: _fetch_remote_budget

This helper downloads a Python file and executes it in a temp module. I then read the expected attributes and build a clean dictionary. A five second timeout prevents the UI from stalling during fetch. If it fails I still have a sane local configuration to continue.

```python
def _fetch_remote_budget(url: str) -> dict:
    mod = types.ModuleType("budget_remote")
    with urllib.request.urlopen(url, timeout=5) as r:
        code = r.read().decode("utf-8")
    exec(compile(code, "budget_remote", "exec"), mod.__dict__)
    return {k: getattr(mod, k, _BUDGET_DEFAULTS[k]) for k in _BUDGET_DEFAULTS}
```

### Helper: get_budget

This function caches the remote budget dictionary in session state. I use a time to live so repeated reads remain fast for the user. Any environment overrides are applied before returning the structure. If fetching fails I return the local defaults to keep things stable.

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

## Data Model: Article

I use a dataclass for each generated draft to keep rendering clean. The class holds a headline, a category, and a short summary string. A typed model keeps me honest about fields across both generation paths. It also makes the UI loop simpler because the shape never surprises me.

```python
# ======================= Data Models =======================
@dataclass
class Article:
    headline: str
    category: str
    summary: str
```

## UI Helpers

I prefer tiny helpers to keep the main script readable and calm. The first helper sets a brand colored h2 with inline HTML markup. The second helper renders a bordered card with optional links row. Simple helpers cut repetition and reduce layout related copy paste errors.

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

### Helper: brand_h2

This helper prints a styled h2 with a custom color parameter. I use Streamlit markdown with HTML allowed for a clean header. I return nothing because I only want the side effect on screen. This keeps title styles consistent across the entire page layout.

```python
def brand_h2(text: str, color: str):
    st.markdown(f"<h2 style='margin:.25rem 0 .75rem 0; color:{color}'>{text}</h2>", unsafe_allow_html=True)
```

### Helper: section_card

This helper draws a small bordered card with a bold title line. I accept a subtitle HTML string and a list of link tuples. I join link labels with a dot separator to keep the row compact. I inline style the card so it looks consistent across environments.

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

# ======================= Offline Generator =======================
```

## Offline Generation Path

I ship a tiny offline catalog for when the network is flaky. I shuffle the list with a seeded random so results feel varied. I wrap each tuple into the Article dataclass before returning results. This path guarantees the UI never breaks when the model cannot respond.

```python
# ======================= Offline Generator =======================
OFFLINE_ARTICLES = [
    ("Tech Stocks Rally Amid Market Optimism", "Finance", "Analysts note growth in AI-driven sectors."),
    ("Breakthrough in Renewable Energy Storage", "Science", "New battery tech promises longer grid resilience."),
    ("Local Elections See Record Turnout", "Politics", "Voters emphasize transparency and accountability."),
    ("Art Exhibit Showcases Digital Creators", "Culture", "Exploring AI’s role in modern art forms."),
    ("New Study Highlights Ocean Warming", "Environment", "Impacts on marine biodiversity under scrutiny."),
]

def offline_generate(topic: str, tone: str, seed: int):
    rng = random.Random(seed + len(topic) + len(tone))
    shuffled = OFFLINE_ARTICLES.copy()
    rng.shuffle(shuffled)
    # Wrap tuples into Article dataclasses
    return [Article(headline=h, category=c, summary=s) for h, c, s in shuffled[:5]]
```

### Helper: offline_generate

This function picks five items from the shuffled offline catalog. I seed the random with topic and tone so users can nudge variety. I convert raw tuples to Article instances to match the OpenAI path. Stable shapes let the rendering loop ignore which path produced data.

```python
def offline_generate(topic: str, tone: str, seed: int):
    rng = random.Random(seed + len(topic) + len(tone))
    shuffled = OFFLINE_ARTICLES.copy()
    rng.shuffle(shuffled)
    # Wrap tuples into Article dataclasses
    return [Article(headline=h, category=c, summary=s) for h, c, s in shuffled[:5]]

# ======================= OpenAI Call =======================
```

## OpenAI Generation Path

I keep the OpenAI call tight and deterministic for safer parsing. I require an API key in Streamlit secrets and fail early without it. The system prompt demands strict JSON with a single top level key. I clip the list to ten articles so the page remains easy to scan.

```python
# ======================= OpenAI Call =======================
def call_openai(model: str, topic: str, tone: str, length: str, temp: float, max_tokens: int):
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in Streamlit Secrets.")
    if OpenAI is None:
        raise RuntimeError("openai package not available.")

    client = OpenAI(api_key=api_key)

    sys = (
        "You are an AI journalist. Generate structured news article drafts.\n"
        "Return JSON only with one key: 'articles' "
        "(list of objects with keys: headline, category, summary).\n"
        "No prose, no markdown fences."
    )
    usr = (
        f"Topic: {topic}\n"
        f"Tone: {tone}\n"
        f"Length: {length}\n"
        "Provide 5–10 articles with engaging headlines, categories, and concise summaries."
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
    raw = data.get("articles", [])
    return [Article(a.get("headline",""), a.get("category",""), a.get("summary","")) for a in raw][:10]
```

### Helper: call_openai

This function builds messages and calls the chat completions API. I strip markdown fences if the model wraps JSON in a code block. I then parse JSON into Article objects and return the trimmed list. Short error messages bubble up so the caller can fall back cleanly.

```python
def call_openai(model: str, topic: str, tone: str, length: str, temp: float, max_tokens: int):
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in Streamlit Secrets.")
    if OpenAI is None:
        raise RuntimeError("openai package not available.")

    client = OpenAI(api_key=api_key)

    sys = (
        "You are an AI journalist. Generate structured news article drafts.\n"
        "Return JSON only with one key: 'articles' "
        "(list of objects with keys: headline, category, summary).\n"
        "No prose, no markdown fences."
    )
    usr = (
        f"Topic: {topic}\n"
        f"Tone: {tone}\n"
        f"Length: {length}\n"
        "Provide 5–10 articles with engaging headlines, categories, and concise summaries."
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
    raw = data.get("articles", [])
    return [Article(a.get("headline",""), a.get("category",""), a.get("summary","")) for a in raw][:10]

# ======================= Inputs & Sidebar =======================
```

## Sidebar Controls And Inputs

I split the page with columns so inputs feel balanced and compact. The sidebar controls provider, model, temperature, and maximum tokens. I show daily counts and hourly capacity so use feels transparent. I also display budget burn so people see costs before hitting limits.

```python
# ======================= Inputs & Sidebar =======================
st.title("AI Journalist")

with st.sidebar:
    st.subheader("Generator")
    provider = st.selectbox("Provider", ["OpenAI", "Offline (rule-based)"])
    model = st.selectbox("Model (OpenAI)", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])
    brand = "#0F62FE"
    temp = st.slider("Creativity (OpenAI)", 0.0, 1.0, 0.5, 0.05)
    max_tokens = st.slider("Max tokens (OpenAI)", 256, 2048, 900, 32)

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
    topic = st.text_input("Topic", placeholder="e.g., Climate change, AI in healthcare")
    tone = st.selectbox("Tone", ["Neutral", "Investigative", "Optimistic", "Critical"])
with colB:
    length = st.selectbox("Length", ["Short (2-3 paragraphs)", "Medium (4-6 paragraphs)", "Long (7+ paragraphs)"])

col1, col2, col3 = st.columns([1,1,1])
allowed, reason, _ = can_call_now()
with col1:
    gen = st.button("Generate Articles", type="primary", disabled=(not topic.strip()) or (not allowed))
with col2:
    regen = st.button("Regenerate Suggestions")
with col3:
    clear = st.button("Clear")

if "seed" not in st.session_state:
    st.session_state.seed = 42
if clear:
    st.session_state.pop("results", None)
```

## Rate Limits And Shared Capacity

I enforce a simple cooldown and a daily generation count per session. I also track a shared hourly capacity using a cached singleton map. This prevents a burst of heavy traffic from degrading everyone’s use. I prefer soft guardrails that communicate limits without blocking exploration.


### Helper: _hour_bucket

This helper returns a UTC timestamp bucket grouped by the hour. I keep the format readable so inspecting counters stays trivial. The helper accepts an optional datetime for reliable tests and demos. Small pure functions like this are easy to reason about and trust.

```python
def _hour_bucket(now=None):
    now = now or dt.datetime.utcnow()
    return now.strftime("%Y-%m-%d-%H")

@st.cache_resource
```

### Helper: _shared_hourly_counters

This helper returns a module level cache for hourly capacity. Streamlit cache resource ensures a singleton dictionary per session. I read and update counts by bucket to enforce a shared ceiling. A dictionary is enough here because I only need current hour checks.

```python
@st.cache_resource
def _shared_hourly_counters():
    return {}
```

### Helper: init_rate_limit_state

This function initializes per session counters in Streamlit state. I reset the counters if the date has changed since the last call. I also seed the last timestamp and call count if they are missing. Careful initialization avoids edge cases around first renders and reloads.

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

### Helper: can_call_now

This function evaluates all limits and returns a structured decision. I compute cooldown time remaining and estimate spend from past calls. I also check the shared hourly ceiling and the daily generation cap. The function returns a boolean, a message, and a number for waits.

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

### Helper: record_successful_call

This function records a successful generation into session and shared maps. I update the last call timestamp and increment the daily count metric. I also bump the current hour bucket to enforce the pooled capacity. Keeping updates in one place reduces drift between checks and writes.

```python
def record_successful_call():
    ss = st.session_state
    ss["rl_last_ts"] = time.time()
    ss["rl_calls_today"] += 1
    if HOURLY_SHARED_CAP > 0:
        bucket = _hour_bucket()
        counters = _shared_hourly_counters()
        counters[bucket] = counters.get(bucket, 0) + 1

# ======================= Data Models =======================
@dataclass
```

## Orchestrator And Buttons

I wire three buttons for generate, regenerate, and clear interactions. The orchestrator decides whether to call offline or OpenAI mode. On exceptions I show a short error and fall back to the offline path. I also walk the seed forward on regenerate to change the shuffle order.

```python
# ======================= Orchestrator =======================
def generate():
    if provider == "Offline (rule-based)":
        return offline_generate(topic.strip(), tone, st.session_state.seed), "offline"
    else:
        try:
            return call_openai(model, topic.strip(), tone, length, temp, max_tokens), "openai"
        except Exception as e:
            st.error(f"OpenAI error: {e}. Falling back to Offline mode.")
            return offline_generate(topic.strip(), tone, st.session_state.seed), "offline-fallback"

if (gen or regen) and topic.strip():
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

### Helper: generate

This function picks the correct generation path and returns results. I pass trimmed topic text and current controls to the chosen path. On any error I show feedback and produce data from the offline path. The function returns the list and a mode string for a small caption.

```python
def generate():
    if provider == "Offline (rule-based)":
        return offline_generate(topic.strip(), tone, st.session_state.seed), "offline"
    else:
        try:
            return call_openai(model, topic.strip(), tone, length, temp, max_tokens), "openai"
        except Exception as e:
            st.error(f"OpenAI error: {e}. Falling back to Offline mode.")
            return offline_generate(topic.strip(), tone, st.session_state.seed), "offline-fallback"
```

## Rendering Results

I show a small caption with mode so people know what produced results. I render a heading with the brand color and a detail line for tone. I draw two columns and drop cards in alternating positions for balance. If there are no results I show an info line instead of empty space.

```python
# ======================= Results =======================
if "results" in st.session_state:
    results, mode = st.session_state.results
    st.caption(f"Mode: {mode}")
    brand_h2(f"Generated Articles on {topic}", brand)
    st.write(f"Tone: {tone} | Length: {length}")
    brand_h2("Draft Headlines & Summaries", brand)
    if not results:
        st.info("No articles found.")
    else:
        cols = st.columns(2)
        for i, a in enumerate(results):
            with cols[i % 2]:
                subtitle = f"Category: {a.category}<br/>{a.summary}"
                section_card(a.headline, subtitle_html=subtitle)
else:
    st.info("Enter a topic and click Generate Articles.")
```

## How I Would Extend This

I plan to add link discovery so cards can include source references. I also want to batch multiple topics and stream incremental results. A saved drafts sidebar could improve reuse and reduce repeated calls. Finally I will add tests around rate limits to validate edge situations.


## Notes And Constraints

OpenAI mode requires a valid key in Streamlit secrets to operate. Offline mode always works and is the designed safety net for demos. Remote budget files should be readable and fast to fetch over HTTPS. Shared capacity is a memory structure and resets when sessions restart.
