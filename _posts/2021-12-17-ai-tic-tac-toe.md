---
layout: default
title: "My Virtual AI Powered Tic Tac Toe"
date: 2021-12-17 11:21:45
categories: [ai]
tags: [python,streamlit,openai]
thumbnail: /assets/images/ai_rock_paper_scissor.webp
demo_link: https://rahuls-ai-tic-tac-toe.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_tic_tac_toe
featured: true
---

I wrote a small Streamlit app that plays Tic Tac Toe against me. The app has two brains. One brain is an offline minimax engine. The second brain is an OpenAI call that returns a JSON move. Both brains respect budget guardrails and a simple rate limiter. In this post I explain every code block and why I wrote it that way. I avoid long sentences and keep the style simple and direct.

> Repo pieces used here: `app.py`, `README.md`, and `requirements.txt`.

---

## What I built

- A single-page Streamlit app named **Tic Tac Toe**. - A perfect-play offline AI using **minimax**. - An optional **OpenAI** path that returns a move as JSON. - A small **remote budget** config with daily and hourly caps. - A **rate limiter** in session state so users cannot spam moves. - A clean UI with buttons and status cards.

I run it locally with Streamlit. I can also deploy it if needed. The offline mode does not need an API key. The OpenAI mode does.

---

## Requirements and quick start

The app depends on Streamlit and the official `openai` client. I keep it minimal so setup stays simple.

```python
# requirements.txt
streamlit>=1.35.0
openai>=1.3.5
```

I install, set an optional key, and run the app.

```python
# README.md (quickstart distilled)
# 1) create venv, 2) install, 3) optional OpenAI secret, 4) run
# bash
# python -m venv .venv && source .venv/bin/activate
# pip install -r requirements.txt
# mkdir -p .streamlit && printf 'OPENAI_API_KEY = "sk-..."
' > .streamlit/secrets.toml  # optional
# streamlit run app.py
```

I keep OpenAI optional because the offline engine is strong. This helps demos run even with no network or no secret.

---

## App header and imports

I start with standard modules and Streamlit. I also import the OpenAI client safely, so missing installs do not break.

```python
import os, time, json, random, datetime as dt, types, urllib.request
from dataclasses import dataclass
from typing import List, Tuple, Optional
import streamlit as st
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title="Tic Tac Toe", layout="wide")
```

**Why I wrote it this way:** I keep imports compact to reduce noise. I wrap the OpenAI import in `try/except`. If the package is missing, the app still runs offline. The page config widens the layout. I do not need multiple pages, so a simple wide layout works well.

---

## Remote budget config and sane defaults

I pull a small config file from a GitHub raw URL. If the fetch fails, I fall back to local defaults. This keeps limits flexible without redeploys.

```python
BUDGET_URL = os.getenv("BUDGET_URL","https://raw.githubusercontent.com/RahulBhattacharya1/shared_config/main/budget.py")
_DEFAULTS = {"COOLDOWN_SECONDS":30,"DAILY_LIMIT":40,"HOURLY_SHARED_CAP":250,"DAILY_BUDGET":1.00,"EST_COST_PER_GEN":1.00,"VERSION":"fallback-local"}
def _fetch(url:str)->dict:
    mod=types.ModuleType("b"); 
    with urllib.request.urlopen(url,timeout=5) as r: code=r.read().decode()
    exec(compile(code,"b","exec"),mod.__dict__); 
    return {k:getattr(mod,k,_DEFAULTS[k]) for k in _DEFAULTS}
```

**What this helper does:** `_fetch` loads a tiny Python module over HTTP and executes it in memory. I only extract known keys from that module to avoid surprises. A five‑second timeout keeps the UI snappy when the network drags. This helper lets me tune limits without touching the app code.

---

## Caching and merging config with environment

I keep the fetched config in session state with a short TTL. I let environment variables override a few numeric values. This gives me three layers: defaults, remote file, and env overrides.

```python
def _cfg(ttl=300):
    now=time.time(); cache=st.session_state.get("_b"); ts=st.session_state.get("_bts",0)
    if cache and (now-ts)<ttl: return cache
    try: cfg=_fetch(BUDGET_URL)
    except Exception: cfg=_DEFAULTS.copy()
    cfg["DAILY_BUDGET"]=float(os.getenv("DAILY_BUDGET",cfg["DAILY_BUDGET"]))
    cfg["EST_COST_PER_GEN"]=float(os.getenv("EST_COST_PER_GEN",cfg["EST_COST_PER_GEN"]))
    st.session_state["_b"]=cfg; st.session_state["_bts"]=now; return cfg

_cfg=_cfg(); COOLDOWN_SECONDS=int(_cfg["COOLDOWN_SECONDS"]); DAILY_LIMIT=int(_cfg["DAILY_LIMIT"])
HOURLY_SHARED_CAP=int(_cfg["HOURLY_SHARED_CAP"]); DAILY_BUDGET=float(_cfg["DAILY_BUDGET"])
EST_COST_PER_GEN=float(_cfg["EST_COST_PER_GEN"]); CONFIG_VERSION=str(_cfg["VERSION"])
```

**Why this design helps:** The `_cfg` function caches the budget dictionary for five minutes. This avoids hitting the remote URL on every rerun. I cast numeric fields to the right types for safety. I then bind simple module‑level constants for easy reads in the UI.

**Note on shadowing:** After calling `_cfg()`, I assign the result back into `_cfg`. This turns the name into a dictionary, not a function. I accept the shadowing because I only need to call it once. If I wanted to refresh later, I would keep distinct names.

---

## Bucketing, shared counters, and simple rate limiting

I track per‑hour capacity in a plain dict cached as a resource. I also track per‑session daily calls and cooldown timestamps. This is enough for a small demo with polite usage controls.

```python
def _hour_bucket(now=None): now=now or dt.datetime.utcnow(); return now.strftime("%Y-%m-%d-%H")
@st.cache_resource
def _shared_hourly_counters(): return {}
def _init_rl():
    ss=st.session_state; today=dt.date.today().isoformat()
    if ss.get("rl_date")!=today: ss["rl_date"]=today; ss["rl_calls_today"]=0; ss["rl_last_ts"]=0.0
    ss.setdefault("rl_last_ts",0.0); ss.setdefault("rl_calls_today",0)
def _can():
    _init_rl(); ss=st.session_state; now=time.time()
    rem=int(max(0, ss["rl_last_ts"]+COOLDOWN_SECONDS-now)); 
    if rem>0: return False, f"Wait {rem}s before next move.", rem
    if ss["rl_calls_today"]*EST_COST_PER_GEN>=DAILY_BUDGET: return False, f"Daily cost limit reached (${DAILY_BUDGET:.2f}).",0
    if ss["rl_calls_today"]>=DAILY_LIMIT: return False, f"Daily limit reached ({DAILY_LIMIT}).",0
    if HOURLY_SHARED_CAP>0:
        b=_hour_bucket(); c=_shared_hourly_counters()
        if c.get(b,0)>=HOURLY_SHARED_CAP: return False,"Hourly capacity reached.",0
    return True,"",0
def _record():
    ss=st.session_state; ss["rl_last_ts"]=time.time(); ss["rl_calls_today"]+=1
    if HOURLY_SHARED_CAP>0:
        b=_hour_bucket(); c=_shared_hourly_counters(); c[b]=c.get(b,0)+1
```

**How these helpers work together:** `_hour_bucket` returns a compact hour key for shared limits. `_shared_hourly_counters` holds a process‑wide dict for that hour. `_init_rl` resets session counters when the day changes. `_can` enforces cooldown, daily budget, daily calls, and hour caps. `_record` stamps the last call and increments counters after a move.

**Why this matters in the app:** The AI move is the expensive step in this UI. I want clear messages when limits block a move. Users see the hour and day usage in the sidebar. This keeps expectations clear and avoids silent failures.

---

## Game model: dataclass and winning lines

I define a tiny `Move` dataclass for clarity. I also declare all eight winning triplets for the board.

```python
@dataclass
class Move: index:int

WIN = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
```

**Why this helps:** `Move` makes returns explicit and easy to extend if needed. The `WIN` constant avoids recomputing combinations on every check. Both choices keep the minimax code clean and readable later.

---

## Core rules: winner and available squares

I use a simple function to detect a winner or a draw. I also compute a fresh list of empty indexes for the next move.

```python
def winner(b:List[str])->Optional[str]:
    for a,b2,c in WIN:
        if b[a] and b[a]==b[b2]==b[c]: return b[a]
    return "draw" if all(b) else None
def avail(b): return [i for i,v in enumerate(b) if v==""]
```

**Explanation of conditionals:** The first `if` checks that the first cell is non‑empty and matches the line. If that is true, the function returns that symbol as the winner. If no line matches and all cells are taken, I return `"draw"`. Otherwise I return `None` which means the game continues.

**Why `avail` is separate:** Minimax needs a list of legal actions each turn. By isolating it, I avoid duplicating list logic. It keeps the recursive search easy to follow. It also helps unit testing in isolation.

---

## Minimax: perfect play without randomness

This is the offline brain. It searches the game tree. It scores terminal states and picks the best move.

```python
def minimax(b, ai, hu, maxing):
    w=winner(b); 
    if w==ai: return 1,None
    if w==hu: return -1,None
    if w=="draw": return 0,None
    best=None
    if maxing:
        sc=-10
        for i in avail(b):
            b[i]=ai; s,_=minimax(b,ai,hu,False); b[i]=""
            if s>sc: sc, best=s,i
        return sc,best
    else:
        sc=10
        for i in avail(b):
            b[i]=hu; s,_=minimax(b,ai,hu,True); b[i]=""
            if s<sc: sc, best=s,i
        return sc,best
```

**How the branches work:** I check for a terminal state at the top and return a score. When `maxing` is `True`, the AI tries to maximize the score. When `maxing` is `False`, the human branch minimizes that score. I place a symbol, recurse, then undo the move to keep the board clean. The function returns both the score and the best index.

**Why I chose these scores:** Win is `+1` for the AI and `-1` for the human. A draw is `0` since no side gains an edge. This scale is enough for Tic Tac Toe. It keeps comparisons fast and intent clear.

---

## Offline move: deterministic and safe fallback

I wrap the minimax call in a small helper. If minimax returns `None` for the index, I choose a random legal move. This path also acts as a fallback for bad model outputs.

```python
def offline_move(b, ai, hu)->Move:
    _,mv=minimax(b[:],ai,hu,True); 
    if mv is None: a=avail(b); mv=random.choice(a) if a else 0
    return Move(mv)
```

**Why this is useful:** I copy the board so the search does not mutate the original. The random branch only fires if something odd happens. Returning `Move` keeps the type stable across providers. This function gives me a reliable offline experience.

---

## OpenAI move: JSON in and validation out

I define a clean adapter for OpenAI Chat Completions. The model returns a tiny JSON string like `{"move": 4}`. I still validate the index and fall back if needed.

```python
def call_openai(model,b,ai,hu,temp,max_tokens)->Move:
    api=st.secrets.get("OPENAI_API_KEY",""); 
    if not api: raise RuntimeError("OPENAI_API_KEY missing")
    if OpenAI is None: raise RuntimeError("openai package not available")
    client=OpenAI(api_key=api)
    sys="You play perfect Tic Tac Toe. Return JSON with key 'move' (0..8). No prose."
    usr=f"Board: {b}\nAI: {ai}\nHuman: {hu}"
    r=client.chat.completions.create(model=model,temperature=float(temp),max_tokens=int(max_tokens),
                                     messages=[{"role":"system","content":sys},{"role":"user","content":usr}])
    t=r.choices[0].message.content.strip()
    if t.startswith("```"): t=t.strip("`").split("\n",1)[-1].strip()
    mv=int(json.loads(t).get("move",-1))
    if mv not in avail(b): return offline_move(b,ai,hu)
    return Move(mv)
```

**Why these guardrails matter:** I read the API key from Streamlit secrets, not env, for local safety. I raise clear exceptions if the key or package is missing. I strip code fences in case the model formats the JSON as a block. I parse and validate the index before I trust the response. If the move is illegal, I fall back to `offline_move` immediately.

---

## Small UI helpers for headings and cards

I like a simple visual rhythm in the sidebar and status. Two helpers render headings and a bordered card with an optional subtext.

```python
def h2(txt,col): st.markdown(f"<h2 style='margin:.25rem 0 .75rem 0; color:{col}'>{txt}</h2>", unsafe_allow_html=True)
def card(title, sub=""):
    st.markdown(f'''<div style="border:1px solid #e5e7eb;padding:.75rem 1rem;border-radius:10px;margin:.75rem 0;">
    <div style="font-weight:600">{title}</div>{f'<div style="margin-top:.25rem">{sub}</div>' if sub else ''}</div>''', unsafe_allow_html=True)
```

**Why I keep them thin:** I do not want a UI framework inside another UI framework. Streamlit Markdown is enough for headings and boxes. These helpers reduce duplication and keep the code tidy. They also make later styling tweaks much easier.

---

## Top‑level title and sidebar controls

I draw the title, then shape the sidebar controls. I expose the provider, model, and temperature settings. I also show live usage counters to set expectations.

```python
st.title("Tic Tac Toe")
with st.sidebar:
    st.subheader("Generator")
    provider = st.selectbox("Provider", ["OpenAI","Offline (rule-based)"])
    model    = st.selectbox("Model (OpenAI)", ["gpt-4o-mini","gpt-4o","gpt-4.1-mini"])
    brand    = "#0F62FE"
    temp     = st.slider("Creativity (OpenAI)",0.0,1.0,0.4,0.05)
    max_tok  = st.slider("Max tokens (OpenAI)",256,2048,600,32)

    _init_rl(); ss=st.session_state
    st.markdown("**Usage limits**")
    st.write(f"<span style='font-size:.9rem'>Today: {ss['rl_calls_today']} / {DAILY_LIMIT}</span>", unsafe_allow_html=True)
    if HOURLY_SHARED_CAP>0:
        used=_shared_hourly_counters().get(_hour_bucket(),0)
        st.write(f"<span style='font-size:.9rem'>Hour: {used} / {HOURLY_SHARED_CAP}</span>", unsafe_allow_html=True)
    est=ss['rl_calls_today']*EST_COST_PER_GEN
    st.markdown(f"<span style='font-size:.9rem'>Budget: ${est:.2f} / ${DAILY_BUDGET:.2f}</span><br/>"
                f"<span style='font-size:.8rem;opacity:.8'>Version: {CONFIG_VERSION}</span>", unsafe_allow_html=True)
```

**What each control does:** The provider toggle selects OpenAI or the offline brain. The model list narrows to stable, fast options I like. The sliders let me steer temperature and token caps. The counters show calls today, hour usage, and budget spend. I display the remote config version for quick debugging.

---

## Game state, reset, and symbol choice

I keep the whole game in `st.session_state` under the key `ttt`. I also provide a reset button and a radio to pick my symbol.

```python
gs = st.session_state.setdefault("ttt", {"board":[""]*9,"human":"X","ai":"O","turn":"human","over":False,"result":""})
colA,colB=st.columns([1.2,1])
with colB:
    if st.button("New Game"): gs.update({"board":[""]*9,"turn":"human","over":False,"result":""})
    sym = st.radio("You play as",["X","O"],index=0,horizontal=True)
    if sym!=gs["human"] and not any(gs["board"]): gs["human"]=sym; gs["ai"]="O" if sym=="X" else "X"
```

**Why I gate symbol changes:** I only allow symbol switches before the first move. I check `not any(gs["board"])` to enforce that rule. This avoids mid‑game flips that would break state logic. The reset button clears the board and returns turn to human.

---

## Board rendering and click handler

I render the board as nine buttons in three columns. Each cell button is disabled if the game is over or not my turn.

```python
def cell(i):
    dis=gs["over"] or gs["board"][i]!="" or gs["turn"]!="human"
    if st.button(gs["board"][i] or " ", key=f"c{i}", use_container_width=True, disabled=dis):
        gs["board"][i]=gs["human"]; gs["turn"]="ai"

c1,c2,c3=st.columns(3)
with c1: cell(0); cell(3); cell(6)
with c2: cell(1); cell(4); cell(7)
with c3: cell(2); cell(5); cell(8)
```

**Why these conditionals matter:** I disable a cell if it is already played, not my turn, or game is over. On click, I mark my symbol and flip turn to the AI. I use `use_container_width=True` so buttons fill each column. The grid layout keeps the look close to a classic board.

---

## Post‑move checks and AI turn

After my click, I check for a winner or a draw. If the game continues and it is the AI turn, I ask for a move. I respect limits, handle errors, and update state carefully.

```python
w = winner(gs["board"])
if w and not gs["over"]:
    gs["over"]=True; gs["result"]="Draw" if w=="draw" else f"{w} wins"

if not gs["over"] and gs["turn"]=="ai":
    ok,msg,_=_can()
    try:
        if provider=="OpenAI" and ok:
            mv=call_openai(model,gs["board"],gs["ai"],gs["human"],temp,max_tok); _record()
        else:
            if provider=="OpenAI" and not ok: st.warning(msg)
            mv=offline_move(gs["board"],gs["ai"],gs["human"])
        gs["board"][mv.index]=gs["ai"]; gs["turn"]="human"
    except Exception as e:
        st.error(f"AI move error: {e}. Using offline logic.")
        mv=offline_move(gs["board"],gs["ai"],gs["human"])
        gs["board"][mv.index]=gs["ai"]; gs["turn"]="human"
    w=winner(gs["board"])
    if w and not gs["over"]:
        gs["over"]=True; gs["result"]="Draw" if w=="draw" else f"{w} wins"
```

**Detailed flow explanation:** I call `winner` right after my move. If someone won, I stop here. If the game is still live and it is the AI turn, I check `_can()`. When provider is OpenAI and limits allow, I call the API and `_record()`. Otherwise I warn about limits and use the offline brain. On any exception from OpenAI, I show an error and fall back. I always validate the board again after the AI places its mark. This double check ensures status stays in sync with the UI.

---

## Status card

I show a clear status under the board at all times. It tells me whose turn it is or who won the game.

```python
stat = "Your turn" if (not gs["over"] and gs["turn"]=="human") else ("AI thinking..." if not gs["over"] else gs["result"])
card("Game Status", stat)
```

**Why I format it this way:** I keep the message set small and predictable. The status stays visible during the OpenAI call. The same card shows the final result without extra UI. This keeps the page clean and the flow easy to read.

---

## How the offline AI behaves

The minimax engine never loses if I also play perfectly. It will force a draw when I open in the center. It will win if I blunder and allow a fork. This makes the app a nice demo of search on a small state space.

**Why not alpha‑beta here:** For Tic Tac Toe, plain minimax is fast enough. The branching factor is small and the depth is shallow. Adding alpha‑beta would add code with little benefit here. If I switched to Connect Four, I would add pruning and depth caps.

---

## How the OpenAI path behaves

The OpenAI path returns a single integer in JSON. I craft a short system prompt to force structured output. I also cap tokens so responses stay small and cheap. Temperature defaults to a modest value to reduce drift.

**Why I still validate:** Models can return prose, code blocks, or malformed JSON. I handle code fences and parse the JSON safely. I also validate that the move is in `avail(b)`. If anything looks off, I go offline and carry on.

---

## Budget guardrails in practice

Guards are visible and enforced from the sidebar. A cooldown avoids frantic clicking. A daily limit and daily budget prevent long sessions. An hourly cap protects a shared demo on one host.

**What I would add later:** Per‑user counters backed by a small key‑value store. A global budget endpoint with time‑based resets. Separate caps for OpenAI and offline calls. Better UI to explain which rule blocked the move.

---

## Testing ideas I used

I kept tests informal during development. Manual tests covered wins, draws, and illegal moves. I also forced the OpenAI path while hiding the key. The app handled the errors and fell back as expected.

**Quick checks to try yourself:** Start with the center and play perfect defense. You should draw every time against the engine. Try corner openings and allow a fork on purpose. You should see the engine convert the win cleanly.

---

## Ways I would extend this

- Add a training page that visualizes minimax recursion. - Add a two‑player mode with hot‑seat switching. - Persist a simple leaderboard in a small SQLite file. - Theme the board with a dark palette and larger touch targets. - Offer a four‑in‑a‑row variant with a taller grid.

The current code is a compact teaching example. It stays readable while covering many core patterns.

---

## Full source for reference

For convenience I am repeating the full `app.py` below. This matches the code blocks I explained above.

```python
import os, time, json, random, datetime as dt, types, urllib.request
from dataclasses import dataclass
from typing import List, Tuple, Optional
import streamlit as st
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title="Tic Tac Toe", layout="wide")

BUDGET_URL = os.getenv("BUDGET_URL","https://raw.githubusercontent.com/RahulBhattacharya1/shared_config/main/budget.py")
_DEFAULTS = {"COOLDOWN_SECONDS":30,"DAILY_LIMIT":40,"HOURLY_SHARED_CAP":250,"DAILY_BUDGET":1.00,"EST_COST_PER_GEN":1.00,"VERSION":"fallback-local"}
def _fetch(url:str)->dict:
    mod=types.ModuleType("b"); 
    with urllib.request.urlopen(url,timeout=5) as r: code=r.read().decode()
    exec(compile(code,"b","exec"),mod.__dict__); 
    return {k:getattr(mod,k,_DEFAULTS[k]) for k in _DEFAULTS}
def _cfg(ttl=300):
    now=time.time(); cache=st.session_state.get("_b"); ts=st.session_state.get("_bts",0)
    if cache and (now-ts)<ttl: return cache
    try: cfg=_fetch(BUDGET_URL)
    except Exception: cfg=_DEFAULTS.copy()
    cfg["DAILY_BUDGET"]=float(os.getenv("DAILY_BUDGET",cfg["DAILY_BUDGET"]))
    cfg["EST_COST_PER_GEN"]=float(os.getenv("EST_COST_PER_GEN",cfg["EST_COST_PER_GEN"]))
    st.session_state["_b"]=cfg; st.session_state["_bts"]=now; return cfg
_cfg=_cfg(); COOLDOWN_SECONDS=int(_cfg["COOLDOWN_SECONDS"]); DAILY_LIMIT=int(_cfg["DAILY_LIMIT"])
HOURLY_SHARED_CAP=int(_cfg["HOURLY_SHARED_CAP"]); DAILY_BUDGET=float(_cfg["DAILY_BUDGET"])
EST_COST_PER_GEN=float(_cfg["EST_COST_PER_GEN"]); CONFIG_VERSION=str(_cfg["VERSION"])
def _hour_bucket(now=None): now=now or dt.datetime.utcnow(); return now.strftime("%Y-%m-%d-%H")
@st.cache_resource
def _shared_hourly_counters(): return {}
def _init_rl():
    ss=st.session_state; today=dt.date.today().isoformat()
    if ss.get("rl_date")!=today: ss["rl_date"]=today; ss["rl_calls_today"]=0; ss["rl_last_ts"]=0.0
    ss.setdefault("rl_last_ts",0.0); ss.setdefault("rl_calls_today",0)
def _can():
    _init_rl(); ss=st.session_state; now=time.time()
    rem=int(max(0, ss["rl_last_ts"]+COOLDOWN_SECONDS-now)); 
    if rem>0: return False, f"Wait {rem}s before next move.", rem
    if ss["rl_calls_today"]*EST_COST_PER_GEN>=DAILY_BUDGET: return False, f"Daily cost limit reached (${DAILY_BUDGET:.2f}).",0
    if ss["rl_calls_today"]>=DAILY_LIMIT: return False, f"Daily limit reached ({DAILY_LIMIT}).",0
    if HOURLY_SHARED_CAP>0:
        b=_hour_bucket(); c=_shared_hourly_counters()
        if c.get(b,0)>=HOURLY_SHARED_CAP: return False,"Hourly capacity reached.",0
    return True,"",0
def _record():
    ss=st.session_state; ss["rl_last_ts"]=time.time(); ss["rl_calls_today"]+=1
    if HOURLY_SHARED_CAP>0:
        b=_hour_bucket(); c=_shared_hourly_counters(); c[b]=c.get(b,0)+1

@dataclass
class Move: index:int

WIN = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
def winner(b:List[str])->Optional[str]:
    for a,b2,c in WIN:
        if b[a] and b[a]==b[b2]==b[c]: return b[a]
    return "draw" if all(b) else None
def avail(b): return [i for i,v in enumerate(b) if v==""]
def minimax(b, ai, hu, maxing):
    w=winner(b); 
    if w==ai: return 1,None
    if w==hu: return -1,None
    if w=="draw": return 0,None
    best=None
    if maxing:
        sc=-10
        for i in avail(b):
            b[i]=ai; s,_=minimax(b,ai,hu,False); b[i]=""
            if s>sc: sc, best=s,i
        return sc,best
    else:
        sc=10
        for i in avail(b):
            b[i]=hu; s,_=minimax(b,ai,hu,True); b[i]=""
            if s<sc: sc, best=s,i
        return sc,best
def offline_move(b, ai, hu)->Move:
    _,mv=minimax(b[:],ai,hu,True); 
    if mv is None: a=avail(b); mv=random.choice(a) if a else 0
    return Move(mv)

def call_openai(model,b,ai,hu,temp,max_tokens)->Move:
    api=st.secrets.get("OPENAI_API_KEY",""); 
    if not api: raise RuntimeError("OPENAI_API_KEY missing")
    if OpenAI is None: raise RuntimeError("openai package not available")
    client=OpenAI(api_key=api)
    sys="You play perfect Tic Tac Toe. Return JSON with key 'move' (0..8). No prose."
    usr=f"Board: {b}\nAI: {ai}\nHuman: {hu}"
    r=client.chat.completions.create(model=model,temperature=float(temp),max_tokens=int(max_tokens),
                                     messages=[{"role":"system","content":sys},{"role":"user","content":usr}])
    t=r.choices[0].message.content.strip()
    if t.startswith("```"): t=t.strip("`").split("\n",1)[-1].strip()
    mv=int(json.loads(t).get("move",-1))
    if mv not in avail(b): return offline_move(b,ai,hu)
    return Move(mv)

def h2(txt,col): st.markdown(f"<h2 style='margin:.25rem 0 .75rem 0; color:{col}'>{txt}</h2>", unsafe_allow_html=True)
def card(title, sub=""):
    st.markdown(f'''<div style="border:1px solid #e5e7eb;padding:.75rem 1rem;border-radius:10px;margin:.75rem 0;">
    <div style="font-weight:600">{title}</div>{f'<div style="margin-top:.25rem">{sub}</div>' if sub else ''}</div>''', unsafe_allow_html=True)

st.title("Tic Tac Toe")
with st.sidebar:
    st.subheader("Generator")
    provider = st.selectbox("Provider", ["OpenAI","Offline (rule-based)"])
    model    = st.selectbox("Model (OpenAI)", ["gpt-4o-mini","gpt-4o","gpt-4.1-mini"])
    brand    = "#0F62FE"
    temp     = st.slider("Creativity (OpenAI)",0.0,1.0,0.4,0.05)
    max_tok  = st.slider("Max tokens (OpenAI)",256,2048,600,32)

    _init_rl(); ss=st.session_state
    st.markdown("**Usage limits**")
    st.write(f"<span style='font-size:.9rem'>Today: {ss['rl_calls_today']} / {DAILY_LIMIT}</span>", unsafe_allow_html=True)
    if HOURLY_SHARED_CAP>0:
        used=_shared_hourly_counters().get(_hour_bucket(),0)
        st.write(f"<span style='font-size:.9rem'>Hour: {used} / {HOURLY_SHARED_CAP}</span>", unsafe_allow_html=True)
    est=ss['rl_calls_today']*EST_COST_PER_GEN
    st.markdown(f"<span style='font-size:.9rem'>Budget: ${est:.2f} / ${DAILY_BUDGET:.2f}</span><br/>"
                f"<span style='font-size:.8rem;opacity:.8'>Version: {CONFIG_VERSION}</span>", unsafe_allow_html=True)

gs = st.session_state.setdefault("ttt", {"board":[""]*9,"human":"X","ai":"O","turn":"human","over":False,"result":""})
colA,colB=st.columns([1.2,1])
with colB:
    if st.button("New Game"): gs.update({"board":[""]*9,"turn":"human","over":False,"result":""})
    sym = st.radio("You play as",["X","O"],index=0,horizontal=True)
    if sym!=gs["human"] and not any(gs["board"]): gs["human"]=sym; gs["ai"]="O" if sym=="X" else "X"

def cell(i):
    dis=gs["over"] or gs["board"][i]!="" or gs["turn"]!="human"
    if st.button(gs["board"][i] or " ", key=f"c{i}", use_container_width=True, disabled=dis):
        gs["board"][i]=gs["human"]; gs["turn"]="ai"

c1,c2,c3=st.columns(3)
with c1: cell(0); cell(3); cell(6)
with c2: cell(1); cell(4); cell(7)
with c3: cell(2); cell(5); cell(8)

w = winner(gs["board"])
if w and not gs["over"]:
    gs["over"]=True; gs["result"]="Draw" if w=="draw" else f"{w} wins"

if not gs["over"] and gs["turn"]=="ai":
    ok,msg,_=_can()
    try:
        if provider=="OpenAI" and ok:
            mv=call_openai(model,gs["board"],gs["ai"],gs["human"],temp,max_tok); _record()
        else:
            if provider=="OpenAI" and not ok: st.warning(msg)
            mv=offline_move(gs["board"],gs["ai"],gs["human"])
        gs["board"][mv.index]=gs["ai"]; gs["turn"]="human"
    except Exception as e:
        st.error(f"AI move error: {e}. Using offline logic.")
        mv=offline_move(gs["board"],gs["ai"],gs["human"])
        gs["board"][mv.index]=gs["ai"]; gs["turn"]="human"
    w=winner(gs["board"])
    if w and not gs["over"]:
        gs["over"]=True; gs["result"]="Draw" if w=="draw" else f"{w} wins"

stat = "Your turn" if (not gs["over"] and gs["turn"]=="human") else ("AI thinking..." if not gs["over"] else gs["result"])
card("Game Status", stat)
```

---

## Final notes

I aimed for clarity over cleverness in this code. The patterns generalize to many small Streamlit tools. Feel free to fork and adapt for your own demos. If you add variants or themes, I would love to learn from them.
