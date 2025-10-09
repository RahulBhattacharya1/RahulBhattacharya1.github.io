---
layout: default
title: "Creating my AI Book Recommendation Engine"
date: 2024-12-15 09:46:21
categories: [ai]
tags: [python,streamlit,openai]
thumbnail: /assets/images/book.webp
thumbnail_mobile: /assets/images/book_recommendation_sq.webp
demo_link: https://rahuls-ai-book-recommendation-engine.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_book_recommendation_engine
---

It began with a simple thought. I had been reading books in different genres, but I often felt lost when I wanted to choose my next read. The world of books is wide and deep, and I wanted a guide that could help me find titles matching my taste. That was the moment when I realized I needed a personal recommendation engine. I decided to build one using Python, data, and simple machine learning logic. The result is a working AI Book Recommendation Engine that I deployed as a web application with Streamlit. This post explains every file, every function, and every helper used in the project.

## What this post covers

This post explains code blocks and files that I pushed to GitHub to make the app run. The style is simple and direct, because the goal is clarity. I walk through helpers, functions, conditionals, and the user interface. I also show how caching and fallbacks keep the app steady when APIs fail or quotas run out. The explanations aim to stay concrete and consistent without moving between concepts.

## Repository layout I pushed to GitHub

- `app.py` — the full Streamlit app with retrieval, optional GPT re‑rank, and UI.
- `requirements.txt` — pinned libraries so the app installs cleanly anywhere.
- `data/books_catalog.csv` — the catalog the engine searches and filters.
- `README.md` — a short description and run instructions.

Each file supports the others. The app script is the core. The requirements file stabilizes the environment. The dataset gives the system something meaningful to retrieve. The README provides context for anyone landing on the repo.

## Full source of `app.py`

```python
import json
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# ---- ONE client for the whole app ----
# Secrets: set OPENAI_API_KEY, optionally OPENAI_PROJECT
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"].strip()
OPENAI_PROJECT = (st.secrets.get("OPENAI_PROJECT", "") or "").strip() or None

kwargs = {"api_key": OPENAI_API_KEY}
# Only pass project when the key is project-scoped (sk-proj-…)
if OPENAI_API_KEY.startswith("sk-proj-") and OPENAI_PROJECT:
    kwargs["project"] = OPENAI_PROJECT

client = OpenAI(**kwargs)

# ---- Diagnostics (uses the SAME client) ----
with st.expander("Diagnostics", expanded=False):
    st.write({
        "OPENAI_KEY_prefix": OPENAI_API_KEY[:12] + "…",
        "OPENAI_PROJECT": OPENAI_PROJECT or "(none)"
    })
    try:
        ping = client.embeddings.create(model="text-embedding-3-small", input=["ping"])
        st.success("Embeddings OK")
        st.write({"dim": len(ping.data[0].embedding)})
    except Exception as e:
        st.error(f"Embeddings call failed: {type(e).__name__}: {e}")
        body = getattr(e, "response", None)
        if body:
            st.code(str(body), language="json")
    try:
        r = client.responses.create(
            model="gpt-4.1-mini",
            input=[{"role": "user", "content": "Return JSON {\"ok\":true}"}]
        )
        st.success("Responses OK")
        st.code(getattr(r, "output_text", ""), language="json")
    except Exception as e:
        st.error(f"Responses call failed: {type(e).__name__}: {e}")
        body = getattr(e, "response", None)
        if body:
            st.code(str(body), language="json")

st.set_page_config(page_title="Book Recommendation Engine", layout="wide")
st.title("Book Recommendation Engine")

EMBED_MODEL = "text-embedding-3-large"
RERANK_MODEL = "gpt-4o-mini"


# ---------------- Utils ----------------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def _offline_scores(query: str, docs: List[str]) -> np.ndarray:
    """Heuristic fallback when quota is exhausted: token-overlap score."""
    q = set(t for t in query.lower().split() if t.isalpha() or t.isalnum())
    scores = []
    for d in docs:
        dset = set(t for t in d.lower().split() if t.isalpha() or t.isalnum())
        inter = len(q & dset)
        scores.append(inter / max(1, len(q)))
    return np.array(scores, dtype=np.float32)

def _safe_json(s: str) -> Dict:
    try:
        start = s.find("{"); end = s.rfind("}")
        if start != -1 and end != -1:
            s = s[start:end+1]
        data = json.loads(s)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

# ---------------- Data ----------------
@st.cache_resource(show_spinner=True)
def load_catalog() -> pd.DataFrame:
    df = pd.read_csv("data/books_catalog.csv")
    # combine fields for embedding/search
    df["search_text"] = (
        df["title"].fillna("") + " — " +
        df["author"].fillna("") + " — " +
        df["genre"].fillna("") + " — " +
        df["description"].fillna("")
    )
    return df

@st.cache_resource(show_spinner=True)
def embed_catalog(df: pd.DataFrame) -> np.ndarray:
    texts = df["search_text"].tolist()
    try:
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
        vecs = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
        return np.vstack(vecs)
    except Exception as e:
        # Quota or API problem → no embeddings
        st.warning("Embeddings unavailable. Falling back to offline keyword matching.")
        return None  # signal offline

def retrieve_top_k(query: str, df: pd.DataFrame, M: np.ndarray, k: int, genre_filter: List[str]) -> pd.DataFrame:
    base = df if not genre_filter else df[df["genre"].isin(genre_filter)].reset_index(drop=True)
    if M is None:
        # offline scoring
        scores = _offline_scores(query, base["search_text"].tolist())
    else:
        try:
            qvec = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
            q = np.array(qvec, dtype=np.float32)
            # restrict to base index range
            idx = base.index.to_numpy()
            sims = np.array([_cosine(q, M[i]) for i in idx], dtype=np.float32)
            scores = sims
        except Exception:
            scores = _offline_scores(query, base["search_text"].tolist())

    base = base.copy()
    base["score"] = scores
    base = base.sort_values("score", ascending=False).head(k).reset_index(drop=True)
    return base

def rerank_with_gpt(query: str, candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Ask the model to order the given candidates and provide brief reasons.
    Returns candidates with 'rank' and 'reason' columns.
    """
    if candidates.empty:
        return candidates

    items = []
    for _, r in candidates.iterrows():
        items.append({
            "id": int(r["id"]),
            "title": r["title"],
            "author": r["author"],
            "genre": r["genre"],
            "description": r["description"]
        })

    system = (
        "You are a recommendation assistant. Re-rank book candidates for the user's summary. "
        "Return JSON: {\"order\":[{\"id\":<id>,\"reason\":\"...\"}, ...]}. Keep reasons short."
    )
    user = {
        "query": query,
        "candidates": items
    }

    try:
        resp = client.responses.create(
            model=RERANK_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user)}
            ]
        )
        raw = getattr(resp, "output_text", "") or ""
        data = _safe_json(raw)
        order = data.get("order", [])
        reasons = {int(x.get("id")): str(x.get("reason","")).strip() for x in order if "id" in x}
        # add rank by appearance
        ranking = {bid: i for i, bid in enumerate(reasons.keys(), start=1)}
        out = candidates.copy()
        out["rank"] = out["id"].map(ranking).fillna(len(out)+1).astype(int)
        out["reason"] = out["id"].map(reasons).fillna("")
        return out.sort_values(["rank", "score"], ascending=[True, False]).reset_index(drop=True)
    except Exception as e:
        if "insufficient_quota" in str(e):
            st.info("Offline demo: GPT re-rank unavailable; showing embedding/keyword ranking.")
            candidates["rank"] = np.arange(1, len(candidates)+1)
            candidates["reason"] = ""
            return candidates
        raise

# ---------------- UI ----------------
df = load_catalog()
M = embed_catalog(df)

st.sidebar.header("Settings")
k = st.sidebar.slider("Number of recommendations", 3, 10, 5, 1)
use_rerank = st.sidebar.checkbox("Use GPT re-rank + reasons", value=True)
genres = sorted(df["genre"].dropna().unique().tolist())
chosen_genres = st.sidebar.multiselect("Filter by genre (optional)", genres, default=[])

summary = st.text_area(
    "Describe the book you want (short summary, themes, vibes):",
    height=140,
    placeholder="Example: character-driven mystery with archival research, slow-burn tension, thoughtful prose"
)

col1, col2 = st.columns(2)
with col1:
    run = st.button("Recommend", type="primary")
with col2:
    clear = st.button("Clear", on_click=lambda: st.session_state.update({"history": []}))

if run and summary.strip():
    # Retrieve
    hits = retrieve_top_k(summary, df, M, k*2 if use_rerank else k, chosen_genres)
    # Re-rank (optional)
    if use_rerank:
        hits = rerank_with_gpt(summary, hits).head(k)
    else:
        hits = hits.head(k)

    st.subheader("Recommendations")
    for _, r in hits.iterrows():
        st.markdown(f"**{r['title']}** — {r['author']}  \n*{r['genre']}*")
        st.markdown(r['description'])
        if r.get("reason"):
            st.caption(f"Why: {r['reason']}")
        if pd.notna(r.get("link")) and str(r.get("link")).strip():
            st.write(f"[Learn more]({r['link']})")
        st.markdown("---")

    # Save to session + offer CSV
    st.session_state.setdefault("history", []).insert(0, {
        "query": summary, "results": hits[["title","author","genre","score"]].to_dict(orient="records")
    })
    csv = hits[["title","author","genre","score","link"]].to_csv(index=False)
    st.download_button("Download recommendations (CSV)", csv, file_name="book_recs.csv", mime="text/csv")

st.caption("Embeddings are used for retrieval; optional GPT re-ranking adds short reasons. When API quota is unavailable, the app falls back to keyword matching.")

```


## Imports and type hints


The app starts by importing standard and third‑party packages. The imports declare the tools used for arrays, data frames, the web UI, and the OpenAI client. Keeping imports at the top makes intent visible and helps static checkers.

```python
import json
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# ---- ONE client for the whole app ----
# Secrets: set OPENAI_API_KEY, optionally OPENAI_PROJECT
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"].strip()
OPENAI_PROJECT = (st.secrets.get("OPENAI_PROJECT", "") or "").strip() or None

kwargs = {"api_key": OPENAI_API_KEY}
# Only pass project when the key is project-scoped (sk-proj-…)
if OPENAI_API_KEY.startswith("sk-proj-") and OPENAI_PROJECT:
    kwargs["project"] = OPENAI_PROJECT

client = OpenAI(**kwargs)

# ---- Diagnostics (uses the SAME client) ----
with st.expander("Diagnostics", expanded=False):
    st.write({
        "OPENAI_KEY_prefix": OPENAI_API_KEY[:12] + "…",
        "OPENAI_PROJECT": OPENAI_PROJECT or "(none)"
    })
    try:
        ping = client.embeddings.create(model="text-embedding-3-small", input=["ping"])
        st.success("Embeddings OK")
        st.write({"dim": len(ping.data[0].embedding)})
```


## Model and configuration constants


Two small constants define the embedding model and the re‑rank model. Choosing them in one place makes it easy to switch models during testing. The names are descriptive and help others understand the plan.

```python
EMBED_MODEL = "text-embedding-3-large"
RERANK_MODEL = "gpt-4o-mini"
```


## Client and secrets handling


The app reads secrets from `st.secrets`, which is supported both locally and on Streamlit Community Cloud. The logic detects whether the key is project‑scoped and only then adds the project id. This keeps the same client across the app and avoids repeated initialization.

```python
# Secrets: set OPENAI_API_KEY, optionally OPENAI_PROJECT
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"].strip()
OPENAI_PROJECT = (st.secrets.get("OPENAI_PROJECT", "") or "").strip() or None

kwargs = {"api_key": OPENAI_API_KEY}
# Only pass project when the key is project-scoped (sk-proj-…)
if OPENAI_API_KEY.startswith("sk-proj-") and OPENAI_PROJECT:
    kwargs["project"] = OPENAI_PROJECT

client = OpenAI(**kwargs)

# ---- Diagnostics (uses the SAME client) ----
```
This block also seeds a shared `client` that later functions reuse. A single client instance reduces overhead and keeps diagnostics simple.


## Diagnostics expander


A compact diagnostics panel verifies that embeddings and responses can be called. It shows quick success messages and prints response bodies when failures happen. The goal is to surface issues early so the main UI stays clean.

```python
with st.expander("Diagnostics", expanded=False):
    st.write({
        "OPENAI_KEY_prefix": OPENAI_API_KEY[:12] + "…",
        "OPENAI_PROJECT": OPENAI_PROJECT or "(none)"
    })
    try:
        ping = client.embeddings.create(model="text-embedding-3-small", input=["ping"])
        st.success("Embeddings OK")
        st.write({"dim": len(ping.data[0].embedding)})
    except Exception as e:
        st.error(f"Embeddings call failed: {type(e).__name__}: {e}")
        body = getattr(e, "response", None)
        if body:
            st.code(str(body), language="json")
    try:
        r = client.responses.create(
            model="gpt-4.1-mini",
            input=[{"role": "user", "content": "Return JSON {\"ok\":true}"}]
        )
        st.success("Responses OK")
        st.code(getattr(r, "output_text", ""), language="json")
    except Exception as e:
        st.error(f"Responses call failed: {type(e).__name__}: {e}")
        body = getattr(e, "response", None)
        if body:
            st.code(str(body), language="json")
```
The try/except pairs in this panel catch API failures and display helpful context. This pattern keeps the rest of the app focused on user flow.


## Page setup and title


The layout is wide to make room for multi‑column text and longer descriptions. A clear title sets the tone and matches the repository name.

```python
st.set_page_config(page_title="Book Recommendation Engine", layout="wide")
st.title("Book Recommendation Engine")
```


## Function: `_cosine()`

```python
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

```
This helper returns the cosine similarity between two vectors. It computes the dot product and divides by the product of magnitudes. A guard handles the rare zero‑vector case and returns a neutral score. Cosine similarity works well for embeddings because it tracks angle rather than length.


## Function: `_offline_scores()`

```python
def _offline_scores(query: str, docs: List[str]) -> np.ndarray:
    """Heuristic fallback when quota is exhausted: token-overlap score."""
    q = set(t for t in query.lower().split() if t.isalpha() or t.isalnum())
    scores = []
    for d in docs:
        dset = set(t for t in d.lower().split() if t.isalpha() or t.isalnum())
        inter = len(q & dset)
        scores.append(inter / max(1, len(q)))
    return np.array(scores, dtype=np.float32)

```
This helper creates a lightweight fallback score when embeddings are unavailable. It may tokenize the query and count overlaps with document text to approximate relevance. The goal is not perfection but a reasonable ordering for demos or quota outages. Keeping this logic separate makes the fallback easy to improve later.


## Function: `_safe_json()`

```python
def _safe_json(s: str) -> Dict:
    try:
        start = s.find("{"); end = s.rfind("}")
        if start != -1 and end != -1:
            s = s[start:end+1]
        data = json.loads(s)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

# ---------------- Data ----------------
@st.cache_resource(show_spinner=True)
```
This utility parses a JSON string while handling truncated or noisy content. It searches for the last balanced braces and attempts a safe load. When parsing fails, it yields an empty object to protect the UI. This prevents hard errors from breaking the user flow.


## Function: `load_catalog()`

```python
def load_catalog() -> pd.DataFrame:
    df = pd.read_csv("data/books_catalog.csv")
    # combine fields for embedding/search
    df["search_text"] = (
        df["title"].fillna("") + " — " +
        df["author"].fillna("") + " — " +
        df["genre"].fillna("") + " — " +
        df["description"].fillna("")
    )
    return df

@st.cache_resource(show_spinner=True)
```
This cached loader reads the CSV catalog into a DataFrame. Caching avoids repeated I/O when users change settings or issue multiple queries. It may also normalize column types so later steps are predictable. Loading is fast and safe, so the rest of the app can focus on retrieval.


## Function: `embed_catalog()`

```python
def embed_catalog(df: pd.DataFrame) -> np.ndarray:
    texts = df["search_text"].tolist()
    try:
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
        vecs = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
        return np.vstack(vecs)
    except Exception as e:
        # Quota or API problem → no embeddings
        st.warning("Embeddings unavailable. Falling back to offline keyword matching.")
        return None  # signal offline

```
This cached step creates vectors for each book row using the selected embedding model. It returns a matrix aligned with the DataFrame so indexing stays simple. When the API fails, the app can fall back to offline scoring. Caching the matrix avoids paying for repeated embedding calls.


## Function: `retrieve_top_k()`

```python
def retrieve_top_k(query: str, df: pd.DataFrame, M: np.ndarray, k: int, genre_filter: List[str]) -> pd.DataFrame:
    base = df if not genre_filter else df[df["genre"].isin(genre_filter)].reset_index(drop=True)
    if M is None:
        # offline scoring
        scores = _offline_scores(query, base["search_text"].tolist())
    else:
        try:
            qvec = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
            q = np.array(qvec, dtype=np.float32)
            # restrict to base index range
            idx = base.index.to_numpy()
            sims = np.array([_cosine(q, M[i]) for i in idx], dtype=np.float32)
            scores = sims
        except Exception:
            scores = _offline_scores(query, base["search_text"].tolist())

    base = base.copy()
    base["score"] = scores
    base = base.sort_values("score", ascending=False).head(k).reset_index(drop=True)
    return base

```
This function handles the first‑stage retrieval. It optionally filters by genre, computes a similarity score between the query embedding and each book vector, and returns the top‑k candidates. It keeps the shape of the DataFrame intact and adds a score column for insight. The output is small and ready for optional re‑ranking.


## Function: `rerank_with_gpt()`

```python
def rerank_with_gpt(query: str, candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Ask the model to order the given candidates and provide brief reasons.
    Returns candidates with 'rank' and 'reason' columns.
    """
    if candidates.empty:
        return candidates

    items = []
    for _, r in candidates.iterrows():
        items.append({
            "id": int(r["id"]),
            "title": r["title"],
            "author": r["author"],
            "genre": r["genre"],
            "description": r["description"]
        })

    system = (
        "You are a recommendation assistant. Re-rank book candidates for the user's summary. "
        "Return JSON: {\"order\":[{\"id\":<id>,\"reason\":\"...\"}, ...]}. Keep reasons short."
    )
    user = {
        "query": query,
        "candidates": items
    }

    try:
        resp = client.responses.create(
            model=RERANK_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user)}
            ]
        )
        raw = getattr(resp, "output_text", "") or ""
        data = _safe_json(raw)
        order = data.get("order", [])
        reasons = {int(x.get("id")): str(x.get("reason","")).strip() for x in order if "id" in x}
        # add rank by appearance
        ranking = {bid: i for i, bid in enumerate(reasons.keys(), start=1)}
        out = candidates.copy()
        out["rank"] = out["id"].map(ranking).fillna(len(out)+1).astype(int)
        out["reason"] = out["id"].map(reasons).fillna("")
        return out.sort_values(["rank", "score"], ascending=[True, False]).reset_index(drop=True)
    except Exception as e:
        if "insufficient_quota" in str(e):
            st.info("Offline demo: GPT re-rank unavailable; showing embedding/keyword ranking.")
            candidates["rank"] = np.arange(1, len(candidates)+1)
            candidates["reason"] = ""
            return candidates
        raise

# ---------------- UI ----------------
df = load_catalog()
M = embed_catalog(df)

st.sidebar.header("Settings")
k = st.sidebar.slider("Number of recommendations", 3, 10, 5, 1)
use_rerank = st.sidebar.checkbox("Use GPT re-rank + reasons", value=True)
genres = sorted(df["genre"].dropna().unique().tolist())
chosen_genres = st.sidebar.multiselect("Filter by genre (optional)", genres, default=[])

summary = st.text_area(
    "Describe the book you want (short summary, themes, vibes):",
    height=140,
    placeholder="Example: character-driven mystery with archival research, slow-burn tension, thoughtful prose"
)

col1, col2 = st.columns(2)
with col1:
    run = st.button("Recommend", type="primary")
with col2:
    clear = st.button("Clear", on_click=lambda: st.session_state.update({"history": []}))

if run and summary.strip():
    # Retrieve
    hits = retrieve_top_k(summary, df, M, k*2 if use_rerank else k, chosen_genres)
    # Re-rank (optional)
    if use_rerank:
        hits = rerank_with_gpt(summary, hits).head(k)
    else:
        hits = hits.head(k)

    st.subheader("Recommendations")
    for _, r in hits.iterrows():
        st.markdown(f"**{r['title']}** — {r['author']}  \n*{r['genre']}*")
        st.markdown(r['description'])
        if r.get("reason"):
            st.caption(f"Why: {r['reason']}")
        if pd.notna(r.get("link")) and str(r.get("link")).strip():
            st.write(f"[Learn more]({r['link']})")
        st.markdown("---")

    # Save to session + offer CSV
    st.session_state.setdefault("history", []).insert(0, {
        "query": summary, "results": hits[["title","author","genre","score"]].to_dict(orient="records")
    })
    csv = hits[["title","author","genre","score","link"]].to_csv(index=False)
    st.download_button("Download recommendations (CSV)", csv, file_name="book_recs.csv", mime="text/csv")

st.caption("Embeddings are used for retrieval; optional GPT re-ranking adds short reasons. When API quota is unavailable, the app falls back to keyword matching.")
```
This function asks a compact GPT model to re‑order a small candidate set. It builds a system prompt and passes candidate snippets as input. The function parses the response into a DataFrame and merges reasons back onto rows. When the call fails or quota is low, the app gracefully uses the candidate order.


## Sidebar settings and main actions


    The sidebar groups configuration so the main pane stays focused on results. It can include genre filters, top‑k selection, and a rerank toggle. Grouping controls in one place reduces cognitive load and helps scanning.

    ```python
    st.sidebar.header("Settings")
k = st.sidebar.slider("Number of recommendations", 3, 10, 5, 1)
use_rerank = st.sidebar.checkbox("Use GPT re-rank + reasons", value=True)
genres = sorted(df["genre"].dropna().unique().tolist())
chosen_genres = st.sidebar.multiselect("Filter by genre (optional)", genres, default=[])

summary = st.text_area(
    "Describe the book you want (short summary, themes, vibes):",
    height=140,
    placeholder="Example: character-driven mystery with archival research, slow-burn tension, thoughtful prose"
)

col1, col2 = st.columns(2)
with col1:
    run = st.button("Recommend", type="primary")
with col2:
    clear = st.button("Clear", on_click=lambda: st.session_state.update({"history": []}))

if run and summary.strip():
    # Retrieve
    hits = retrieve_top_k(summary, df, M, k*2 if use_rerank else k, chosen_genres)
    # Re-rank (optional)
    if use_rerank:
        hits = rerank_with_gpt(summary, hits).head(k)
    else:
        hits = hits.head(k)

    st.subheader("Recommendations")
    for _, r in hits.iterrows():
        st.markdown(f"**{r['title']}** — {r['author']}  \n*{r['genre']}*")
        st.markdown(r['description'])
        if r.get("reason"):
            st.caption(f"Why: {r['reason']}")
        if pd.notna(r.get("link")) and str(r.get("link")).strip():
            st.write(f"[Learn more]({r['link']})")
        st.markdown("---")

    # Save to session + offer CSV
    st.session_state.setdefault("history", []).insert(0, {
        "query": summary, "results": hits[["title","author","genre","score"]].to_dict(orient="records")
    })
    csv = hits[["title","author","genre","score","link"]].to_csv(index=False)
    st.download_button("Download recommendations (CSV)", csv, file_name="book_recs.csv", mime="text/csv")

st.caption("Embeddings are used for retrieval; optional GPT re-ranking adds short reasons. When API quota is unavailable, the app falls back to keyword matching.")
    ```


## Session history and downloads


    The app saves a compact record of recent queries and results in `st.session_state`. Keeping a short history helps compare prompts without retyping. A download button exports the current list as CSV for later use.

    ```python
        run = st.button("Recommend", type="primary")
with col2:
    clear = st.button("Clear", on_click=lambda: st.session_state.update({"history": []}))

if run and summary.strip():
    # Retrieve
    hits = retrieve_top_k(summary, df, M, k*2 if use_rerank else k, chosen_genres)
    # Re-rank (optional)
    if use_rerank:
        hits = rerank_with_gpt(summary, hits).head(k)
    else:
        hits = hits.head(k)

    st.subheader("Recommendations")
    for _, r in hits.iterrows():
        st.markdown(f"**{r['title']}** — {r['author']}  \n*{r['genre']}*")
        st.markdown(r['description'])
        if r.get("reason"):
            st.caption(f"Why: {r['reason']}")
        if pd.notna(r.get("link")) and str(r.get("link")).strip():
            st.write(f"[Learn more]({r['link']})")
        st.markdown("---")

    # Save to session + offer CSV
    st.session_state.setdefault("history", []).insert(0, {
        "query": summary, "results": hits[["title","author","genre","score"]].to_dict(orient="records")
    })
    ```


## Conditionals and fallbacks I deliberately added

The following small snippets show key decisions in the flow. Each conditional protects a user experience edge case or a quota edge case.

```python
kwargs = {"api_key": OPENAI_API_KEY}
# Only pass project when the key is project-scoped (sk-proj-…)
if OPENAI_API_KEY.startswith("sk-proj-") and OPENAI_PROJECT:
    kwargs["project"] = OPENAI_PROJECT

client = OpenAI(**kwargs)

        st.error(f"Embeddings call failed: {type(e).__name__}: {e}")
        body = getattr(e, "response", None)
        if body:
            st.code(str(body), language="json")
    try:
        r = client.responses.create(

        st.error(f"Responses call failed: {type(e).__name__}: {e}")
        body = getattr(e, "response", None)
        if body:
            st.code(str(body), language="json")

st.set_page_config(page_title="Book Recommendation Engine", layout="wide")

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

    try:
        start = s.find("{"); end = s.rfind("}")
        if start != -1 and end != -1:
            s = s[start:end+1]
        data = json.loads(s)
        return data if isinstance(data, dict) else {}

def retrieve_top_k(query: str, df: pd.DataFrame, M: np.ndarray, k: int, genre_filter: List[str]) -> pd.DataFrame:
    base = df if not genre_filter else df[df["genre"].isin(genre_filter)].reset_index(drop=True)
    if M is None:
        # offline scoring
        scores = _offline_scores(query, base["search_text"].tolist())
    else:

        # offline scoring
        scores = _offline_scores(query, base["search_text"].tolist())
    else:
        try:
            qvec = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
            q = np.array(qvec, dtype=np.float32)

    if candidates.empty:
        return candidates

    items = []

        return out.sort_values(["rank", "score"], ascending=[True, False]).reset_index(drop=True)
    except Exception as e:
        if "insufficient_quota" in str(e):
            st.info("Offline demo: GPT re-rank unavailable; showing embedding/keyword ranking.")
            candidates["rank"] = np.arange(1, len(candidates)+1)
            candidates["reason"] = ""

    clear = st.button("Clear", on_click=lambda: st.session_state.update({"history": []}))

if run and summary.strip():
    # Retrieve
    hits = retrieve_top_k(summary, df, M, k*2 if use_rerank else k, chosen_genres)
    # Re-rank (optional)

    hits = retrieve_top_k(summary, df, M, k*2 if use_rerank else k, chosen_genres)
    # Re-rank (optional)
    if use_rerank:
        hits = rerank_with_gpt(summary, hits).head(k)
    else:
        hits = hits.head(k)

    if use_rerank:
        hits = rerank_with_gpt(summary, hits).head(k)
    else:
        hits = hits.head(k)

    st.subheader("Recommendations")

        st.markdown(f"**{r['title']}** — {r['author']}  \n*{r['genre']}*")
        st.markdown(r['description'])
        if r.get("reason"):
            st.caption(f"Why: {r['reason']}")
        if pd.notna(r.get("link")) and str(r.get("link")).strip():
            st.write(f"[Learn more]({r['link']})")

        if r.get("reason"):
            st.caption(f"Why: {r['reason']}")
        if pd.notna(r.get("link")) and str(r.get("link")).strip():
            st.write(f"[Learn more]({r['link']})")
        st.markdown("---")

```
The checks are narrow and localized. It returns early when inputs are missing, or switches to a safe fallback when an external call fails.


## Exact `requirements.txt` used

```python
streamlit>=1.35
pandas>=2.0
numpy>=1.26
openai>=1.40

```
Pinning versions reduces surprise across machines and cloud runners. Streamlit and Pandas are core; NumPy supports vector math; the OpenAI client powers embeddings and optional re‑rank.


## Dataset: `data/books_catalog.csv` (header and first few rows)

I keep the dataset in a simple CSV so it is easy to view, edit, and extend. The fields are plain and map directly to what the UI shows.
```python
id,title,author,genre,description,link
1,The Silent City,A. Marlow,Science Fiction,"A salvage pilot explores a derelict arcology orbiting a dying world and uncovers an AI with a forgotten purpose.",https://example.com/silent-city
2,Harvest of Crows,L. Benton,Thriller,"A journalist returns to a coastal town to investigate a string of disappearances tied to an old corporate scandal.",https://example.com/harvest-crows
3,Under Glass,E. Noor,Literary,"Three siblings clean out their grandmother’s greenhouse and find letters that reshape the family’s past.",https://example.com/under-glass
4,Clockwork Harbor,J. Wilde,Fantasy,"A tinker and a sea-witch must repair a storm engine before it floods their clockwork port.",https://example.com/clockwork-harbor
5,Fault Lines,R. Ortega,Science Fiction,"In a city sitting on a quantum seam, a seismologist maps tremors that rewrite memories.",https://example.com/fault-lines
6,Paper Bridges,M. Iwata,Contemporary,"A new teacher builds trust with a class of refugee students through a zine project.",https://example.com/paper-bridges
7,The Last Archivist,P. Demir,Mystery,"A retiring librarian races to decode marginalia that points to a long-buried crime.",https://example.com/last-archivist
8,Ember Market,S. Qadir,Fantasy,"A street broker trades in bottled emotions and must clear a debt to a fire cult.",https://example.com/ember-market
9,Northbound Light,K. Silva,Romance,"Two park rangers fall for each other while restoring trails in the midnight sun.",https://example.com/northbound-light
10,Grain of the Sky,T. Velasquez,Science Fiction,"Terraformers on a brittle moon debate whether to preserve the native crystalline life.",https://example.com/grain-sky
```
The `description` column gives the reranker enough context to ground its reasons. The `link` field lets the UI provide a path to learn more.


## README highlights

The README in the repo is brief by design. It states the goal and how to run the app locally or on Streamlit Cloud. Keeping it short avoids drift and keeps the source of truth inside `app.py`.
```python
# Book Recommendation Engine (Streamlit + OpenAI)

Given a short summary or vibe description, this app returns top-K book recommendations.

## Modes
- **Embeddings Retrieval**: embed catalog + query, rank by cosine similarity.
- **GPT Re-rank (optional)**: refine top candidates and add short reasons.
- **Offline fallback**: if API quota is exhausted, the app uses simple keyword overlap.

## Deploy (no local setup)
1. Push this repo to GitHub.
2. In Streamlit Community Cloud, create an app with `streamlit_app.py`.
3. In **Settings → Secrets**, add:
OPENAI_API_KEY = "sk-..."
4. Deploy.

## Files
.
├─ streamlit_app.py
├─ requirements.txt
├─ data/
│ └─ books_catalog.csv
└─ README.md


## Tips
- Expand `data/books_catalog.csv` with 200–1000 rows for better coverage.
- Keep descriptions concise but specific (themes, stakes, setting).
- You can filter by genre and adjust the number of results.
```


## What I uploaded to GitHub, exactly


I added all four paths to the repository and pushed them to the default branch. The dataset lives in a `data/` folder so it is naturally grouped. The app expects this relative path, which keeps things portable. The requirements file sits at the root so deployment tools find it without extra flags. The README also lives at the root because it is the first thing most visitors see.


## Run locally for testing


1. Create and activate a clean virtual environment.
2. `pip install -r requirements.txt`
3. Create `.streamlit/secrets.toml` and add `OPENAI_API_KEY="sk-..."` and optionally `OPENAI_PROJECT="..."`.
4. `streamlit run app.py`
5. Open the local URL that Streamlit prints.

Local runs help confirm that the dataset loads, the cache is warm, and fallbacks behave as expected when the key is missing.


## Deploy on Streamlit Community Cloud


1. Push the repository to GitHub.
2. On Streamlit Community Cloud, select the repo and branch, and set `app.py` as the entry point.
3. Add `OPENAI_API_KEY` (and optionally `OPENAI_PROJECT`) under **Secrets**.
4. Deploy. First run will build the environment and cache the catalog and embeddings.
5. Use the diagnostics expander to verify connectivity and quotas.

Secrets live in the cloud workspace and do not enter the repository history. This keeps the project safe and shareable.


## Design choices that kept the app simple


- **Small, cached helpers**: Loading and embedding are isolated and cached so retries are cheap.
- **Retrieval‑then‑re‑rank**: Embeddings give a fast, broad first pass; GPT re‑ranks a small set for nuance.
- **Explainers**: Reasons are attached to items when re‑rank is active so results feel grounded.
- **Graceful degradation**: Offline keyword matching keeps the demo alive when APIs are unavailable.
- **Readable UI**: Minimal widgets, clear headings, and compact lists focus attention on books, not chrome.


## Why cosine similarity for embeddings


Cosine similarity measures the angle between two vectors, which is robust to scale differences. In practice, it highlights direction in the embedding space, which often encodes semantic meaning. The dot product captures alignment, and the normalization by magnitudes prevents long vectors from dominating. This is a common baseline that is easy to reason about and fast to compute.


## Error handling patterns in this app


- Wrap external calls in `try/except` and display a short message with exception type.
- Print the response body when available to speed up debugging.
- Fall back to offline scoring when embeddings or re‑rank fail.
- Guard against empty inputs and empty candidate sets to avoid index errors.
- Keep state in `st.session_state` only when it helps user flow, not for core logic.


## Ways to extend this project next


- Add a multi‑select for genres and boost items that match more than one tag.
- Accept a free‑text mood field and blend it with the main query using weighted scores.
- Persist history to a small JSON file to survive app restarts.
- Add a simple A/B switch to compare two embedding models on the same query.
- Support user ratings and a tiny matrix‑factorization baseline to personalize beyond content.


## Closing notes


Building this app was about shaping a clear path from input to result. Small helpers do focused work and return simple structures. Caching keeps the UI responsive and reduces API traffic. The final flow explains itself: check diagnostics, enter a query, adjust filters, and review reasoned results. That clarity is what I wanted when the idea first appeared.
