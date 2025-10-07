---
layout: default
title: "Creating my Fake News Detector"
date: 2024-08-23 12:34:19
categories: [ai]
tags: [python,streamlit,openai]
thumbnail: /assets/images/news.webp
thumbnail_mobile: /assets/images/fake_news_sq.webp
demo_link: https://rahuls-ai-fake-news-detector.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_fake_news_detector
---

I once stumbled upon an online article that looked reliable at first. The formatting was polished and the headline was striking. Yet as I kept reading, some statements did not align with facts I already knew. That personal experience inspired me to imagine a tool that could highlight questionable claims. I decided to create a Fake News Detector using Streamlit and OpenAI.


The project became a way to combine a lightweight user interface with the reasoning of language models. I structured the repository with a main Python script, a dataset of seed examples, and configuration files. The rest of this post walks step by step through every block of code. Each code snippet is followed by a clear explanation of what it does and how it fits in.


## requirements.txt


```python
streamlit
openai
pandas
numpy
```


This file lists the essential dependencies. Streamlit builds the interactive web application. OpenAI provides access to large language models and embeddings. Pandas and NumPy support data handling and numeric operations. Keeping the list short ensures installation is straightforward.


## README.md


```python
# Fake News Detector
A simple web app to detect potential fake news using OpenAI models and Streamlit.
```


The README introduces the project briefly. It makes clear that this is a Streamlit app powered by OpenAI models. Such a minimal description is enough for someone browsing GitHub to understand the intent. Documentation, even if short, is a sign of a usable repository.


## data/seed_examples_fake_news.csv


This CSV file stores labeled examples of fake and real news. These examples serve as a reference when comparing embeddings. By having some initial labeled cases, the app can provide more consistent classification. Even a small dataset can make outputs steadier across runs.


## app.py Detailed Walkthrough


### Code Block 1


```python
import json
import math
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI


# ---------- Page + globals ----------
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("Fake News Detector")

# OpenAI client from Streamlit secrets (set in Streamlit Cloud)
# Secrets UI keeps your key out of Git history.
# Docs: https://platform.openai.com/docs/api-reference  and Streamlit secrets docs.
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

GPT_MODEL_DEFAULT = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-small"  # fast, inexpensive


# ---------- Utilities ----------
```

This block handles configuration or inline logic. 

It sets up environment, constants, or streamlit UI elements. 

Such code runs at the top level whenever the app executes. 

It prepares the context so later functions can work without errors.


### Code Block 2


```python
def _safe_parse_json(s: str) -> Dict:
    """
    Accepts a model's raw string and tries to produce a dict with the keys we need.
    Falls back to a conservative guess if parsing fails.
    """
    try:
        # Try to find a JSON object if extra text surrounds it
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1:
            s = s[start:end + 1]
        data = json.loads(s)
        if not isinstance(data, dict):
            raise ValueError("Top-level JSON is not an object.")
        return data
    except Exception:
        return {"label": "REAL", "confidence": 0.5, "rationale": "Could not parse model JSON; defaulted."}
```

The function **_safe_parse_json** is defined here. It plays a specific role in the workflow. 

It accepts inputs, performs structured logic, and returns outputs in a predictable form. 

By isolating this logic in a helper, the rest of the application code stays simpler and easier to read. 

This modular approach also allows re-use and safer debugging.


### Code Block 3


```python
def _format_output_card(label: str, confidence: float, rationale: str, show_reason: bool):
    label_badge = "🟢 REAL" if label.upper() == "REAL" else "🔴 FAKE"
    st.subheader("Result")
    st.markdown(f"**Label:** {label_badge}")
    st.markdown(f"**Confidence:** {confidence:.2f}")
    if show_reason and rationale:
        st.markdown(f"**Rationale:** {rationale}")
```

The function **_format_output_card** is defined here. It plays a specific role in the workflow. 

It accepts inputs, performs structured logic, and returns outputs in a predictable form. 

By isolating this logic in a helper, the rest of the application code stays simpler and easier to read. 

This modular approach also allows re-use and safer debugging.


### Code Block 4


```python
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------- Embeddings seed bank (cached) ----------
@st.cache_resource(show_spinner=True)
```

The function **_cosine** is defined here. It plays a specific role in the workflow. 

It accepts inputs, performs structured logic, and returns outputs in a predictable form. 

By isolating this logic in a helper, the rest of the application code stays simpler and easier to read. 

This modular approach also allows re-use and safer debugging.


### Code Block 5


```python
def load_seed_bank() -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]:
    """
    Loads seed examples from CSV, gets embeddings once, and pre-computes label centroids.
    Returns (df_with_vectors, matrix_vectors, centroids_by_label).
    """
    df = pd.read_csv("data/seed_examples_fake_news.csv")
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)

    # Create embeddings in small batches to stay efficient.
    # API ref: https://platform.openai.com/docs/api-reference/embeddings
    texts = df["text"].tolist()
    vecs: List[np.ndarray] = []

    # The embeddings endpoint supports batching; we keep it simple here.
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    for item in resp.data:
        vecs.append(np.array(item.embedding, dtype=np.float32))

    M = np.vstack(vecs)  # [N, D]
    df["__vec_index"] = np.arange(len(df))

    # Compute centroids per label
    centroids: Dict[str, np.ndarray] = {}
    for label in sorted(df["label"].unique()):
        idx = df.index[df["label"] == label].tolist()
        if not idx:
            continue
        arr = M[idx]
        centroids[label] = arr.mean(axis=0)
    return df, M, centroids


# ---------- Classifiers ----------
```

The function **load_seed_bank** is defined here. It plays a specific role in the workflow. 

It accepts inputs, performs structured logic, and returns outputs in a predictable form. 

By isolating this logic in a helper, the rest of the application code stays simpler and easier to read. 

This modular approach also allows re-use and safer debugging.


### Code Block 6


```python
def classify_with_gpt(text: str, model: str = GPT_MODEL_DEFAULT) -> Dict:
    """
    Few-shot JSON classification using the Responses API.
    Returns dict with keys: label, confidence, rationale.
    """
    system = (
        "You are a strict news veracity classifier. "
        "Return compact JSON with keys: label (REAL|FAKE), confidence (0..1), rationale (short). "
        "Be decisive. No surrounding prose."
    )

    fewshots = [
        {"role": "user", "content": "TEXT: CDC confirms water can cure COVID-19 overnight"},
        {"role": "assistant", "content": '{"label":"FAKE","confidence":0.92,"rationale":"Extraordinary medical claim with no credible source."}'},
        {"role": "user", "content": "TEXT: WHO releases updated influenza guidance for the upcoming season"},
        {"role": "assistant", "content": '{"label":"REAL","confidence":0.88,"rationale":"Routine guidance from a credible public health body."}'},
    ]

    messages = [{"role": "system", "content": system}] + fewshots + [
        {"role": "user", "content": f"TEXT: {text}"}
    ]

    # Responses API (Python): https://platform.openai.com/docs/api-reference
    resp = client.responses.create(model=model, input=messages)

    # The SDK exposes response text via .output_text; if not present, stitch from content parts.
    raw = getattr(resp, "output_text", None)
    if raw is None:
        try:
            parts = []
            for item in resp.output:
                if hasattr(item, "content") and item.content:
                    for c in item.content:
                        if getattr(c, "type", "") == "output_text":
                            parts.append(c.text)
            raw = "".join(parts) if parts else ""
        except Exception:
            raw = ""

    data = _safe_parse_json(raw or "")
    # Normalize / validate
    label = str(data.get("label", "REAL")).upper()
    if label not in {"REAL", "FAKE"}:
        label = "REAL"
    try:
        conf = float(data.get("confidence", 0.5))
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    rationale = str(data.get("rationale", ""))
    return {"label": label, "confidence": conf, "rationale": rationale}
```

The function **classify_with_gpt** is defined here. It plays a specific role in the workflow. 

It accepts inputs, performs structured logic, and returns outputs in a predictable form. 

By isolating this logic in a helper, the rest of the application code stays simpler and easier to read. 

This modular approach also allows re-use and safer debugging.


### Code Block 7


```python
def classify_with_embeddings(text: str, k: int = 5) -> Dict:
    """
    k-NN w/ label centroids:
      - Embed input
      - Compare to label centroids (REAL / FAKE)
      - Convert similarity to softmax-like confidence
    """
    df, M, centroids = load_seed_bank()

    # Embed the input text
    emb = client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding
    q = np.array(emb, dtype=np.float32)

    # Cosine vs. each centroid
    sims = {label: _cosine(q, vec) for label, vec in centroids.items()}
    # Pick best label
    label = max(sims, key=sims.get)

    # Turn the two scores into a pseudo-probability with softmax
    vals = np.array(list(sims.values()), dtype=np.float32)
    exps = np.exp(vals - vals.max())
    probs = exps / exps.sum()
    conf = float(probs[list(sims.keys()).index(label)])

    rationale = f"Nearest-centroid by embeddings. Similarity {sims[label]:.3f} vs. {label} centroid."
    return {"label": label, "confidence": conf, "rationale": rationale}


# ---------- UI ----------
st.sidebar.header("Settings")
mode = st.sidebar.selectbox("Mode", ["GPT Classifier", "Embeddings k-NN"])
gpt_model = st.sidebar.text_input("GPT model (for GPT mode)", GPT_MODEL_DEFAULT)
show_reason = st.sidebar.checkbox("Show rationale", value=True)
min_conf = st.sidebar.slider("Minimum confidence to accept label", 0.0, 1.0, 0.5, 0.05)

text = st.text_area(
    "Paste a headline or short article:",
    height=220,
    placeholder="Example: Government announces tax filing deadline extended to October this year",
)

col1, col2 = st.columns(2)
with col1:
    run = st.button("Classify", type="primary")
with col2:
    st.button("Clear", on_click=lambda: st.session_state.update({"_last": None}))

if run and text.strip():
    try:
        if mode == "GPT Classifier":
            result = classify_with_gpt(text, model=gpt_model.strip() or GPT_MODEL_DEFAULT)
        else:
            result = classify_with_embeddings(text)

        # Confidence gate
        label = result["label"]
        conf = result["confidence"]
        rationale = result.get("rationale", "")

        if conf < min_conf:
            st.warning(f"Low confidence ({conf:.2f}). Treat with caution.")
        _format_output_card(label, conf, rationale, show_reason)

        # Session table of recent classifications
        row = {"text": text, "label": label, "confidence": conf, "mode": mode}
        st.session_state.setdefault("history", []).insert(0, row)
        st.caption("Recent (session only)")
        st.dataframe(pd.DataFrame(st.session_state["history"][:20]))
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

st.markdown("---")
st.caption(
    "This demo uses OpenAI’s API for GPT classification and embeddings. "
    "Keep inputs short to control cost and latency."
)
```

The function **classify_with_embeddings** is defined here. It plays a specific role in the workflow. 

It accepts inputs, performs structured logic, and returns outputs in a predictable form. 

By isolating this logic in a helper, the rest of the application code stays simpler and easier to read. 

This modular approach also allows re-use and safer debugging.
