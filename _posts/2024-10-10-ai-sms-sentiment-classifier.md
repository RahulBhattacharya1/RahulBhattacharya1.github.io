---
layout: default
title: "Creating my AI SMS Sentiment Classifier"
date: 2024-10-10 14:23:51
categories: [ai]
tags: [python,streamlit,openai]
thumbnail: /assets/images/tictactoe.webp
demo_link: https://rahuls-ai-sms-sentiment-classifier.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_sms_sentiment_classifier
featured: true
---

 Sometimes daily events spark unexpected ideas. One day I received several promotional and personal text messages back to back. I found myself quickly deciding which ones mattered and which ones I could ignore. That simple act of judgment made me wonder how a machine could learn the same distinction. I wanted to see if I could train a program to understand whether a message feels positive or negative. That thought became the seed of this project.

The idea grew stronger over the next few days. I kept noticing how often I filter messages by tone and sentiment without thinking. I imagined how useful it would be if a system could do the same automatically. It would save time, make interfaces smarter, and reduce the burden of unnecessary reading. That is when I decided to create this SMS Sentiment Classifier. I kept the design simple and accessible, so that it could be run from a GitHub repository and deployed with Streamlit.
 
 ------------------------------------------------------------------------
 ## Files in the Repository
 The project folder contained four main files and one subfolder. Each file plays an important role in making the app work.
1.  `README.md` --- explains what the project is and how to use it.
2.  `app.py` --- the core Streamlit application script.
3.  `requirements.txt` --- lists the dependencies needed to run the
    project.
4.  `data/seed_examples_sms.csv` --- provides sample text messages to
    test the classifier.


 
 ---

 ## Requirements
 The project depends on four libraries. Streamlit powers the UI and session state. Pandas supports simple tabular views and quick data filters. Numpy helps with vector math. The OpenAI client handles GPT calls and embeddings when the online mode is active.

```python
streamlit>=1.35
pandas>=2.0
numpy>=1.26
openai>=1.40

```
 ## Dataset Seed (for demos)
 The CSV holds brief example messages with labels. It is not used for training in this app. It supports smoke testing and quick demos without any internet connection. I load it into a dataframe only when I want to show a short table of recent in-app classifications. This file also documents the expected schema for any future batch tests.

```python
text,label
"thank you so much for your help today","POSITIVE"
"that was amazing news, congrats","POSITIVE"
"had a great time, let's do it again","POSITIVE"
"really appreciate your quick response","POSITIVE"
"this is the best update we've had","POSITIVE"
```
 ## Imports
 These imports bring in JSON utilities, typed hints, numeric arrays, dataframes, the Streamlit runtime, and the OpenAI client. Each library plays a small role and is used directly in later blocks.

```python
import json
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
```
 ## Page Setup and Title
 The app config sets a centered layout and a clear browser tab title. A simple page title keeps the top of the UI readable. Using a centered layout helps focus attention on the input and the output blocks.

```python
st.set_page_config(page_title="SMS Sentiment Classifier", layout="centered")
st.title("SMS Sentiment Classifier")

# Read key from Streamlit Secrets (add OPENAI_API_KEY in Cloud → Settings → Secrets)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
```
 ## Secrets, Client, and Constants
 The client reads the API key from Streamlit Secrets. This avoids hardcoding any credentials in the repository. Two constants declare the default GPT model and the embeddings model. These values are kept at the top of the file so that swaps or upgrades are a one-line change.

```python
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

GPT_MODEL_DEFAULT = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-small"

# ---------- Utils ----------
```
 ## Helper: Safe JSON Parse (`_safe_parse_json`)
 This helper extracts a JSON object from a string that may include extra tokens. It trims text before the first `{` and after the last `}` so stray logs do not break parsing. If parsing fails, it returns a safe default label, a mid confidence value, and a short rationale. The function protects the UI from unhandled exceptions and helps keep the page stable.

```python
def _safe_parse_json(s: str) -> Dict:
    try:
        start = s.find("{"); end = s.rfind("}")
        if start != -1 and end != -1:
            s = s[start:end + 1]
        data = json.loads(s)
        if not isinstance(data, dict):
            raise ValueError("Top-level JSON is not an object.")
        return data
    except Exception:
        return {"label": "POSITIVE", "confidence": 0.5, "rationale": "Parse error; defaulted."}
```
 ## Helper: Output Formatter (`_format_output`)
 This helper centralizes how the model result is displayed. It prints a bold label, shows a percentage style confidence, and includes an optional rationale. A single output path makes it easy to keep formatting consistent when models change.

```python
def _format_output(label: str, confidence: float, rationale: str, show_reason: bool):
    st.subheader("Result")
    st.markdown(f"**Label:** {label}")
    st.markdown(f"**Confidence:** {confidence:.2f}")
    if show_reason and rationale:
        st.markdown(f"**Rationale:** {rationale}")
```
 ## Helper: Cosine Similarity (`_cosine`)
 This numeric helper computes cosine similarity between two vectors. It normalizes the dot product by the vector norms. Cosine works well for embeddings where direction matters more than magnitude. The function is short, fast, and easy to test.

```python
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# ---------- Seeds cache ----------
@st.cache_resource(show_spinner=True)
```
 ## Helper: Seed Bank Loader (`load_seed_bank`)
 This helper loads the seed CSV, builds embeddings for each row when online, and prepares centroids per label. It returns the raw dataframe, a matrix of item vectors, and a map of label centroids. These pieces allow a simple nearest-centroid classifier.

```python
def load_seed_bank() -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]:
    df = pd.read_csv("data/seed_examples_sms.csv").dropna(subset=["text", "label"]).reset_index(drop=True)
    texts = df["text"].tolist()

    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
    M = np.vstack(vecs)

    centroids: Dict[str, np.ndarray] = {}
    for label in sorted(df["label"].unique()):
        idx = df.index[df["label"] == label].tolist()
        arr = M[idx]
        centroids[label] = arr.mean(axis=0)
    return df, M, centroids

# ---------- Classifiers ----------
```
 ## Classifier: GPT (`classify_with_gpt`)
 This function sends the user text to the GPT model with a strict instruction to return JSON. It then parses the reply with `_safe_parse_json` and normalizes the fields. If the API is not available, it falls back to a small offline heuristic so demos can proceed. The function returns a clear dict with label, confidence, and rationale.

```python
def classify_with_gpt(text: str, model: str = GPT_MODEL_DEFAULT) -> Dict:
    system = (
        "You classify short SMS/chat messages as POSITIVE or NEGATIVE. "
        "Return compact JSON with keys: label (POSITIVE|NEGATIVE), confidence (0..1), rationale (short). "
        "No surrounding prose."
    )
    fewshots = [
        {"role":"user","content":"TEXT: thanks for the quick fix, that was awesome"},
        {"role":"assistant","content":'{"label":"POSITIVE","confidence":0.92,"rationale":"Gratitude and praise."}'},
        {"role":"user","content":"TEXT: this is ridiculous, nothing works and I am tired of it"},
        {"role":"assistant","content":'{"label":"NEGATIVE","confidence":0.94,"rationale":"Complaints and frustration."}'},
    ]
    messages = [{"role":"system","content":system}] + fewshots + [{"role":"user","content":f"TEXT: {text}"}]

    try:
        resp = client.responses.create(model=model, input=messages)
        raw = getattr(resp, "output_text", "") or ""
        data = _safe_parse_json(raw)
        label = str(data.get("label","POSITIVE")).upper()
        if label not in {"POSITIVE","NEGATIVE"}:
            label = "POSITIVE"
        conf = float(data.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
        return {"label": label, "confidence": conf, "rationale": str(data.get("rationale",""))}
    except Exception as e:
        # Offline demo fallback when quota exhausted
        if "insufficient_quota" in str(e):
            # simple heuristic
            neg_words = ["angry","annoy","bad","hate","worst","tired","delay","terrible","upset","ridiculous"]
            label = "NEGATIVE" if any(w in text.lower() for w in neg_words) else "POSITIVE"
            return {"label": label, "confidence": 0.6, "rationale": "Mock result (offline demo mode)."}
        raise
```
 ## Classifier: Embeddings (`classify_with_embeddings`)
 This function builds a fresh embedding for the user text and compares it to the precomputed centroids. It translates cosine scores to a softmax-like confidence for a readable number. If quota is missing, it uses a keyword heuristic as a backup. The output mirrors the GPT path so the UI stays the same.

```python
def classify_with_embeddings(text: str) -> Dict:
    try:
        _, _, centroids = load_seed_bank()
        q = np.array(client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding, dtype=np.float32)
        sims = {label: _cosine(q, c) for label, c in centroids.items()}
        label = max(sims, key=sims.get)
        vals = np.array(list(sims.values()), dtype=np.float32)
        exps = np.exp(vals - vals.max()); probs = exps / exps.sum()
        conf = float(probs[list(sims.keys()).index(label)])
        return {"label": label, "confidence": conf, "rationale": f"Nearest-centroid by embeddings. Similarity {sims[label]:.3f}."}
    except Exception as e:
        if "insufficient_quota" in str(e):
            label = "NEGATIVE" if any(w in text.lower() for w in ["angry","annoy","bad","hate","worst","tired","delay","terrible","upset","ridiculous"]) else "POSITIVE"
            return {"label": label, "confidence": 0.6, "rationale": "Mock result (offline demo mode)."}
        raise

# ---------- UI ----------
st.sidebar.header("Settings")
mode = st.sidebar.selectbox("Mode", ["GPT Classifier", "Embeddings k-NN"])
gpt_model = st.sidebar.text_input("GPT model (for GPT mode)", GPT_MODEL_DEFAULT)
show_reason = st.sidebar.checkbox("Show rationale", value=True)
min_conf = st.sidebar.slider("Minimum confidence to accept label", 0.0, 1.0, 0.5, 0.05)

text = st.text_area("Paste an SMS/chat line:", height=180, placeholder="Example: thanks a ton for resolving this so quickly")
col1, col2 = st.columns(2)
with col1:
    run = st.button("Classify", type="primary")
with col2:
    st.button("Clear", on_click=lambda: st.session_state.update({"_last": None}))

if run and text.strip():
    try:
        result = classify_with_gpt(text, gpt_model.strip() or GPT_MODEL_DEFAULT) if mode=="GPT Classifier" else classify_with_embeddings(text)
        if result["confidence"] < min_conf:
            st.warning(f"Low confidence ({result['confidence']:.2f}). Treat with caution.")
        _format_output(result["label"], result["confidence"], result.get("rationale",""), show_reason)
        st.session_state.setdefault("history", []).insert(0, {"text": text, **result, "mode": mode})
        st.caption("Recent (session only)")
        st.dataframe(pd.DataFrame(st.session_state["history"][:20]))
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
```
 ## Sidebar and Inputs
 The sidebar collects the operating mode, optional model override, confidence threshold, and a toggle to show the model rationale. The main pane takes the SMS input text. The run button ensures the app only computes when the user decides to submit.

```python
st.sidebar.header("Settings")
mode = st.sidebar.selectbox("Mode", ["GPT Classifier", "Embeddings k-NN"])
gpt_model = st.sidebar.text_input("GPT model (for GPT mode)", GPT_MODEL_DEFAULT)
show_reason = st.sidebar.checkbox("Show rationale", value=True)
min_conf = st.sidebar.slider("Minimum confidence to accept label", 0.0, 1.0, 0.5, 0.05)
```
 ## Run and Display Logic
 This block triggers classification based on the selected mode. It warns when the confidence falls below the chosen threshold. It displays the formatted result and tracks a small session history for convenience. Any exception surfaces as a visible error with context.

```python
    run = st.button("Classify", type="primary")
with col2:
    st.button("Clear", on_click=lambda: st.session_state.update({"_last": None}))

if run and text.strip():
    try:
        result = classify_with_gpt(text, gpt_model.strip() or GPT_MODEL_DEFAULT) if mode=="GPT Classifier" else classify_with_embeddings(text)
        if result["confidence"] < min_conf:
            st.warning(f"Low confidence ({result['confidence']:.2f}). Treat with caution.")
        _format_output(result["label"], result["confidence"], result.get("rationale",""), show_reason)
        st.session_state.setdefault("history", []).insert(0, {"text": text, **result, "mode": mode})
        st.caption("Recent (session only)")
        st.dataframe(pd.DataFrame(st.session_state["history"][:20]))
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.caption("Uses OpenAI for GPT classification and embeddings. For demos with no quota, the app falls back to a simple offline heuristic.")
```
 ## README Notes
 The README explains how to supply the API key in Streamlit Secrets and how the app behaves with and without quotas. It also clarifies that the offline path exists for demos and that better training can be added later as needed.

```python
# SMS Sentiment Classifier (Streamlit + OpenAI)

Classifies SMS/chat messages as **POSITIVE** or **NEGATIVE** using either:
- **GPT Classifier** (few-shot → JSON)
- **Embeddings k-NN** (seed set → nearest-centroid)

## Deploy
1) Push to GitHub.  
2) In Streamlit Community Cloud, create app with `streamlit_app.py`.  
3) Add secret:
OPENAI_API_KEY = "sk-..."

## Notes
- Keep inputs short.  
- Default models: `gpt-4.1-mini`, `text-embedding-3-small`.  
- When quota is unavailable, the app uses a minimal offline heuristic for demo purposes.
```
 ## Closing Thoughts
 I built the project to make setup small and repeatable. The app behaves well without training, yet it leaves room for growth. Replacing the simple centroids with a proper classifier would be straightforward. The UI already supports switchable backends and confidence thresholds. I kept state in memory only, since this keeps the code base lighter and reduces side effects during demos. The structure aims for clarity over cleverness.
 
 The offline heuristic is intentional and practical. It proves that the interface holds even when the online path is unavailable. It also helps when presenting in spaces with restricted networks. The same design pattern could be applied to other tasks like intent detection or topic tags. By keeping the code modular, I made it easy to test smaller units and to evolve the system without breaking the story for the user.
