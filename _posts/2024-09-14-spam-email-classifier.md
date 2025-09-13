---
layout: default
title: "Building my AI Spam Email Classifier"
date: 2024-09-14 16:31:25
categories: [ai]
tags: [python,streamlit,openai]
thumbnail: /assets/images/tictactoe.webp
demo_link: https://rahuls-ai-spam-email-classifier.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_spam_email_classifier
featured: true
---

The idea for this project came to me after a personal experience. I once received a strange email that claimed I had won a lottery prize. The subject line promised an urgent reward, but the message itself was filled with suspicious links and generic greetings. At that moment I wondered how many people might fall for such messages. It made me realize the importance of creating tools that help distinguish between harmless communication and harmful scams. This experience inspired me to create a spam email classifier that can be run in a browser using Streamlit.

I wanted the project to be approachable and transparent. My focus was to make sure each part of the application was easy to follow. I used OpenAI models for text classification and embeddings, and I wrapped the entire system inside a Streamlit interface. This way, the application could provide predictions quickly while also showing how the results were reached. My goal was not just to detect spam but also to explain why a message was marked as spam or ham.

---

## Project Structure

The repository contained four key files and folders:

- `README.md`: Provided a short description and setup notes.
- `requirements.txt`: Listed dependencies such as Streamlit, pandas, numpy, and OpenAI.
- `data/seed_examples_spam.csv`: A dataset of sample emails with spam and ham labels.
- `app.py`: The main Streamlit application script.

In the following sections I will explain each file and then provide a detailed block by block breakdown of the `app.py` code.

---

## README.md

The `README.md` file gave a simple overview. It explained the purpose of the project, how to install dependencies, and how to run the application. A README serves as the entry point for anyone opening the repository, and ensures that they can quickly set up the project.

---

## requirements.txt

This file listed the required Python libraries. It included:

- `streamlit`: for building the web interface.
- `openai`: for calling the GPT and embedding APIs.
- `numpy` and `pandas`: for handling data, vectors, and analysis.

Without these dependencies, the application would not run. The requirements file made installation consistent across environments.

---

## seed_examples_spam.csv

This CSV contained sample labeled emails. Each row included the text of an email and whether it was spam or ham. These examples served as the seed bank used by the embedding-based classifier. By computing embeddings for each text and grouping them by label, the system could later classify new input by comparing its embedding against known centroids.

---

## app.py

The `app.py` file was the heart of the project. It tied together the interface, the machine learning logic, and the presentation of results. Below I expand on each part.

### Imports and Setup

```python
import json
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
```

These imports brought in the essential libraries. `json` allowed parsing structured text. `typing` made function signatures more explicit. `numpy` and `pandas` supported numerical work and dataframes. `streamlit` powered the user interface. `openai` was the client library for calling GPT and embeddings.

```python
st.set_page_config(page_title="Spam Email Classifier", layout="centered")
st.title("Spam Email Classifier")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

GPT_MODEL_DEFAULT = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-small"
```

Here the Streamlit page was configured with a title. An OpenAI client was created using a secret API key stored securely. Two constants defined the models: one for GPT-based classification and one for embeddings.

### Helper to Parse JSON Safely

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
        return {"label": "HAM", "confidence": 0.5, "rationale": "Could not parse model JSON; defaulted."}
```

This helper tried to parse JSON output from the model. Sometimes GPT returned extra text or malformed structures. The function attempted to extract the JSON portion, then load it. If anything went wrong, it defaulted to a ham label with medium confidence. This was important because it prevented the app from crashing when the model responded unpredictably.

### Formatting Results for Display

```python
def _format_output_card(label: str, confidence: float, rationale: str, show_reason: bool):
    badge = "SPAM" if label.upper() == "SPAM" else "HAM"
    st.subheader("Result")
    st.markdown(f"**Label:** {badge}")
    st.markdown(f"**Confidence:** {confidence:.2f}")
    if show_reason and rationale:
        st.markdown(f"**Rationale:** {rationale}")
```

This function displayed results inside Streamlit. It showed the label, the confidence score, and optionally the rationale. The helper made sure the output was presented cleanly every time without duplicating code.

### Cosine Similarity Helper

```python
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
```

This helper computed cosine similarity between two vectors. It was used by the embedding classifier to compare an input email with centroid vectors. A zero denominator check avoided division errors.

### Loading the Seed Bank

```python
@st.cache_resource(show_spinner=True)
def load_seed_bank() -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]:
    df = pd.read_csv("data/seed_examples_spam.csv")
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)

    texts = df["text"].tolist()
    vecs: List[np.ndarray] = []

    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    for item in resp.data:
        vecs.append(np.array(item.embedding, dtype=np.float32))

    M = np.vstack(vecs)
    df["__vec_index"] = np.arange(len(df))

    centroids: Dict[str, np.ndarray] = {}
    for label in sorted(df["label"].unique()):
        idx = df.index[df["label"] == label].tolist()
        if not idx:
            continue
        arr = M[idx]
        centroids[label] = arr.mean(axis=0)
    return df, M, centroids
```

This function loaded the seed dataset, created embeddings for each text, and built centroid vectors grouped by label. Caching was used so that embeddings were only computed once. The result was a dataframe with emails, a full matrix of embeddings, and centroids for ham and spam. This enabled fast classification later.

### GPT-based Classification

```python
def classify_with_gpt(text: str, model: str = GPT_MODEL_DEFAULT) -> Dict:
    system = (
        "You classify emails as SPAM or HAM. "
        "Return compact JSON: {"label":"SPAM|HAM","confidence":0..1,"rationale":"short"}. "
        "Penalize phishing, prize scams, urgent tone, deceptive links, obfuscation."
    )

    fewshots = [
        {"role":"user","content":"SUBJECT: Win an iPhone NOW!!! BODY: Click http://scam.co to claim"},
        {"role":"assistant","content":'{"label":"SPAM","confidence":0.96,"rationale":"Prize lure, urgency, suspicious link."}'},
        {"role":"user","content":"SUBJECT: Team meeting notes BODY: Sharing action items from today."},
        {"role":"assistant","content":'{"label":"HAM","confidence":0.90,"rationale":"Work context, neutral language, no lures."}'},
    ]

    messages = [{"role":"system","content":system}] + fewshots + [{"role":"user","content":text}]

    resp = client.responses.create(model=model, input=messages)

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
    label = str(data.get("label","HAM")).upper()
    if label not in {"SPAM","HAM"}:
        label = "HAM"
    try:
        conf = float(data.get("confidence", 0.5))
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    rationale = str(data.get("rationale",""))
    return {"label": label, "confidence": conf, "rationale": rationale}
```

This function used GPT to classify a message. It defined system instructions, provided few-shot examples, and then appended the user email. The API response was parsed, and if no clean JSON came back, the `_safe_parse_json` helper handled it. The function returned a label, confidence, and rationale. This method mimicked how a human might reason about spam cues.

### Embedding-based Classification

```python
def classify_with_embeddings(text: str) -> Dict:
    df, M, centroids = load_seed_bank()

    emb = client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding
    q = np.array(emb, dtype=np.float32)

    sims = {label: _cosine(q, vec) for label, vec in centroids.items()}
    label = max(sims, key=sims.get)

    vals = np.array(list(sims.values()), dtype=np.float32)
    exps = np.exp(vals - vals.max())
    probs = exps / exps.sum()
    conf = float(probs[list(sims.keys()).index(label)])

    rationale = f"Nearest-centroid by embeddings. Similarity {sims[label]:.3f} vs. {label} centroid."
    return {"label": label, "confidence": conf, "rationale": rationale}
```

This function used embeddings to classify. It loaded the seed bank, embedded the input, computed cosine similarities with each centroid, and chose the closest. Then it normalized the similarities into probabilities using softmax. The rationale was explained in plain text. This method provided a lightweight alternative to GPT classification.

### Sidebar Settings

```python
st.sidebar.header("Settings")
mode = st.sidebar.selectbox("Mode", ["GPT Classifier", "Embeddings k-NN"])
gpt_model = st.sidebar.text_input("GPT model (for GPT mode)", GPT_MODEL_DEFAULT)
show_reason = st.sidebar.checkbox("Show rationale", value=True)
min_conf = st.sidebar.slider("Minimum confidence to flag as spam", 0.0, 1.0, 0.5, 0.05)
```

These lines built the sidebar in Streamlit. They allowed the user to choose classification mode, set the GPT model, toggle rationale display, and adjust the minimum spam confidence threshold.

### Main Input and Buttons

```python
raw = st.text_area(
    "Paste email (subject + body):",
    height=240,
    placeholder="SUBJECT: ...\nBODY: ...",
)

col1, col2 = st.columns(2)
with col1:
    run = st.button("Classify", type="primary")
with col2:
    st.button("Clear", on_click=lambda: st.session_state.update({"_last": None}))
```

This section provided a text area for pasting emails and buttons for classifying or clearing input. The two-column layout made the interface clean.

### Running Classification

```python
if run and raw.strip():
    try:
        if mode == "GPT Classifier":
            result = classify_with_gpt(raw, model=gpt_model.strip() or GPT_MODEL_DEFAULT)
        else:
            result = classify_with_embeddings(raw)

        label = result["label"]
        conf = result["confidence"]
        rationale = result.get("rationale","")

        if conf < min_conf and label == "SPAM":
            st.warning(f"Low confidence ({conf:.2f}). Treat with caution.")
        _format_output_card(label, conf, rationale, show_reason)

        row = {"email": raw, "label": label, "confidence": conf, "mode": mode}
        st.session_state.setdefault("history", []).insert(0, row)
        st.caption("Recent (session only)")
        st.dataframe(pd.DataFrame(st.session_state["history"][:20]))
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
```

This block executed classification when the button was pressed. It selected GPT or embeddings mode, processed the input, and displayed results. If confidence was below the minimum threshold for spam, a warning appeared. Results were stored in session state for history, and displayed in a table. Exceptions were caught and shown as error messages.

### Footer

```python
st.markdown("---")
st.caption("Uses OpenAI for GPT classification and embeddings. Keep inputs short to control cost and latency.")
```

The footer reminded users about the underlying models and gave a note on usage cost.

---

## Conclusion

This project combined a practical problem with modern tools. By using Streamlit, OpenAI models, and embeddings, I created an application that can quickly distinguish spam from ham. Each function and helper played a role: safe parsing avoided crashes, cosine similarity powered embedding classification, and formatted outputs ensured clarity. Together these pieces formed a complete, working application that grew out of a simple personal observation.

