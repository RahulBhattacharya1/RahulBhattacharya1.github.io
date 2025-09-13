---
layout: default
title: "Building my AI Powered UPSC Mock Exams"
date: 2024-11-16 10:22:51
categories: [ai]
tags: [python,streamlit,openai]
thumbnail: /assets/images/tictactoe.webp
demo_link: https://rahuls-upsc-mock-test.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/upsc_mock_test
featured: true
---

It started with a simple experience while browsing headlines online. I noticed how some headlines grabbed attention with exaggerated words while others felt straightforward. That moment gave me the thought that I should try detecting clickbait automatically. I wanted a small tool that could separate genuine news headlines from overly sensational ones. The idea felt practical because news and media today often mix informative writing with clickbait techniques.

The motivation was not only curiosity but also a wish for clarity. Many times I wasted time on headlines that promised much but delivered little. I imagined having a personal filter that could score headlines before clicking them. That is how I began building this project. I created a simple Streamlit application powered by OpenAI and some seed examples. The project is light, easy to run, and does not need complicated setup.

## Project Files Overview

The project required uploading four files to GitHub:

1.  **README.md** --- A description of what the project does and how to
deploy.\
2.  **requirements.txt** --- A list of all Python libraries that must be
installed.\
3.  **app.py** --- The main Streamlit application code that runs the
detector.\
4.  **data/seed_examples_clickbait.csv** --- A small CSV dataset with
seed examples of clickbait and genuine headlines.

Together these files make the project complete and ready for deployment on GitHub Pages with Streamlit.

------------------------------------------------------------------------

## The README File

The README provides instructions and context. Below is the content:

``` python
# Headline Clickbait Detector (Streamlit + OpenAI)

Predicts whether a headline is **CLICKBAIT** or **GENUINE** using:
- **GPT Classifier** (few-shot → JSON)
- **Embeddings k-NN** (seed set → nearest-centroid)

## Deploy
1) Push to GitHub.  
2) Create Streamlit app with `streamlit_app.py`.  
3) Add secret:
OPENAI_API_KEY = "sk-..."

## Notes
- Keep headlines short.  
- Default models: `gpt-4.1-mini`, `text-embedding-3-small`.  
- When quota is unavailable, the app uses a minimal offline heuristic for demo purposes.
```

The README explains two classification strategies: a GPT-based classifier and an embeddings nearest-centroid classifier. It also guides the user to push code to GitHub and configure the OpenAI key. The note about offline heuristic ensures the app still runs in demo situations.

------------------------------------------------------------------------

## The Requirements File

The dependencies are listed in `requirements.txt`. Without these, the app cannot run.

``` python
streamlit>=1.35
pandas>=2.0
numpy>=1.26
openai>=1.40
```

Each library plays a clear role. Streamlit builds the web interface. Pandas handles the dataset. Numpy helps with array math. OpenAI connects the application to the GPT models and embedding services.

------------------------------------------------------------------------

## The Dataset File

The CSV provides seed examples for both clickbait and genuine headlines.

``` python
text,label
"you won't believe what happened next","CLICKBAIT"
"this simple trick will change your life forever","CLICKBAIT"
"10 shocking facts doctors don't want you to know","CLICKBAIT"
"here's why everyone is talking about this one secret","CLICKBAIT"
"man tries this and the results are unbelievable","CLICKBAIT"
"federal reserve announces rate policy update for next quarter","GENUINE"
"local council approves new public park with community funding","GENUINE"
"researchers publish peer reviewed study on climate patterns","GENUINE"
"company releases quarterly earnings and guidance","GENUINE"
"health department issues advisory about seasonal flu","GENUINE"
```

The dataset has two columns: text and label. The text column holds example headlines. The label column indicates whether the headline is clickbait or genuine. These examples form a seed set for the embedding classifier.

------------------------------------------------------------------------

## The Main Application File (app.py)

The real logic lives inside `app.py`. Below is the full code with explanations broken into blocks.

### Imports and Setup

``` python
import json
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Headline Clickbait Detector", layout="centered")
st.title("Headline Clickbait Detector")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

GPT_MODEL_DEFAULT = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-small"
```

Here I import all required libraries. JSON handles structured parsing. Typing helps annotate return types. Numpy and Pandas are essential for math and data manipulation. Streamlit builds the interface. The OpenAI library connects to GPT models. Then I configure the page title and layout for Streamlit. The `client` is initialized with the OpenAI key stored in Streamlit secrets. Finally, two constants define the default GPT model and the embedding model.

------------------------------------------------------------------------

### Helper Function: Safe JSON Parser

``` python
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
        return {"label": "GENUINE", "confidence": 0.5, "rationale": ""}
```

This helper is important because GPT sometimes returns text with extra characters. The function tries to extract JSON content safely. It looks for the first opening brace and the last closing brace. Then it loads the substring as JSON. If the top-level object is not a dictionary, it raises an error. If anything fails, it returns a fallback dictionary with a default label and confidence. This prevents the app from crashing when parsing fails.

------------------------------------------------------------------------

### Seed Examples Loader

``` python
@st.cache_data(show_spinner=True)
def load_seed_examples() -> pd.DataFrame:
    return pd.read_csv("data/seed_examples_clickbait.csv")
```

This function loads the CSV seed dataset. Streamlit caches the result to avoid reloading on every run. The dataset provides the examples required for the embedding classifier. The decorator `@st.cache_data` makes the function efficient for repeated usage.

------------------------------------------------------------------------

### Embedding Function

``` python
def embed_texts(texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([r.embedding for r in resp.data])
```

This function takes a list of strings and converts them into embeddings. The OpenAI client sends the texts to the embedding model. The response contains embedding vectors for each text. Those vectors are collected into a Numpy array. This array is then used for classification tasks. By centralizing embedding calls here, the code stays clean and modular.

------------------------------------------------------------------------

### Nearest Centroid Classifier

``` python
def nearest_centroid_classifier(seed_df: pd.DataFrame, headline: str) -> Dict:
    seed_embeddings = embed_texts(seed_df["text"].tolist())
    headline_embedding = embed_texts([headline])[0]

    labels = seed_df["label"].unique()
    results = {}

    for label in labels:
        idxs = seed_df[seed_df["label"] == label].index
        centroid = seed_embeddings[idxs].mean(axis=0)
        dist = np.linalg.norm(headline_embedding - centroid)
        results[label] = dist

    pred = min(results, key=results.get)
    return {"label": pred, "confidence": 1.0 / (1.0 + results[pred]), "rationale": "nearest centroid"}
```

This function implements a simple nearest centroid classifier. First it embeds all seed examples. Then it embeds the new headline. For each label, it calculates the centroid of embeddings belonging to that label. The distance from the headline embedding to each centroid is measured. The label with the smallest distance becomes the prediction. The confidence is computed as an inverse function of distance. This provides a basic yet effective baseline classifier.

------------------------------------------------------------------------

### GPT Few-Shot Classifier

``` python
def gpt_fewshot_classifier(headline: str) -> Dict:
    prompt = f"Classify the headline as CLICKBAIT or GENUINE with rationale: {headline}"
    resp = client.chat.completions.create(
        model=GPT_MODEL_DEFAULT,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return _safe_parse_json(resp.choices[0].message.content)
```

This function builds a prompt asking GPT to classify the headline. It requests a JSON object response with label and rationale. The OpenAI client sends the chat completion request. The response is then passed to `_safe_parse_json` to ensure valid JSON output. This classifier leverages GPT reasoning to produce both label and explanation.

------------------------------------------------------------------------

### Offline Heuristic

``` python
def offline_heuristic(headline: str) -> Dict:
    if "!" in headline or "believe" in headline.lower():
        return {"label": "CLICKBAIT", "confidence": 0.6, "rationale": "simple keyword rule"}
    return {"label": "GENUINE", "confidence": 0.5, "rationale": "default fallback"}
```

This function provides a very simple fallback when API calls fail. It checks for exclamation marks or the word "believe". If found, it predicts clickbait with moderate confidence. Otherwise, it defaults to genuine. This keeps the app functional even without an API key.

------------------------------------------------------------------------

### User Interface and Flow

``` python
headline = st.text_input("Enter a headline to classify:")

if st.button("Classify"):
    seed_df = load_seed_examples()
    try:
        gpt_result = gpt_fewshot_classifier(headline)
    except Exception:
        gpt_result = offline_heuristic(headline)

    embed_result = nearest_centroid_classifier(seed_df, headline)

    st.subheader("GPT Result")
    st.write(gpt_result)

    st.subheader("Embedding Result")
    st.write(embed_result)
```

This is the main interactive section. A text input lets the user provide a headline. When the classify button is pressed, the app loads seed examples. It attempts to classify with GPT. If GPT fails, it falls back to offline heuristic. Then it always computes an embedding-based classification. Both results are displayed with subheaders. This flow combines robustness with clarity for the user.

------------------------------------------------------------------------

## Conclusion

This project combines three approaches: GPT few-shot reasoning, embedding nearest centroid, and offline heuristics. Each plays a role in maintaining accuracy, speed, and reliability. The structure of the code separates helpers, classifiers, and interface clearly. The dataset, requirements, and README complete the package. Together these files form a simple yet insightful project. The design shows how AI and simple heuristics can work together to solve a small but meaningful problem.
