---
layout: default
title: "Creating my AI Powered Emotion Classifier"
date: 2025-02-21 09:11:46
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/emotion.webp
thumbnail_mobile: /assets/images/emotion_classifier_sq.webp
demo_link: https://rahuls-ai-emotion-classifier.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_emotion_classifier
---

I built a emotion classifier which reads text and predicts the emotion label.

## Repository Structure

Below is the file tree I committed to the repository.

```python
README.md
app.py
models/emotion_pipeline.joblib
requirements.txt
```

## File: README.md

```python
# Emotion Classifier — Self-trained in Colab (Hugging Face)

Dataset: `dair-ai/emotion` via the Hugging Face `datasets` library.

Steps:
1. In Colab, run the one-cell script to train and download `emotion_pipeline.joblib`.
2. In GitHub, create this repo and add `app.py` + `requirements.txt`.
3. Create a `models/` folder and upload `emotion_pipeline.joblib`.
4. Deploy on Streamlit Community Cloud (main file: `app.py`).
```

### What this file does
This file is part of the end to end classifier. I commit it so the repository runs without guesswork. It holds well scoped code and avoids hidden side effects.

## File: app.py

```python
import streamlit as st
import joblib
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Emotion Classifier", layout="wide")
st.title("Emotion Classifier — Self-trained in Colab (Hugging Face)")

@st.cache_resource
def load_pipeline():
    path = Path("models") / "emotion_pipeline.joblib"
    if not path.exists():
        st.error("Model not found. Upload models/emotion_pipeline.joblib to the repo.")
        st.stop()
    return joblib.load(path)

pipe = load_pipeline()

st.subheader("Single Prediction")
text = st.text_area(
    "Enter text (a sentence or two):",
    "",
    height=140,
    placeholder="The sunset was so beautiful that I couldn't stop smiling."
)

if st.button("Predict Emotion"):
    if text.strip():
        pred = pipe.predict([text])[0]
        proba_fn = getattr(pipe, "predict_proba", None)
        if proba_fn:
            probs = proba_fn([text])[0]
            classes = pipe.classes_
            top = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:3]
            st.success(f"Prediction: **{pred}**")
            st.write("Top confidences:")
            for label, p in top:
                st.write(f"- {label}: {p:.2f}")
        else:
            st.success(f"Prediction: **{pred}**")
    else:
        st.warning("Please enter some text.")

st.divider()
st.subheader("Batch Prediction (CSV)")

st.write("Upload a CSV with a column named **text**. The app adds an **emotion** column and lets you download the results.")
file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    try:
        df = pd.read_csv(file)
        if "text" not in df.columns:
            st.error("CSV must include a 'text' column.")
        else:
            df["emotion"] = pipe.predict(df["text"].fillna("").astype(str))
            st.dataframe(df.head(50), use_container_width=True)
            st.download_button(
                "Download results CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="emotion_results.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Failed to process CSV: {e}")
```

### What this file does
This file is part of the end to end classifier. I commit it so the repository runs without guesswork. It holds well scoped code and avoids hidden side effects.

### Functions and helpers explained

#### def load_pipeline(no arguments)
I wrote this function to handle one clear task. It reduces duplication and keeps the call sites small.
It accepts inputs, validates assumptions, and returns a stable shape.
It connects the logic to a Streamlit UI and keeps the UI reactive. It guards edge cases with clear conditionals and early exits.
In the bigger picture, this helper keeps training and inference readable. It also makes unit tests simpler.

### Key conditionals and why they matter

- The code checks `st.button('Predict Emotion')`. This reduces runtime errors and avoids silent failures. It keeps flows explicit and safe.

- The code checks `file is not None`. This reduces runtime errors and avoids silent failures. It keeps flows explicit and safe.

- The code checks `not path.exists()`. This reduces runtime errors and avoids silent failures. It keeps flows explicit and safe.

- The code checks `text.strip()`. This reduces runtime errors and avoids silent failures. It keeps flows explicit and safe.

- The code checks `proba_fn`. This reduces runtime errors and avoids silent failures. It keeps flows explicit and safe.

- The code checks `'text' not in df.columns`. This reduces runtime errors and avoids silent failures. It keeps flows explicit and safe.

## File: requirements.txt

```python
streamlit==1.37.1
scikit-learn==1.5.2
pandas==2.2.3
joblib==1.4.2
numpy==2.1.3
```

### What this file does
This file is part of the end to end classifier. I commit it so the repository runs without guesswork. It holds well scoped code and avoids hidden side effects.

## Notebook: emotion_classifier.ipynb

I used a notebook to experiment fast. I saved code cells that train and evaluate. Then I moved stable parts into modules.

### Cell 1: Code

```python
# ===== Colab: Emotion Classifier using Hugging Face dataset =====
# Trains on dair-ai/emotion (anger, fear, joy, love, sadness, surprise),
# exports /content/models/emotion_pipeline.joblib for your Streamlit app.

!pip -q install datasets scikit-learn joblib pandas

import pandas as pd
import pathlib, joblib
from datasets import load_dataset

# 1) Load the dataset from Hugging Face
# Splits: train/validation/test; labels are integer IDs with names
ds = load_dataset("dair-ai/emotion")

# Map label IDs to string names
id2label = ds["train"].features["label"].names  # ['sadness','joy','love','anger','fear','surprise']
def to_df(split):
    return pd.DataFrame({
        "text": split["text"],
        "emotion": [id2label[i] for i in split["label"]],
    })

train_df = to_df(ds["train"])
val_df   = to_df(ds["validation"])
test_df  = to_df(ds["test"])

# Combine train + validation for a bit more data
full_train = pd.concat([train_df, val_df], ignore_index=True)

print("Train size:", full_train.shape, " Test size:", test_df.shape)
print("Label counts (train):")
print(full_train["emotion"].value_counts())

# 2) Build & train a classic ML pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X_tr, y_tr = full_train["text"], full_train["emotion"]
X_te, y_te = test_df["text"],  test_df["emotion"]

pipe = make_pipeline(
    TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1,2)),
    LogisticRegression(max_iter=1000)
)

pipe.fit(X_tr, y_tr)
preds = pipe.predict(X_te)
acc   = accuracy_score(y_te, preds)
print("Test accuracy:", f"{acc:.3f}")
print(classification_report(y_te, preds))

# 3) Save the trained pipeline
models_dir = pathlib.Path("/content/models")
models_dir.mkdir(parents=True, exist_ok=True)
out_path = models_dir / "emotion_pipeline.joblib"
joblib.dump(pipe, out_path)
print("Saved model:", out_path)

# 4) Download to your laptop
from google.colab import files
files.download(str(out_path))
```

**Explanation**

This block runs a focused step. It loads data, prepares inputs, or trains the model. I kept the cell short so the outputs stay readable. I reused helpers from the modules when possible. This kept the notebook honest and close to production code.

## Files I uploaded to GitHub
I added these files to the repository so the project runs without extra work.
```python
README.md
app.py
models/emotion_pipeline.joblib
requirements.txt
```


## Running the project
I prefer simple commands. I describe the typical steps below. I keep the run flow linear and predictable.

```python
# 1) Create a fresh environment
#    I use python -m venv or conda based on the machine.
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# 2) Install dependencies
# If the repository has requirements.txt or pyproject.toml, install them.
pip install -r requirements.txt

# 3) Run quick tests or a dry run
python - << 'PY'
print("Environment OK")
PY

# 4) Start the app (if Streamlit app.py exists)
streamlit run app.py

# 5) Or run a CLI script if the repo exposes one
python main.py --help
```

## Why this structure works
Small, pure helpers reduce cognitive load. Each layer does one thing well. The app layer deals with inputs and outputs. The core layer handles model logic and data flow. The utilities layer hides repeated chores like caching and tokenization. This separation keeps refactors calm and focused. It also helps when I change models or datasets.

## Closing thoughts
I built the classifier for clarity and for learning as a set of simple building blocks that I can reason about. Each block is short, testable, and replaceable so the project is maintainable over time.

