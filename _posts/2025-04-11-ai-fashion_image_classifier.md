---
layout: default
title: "Building my AI Fashion Image Classifier"
date: 2025-04-11 13:46:39
categories: [ai]
tags: [python,streamlit,bert,self-trained]
thumbnail: /assets/images/ai_rock_paper_scissor.webp
demo_link: https://rahuls-ai-sentiment-analysis.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_sentiment_analysis
featured: true
---

I wanted a portfolio project that does not rely on OpenAI’s API. I picked a classic problem that still feels practical for recruiters. I built a topic classifier that reads a short news headline or snippet and predicts one of four AG News categories. I trained the model in Google Colab. I pushed a small set of files to GitHub. I deployed a clean demo on Streamlit Community Cloud.

---

## What I Uploaded To GitHub

I kept the repository small and clear. These files are enough for Streamlit Cloud to run the app without extra steps.

- `app.py` — Streamlit application that loads the trained model and serves predictions.
- `requirements.txt` — Package list that Streamlit Cloud installs before starting the app.
- `models/ag_news_tfidf_logreg.pkl` — Trained scikit‑learn pipeline saved with joblib.
- `models/ag_news_labels.json` — Label names to map numeric predictions to readable text.
- `README.md` — A short explanation for visitors and recruiters on GitHub.

I trained in Colab and exported the artifacts into the `models` folder. I committed those artifacts so the app starts fast and does not need to train again during deployment.

---

## Colab‑Only Workflow Overview

I ran everything in Colab. I did not install anything on my laptop. The flow is simple and repeatable.

1. Install Python packages inside Colab using pip from Python.
2. Create project folders in `/content` for a clean structure.
3. Load the AG News dataset, normalize the text, and build features.
4. Train a Logistic Regression classifier on TF‑IDF vectors.
5. Evaluate the model and save the trained pipeline with readable labels.
6. Generate the `app.py`, `requirements.txt`, and `README.md` files inside Colab.
7. Download the folder as a zip and upload to a new GitHub repository.
8. Point Streamlit Community Cloud at `app.py` and deploy the live demo.

---

## Installing Dependencies In Colab

I install packages using `subprocess` so the code runs as plain Python. This avoids notebook magics and keeps the block portable.

```python
import sys, subprocess

pkgs = [
    "scikit-learn>=1.3",
    "datasets>=2.19",
    "joblib>=1.3",
    "numpy>=1.23",
    "pandas>=1.5",
    "streamlit>=1.37"
]

cmd = [sys.executable, "-m", "pip", "install", "-q"] + pkgs
subprocess.check_call(cmd)
print("Installed packages:", pkgs)
```

**Why this block exists and how it helps**  
This block ensures Colab has the same packages that Streamlit Cloud will install. I avoid surprises between training and serving. I pin minimum versions to avoid older bugs. I use Python to call pip so the code works outside Colab if needed.

---

## Project Folders And A Reusable Normalizer

I create a root folder and a `models` folder under `/content`. I also define a helper that cleans raw text. This helper runs during training and again during inference, which keeps behavior consistent.

```python
import re, os, json, zipfile
from pathlib import Path

import numpy as np
import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

ROOT = Path("/content/ag_news_topic_classifier")
MODELS = ROOT / "models"
ROOT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

def normalize(text: str) -> str:
    """
    Clean a news headline or snippet for vectorization.
    1) Remove HTML break tags since they add noise but no meaning.
    2) Keep letters, numbers, spaces, and apostrophes, drop other symbols.
    3) Collapse repeated whitespace to a single space and strip edges.
    4) Lowercase everything to reduce duplicate variants of the same token.
    """
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

print("Project root:", ROOT)
```

**How `normalize` supports the whole project**  
The model learns patterns from normalized text. The app also predicts on normalized text. This keeps training and serving aligned. The function is short but removes noise that hurts generalization. It lowers risk of brittle features and unstable predictions.

---

## Loading AG News And Preparing Arrays

I load the dataset from the `datasets` hub. I apply my normalizer to both splits. I also fetch the integer labels and the human names. I keep everything in memory for speed and simplicity.

```python
ds = load_dataset("ag_news")

train_texts = [normalize(t) for t in ds["train"]["text"]]
test_texts  = [normalize(t) for t in ds["test"]["text"]]

y_train = np.array(ds["train"]["label"], dtype=int)
y_test  = np.array(ds["test"]["label"], dtype=int)

label_names = ds["train"].features["label"].names

print("Label names:", label_names)
print("Training examples:", len(train_texts), "Test examples:", len(test_texts))
```

**Why this block looks like this**  
I normalize early to give the vectorizer clean material. I convert labels to arrays for efficient operations. I store the label names for reporting and for the app. The names make the output clear for non‑technical readers.

---

## Building Features With TF‑IDF And Choosing A Classifier

I use a TF‑IDF vectorizer to convert text into sparse numeric features. I keep a generous feature cap and include bigrams. I choose Logistic Regression because it is fast, strong, and exposes probabilities.

```python
vectorizer = TfidfVectorizer(
    max_features=60000,
    ngram_range=(1, 2),
    min_df=2
)

clf = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    multi_class="auto"
)

pipe = Pipeline([
    ("tfidf", vectorizer),
    ("clf", clf)
])
```

**How this design helps my goals**  
TF‑IDF is strong for short text classification. Bigrams catch short phrases like player names or product names. The pipeline keeps preprocessing and modeling together. It reduces the chance of mismatched steps during serving. Logistic Regression gives calibrated and readable probabilities for the demo.

---

## Training, Evaluating, And Printing A Report

I fit the pipeline on the training texts and labels. I then evaluate on the test split. I print the accuracy and a classification report with per class metrics. I want a quick snapshot of strengths and weaknesses.

```python
pipe.fit(train_texts, y_train)

y_pred = pipe.predict(test_texts)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, digits=3, target_names=label_names))
```

**Why this block matters for a portfolio**  
This block proves the model actually learned from data. The report shows precision and recall for each topic, which non‑technical readers still recognize as quality signals. I avoid heavy charts and keep output readable inside Colab logs.

---

## Saving The Trained Artifacts For Deployment

I export the fitted pipeline and the label names. I store them under `models`. The Streamlit app will load these files at startup. This lets the app respond fast without retraining.

```python
model_path = MODELS / "ag_news_tfidf_logreg.pkl"
labels_path = MODELS / "ag_news_labels.json"

joblib.dump(pipe, model_path)

with open(labels_path, "w", encoding="utf-8") as f:
    json.dump({"label_names": label_names}, f, ensure_ascii=False, indent=2)

print("Saved model:", model_path)
print("Saved labels:", labels_path)
```

**How these files fit into the bigger picture**  
The pickle bundles vectorization and classification. The JSON keeps human names separate and easy to adjust. Keeping artifacts small and readable makes deployment smooth. It also makes future updates less risky.

---

## The Streamlit App (`app.py`) — Full File And Commentary

This is the user interface. Streamlit Cloud runs this file. It loads the model, renders a small form, predicts on demand, and prints clear results. I keep it small and friendly.

```python
import re, json
from pathlib import Path
import joblib
import streamlit as st
import numpy as np

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "ag_news_tfidf_logreg.pkl"
LABELS_PATH = ROOT / "models" / "ag_news_labels.json"

def normalize(text: str) -> str:
    """
    Keep the same cleaning used during training.
    This avoids differences between train and serve time.
    """
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

@st.cache_resource
def load_model_and_labels():
    """
    Load the pipeline and the label names once per process.
    Streamlit caches the result so repeated clicks stay fast.
    """
    model = joblib.load(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        label_names = json.load(f)["label_names"]
    return model, label_names

st.set_page_config(page_title="AG News Topic Classifier")
st.title("AG News Topic Classifier")

# Validate required artifacts early and fail clearly if missing.
if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    st.error("Model or labels not found. Please make sure /models has the files.")
    st.stop()

model, label_names = load_model_and_labels()

st.write("Enter a news headline or short article text. The model will classify it into one of the AG News categories.")

example = "NASA announces new mission to study the atmosphere of Mars next year."
text = st.text_area("Text to classify:", value=example, height=120)

# Only predict when the user provides non-empty text and clicks the button.
if st.button("Classify") and text.strip():
    x = normalize(text)
    pred_idx = int(model.predict([x])[0])

    # If classifier exposes probabilities, show a ranked confidence breakdown.
    if hasattr(model.named_steps["clf"], "predict_proba"):
        proba = model.predict_proba([x])[0]
        order = np.argsort(proba)[::-1]
        st.subheader(f"Prediction: {label_names[pred_idx]}")
        st.write("Confidence by class:")
        for i in order:
            st.write(f"- {label_names[i]}: {proba[i]:.3f}")
    else:
        st.subheader(f"Prediction: {label_names[pred_idx]}")
        st.caption("This classifier does not expose probabilities.")
```

**Block by block explanation**  
- **Imports and constants**. I import `joblib` to load the pickle, `streamlit` for the UI, and `numpy` for small array work. I compute file paths relative to the script so deployment stays portable.  
- **`normalize` function**. I reuse training time cleaning at serve time. This keeps features aligned and reduces surprises. Short helpers like this pay off in reliability.  
- **`load_model_and_labels` function**. I cache the loaded artifacts with `@st.cache_resource`. Streamlit stores them across interactions. This keeps the app quick even when users click multiple times.  
- **Page setup and artifact guard**. I set the page title and then check that artifacts exist. If anything is missing I show a direct error and stop. This avoids confusing partial screens for viewers.  
- **Text area and main conditional**. I show a default example to guide first use. I only run a prediction after the user clicks the button and provides non empty text. I keep control flow explicit and easy to follow.  
- **Probability branch**. I show a ranked confidence table when the classifier supports probabilities. This helps non‑technical users understand uncertainty. If probabilities are not available, I still present a clear prediction with a short note.

---

## The Requirements File (`requirements.txt`) — Exact Content And Rationale

Streamlit Cloud reads this file and installs listed packages. I keep versions modern and aligned with my Colab session. I only include packages the app imports.

```python
requirements_txt = """
streamlit>=1.37
scikit-learn>=1.3
datasets>=2.19
joblib>=1.3
numpy>=1.23
pandas>=1.5
""".strip()

print(requirements_txt)
```

**Why I include only these packages**  
Lean environments build faster and break less. Every listed package appears in the code. I avoid hidden extras or dev tools. This keeps the deployment surface small and predictable.

---

## The Labels File (`models/ag_news_labels.json`) — Structure And Role

The classifier returns a numeric class id. I map that id to a human label using this file. Keeping names outside the pickle makes it easier to rename or reorder if needed.

```python
labels_json = {
    "label_names": ["World", "Sports", "Business", "Sci/Tech"]
}

import json, os
os.makedirs("/content/ag_news_topic_classifier/models", exist_ok=True)
with open("/content/ag_news_topic_classifier/models/ag_news_labels.json", "w", encoding="utf-8") as f:
    json.dump(labels_json, f, ensure_ascii=False, indent=2)

print("Wrote labels JSON to /content/ag_news_topic_classifier/models/ag_news_labels.json")
```

**Why this separation helps**  
Readable metadata should live in readable files. I can change a label name without touching the binary model. This keeps tiny content edits safe and version friendly.

---

## The Model Artifact (`models/ag_news_tfidf_logreg.pkl`) — What It Contains

The pickle is a scikit‑learn `Pipeline`. It holds a TF‑IDF vectorizer and a Logistic Regression classifier. The app loads this with `joblib.load` and then calls `predict` or `predict_proba` on normalized text.

```python
# Demonstrate a safe reload within Colab to verify the artifact.
from pathlib import Path
reloaded = joblib.load(str(model_path))
test_pred = reloaded.predict(["economy shows strong growth as markets rally"])
print("Reloaded model predicts class id:", int(test_pred[0]))
```

**Why I store a pipeline not separate parts**  
The pipeline keeps the order of operations attached to the model. It avoids subtle bugs where the app vectorizes differently than training. It also keeps the serving code small and readable.

---

## The Repository README (`README.md`) — Purpose And Outline

The README helps visitors get the scope fast. It explains that the model trains in Colab, and the app runs on Streamlit Cloud. It names the core files and points to `app.py`.

```python
readme_text = """
# AG News Topic Classifier (TF-IDF + Logistic Regression)

A lightweight topic classifier trained in Google Colab and deployed on Streamlit Community Cloud.
No OpenAI API is used. The app loads a saved scikit-learn pipeline and predicts one of four AG News categories.

## Files
- app.py — Streamlit application.
- requirements.txt — Python packages for deployment.
- models/ag_news_tfidf_logreg.pkl — Trained pipeline.
- models/ag_news_labels.json — Human-readable label names.

## Deploy
- Push these files to GitHub.
- On Streamlit Community Cloud, create a new app and choose app.py.
- The app should start after requirements install.
""".strip()

print(readme_text[:240] + " ...")
```

**Why I keep this file short**  
Recruiters skim quickly. The README should explain the what and the how in seconds. The deeper explanations live in this blog post.

---

## Explaining Every Function And Conditional In Plain Language

I document helpers and conditionals that control behavior. I keep each explanation short and focused on intent.

### `normalize(text: str) -> str`

This helper removes markup, drops noisy symbols, squeezes whitespace, and lowercases text. It narrows variation across inputs and makes tokenization stable. The vectorizer sees cleaner features and learns better associations. I call it during training and serving to keep parity.

### `load_model_and_labels() -> tuple`

This function loads the pickle and the labels JSON once. The `@st.cache_resource` decorator tells Streamlit to keep the result in memory. Repeated button clicks reuse the cached objects. This removes disk churn and keeps the app quick for visitors.

### `if not MODEL_PATH.exists() or not LABELS_PATH.exists():`

This guard checks that the two required artifacts exist. If one is missing the app will not work. I show a direct error about the missing files and call `st.stop`. The app fails early and loudly rather than confusing users with partial screens.

### `if st.button("Classify") and text.strip():`

This condition ensures the model runs only after a clear user action. It also ensures the text is not empty. The block prepares normalized input, runs the model, and formats output. This reduces accidental submits and keeps the interaction crisp.

### `if hasattr(model.named_steps["clf"], "predict_proba"):`

Some classifiers expose probabilities. Others do not. I check this at runtime and branch accordingly. When available I show probabilities ranked by confidence. When not available I show only the predicted label and a short note.

---

## What I Actually Uploaded To GitHub

I prepared these exact files from Colab and uploaded them to a new public repository.

```python
# Simulated file list as Python data for clarity.
files_to_upload = [
    "app.py",
    "requirements.txt",
    "models/ag_news_tfidf_logreg.pkl",
    "models/ag_news_labels.json",
    "README.md"
]

for p in files_to_upload:
    print("Include:", p)
```

**Why I avoid extra files**  
Small repositories are easier to review and maintain. I do not commit the notebook. I keep the focus on serving the demo. Training remains a Colab step I can repeat later when I want improvements.

---

## How I Wrote The Files Inside Colab (Optional Utility Blocks)

I prefer to keep file content versioned in GitHub. In Colab I sometimes generate files from strings to avoid manual copy paste. This block shows how I could write the app and requirements directly. It is optional and useful during iteration.

```python
# Write app.py from a Python triple-quoted string (for convenience in Colab).
from pathlib import Path

app_py_src = """
import re, json
from pathlib import Path
import joblib
import streamlit as st
import numpy as np

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "ag_news_tfidf_logreg.pkl"
LABELS_PATH = ROOT / "models" / "ag_news_labels.json"

def normalize(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

@st.cache_resource
def load_model_and_labels():
    model = joblib.load(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        label_names = json.load(f)["label_names"]
    return model, label_names

st.set_page_config(page_title="AG News Topic Classifier")
st.title("AG News Topic Classifier")

if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    st.error("Model or labels not found. Please make sure /models has the files.")
    st.stop()

model, label_names = load_model_and_labels()

st.write("Enter a news headline or short article text. The model will classify it into one of the AG News categories.")

example = "NASA announces new mission to study the atmosphere of Mars next year."
text = st.text_area("Text to classify:", value=example, height=120)

if st.button("Classify") and text.strip():
    x = normalize(text)
    pred_idx = int(model.predict([x])[0])
    if hasattr(model.named_steps["clf"], "predict_proba"):
        proba = model.predict_proba([x])[0]
        order = np.argsort(proba)[::-1]
        st.subheader(f"Prediction: {label_names[pred_idx]}")
        st.write("Confidence by class:")
        for i in order:
            st.write(f"- {label_names[i]}: {proba[i]:.3f}")
    else:
        st.subheader(f"Prediction: {label_names[pred_idx]}")
        st.caption("This classifier does not expose probabilities.")
"""
(Path("/content/ag_news_topic_classifier/app.py")).write_text(app_py_src, encoding="utf-8")

# Write requirements.txt
req_txt_src = """streamlit>=1.37
scikit-learn>=1.3
datasets>=2.19
joblib>=1.3
numpy>=1.23
pandas>=1.5
"""
(Path("/content/ag_news_topic_classifier/requirements.txt")).write_text(req_txt_src, encoding="utf-8")
```

**Why these utility writers are handy**  
They remove manual copying inside the notebook. They also reduce typos when I tweak code. I still commit the output files to GitHub, not the generation cells themselves.

---

## Deployment Steps I Followed

I zipped the folder from Colab and uploaded it to GitHub. Then I deployed on Streamlit Community Cloud. The process is simple and repeatable.

1. Create a new public repository on GitHub.  
2. Upload `app.py`, `requirements.txt`, and the `models` folder with both files.  
3. Add a short `README.md` for context.  
4. In Streamlit Community Cloud, create a new app and pick `app.py`.  
5. Wait for packages to install and the app to start.  
6. Copy the public URL and add it to my blog front matter under `demo_link`.

---

## Troubleshooting I Prepared For

- **Model or labels not found**. I verify the `models` folder exists in the repo and contains both files. I confirm the paths in `app.py` match exactly.  
- **Build fails on Streamlit Cloud**. I confirm `requirements.txt` is at the repository root. I avoid extra files that could confuse the build.  
- **Slow first request**. The first call warms the app. It is normal. Later calls are faster because artifacts are cached in memory.  
- **Text prediction looks odd**. I confirm the input is a short headline or snippet. Very long paragraphs reduce clarity. The model was trained on short news pieces.

---

## Front Matter For My Blog Post

I use front matter in my GitHub Pages site. I include a thumbnail and both links. I add tags that mark this as a self‑trained project and not an API demo.

```python
front_matter = """
---
layout: post
title: "AG News Topic Classifier — My End-to-End Build (No API)"
date: 2025-09-11
categories: [ai]
tags: [self-trained, scikit-learn, tfidf]
thumbnail: /assets/images/ai_news.webp
demo_link: https://YOUR-STREAMLIT-URL
github_link: https://github.com/YOUR-USER/ag-news-topic-classifier
featured: true
---
""".strip()

print(front_matter)
```

**Why I chose these tags**  
These tags separate this post from OpenAI API posts. They communicate that I trained a model myself. They also match the tools used in the repository and the app.

---

## What This Project Demonstrates For My Portfolio

This project shows I can build an AI application without external APIs. I can clean text, craft features, and train a model. I can package artifacts and deploy a working demo. I can write a readable blog that explains code blocks and helpers. I keep the repository lean and focused on serving. I make decisions that balance simplicity and clarity. I can repeat the process for other datasets with minimal changes.

---

## Appendix A — Minimal End‑To‑End Training Script (Colab)

This block shows an end‑to‑end script that runs in a single cell. I prefer modular cells for clarity, but a compact script is useful for quick sanity checks during iteration.

```python
import re, json, sys, subprocess
from pathlib import Path
import numpy as np, joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# Install
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "scikit-learn>=1.3", "datasets>=2.19", "joblib>=1.3",
                       "numpy>=1.23", "pandas>=1.5", "streamlit>=1.37"])

# Setup
ROOT = Path("/content/ag_news_topic_classifier")
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

def normalize(t: str) -> str:
    t = re.sub(r"<br\s*/?>", " ", t)
    t = re.sub(r"[^A-Za-z0-9\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower()

# Data
ds = load_dataset("ag_news")
train_texts = [normalize(x) for x in ds["train"]["text"]]
test_texts  = [normalize(x) for x in ds["test"]["text"]]
y_train = np.array(ds["train"]["label"], dtype=int)
y_test  = np.array(ds["test"]["label"], dtype=int)
label_names = ds["train"].features["label"].names

# Model
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=60000, ngram_range=(1,2), min_df=2)),
    ("clf", LogisticRegression(solver="lbfgs", max_iter=1000, multi_class="auto"))
]).fit(train_texts, y_train)

# Eval
pred = pipe.predict(test_texts)
print("Accuracy:", round(float(accuracy_score(y_test, pred)), 4))
print(classification_report(y_test, pred, digits=3, target_names=label_names))

# Save
joblib.dump(pipe, MODELS / "ag_news_tfidf_logreg.pkl")
with open(MODELS / "ag_news_labels.json", "w", encoding="utf-8") as f:
    json.dump({"label_names": label_names}, f, indent=2, ensure_ascii=False)

print("Artifacts saved under:", MODELS)
```

**How this script helps**  
It shows the entire training in one place. I can run it fresh if my notebook is messy. It guarantees artifacts exist before I push to GitHub. It keeps my deployment step predictable.

---

## Appendix B — Sanity Tests For The App Artifacts

I like adding small tests to confirm that artifacts match app expectations. These checks save time during deployment because they catch path mistakes early.

```python
from pathlib import Path
import joblib, json

ROOT = Path("/content/ag_news_topic_classifier")
model_file = ROOT / "models" / "ag_news_tfidf_logreg.pkl"
labels_file = ROOT / "models" / "ag_news_labels.json"

assert model_file.exists(), "Missing model file"
assert labels_file.exists(), "Missing labels file"

model = joblib.load(model_file)
with open(labels_file, "r", encoding="utf-8") as f:
    labels = json.load(f)["label_names"]

sample = "Coach resigns after championship; team searches for new leadership"
pred = int(model.predict([sample])[0])
print("Sample prediction id:", pred, "->", labels[pred])
```
 
These tests help verify files exist where the app expects them. They also verify the pickle works in a clean Python process. They give me confidence before I upload to GitHub and deploy the demo.

---

## Final Notes

This project is fully self‑trained and does not use an external API. I trained in Colab, exported artifacts, and deployed on Streamlit Cloud. The repository stays small and readable. The app is responsive and simple to use.
