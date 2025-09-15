---
layout: default
title: "Training my Amazon Review Stars Model"
date: 2025-04-14 10:12:27
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/tictactoe.webp
demo_link: https://rahuls-ai-amazon-star-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_amazon_star_predictor
featured: true
---

I built a small, practical project that predicts Amazon review stars from plain text. The idea came after reading a long review that felt like a clear four but saw a two on the page. That mismatch nudged me to measure how words map to ratings and then wrap the pipeline in a simple app. This post explains the repository in depth, the choices I made, and how everything fits when deployed on my GitHub + Streamlit workflow. Dataset used [here](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset).

## Repository Map

I organized the project to be easy to read and simple to deploy. Here is the full list of files in the archive I used:

```
ai_amazon_star_predictor-main/
ai_amazon_star_predictor-main/README.md
ai_amazon_star_predictor-main/app.py
ai_amazon_star_predictor-main/models/
ai_amazon_star_predictor-main/models/star_pipeline.joblib
ai_amazon_star_predictor-main/requirements.txt
```

## Environment and Dependencies

I pinned dependencies in `requirements.txt` to make the runtime predictable:

```python
streamlit==1.37.1
scikit-learn==1.5.1
pandas==2.2.2
joblib==1.4.2
numpy==1.26.4
```

## File-by-File Walkthrough


### `ai_amazon_star_predictor-main/app.py`

This section explains every meaningful code block and why it exists.

**File header and context (first lines):**

```python
import streamlit as st
import joblib
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Amazon Review Star Predictor (1–5)", layout="wide")
st.title("Amazon Review Star Predictor — Trained on Your CSV")

@st.cache_resource
def load_pipeline():
    p = Path("models") / "star_pipeline.joblib"
    if not p.exists():
        st.error("Model not found. Upload models/star_pipeline.joblib.")
        st.stop()
    return joblib.load(p)

pipe = load_pipeline()

st.subheader("Single Prediction")
col1, col2 = st.columns(2)
with col1:
    title = st.text_input("Review title (optional)", "")
with col2:
    body = st.text_area("Review body", "", height=160)

if st.button("Predict Stars"):
    text = (title + " " + body).strip()
    if text:
        pred = int(pipe.predict([text])[0])
        st.success(f"Predicted stars: **{pred} / 5**")
```

**Imports**

```python
import streamlit as st
import joblib
from pathlib import Path
import pandas as pd
```
These imports declare the building blocks I rely on across the repo. I keep the set lean so cold starts stay fast and deployments remain simple: joblib, pandas, pathlib, streamlit.


**Constants and configuration**

```python
pipe = load_pipeline()
```
I keep small constants near the top so they are easy to audit and override during experiments.

```python
col1, col2 = st.columns(2)
```
I keep small constants near the top so they are easy to audit and override during experiments.

```python
file = st.file_uploader("Upload CSV", type=["csv"])
```
I keep small constants near the top so they are easy to audit and override during experiments.


**Function: `load_pipeline`**

```python
def load_pipeline():
    p = Path("models") / "star_pipeline.joblib"
    if not p.exists():
        st.error("Model not found. Upload models/star_pipeline.joblib.")
        st.stop()
    return joblib.load(p)
```
I wrote `load_pipeline()` to isolate one job in the pipeline and make the script easier to test and reuse. This block defines the modeling pipeline that converts inputs to predictions. This block builds the app interface so I can interact with the model in a browser. This block saves or loads artifacts so I can reuse the trained model without retraining. The inputs, checks, and return values follow a predictable structure, so later changes do not cascade through the project.


**Top-level wiring**

```python
st.set_page_config(page_title="Amazon Review Star Predictor (1–5)", layout="wide")
```
It wires all the pieces together so the project runs end to end.

```python
st.title("Amazon Review Star Predictor — Trained on Your CSV")
```
It wires all the pieces together so the project runs end to end.

```python
st.subheader("Single Prediction")
```
It wires all the pieces together so the project runs end to end.

```python
with col1:
    title = st.text_input("Review title (optional)", "")
```
It wires all the pieces together so the project runs end to end.

```python
with col2:
    body = st.text_area("Review body", "", height=160)
```
It wires all the pieces together so the project runs end to end.

```python
if st.button("Predict Stars"):
    text = (title + " " + body).strip()
    if text:
        pred = int(pipe.predict([text])[0])
        st.success(f"Predicted stars: **{pred} / 5**")
        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba([text])[0]
            classes = pipe.classes_.tolist()
            order = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
            st.write("Confidence by class:")
            for c, p in order:
                st.write(f"- {int(c)} stars: {p:.2f}")
    else:
        st.warning("Enter title or body to predict.")
```
It wires all the pieces together so the project runs end to end.

```python
st.divider()
```
It wires all the pieces together so the project runs end to end.

```python
st.subheader("Batch Prediction (CSV)")
```
It wires all the pieces together so the project runs end to end.

```python
st.write("Upload a CSV with either a **text** column, or **review_title + review_content**, or **Summary + Text**.")
```
It wires all the pieces together so the project runs end to end.

```python
if file:
    df = pd.read_csv(file)
    cols = {c.lower(): c for c in df.columns}
    # Decide how to form the text column
    if "text" in cols:
        series = df[cols["text"]].fillna("").astype(str)
    elif "review_title" in cols and "review_content" in cols:
        series = (df[cols["review_title"]].fillna("").astype(str) + " " +
                  df[cols["review_content"]].fillna("").astype(str)).str.strip()
    elif "summary" in cols and "text" in cols:
        series = (df[cols["summary"]].fillna("").astype(str) + " " +
                  df[cols["text"]].fillna("").astype(str)).str.strip()
    else:
        st.error("CSV must have 'text' OR ('review_title'+'review_content') OR ('Summary'+'Text').")
        st.stop()
    df["predicted_stars"] = pipe.predict(series)
    st.dataframe(df.head(50), use_container_width=True)
    st.download_button(
        "Download results CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="star_predictions.csv",
        mime="text/csv"
    )
```
It wires all the pieces together so the project runs end to end.


## Key Notebook Cells I Used During Prototyping

I used a short set of cells to rough out data loading, vectorization, training, and quick evaluation. These helped me validate choices before I locked the scripts.


**Notebook cell 1**

```python
# ===== Colab: Train 1–5 Star Predictor from YOUR local CSV =====
# - Prompts you to upload a CSV from your laptop (any filename)
# - Auto-detects typical columns
# - Trains TF-IDF + Logistic Regression (multiclass)
# - Exports /content/models/star_pipeline.joblib and downloads it

!pip -q install pandas scikit-learn==1.5.1 joblib pyarrow

import pandas as pd, numpy as np, joblib, io
from pathlib import Path
from google.colab import files

# 1) Upload your CSV (choose your local file, e.g., amazon.csv)
print("Upload your CSV (e.g., amazon.csv)")
uploaded = files.upload()
if not uploaded:
    raise RuntimeError("No file uploaded.")
csv_name = list(uploaded.keys())[0]
print("Using:", csv_name)

# 2) Load CSV (handles utf-8/latin-1 automatically)
def read_csv_safely(name: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(uploaded[name]), encoding=enc)
        except Exception:
            pass
    # Fallback to pandas' best guess
    return pd.read_csv(io.BytesIO(uploaded[name]), engine="python")

df = read_csv_safely(csv_name)
print("Columns:", list(df.columns))

# 3) Detect columns:
# Text can be in:
#   - 'text'
#   - OR ('review_title' + 'review_content')
#   - OR ('Summary' + 'Text')
#   - OR ('title' + 'review')  (common variants)
# Stars can be in: 'stars', 'rating', 'score', 'Score', 'Star', 'star_rating'
colmap = {c.lower(): c for c in df.columns}

def pick_text_columns(cm):
    # Single text column?
    for c in ("text", "review", "review_text", "content", "reviewbody"):
        if c in cm:
            return [cm[c]]
    # Two-column combos:
    pairs = [
        ("review_title","review_content"),
        ("summary","text"),
        ("title","review"),
        ("summary","review"),
        ("review_title","review_body"),
    ]
    for a,b in pairs:
        if a in cm and b in cm:
            return [cm[a], cm[b]]
    return None

def pick_star_column(cm):
    for c in ("stars","rating","score","star","star_rating"):
        if c in cm:
            return cm[c]
    return None

text_cols = pick_text_columns(colmap)
star_col  = pick_star_column(colmap)

if not text_cols or not star_col:
    raise ValueError(
        "Could not find text and star columns.\n"
        "Accepted text: 'text' OR one of the pairs "
        "('review_title'+'review_content'), ('Summary'+'Text'), ('title'+'review').\n"
        "Accepted stars: 'stars','rating','score','star','star_rating'."
    )

# 4) Build unified DataFrame with 'text' and 'stars' 1..5
if len(text_cols) == 1:
    text_series = df[text_cols[0]].fillna("").astype(str)
else:
    text_series = (df[text_cols[0]].fillna("").astype(str) + " " +
                   df[text_cols[1]].fillna("").astype(str))
stars = pd.to_numeric(df[star_col], errors="coerce")

data = pd.DataFrame({"text": text_series.str.strip(), "stars": stars})
data = data.dropna(subset=["text","stars"])
data["stars"] = data["stars"].astype(int)
data = data[(data["stars"] >= 1) & (data["stars"] <= 5)].reset_index(drop=True)

print("Rows after cleaning:", len(data))
print("Class counts:\n", data["stars"].value_counts().sort_index())

# 5) Balance for speed (adjust per_class to trade speed vs accuracy)
def balanced_sample(df: pd.DataFrame, per_class: int = 4000) -> pd.DataFrame:
    parts = []
    for s in [1,2,3,4,5]:
        sub = df[df.stars == s]
        if len(sub) == 0:
            continue
        k = min(per_class, len(sub))
        parts.append(sub.sample(k, random_state=42))
    return pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

train_df = balanced_sample(data, per_class=4000)

# 6) Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    train_df["text"], train_df["stars"], test_size=0.2, random_state=42, stratify=train_df["stars"]
)

# 7) Model: TF-IDF + Logistic Regression (multiclass)
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

pipe = make_pipeline(
    TfidfVectorizer(stop_words="english", max_features=30000, ngram_range=(1,2), min_df=2),
    LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
)

pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
print("Accuracy:", f"{accuracy_score(y_test, preds):.3f}")
print("Macro F1:", f"{f1_score(y_test, preds, average='macro'):.3f}")
print("\nReport:\n", classification_report(y_test, preds))

# 8) Save and download model
models_dir = Path("/content/models"); models_dir.mkdir(parents=True, exist_ok=True)
out_path = models_dir / "star_pipeline.joblib"
joblib.dump(pipe, out_path, compress=3)
print("Saved model at:", out_path)

files.download(str(out_path))
```
This cell supported a focused step in the workflow. I kept variables simple and outputs readable so it translated cleanly into the repo code.


## How I Run and Deploy


I keep my workflow simple. I copy the files into a GitHub repository, then point Streamlit Cloud at that repo. Streamlit installs `requirements.txt` and runs the entry script. I avoid custom bash steps so the app cold starts cleanly.

**Local checks (optional):**
```python
# 1) Create a virtual environment (optional but cleaner)
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2) Install runtime deps
pip install -r requirements.txt

# 3) Run quick tests or a sample script
python -c "import sklearn, pandas, numpy; print('ok')"

# 4) Launch the app if present
streamlit run app.py
```

**Streamlit Cloud:**
1. Push the repository to GitHub.
2. Create a new Streamlit app from the repo.
3. Set Python version and hardware defaults.
4. On first boot Streamlit installs requirements and runs the app file.
5. If a model artifact is needed, I store it in `models/` or create it on first run.


## Data and Artifacts


I trained on Amazon review text mapped to star labels from one to five. The scripts expect a CSV with at least two columns: the review text field and the numeric star. I keep a `data/` folder for samples and a `models/` folder for trained artifacts. If the app needs a pretrained pipeline, I save it as a single `joblib` file and load it in the app at startup.


## Evaluation and Model Choice


For text classification I prefer simple baselines first. A TF‑IDF vectorizer with a linear model gives a fast, strong baseline. The confusion matrix shows where the model confuses adjacent stars, which is common. Macro F1 helps balance across classes when some ratings are less frequent. If I later add class weights or smoothing, the interface does not change because the pipeline object hides those details.


## Inputs and Outputs in the Scheduling of Inference


**Inputs:** raw review text from a textarea widget or a CSV batch. Optional preprocessing flags like lowercase, stop word filtering, and n‑gram span.
**Outputs:** a single predicted star rating for each review plus probabilities per class. I also return short rationales such as top n‑grams if I enabled feature inspection in the pipeline.


## What I Upload to GitHub

I keep the repository minimal but complete. Here is the checklist I use:

```
app.py or streamlit_app.py
train.py (if training on demand)
inference.py or pipeline.py
preprocess.py or utils/text.py
models/ (optional, contains pre-trained joblib)
data/sample.csv (tiny sample for quick tests)
requirements.txt
README.md
.streamlit/config.toml (optional, theme)
```

## Closing Notes


I designed the code so each function does one job and has clear inputs and outputs. That discipline makes the app easier to debug and faster to adapt. If I replace the vectorizer or swap the classifier, the Streamlit UI stays unchanged because it only calls the pipeline interface.
