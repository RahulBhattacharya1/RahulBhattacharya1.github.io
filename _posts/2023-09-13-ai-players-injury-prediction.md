---
layout: default
title: "My AI Soccer Players Injury Prediction"
date: 2023-09-13 21:32:54
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/soccer.webp
thumbnail_mobile: /assets/images/soccer_injury_sq.webp
demo_link: https://rahuls-ai-players-risk-predict.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_players_risk_predict
---

Sometimes ideas arrive from a moment of reflection rather than a moment of need. I once watched a sports match where a key player had to leave because of an injury that seemed predictable if someone had only looked closely at the data. That scene made me wonder if data could be used to anticipate such risks before they happened. I imagined that a simple dashboard could score risk levels and highlight patterns that would otherwise remain hidden. The thought of being able to create such a tool with my own code stayed with me until I decided to attempt it. Dataset used [here](https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2025-2026).

This project is my way of turning that idea into something real. The application is not just about algorithms, but also about building a working end‑to‑end solution. It takes a CSV file of player statistics and processes it with machine learning pipelines. The result is displayed in a Streamlit dashboard where risk scores are shown in tables and charts. It can run in supervised mode if labels are present or in unsupervised mode if they are not. What follows is a breakdown of every file I uploaded to GitHub and a detailed look at every code block inside the app.

## Requirements File

The first file that had to be uploaded was the `requirements.txt`. This file lists all external libraries that my app needs. Without it, the deployment environment would not know what to install. The content is short but critical.

```python
streamlit
scikit-learn
pandas
numpy
plotly
```

Each entry here plays a different role. Streamlit provides the web dashboard. Scikit‑learn provides the machine learning models, preprocessing, and pipelines. Pandas and NumPy are used for data handling and mathematical operations. Plotly is used for interactive visualizations. By declaring them in one file, I ensure that anyone who runs the app installs the same dependencies that I used.

## README File

The `README.md` file introduces the project. It explains in plain text what the application does, what modes it supports, and how to run it. This file is important for GitHub because it is the first thing visitors see when they open the repository.

```python
# Injury Risk Detection Dashboard

An end-to-end Streamlit app that scores player injury risk using your CSV data.

## Features
- Supervised mode (RandomForest) if you have a binary injury label.
- Unsupervised mode (IsolationForest) if you do not.
- Interactive table and top-30 risk chart.
- One-click CSV export with risk scores.

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
```

The README explains that the application supports both supervised and unsupervised risk scoring. It also mentions the interactive components like tables and charts. The quickstart section tells the user to install requirements and run the app with the Streamlit command. This document is not part of the runtime but it is essential for communication and clarity.

## Dataset File

Another file I uploaded was `players_data-2025_2026.csv`. This is a sample dataset with player information. The application reads this file to generate predictions if no custom file is uploaded. The data provides structure so that the app can be demonstrated without needing an external source.

The dataset typically contains columns for player characteristics, performance metrics, and possibly labels indicating whether a player faced injury. Having such data available allows the supervised mode to train and evaluate models. In the absence of labels, the unsupervised mode makes use of clustering or anomaly detection methods to still provide a risk score. The dataset therefore acts as both input and a demonstration resource.

## Application Code (app.py)

The `app.py` file is the heart of this repository. It contains all the logic for the Streamlit dashboard, the data preprocessing, the model training, and the visualization. I will go through it section by section, showing code blocks and then explaining them in detail.

```python
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
import io
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
from typing import List, Tuple, Optional
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
st.set_page_config(page_title="Injury Risk Detection", layout="wide")
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
# -----------------------------
# Helpers
# -----------------------------
def guess_id_columns(cols: List[str]) -> List[str]:
    patterns = [
        r"player[_\s]?id",
        r"player",
        r"athlete",
        r"name$",
        r"full[_\s]?name",
    ]
    out = []
    for c in cols:
        lc = c.lower()
        if any(re.search(p, lc) for p in patterns):
            out.append(c)
    return out
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
def guess_label_column(cols: List[str]) -> Optional[str]:
    patterns = [
        r"injur",         # injury, injured, injuries
        r"out[_\s]?status",
        r"availability[_\s]?risk",
        r"risk[_\s]?label",
        r"missed[_\s]?games[_\s]?flag",
    ]
    for c in cols:
        lc = c.lower()
        if any(re.search(p, lc) for p in patterns):
            return c
    return None
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
def numeric_feature_candidates(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in nums if c not in exclude]
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
def safe_downcast(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.select_dtypes(include=["float64"]).columns:
        out[c] = pd.to_numeric(out[c], downcast="float")
    for c in out.select_dtypes(include=["int64"]).columns:
        out[c] = pd.to_numeric(out[c], downcast="integer")
    return out
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
def build_supervised_pipeline(num_cols: List[str]) -> Pipeline:
    num_proc = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    pre = ColumnTransformer(
        transformers=[("num", num_proc, num_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    return pipe
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
def build_unsupervised_model() -> IsolationForest:
    return IsolationForest(
        contamination="auto",
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
    )
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
def compute_unsupervised_risk_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    # IsolationForest returns higher scores for "normal" via score_samples.
    # Convert to risk where higher = more risky:
    raw = model.score_samples(X)
    risk = (raw.min() - raw)  # invert
    # Normalize 0..1
    if np.nanmax(risk) > np.nanmin(risk):
        risk = (risk - np.nanmin(risk)) / (np.nanmax(risk) - np.nanmin(risk))
    else:
        risk = np.zeros_like(risk)
    return risk
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
def make_feature_preview_chart(df: pd.DataFrame, id_col: Optional[str], risk_col: str):
    tmp = df.copy()
    if id_col is None:
        tmp["Entity"] = np.arange(len(tmp))
        id_col = "Entity"
    fig = px.bar(
        tmp.sort_values(risk_col, ascending=False).head(30),
        x=id_col,
        y=risk_col,
        title="Top 30 Highest Risk (sorted)",
    )
    st.plotly_chart(fig, use_container_width=True)
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
def to_downloadable_csv(df: pd.DataFrame, filename: str = "injury_risk_scored.csv") -> Tuple[bytes, str]:
    buff = io.StringIO()
    df.to_csv(buff, index=False)
    return buff.getvalue().encode("utf-8"), filename
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
# -----------------------------
# UI
# -----------------------------
st.title("Injury Risk Detection Dashboard")
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
st.markdown(
    """
This app supports **two modes**:
- **Supervised**: If your data has an injury/label column (e.g., `injury`, `injured_flag`), it trains a classifier and outputs risk probabilities.
- **Unsupervised**: If there is no injury label, it uses Isolation Forest to flag abnormal workload patterns as higher risk.
"""
)
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
uploaded = st.file_uploader("Upload season or player dataset (CSV)", type=["csv"])
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
def read_csv_safe(src):
    # Path string or Path object
    if isinstance(src, (str, Path)):
        return pd.read_csv(src)
    # Streamlit UploadedFile / file-like
    if hasattr(src, "read"):
        try:
            src.seek(0)
            return pd.read_csv(src)
        except ValueError:
            # Some environments need a BytesIO wrapper
            src.seek(0)
            return pd.read_csv(io.BytesIO(src.read()))
    raise ValueError("Unsupported CSV source type")
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
def load_default_csv() -> pd.DataFrame:
    candidates = [
        "players_data-2025_2026.csv",
        "players_data_light-2025_2026.csv",
        "data/players_data-2025_2026.csv",
        "data/players_data_light-2025_2026.csv",
    ]
    for p in candidates:
        if Path(p).exists():
            st.info(f"Using default dataset bundled with the app: {p}")
            return read_csv_safe(p)
    st.error(
        "No bundled dataset found. Upload a CSV or add one of the expected files "
        "to the repo root (or a data/ folder)."
    )
    st.stop()
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
# Decide source
if uploaded is not None:
    df_raw = read_csv_safe(uploaded)
else:
    df_raw = load_default_csv()
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
df_raw = safe_downcast(df_raw)
st.success(f"Loaded {df_raw.shape[0]} rows × {df_raw.shape[1]} columns.")
with st.expander("Preview data", expanded=False):
    st.dataframe(df_raw.head(20), use_container_width=True)
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
cols = df_raw.columns.tolist()
id_guesses = guess_id_columns(cols)
label_guess = guess_label_column(cols)
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
left, right = st.columns([1, 1])
with left:
    mode = st.radio("Mode", ["Auto (detect label)", "Supervised", "Unsupervised"], index=0)
with right:
    if mode == "Auto (detect label)":
        st.write(f"Detected label column: **{label_guess or 'None'}**")
    pass
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
if mode == "Auto (detect label)":
    if label_guess is not None:
        active_mode = "Supervised"
    else:
        active_mode = "Unsupervised"
else:
    active_mode = mode
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
# Choose ID column (optional)
id_col = st.selectbox(
    "Player/Entity identifier column (optional, used for display)",
    options=["<none>"] + id_guesses + [c for c in cols if c not in id_guesses],
    index=0 if not id_guesses else 1,
)
id_col = None if id_col == "<none>" else id_col
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
# Choose label (only for supervised)
label_col = None
if active_mode == "Supervised":
    defaults = [label_guess] if label_guess and label_guess in cols else []
    label_col = st.selectbox(
        "Injury label column (binary: 1=injured/at-risk, 0=healthy)",
        options=cols,
        index=cols.index(defaults[0]) if defaults else 0,
    )
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
# Feature selection
exclude = [label_col] if label_col else []
num_candidates = numeric_feature_candidates(df_raw, exclude=exclude)
with st.expander("Feature selection", expanded=True):
    st.caption("Pick numeric workload and profile features (minutes, matches, distance, sprints, age, rest days, etc.).")
    features = st.multiselect(
        "Numeric features",
        options=num_candidates,
        default=[c for c in num_candidates if len(features) < 0] if False else num_candidates[: min(12, len(num_candidates))],
        help="You can change this anytime. Choose meaningful workload indicators.",
    )
    if not features:
        st.warning("Select at least one numeric feature.")
        st.stop()
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
# Additional controls
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    if active_mode == "Unsupervised":
        contamination = st.slider("Assumed at-risk share (unsupervised)", 0.01, 0.20, 0.06, 0.01)
    else:
        contamination = None
with c2:
    test_size = st.slider("Test size (supervised)", 0.1, 0.4, 0.2, 0.05)
with c3:
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
st.markdown("---")
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
# -----------------------------
# Run
# -----------------------------
run = st.button("Train / Score")
if not run:
    st.stop()
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
work_df = df_raw.copy()
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
if active_mode == "Supervised":
    y = work_df[label_col].astype(float)
    X = work_df[features]
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
# Guardrail: ensure label is binary
    unique_labels = sorted(pd.Series(y).dropna().unique().tolist())
    if not set(unique_labels).issubset({0, 1}):
        st.error(f"Label {label_col} must be binary 0/1. Found values: {unique_labels[:10]}")
        st.stop()
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
pipe = build_supervised_pipeline(features)
    pipe.fit(X_train, y_train)
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
# Metrics
    y_prob = pipe.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = float("nan")
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
try:
        ap = average_precision_score(y_test, y_prob)
    except ValueError:
        ap = float("nan")
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
st.subheader("Validation Metrics")
    m1, m2 = st.columns([1, 1])
    m1.metric("ROC AUC", f"{auc:.3f}" if np.isfinite(auc) else "NA")
    m2.metric("Average Precision", f"{ap:.3f}" if np.isfinite(ap) else "NA")
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix (threshold=0.5)")
    st.dataframe(pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]))
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
# Score all rows
    all_prob = pipe.predict_proba(work_df[features])[:, 1]
    scored = work_df.copy()
    scored["risk_score"] = all_prob
    if id_col is not None and id_col in scored.columns:
        display_cols = [id_col, "risk_score"] + [c for c in features if c != id_col]
    else:
        display_cols = ["risk_score"] + features
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
st.subheader("Risk Scores (Supervised)")
    st.dataframe(
        scored.sort_values("risk_score", ascending=False)[display_cols].head(200),
        use_container_width=True,
        height=480,
    )
    make_feature_preview_chart(scored, id_col=id_col, risk_col="risk_score")
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
# Download
    csv_bytes, fname = to_downloadable_csv(scored)
    st.download_button("Download Scored CSV", data=csv_bytes, file_name=fname, mime="text/csv")
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
else:
    # Unsupervised
    X = work_df[features].copy()
    pre = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_pre = pre.fit_transform(X)
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
iso = IsolationForest(
        contamination=contamination if contamination is not None else "auto",
        n_estimators=400,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X_pre)
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
risk_scores = compute_unsupervised_risk_scores(iso, X_pre)
    scored = work_df.copy()
    scored["risk_score"] = risk_scores
    if id_col is not None and id_col in scored.columns:
        display_cols = [id_col, "risk_score"] + [c for c in features if c != id_col]
    else:
        display_cols = ["risk_score"] + features
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
st.subheader("Risk Scores (Unsupervised)")
    st.dataframe(
        scored.sort_values("risk_score", ascending=False)[display_cols].head(200),
        use_container_width=True,
        height=480,
    )
    make_feature_preview_chart(scored, id_col=id_col, risk_col="risk_score")
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
csv_bytes, fname = to_downloadable_csv(scored)
    st.download_button("Download Scored CSV", data=csv_bytes, file_name=fname, mime="text/csv")
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

```python
st.markdown("---")
st.caption(
    "Tips: Include workload features like minutes played, matches in last 7/14 days, age, "
    "rest days, sprint counts, high-speed distance, past injuries, and travel. "
    "If you don’t have labels, start with Unsupervised mode to triage risk."
)
```

This block contributes to the overall application. I will now explain what it does in context.

In this section, the code defines or imports functionality. Each import, class, or function contributes to either preparing data, building the model, or rendering the Streamlit interface. The helpers inside handle small tasks like scaling values, imputing missing data, or splitting columns. Functions are often wrapped into pipelines so that preprocessing and modeling can be chained. Streamlit calls render the user interface and handle input like file uploads, mode selections, and parameter adjustments.

## Conclusion

This project grew from a simple thought into a working dashboard. The final repository holds four files: `app.py`, `requirements.txt`, `README.md`, and a sample dataset. Together they form a complete demonstration that others can clone and run. Every function, conditional, and helper in the code contributes to the flow from raw data to risk prediction.

By walking through each part, I have shown how the pieces connect. The machine learning pipeline is not hidden; it is transparent and adjustable. The user interface allows real time interaction with predictions and plots. The app may begin as a sports injury predictor but the same structure could be reused for other kinds of risk scoring. That is the strength of building something carefully and documenting it fully.
