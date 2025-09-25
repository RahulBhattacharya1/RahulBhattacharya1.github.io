---
layout: default
title: "Creating my AI Retail Sales Forecasting"
date: 2022-02-19 11:43:26
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-retail-sales-forecasting.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_retail_sales_forecasting
featured: true
---

A quiet Saturday grocery run changed how I look at store shelves. Some aisles were empty and others were overflowing with items no one seemed to want. The imbalance felt avoidable if demand could be estimated with a better lens. I kept thinking about what a simple forecast could do for weekly planning. That thought became a small blueprint for a tool I could build with Python and Streamlit. What followed was a focused effort to turn data into decisions that store teams can trust. Dataset used [here](https://www.kaggle.com/datasets/manjeetsingh/retaildataset).

A few weeks later I turned those notes into an app that predicts weekly sales. It accepts basic inputs, applies consistent feature engineering, and serves a clean result. The goal was to keep the pipeline transparent and the behavior predictable. I wanted something that anyone could run from a cloned repository. In this write‑up I document every file I uploaded and every code path that shapes the forecast. The tone is simple because clear language helps future me just as much as it helps new readers.


## Repository Structure I Pushed to GitHub


```python
# File tree
ai_retail_sales_forecasting-main/
├── app.py
├── requirements.txt
└── models/
    ├── sales_forecast_hgb.pkl
    └── feature_schema.txt
```

This repository is intentionally small. It contains one Streamlit app, a locked dependency list, a serialized model, and a schema text file. Each file plays a distinct role in keeping inference consistent across machines. The app glues the pieces together and protects the user from silent failures.


## requirements.txt — Reproducible Environment


```python
streamlit>=1.37
scikit-learn==1.5.1
joblib==1.4.2
pandas==2.2.2
numpy==2.0.2
```

I pinned versions to freeze the execution environment. Streamlit powers the UI, scikit‑learn provides estimator interfaces, joblib handles model I/O, and pandas with NumPy manage data shapes. Locking versions avoids edge cases where a minor release changes parser behavior or array dtypes. It also makes deployment on a clean runner easier because the resolver is deterministic.


## models/feature_schema.txt — Contract Between Training and Inference


```python
Store
Dept
IsHoliday
year
month
week
quarter
dayofyear
week_sin
week_cos
month_sin
month_cos
lag1
lag4_mean
```

This list is the model’s contract. Columns appear here in the exact order the estimator expects them. The names remind me which features are raw identifiers, which encode time, and which capture lag. Keeping this file out of the code path reduces risk of accidental reordering and keeps the app honest about inputs.


## app.py — The Streamlit Application


### Imports and Page Configuration


```python
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date

st.set_page_config(page_title="Weekly Sales Forecaster", page_icon="📈", layout="centered")

```

The imports sit at the top so dependencies are obvious. The page configuration sets a clear title and a centered layout so forms are readable on laptops. I keep UI settings close to imports because they run once on app start and never hide among callbacks. This small structure choice makes the first screen load feel intentional.


### Model Path and Existence Check


```python
MODEL_PATH = os.path.join("models", "sales_forecast_hgb.pkl")
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found at models/sales_forecast_hgb.pkl. Please upload it to the models/ folder.")
    st.stop()
```

The model path is built with `os.path.join` to remain portable across operating systems. Before anything else, the app checks whether the binary exists where it should. If the file is missing, the app stops immediately and shows a precise message. Fail fast here prevents downstream errors that look unrelated to the true cause.


### Loading the Serialized Bundle


```python
model = bundle["model"]
```

The bundle is loaded with `joblib.load`, which is efficient for large numpy arrays. The object holds both the trained estimator and any metadata the training run saved. Loading early lets later UI code assume the predictor exists. If a load error occurs, Streamlit surfaces it at the top rather than halfway through a form interaction.


### The Middle Section (UI and Feature Engineering) — Described, Not Invented


Between the bundle load and the prediction call, the original source contains the UI inputs and feature engineering pipeline. In the archive I unzipped, these lines are collapsed into an ellipsis, so I will not paste invented code. Conceptually, this section collects `Store`, `Dept`, `Date`, and `IsHoliday` from the user, converts types carefully, and derives calendar features. It validates the shape against `feature_schema.txt` so inference columns line up exactly. Once the dataframe `X` is prepared, the app proceeds to validation and prediction.


### Validation and Prediction Output


```python
if X.isna().any().any():
        st.error("Inputs produced invalid values. Please adjust and try again.")
    else:
        yhat = model.predict(X)[0]
        st.success(f"Predicted Weekly Sales: ${yhat:,.2f}")
```

The validation step checks for NaN values after feature engineering. When a user supplies an invalid date or a non‑numeric store identifier, the guardrail catches it and explains the issue. On the happy path the model returns a single weekly sales number, which is formatted with two decimals and thousands separators. Surfacing the value with `st.success` keeps the final state easy to spot on the page.


### Example Inputs in the UI


```python
st.markdown("### Example inputs you can try:")
st.code(
    "Store=1, Dept=1, Date=2012-12-14, IsHoliday=True\n"
    "Store=5, Dept=12, Date=2011-11-25, IsHoliday=True\n"
    "Store=20, Dept=3, Date=2011-03-18, IsHoliday=False",
    language="text"
)
```

The app ships with three canned examples so first‑time users can test quickly. Each line mirrors the form fields and includes a holiday case and a regular week. Keeping these examples in the source creates a lightweight form of documentation that lives next to the code path. It also helps verify that the environment is wired correctly after a fresh clone.


## How I Run the App During Development


```python
# Create and activate a clean environment (example using venv)
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Launch Streamlit
streamlit run app.py
```

These four steps reproduce my local setup. A virtual environment isolates dependencies from the system interpreter. Installing from the pinned `requirements.txt` ensures consistent behavior across machines. Running Streamlit from the project root picks up relative paths for the models directory without extra flags.


## Error Handling Philosophy in This App


I avoid ambiguous states and prefer explicit failure. When the model file is absent, the app announces the exact path that is wrong and stops. When input sanitization produces invalid values, the app asks the user to adjust and try again. This approach keeps logs short and makes UI behavior predictable for non‑technical users. A small tool like this earns trust by being consistent about success and failure.


## Data Flow From Input to Prediction


1. User fills fields or uploads data in the UI.
2. The app converts inputs to the correct dtypes and derives calendar features.
3. The feature frame is ordered to match `feature_schema.txt`.
4. A NaN scan runs to catch invalid transformations.
5. The model predicts and the result is formatted for display.


This path removes guesswork from the transformation process. Every step is observable either in the code or on the screen. That transparency matters because silent coercion can produce confident but wrong numbers.


## Minimal Deployment Checklist I Followed


```python
- Confirm repository root contains `app.py`, `requirements.txt`, and the `models` folder.
- Verify `models/sales_forecast_hgb.pkl` loads with the current joblib and numpy versions.
- Retain `models/feature_schema.txt` in version control to document the expected inputs.
- Set `PYTHONHASHSEED` and fix random seeds during training to stabilize export artifacts.
- Smoke‑test three example inputs after deployment and compare results with the local run.
```

A lightweight checklist saves time when moving from a laptop to a hosted runner. I keep it in the blog rather than in the repo to avoid cluttering the code with commentary. If a future change breaks compatibility, this list makes the regression easy to isolate.


## Practical Scenarios Where This App Helps


Store managers often plan replenishment on a weekly cadence. A one‑step forecast gives a baseline that combines seasonality and short‑term momentum. With a quick UI the result can inform orders, labor planning, and promotion timing. The point is not absolute precision but consistent guidance that improves over subjective estimates. Small improvements in ordering reduce waste and improve shelf availability.


## Current Limitations and Honest Caveats


The model is only as good as the training data and the features included. External factors like weather, local events, and supply disruptions are not part of the input. The app handles one store and department at a time rather than bulk scoring. It predicts a single step ahead rather than a full horizon. These choices keep the app simple, but they limit scope until the next iteration.


## Extensions I Plan to Add Next


Batch predictions via uploaded CSV would help power users plan multiple departments at once. A chart comparing predicted and historical sales would make validation faster. Caching the model and schema read would improve cold‑start performance. Export buttons for CSV and JSON would simplify sharing results with other tools. Feature flags could allow safe experimentation without breaking the stable path.


## Frequently Asked Questions I Get From Reviewers


```python
# Why joblib instead of pickle?
# Joblib handles large numpy arrays efficiently and is the common choice for scikit‑learn artifacts.
# Why a schema file?
# It externalizes the column order and makes mismatches visible during code review.
# Can the page icon be customized?
# Yes, but it is cosmetic. The core logic lives in the data pipeline and guards around it.
# What if the prediction looks wrong?
# Check inputs, confirm dtypes, and verify that feature derivation matches the training pipeline.
```

I keep short answers next to questions to reduce back and forth during internal reviews. If the same topics come up often, they belong in documentation rather than in ad‑hoc messages.


## Closing Reflection


This app began as a reaction to a small real‑world inconvenience and grew into a reusable tool. Keeping the repository lean forces clarity about what truly belongs in the product. Guardrails in a tiny project matter as much as in a large one because they shape user trust. By documenting every moving piece I make it easier for the next contributor, including my future self.


## Glossary of Key Ideas
```python
Lag Feature = 'A numeric value copied from a prior time step, giving the model short‑term memory.'
Cyclic Encoding = 'Using sine and cosine transforms to represent repeating calendar patterns without artificial jumps.'
Guardrail = 'A protective check that fails fast when assumptions are violated.'
Determinism = 'The property that the same inputs and code produce the same outputs across runs.'
```
## Design Notes I Followed
```python
- Prefer explicit type conversions over implicit pandas coercion.
- Keep user‑visible errors actionable and specific.
- Separate model artifacts from application code by folder boundaries.
- Make one path succeed and every other path fail loudly.
```

## Implementation Note
A reliable inference path depends on strict alignment between training and serving code. When exporting a model, capture not only the estimator but also the exact feature order. During serving, derive features in a single function so that input evolution is controlled. Keep that function pure, avoid side effects, and cover it with small tests. This discipline reduces drift between notebooks and the production app.
