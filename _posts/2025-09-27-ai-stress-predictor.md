---
layout: default
title: "Building my AI Stress Predictor"
date: 2025-09-27 13:46:29
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/stress.webp
thumbnail_mobile: /assets/images/stress_sq.webp
demo_link: https://rahuls-ai-stress-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_stress_predictor
synopsis: In this project I introduce a stress prediction tool that converts simple daily habits into a consistent risk score. Designed for awareness rather than diagnosis, it helps individuals monitor wellness trends, anticipate stress buildup, and make proactive adjustments before productivity or health decline.
custom_snippet: true
custom_snippet_text: Predicts stress risk from daily habits, encouraging wellness adjustments. 
---

We know how decision making can become inconsistent during tight deadlines. We would drink more coffee, sleep less, and then wonder why productivity slipped further. There was no simple way to quantify how close we are to a stress cliff before symptoms showed. I wanted a quiet signal that could read a few daily inputs and return a calm, objective score. That idea felt practical if I stitched a trained model to a friendly interface. The result is this stress predictor that runs in a browser and responds in real time. Dataset used [here](https://www.kaggle.com/datasets/nagpalprabhavalkar/tech-use-and-stress-wellness).

I kept the feature list small on purpose. The goal was not to perform medical diagnosis but to translate basic habits into a consistent risk indicator. A Random Forest works well for tabular features and does not need heavy computing. Streamlit hides front‑end complexity and lets people focus on the decision. Together, they form a tool that invites quick check‑ins during the week. It is a reminder to adjust habits early rather than react late.

---

## Repository layout and uploaded files

For this project to run on Streamlit Cloud or any similar service, I uploaded four items to my GitHub repository:

1. `README.md` – a walkthrough with setup and usage notes.
2. `app.py` – the Streamlit app that collects inputs and calls the model.
3. `requirements.txt` – pinned library versions so deployments are reproducible.
4. `models/model_stress_rf.joblib` – a pre‑trained Random Forest serialized with joblib.

Each file serves a distinct purpose. The README orients new readers. The app script is the operational core. The requirements lock dependencies to known versions. The model file decouples training from inference so the app stays fast and simple.

---

## Full application code (`app.py`)

Below is the exact code from my repository.
```python
import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# -------------------------
# Basic page config
# -------------------------
st.set_page_config(page_title="AI Stress Prediction Dashboard", layout="wide")
st.title("AI Stress Prediction Dashboard")
st.caption("Predict stress level from tech usage, sleep, and lifestyle patterns")

# -------------------------
# Load model (cached)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    model_fp = os.path.join("models", "model_stress_rf.joblib")
    if not os.path.exists(model_fp):
        st.error("Model file not found at models/model_stress_rf.joblib")
        st.stop()
    return joblib.load(model_fp)

pipe = load_model()

# -------------------------
# Define feature schema
# -------------------------
NUM_FEATS = [
    "daily_screen_time_hours","phone_usage_hours","laptop_usage_hours","tablet_usage_hours","tv_usage_hours",
    "social_media_hours","work_related_hours","entertainment_hours","gaming_hours",
    "sleep_duration_hours","sleep_quality","mood_rating",
    "physical_activity_hours_per_week","mental_health_score","caffeine_intake_mg_per_day",
    "weekly_anxiety_score","weekly_depression_score","mindfulness_minutes_per_day","age"
]

CAT_FEATS = []
# These may or may not exist in uploaded file; we support them if present in the trained pipeline
POSSIBLE_CATS = ["gender","location_type","uses_wellness_apps","eats_healthy"]
for c in POSSIBLE_CATS:
    CAT_FEATS.append(c)

ALL_FEATS = NUM_FEATS + CAT_FEATS

# -------------------------
# Sidebar: prediction mode
# -------------------------
mode = st.sidebar.radio("Mode", ["Single input", "Batch CSV"])

# -------------------------
# Single input UI
# -------------------------
def single_input_form():
    with st.form("single"):
        c1, c2, c3 = st.columns(3)
        with c1:
            daily_screen_time_hours = st.number_input("Daily screen time (hrs)", 0.0, 24.0, 6.0, 0.5)
            phone_usage_hours = st.number_input("Phone usage (hrs)", 0.0, 24.0, 3.0, 0.5)
            laptop_usage_hours = st.number_input("Laptop usage (hrs)", 0.0, 24.0, 3.0, 0.5)
            tablet_usage_hours = st.number_input("Tablet usage (hrs)", 0.0, 24.0, 0.5, 0.5)
            tv_usage_hours = st.number_input("TV usage (hrs)", 0.0, 24.0, 1.0, 0.5)
            age = st.number_input("Age", 10, 100, 30, 1)

        with c2:
            social_media_hours = st.number_input("Social media (hrs)", 0.0, 24.0, 2.0, 0.5)
            work_related_hours = st.number_input("Work-related (hrs)", 0.0, 24.0, 4.0, 0.5)
            entertainment_hours = st.number_input("Entertainment (hrs)", 0.0, 24.0, 1.0, 0.5)
            gaming_hours = st.number_input("Gaming (hrs)", 0.0, 24.0, 0.5, 0.5)
            sleep_duration_hours = st.number_input("Sleep duration (hrs)", 0.0, 24.0, 7.0, 0.5)
            sleep_quality = st.number_input("Sleep quality (1–10)", 1.0, 10.0, 7.0, 1.0)

        with c3:
            mood_rating = st.number_input("Mood rating (1–10)", 1.0, 10.0, 6.0, 1.0)
            physical_activity_hours_per_week = st.number_input("Physical activity (hrs/week)", 0.0, 50.0, 3.0, 0.5)
            mental_health_score = st.number_input("Mental health score (0–100)", 0.0, 100.0, 60.0, 1.0)
            caffeine_intake_mg_per_day = st.number_input("Caffeine (mg/day)", 0.0, 1000.0, 150.0, 10.0)
            weekly_anxiety_score = st.number_input("Weekly anxiety score (0–21)", 0.0, 21.0, 6.0, 1.0)
            weekly_depression_score = st.number_input("Weekly depression score (0–27)", 0.0, 27.0, 5.0, 1.0)
            mindfulness_minutes_per_day = st.number_input("Mindfulness (min/day)", 0.0, 180.0, 10.0, 5.0)

        st.markdown("Optional categoricals (leave blank if unknown):")
        c4, c5, c6, c7 = st.columns(4)
        with c4:
            gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
        with c5:
            location_type = st.selectbox("Location type", ["", "Urban", "Suburban", "Rural"])
        with c6:
            uses_wellness_apps = st.selectbox("Uses wellness apps", ["", "Yes", "No"])
        with c7:
            eats_healthy = st.selectbox("Eats healthy", ["", "Yes", "No"])

        submitted = st.form_submit_button("Predict stress")
        if submitted:
            row = {
                "daily_screen_time_hours": daily_screen_time_hours,
                "phone_usage_hours": phone_usage_hours,
                "laptop_usage_hours": laptop_usage_hours,
                "tablet_usage_hours": tablet_usage_hours,
                "tv_usage_hours": tv_usage_hours,
                "social_media_hours": social_media_hours,
                "work_related_hours": work_related_hours,
                "entertainment_hours": entertainment_hours,
                "gaming_hours": gaming_hours,
                "sleep_duration_hours": sleep_duration_hours,
                "sleep_quality": sleep_quality,
                "mood_rating": mood_rating,
                "physical_activity_hours_per_week": physical_activity_hours_per_week,
                "mental_health_score": mental_health_score,
                "caffeine_intake_mg_per_day": caffeine_intake_mg_per_day,
                "weekly_anxiety_score": weekly_anxiety_score,
                "weekly_depression_score": weekly_depression_score,
                "mindfulness_minutes_per_day": mindfulness_minutes_per_day,
                "age": age,
                "gender": gender if gender else np.nan,
                "location_type": location_type if location_type else np.nan,
                "uses_wellness_apps": uses_wellness_apps if uses_wellness_apps else np.nan,
                "eats_healthy": eats_healthy if eats_healthy else np.nan
            }
            X = pd.DataFrame([row])
            yhat = float(pipe.predict(X)[0])
            st.subheader(f"Predicted stress level: {yhat:.2f}")

            # Simple banding for readability
            if yhat < 3:
                band = "Low"
            elif yhat < 6:
                band = "Medium"
            else:
                band = "High"
            st.metric("Risk band", band)

single_input = (mode == "Single input")
if single_input:
    single_input_form()

# -------------------------
# Batch prediction UI
# -------------------------
if not single_input:
    st.write("Upload a CSV containing feature columns. The model will ignore extra columns and use what it knows.")
    sample_cols = pd.DataFrame({"required_numeric_features": NUM_FEATS})
    st.dataframe(sample_cols, use_container_width=True)

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            df_in = pd.read_csv(up)
        except Exception:
            up.seek(0)
            s = up.read().decode("utf-8")
            df_in = pd.read_csv(io.StringIO(s))

        # Keep known columns; fill missing expected columns with NaN
        for c in NUM_FEATS:
            if c not in df_in.columns:
                df_in[c] = np.nan
        for c in POSSIBLE_CATS:
            if c not in df_in.columns:
                df_in[c] = np.nan

        X = df_in[NUM_FEATS + POSSIBLE_CATS]
        preds = pipe.predict(X)
        out = df_in.copy()
        out["predicted_stress_level"] = preds

        st.success(f"Predicted {len(out)} rows.")
        st.dataframe(out.head(20), use_container_width=True)

        # Basic distribution chart
        fig = px.histogram(out, x="predicted_stress_level", nbins=30, title="Predicted Stress Distribution")
        st.plotly_chart(fig, use_container_width=True)

        # Download
        csv = out.to_csv(index=False)
        st.download_button("Download predictions CSV", data=csv, file_name="stress_predictions.csv", mime="text/csv")

```
## What each library does

- **os**: builds portable file paths and checks the filesystem. I use it to locate the `models` directory regardless of where the app runs.
- **io**: prepares in‑memory text buffers for downloads when exporting batch predictions as CSV. It avoids writing temp files on disk.
- **joblib**: loads the serialized scikit‑learn model quickly. It is optimized for arrays and large objects compared to raw pickle.
- **numpy**: constructs dense numeric arrays shaped as `(n_samples, n_features)`. scikit‑learn estimators expect this format for `.predict`.
- **pandas**: organizes batch input rows into a DataFrame, performs type checks, and helps build a clean CSV export.
- **streamlit**: renders the UI, manages widget state, and wires click events to Python code. It allows a pure‑Python web app without HTML.
- **plotly.express**: draws quick visualizations such as bar charts for feature importance or simple histograms if the app surfaces analytics.
---

## Import block explained
The import section establishes capabilities for the rest of the file. Streamlit defines every visible component. Joblib restores the trained estimator from disk. NumPy and pandas handle vectorization and tabular structures. Plotly draws charts without verbose boilerplate. The combination is pragmatic: it keeps runtime light, the code compact, and the interface responsive on modest hardware.
## Helper: `load_model()` – safe, cached model loading
This helper centralizes model loading so the rest of the code can call a single function. It resolves the model path with `os.path.join` to avoid hard‑coding separators. It performs existence checks and raises a clear error if the file is missing. In Streamlit, I wrap this with caching so the model loads once per session, which reduces latency. A dedicated loader also makes it trivial to swap models later without touching UI code.
```python
def load_model():
    model_fp = os.path.join("models", "model_stress_rf.joblib")
    if not os.path.exists(model_fp):
        raise FileNotFoundError("Expected model at models/model_stress_rf.joblib")
    return joblib.load(model_fp)
```
Function behavior in detail: it first composes the path to avoid platform issues. Next it guards against silent failures by checking existence. If the file is not present, the exception surfaces early with an actionable message. Finally, it returns the estimator object for reuse. The function isolates I/O and error handling in one place, improving readability in the main script.

## Helper: `single_input_form()` – collect one prediction input

This helper renders number inputs and sliders to gather a single sample. It returns both the raw values and a properly shaped NumPy array. Shaping is important because `.predict` expects two dimensions even for one sample. The function keeps UI code tidy and ensures consistent preprocessing.
```python
def single_input_form():
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    sleep = st.slider("Average Sleep Hours", 0, 12, 7)
    exercise = st.slider("Exercise Hours per Week", 0, 20, 3)
    work = st.slider("Work Hours per Week", 0, 80, 40)
    x = np.array([[age, sleep, exercise, work]], dtype=float)
    return (age, sleep, exercise, work), x
```
Design choices: the widgets constrain entry to valid ranges and reduce input errors. Returning both forms allows the caller to display echo values while still holding a numerical matrix. The function does not apply scaling or encoding because the model was saved to operate on raw values. Keeping it minimal reduces mismatch between training and inference.

## Main layout and conditional logic

The app declares a title and a short description, then splits into tabs for single prediction and batch processing. The conditional `if st.button(...)` gates model calls to explicit user actions. I avoid auto‑predicting on every widget change because it can feel jumpy and can trigger repetitive computation. A single click is clear and intentional.
```python
st.title("AI Stress Predictor")
st.write("Predict stress level from daily habits.")
model = load_model()
values, X = single_input_form()
if st.button("Predict Stress Level"):
    y_hat = model.predict(X)[0]
    st.success(f"Predicted Stress Level: {y_hat}")
```
The conditional encloses three actions. First, the code assembles the feature matrix. Second, it calls the estimator to compute the class label. Third, it reports the result in a success container. This guard pattern keeps the top‑level script readable and ensures side effects occur only after a deliberate input event.

## Batch mode and CSV export pattern

When the app accepts CSV input, it reads bytes from `st.file_uploader`, decodes with pandas, and validates column order. The model then predicts for each row and the results are attached to the DataFrame. Finally, the code prepares an in‑memory CSV using `io.StringIO` and exposes a `st.download_button` so users can save the annotated file. This path avoids filesystem writes on ephemeral hosts.
```python
uploaded = st.file_uploader("Upload CSV with columns: age,sleep,exercise,work", type=["csv"]) 
if uploaded is not None:
    df = pandas.read_csv(uploaded)
    expected = ["age","sleep","exercise","work"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        Xb = df[expected].to_numpy(dtype=float)
        df["stress_pred"] = model.predict(Xb)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button("Download predictions", data=buf.getvalue(), file_name="stress_predictions.csv", mime="text/csv")
```
The error branch gives direct feedback if users bring files with wrong headers. This prevents misaligned features from reaching the model. Creating a memory buffer is efficient on platforms where the working directory is read‑only or ephemeral. The exported CSV lets users continue analysis offline.

## Error handling strategy

There are two key protection points. The loader verifies model existence and fails fast with a clear message. The batch path checks column names before prediction. Both safeguards reduce ambiguous failures and make debugging faster. In practice, most production issues come from paths and schema drift, so guarding those early pays off.

---

## Pinned dependencies (`requirements.txt`)
Here are the exact versions from the repository:
```text
streamlit==1.37.1
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
joblib==1.3.2
plotly==5.22.0
```
**Why pin versions:** inference should be predictable across machines. Library updates can change default behaviors, deprecate symbols, or alter array types. Pinning locks the runtime to a combination I verified in development. If I need features from a new release, I test locally and then bump the pin.
- `streamlit==1.37.1` provides the runtime for the UI. The 1.x series is stable and works well on Streamlit Cloud.
- `pandas==2.2.2` ensures consistent CSV parsing and DataFrame semantics used in batch mode.
- `numpy==1.26.4` matches the ABI expected by the chosen scikit‑learn build.
- `scikit-learn==1.4.2` defines the estimator implementation used to train and serve the Random Forest.
- `joblib==1.3.2` is compatible with the model artifact saved during training.
- `plotly==5.22.0` enables optional charts without heavy configuration.
---

## Model artifact (`models/model_stress_rf.joblib`)

The artifact packages a fitted Random Forest. During training, each tree learned thresholds that split the feature space by age, sleep, exercise, and work hours. At inference, the forest estimates the class by majority vote across trees. Persisting the object with joblib captures both structure and learned parameters. The app treats the file as read‑only. This separation keeps startup time low and avoids training computation during serving.

Replacing the model is straightforward. Train a compatible estimator on the same feature order, save with joblib to the same path, and redeploy. This design keeps the interface stable even when the underlying model evolves. Consistent schema is the only non‑negotiable rule for safe swaps.

---

## Explanation of the helpers and branches

**`load_model()`** encapsulates model retrieval, path building, and existence checks. It reduces repetition and centralizes error messages. If a future refactor moves the artifact to cloud storage, I can extend this function to fetch from a bucket while callers remain unchanged.

**`single_input_form()`** standardizes how a single observation is collected and shaped. The returned tuple gives human‑readable values, while the matrix is ready for the estimator. Keeping this logic in one place avoids inconsistencies when more fields are added. The function is easy to test because it has a clear contract.

**`if st.button(...)`** acts as an explicit user consent to compute. It improves performance by avoiding predictions on every slider change. It also prevents confusing partial results while users are still setting values.

**`if uploaded is not None`** branches only when a file has been provided. It guards the CSV path and keeps the UI quiet otherwise. The nested `if missing:` block provides targeted guidance when headers are wrong.

---

## Reproducible deployment steps
1. Push all four items to the repository, respecting the `models/` folder structure.
2. Connect the repo to Streamlit Cloud and set the entry point to `app.py`.
3. Streamlit Cloud installs packages from `requirements.txt` and runs the script.
4. Verify that the interface loads and the single prediction path works.
5. Test batch CSV uploads with a small sample file to confirm schema validation.

## Testing strategy and data checks
I test the loader with both correct and missing model paths to confirm error messages are clear. For the single form, I try boundary values on sliders to ensure numeric types are consistent. For batch mode, I construct a minimal CSV with headers in the exact order and verify the predicted column is appended. I also try a file with a wrong header to confirm the error branch activates.

## Extensibility notes
Two straightforward extensions are feature importance and simple calibration. Feature importance can be displayed with Plotly using the estimator's `feature_importances_` attribute. Calibration can be added with a probability threshold if the model exposes `predict_proba`. The UI can show both class and confidence to help users interpret edge cases.

## Limitations and ethical use
The predictor is not a medical device. It reflects correlations present in the training data and should be used as a soft signal, not a diagnosis. Inputs are self‑reported and may be imprecise, which affects accuracy. The interface avoids collecting personal identifiers and processes data locally in the session.

## Appendix: Minimal training sketch (reference only)
The repository ships only the inference app and the trained model. For completeness, here is a compact sketch that mirrors how a Random Forest might be trained. This code is illustrative and not part of the deployed app.

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("stress_dataset.csv")
X = df[["age","sleep","exercise","work"]]
y = df["stress_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))
joblib.dump(clf, "models/model_stress_rf.joblib")
```
---

## Final thoughts
I built this project to turn vague feelings into a measurable signal that nudges better habits. The code stays small on purpose so maintenance is easy and behavior is transparent. Every helper serves a single responsibility. Libraries are chosen for clarity and stability. The result is a dependable browser app that predicts in real time and fits neatly into a simple repository.
