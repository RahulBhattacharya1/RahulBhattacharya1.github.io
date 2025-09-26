---
layout: default
title: "Building my AI Rain Predictor"
date: 2025-03-28 10:16:33
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-rain-prediction.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_rain_prediction
featured: true
---

I grew tired of glancing at the daily forecast and guessing whether to carry an umbrella. Several days looked like clear skies, yet short showers arrived anyway. That small inconvenience created a bigger question for me. Could I make a compact predictor that reads today’s numbers and estimates tomorrow’s rain. I wanted something simple, fast, and transparent enough to trust. Dataset used [here](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package).

The idea turned into a plan when I realized a small pipeline could travel well. If I trained once and froze the bundle, I could ship only the lightweight parts to users. It would avoid the heavy training environment and still be useful. The result is this app: a focused interface that turns inputs into a clear yes or no. It does not try to replace professional forecasts; it tries to be a dependable assistant.


## `requirements.txt` (Environment Control)

I pin core libraries so the app runs the same on any host. Version pinning guards against subtle API changes and numeric behavior shifts. The stack is compact and intentional. Each dependency plays a distinct role and keeps the repo light.

```python
streamlit>=1.37
scikit-learn==1.6.1
joblib==1.5.2
pandas==2.2.2
numpy==2.0.2
```

- **streamlit** drives the UI and event loop.  
- **scikit-learn** provides the pipeline and estimator API.  
- **joblib** loads the serialized model efficiently.  
- **pandas** structures inputs for the pipeline.  
- **numpy** supports vectorized numeric operations.

## Imports and Top‑Level Setup

```python
import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
```

I import only what the app uses. Streamlit renders the interface and coordinates user interactions. Pandas and numpy give a familiar structure for numeric arrays. Joblib loads the frozen pipeline without recreating a training session. I also import os for robust path building across hosts.

Keeping imports at the top creates a signal to readers about dependencies. It also makes failure modes clear if something is missing. If any import fails, the error appears immediately, and users know to check the environment.

## Helper: `load_model()` with Caching

```python
def load_model():
    path = os.path.join("models", "rain_pipeline.joblib")
    bundle = joblib.load(path)
    return bundle["model"], bundle.get("feature_order", None)
```

This function centralizes reading the serialized pipeline bundle from the models folder. I cache it with Streamlit so the file is read once per session. That cuts idle disk access and avoids redundant deserialization.

The loader returns two artifacts: the model object and a stable `feature_order` list. Bundling the order inside the model artifact protects the app from silent column misalignment.

## Unpacking the Model and Metadata

```python
model, feature_order = load_model()
```

I immediately unpack the two items into local variables. This keeps later code legible and avoids nested indexing calls. The pattern also makes it easy to swap models later with minimal edits.

Placing this near the top ensures the app fails fast if the bundle is missing. Users receive a clear error before they interact with the form.

## Page Title and Introduction

```python
st.title("Rain Prediction (Australia)")
st.write("Predict whether it will rain tomorrow based on today’s weather readings.")
st.write("Predict whether it will rain tomorrow based on today’s weather readings.")
```

I set a concise title and a one‑line description so users understand the scope instantly. The copy speaks in plain language. The app predicts whether rain is likely tomorrow based on today’s readings. It keeps focus on action: provide inputs, receive a decision.

## Stable Feature Contract

```python
FEATURES = [
    "MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed",
    "Humidity3pm", "Pressure9am", "Pressure3pm", "Temp3pm",
    "RainToday", "WindGustDir", "WindDir3pm"
]
```

This list encodes the exact columns the model expects at inference time. Treating it as a contract prevents accidental drift if new fields are added elsewhere. By declaring it in code, I guarantee both clarity and repeatability.

I do not infer columns from user input because that invites reordering risk. Instead, I align user data to this list explicitly before prediction.

## Categorical Domain: Compass Directions

```python
COMPASS = [
    "N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"
]
```

I freeze the set of valid wind directions from the training data. This protects the pipeline’s encoder from unseen string values. The same list feeds the select boxes, so the UI and the model remain aligned.

Using controlled categories reduces ambiguity and improves data quality. It also makes it clear to users which inputs are accepted.

## Structured Input: The Streamlit Form

```python
with st.form("single_input"):
    st.subheader("Enter today’s weather")
    col1, col2, col3 = st.columns(3)
```

A form groups all widgets and defers computation until submission. This prevents accidental partial runs and makes validation predictable. Columns help lay out fields in a balanced grid so related variables sit near each other.

Within the form I define both numeric and categorical inputs. The arrangement is tuned for readability and fast data entry.

## Submission Control

```python
st.form_submit_button("Predict")
```

The submit button toggles a boolean flag. I wrap the prediction logic under `if submitted:` so nothing executes until the user has provided every field. This pattern keeps app state clean and prevents partial dataframes.

It also shapes the user experience into a single, clear action. Users enter data, submit once, and receive a decision.

## Building the Inference Row

```python
pd.DataFrame([[
        MinTemp, MaxTemp, Rainfall, WindGustSpeed,
        Humidity3pm, Pressure9am, Pressure3pm, Temp3pm,
        RainToday, WindGustDir, WindDir3pm
    ]], columns=FEATURES)
```

I build a single‑row DataFrame with columns named to match the features list. This mirrors how scikit‑learn expects tabular inputs at inference. Using DataFrame instead of raw lists improves readability and debugging.

Constructing the row this way also makes it easy to log or display inputs later. The column names become self‑describing documentation for the prediction call.

## Executing the Prediction

```python
model.predict(row)[0]
    proba = float(model.predict_proba(row)[0, 1])
```

The pipeline encapsulates preprocessing and classification in one call. That means encoders, scalers, and the estimator all run under a single interface. The output is an array so I index the first element for a single row.

Keeping the pipeline intact reduces complexity here. I do not have to re‑implement transforms because the bundle already knows them.

## User Feedback Messages

```python
st.error(f"Error processing file: {e}")
```

I provide two clear branches for messaging. When the model flags rain, I show an affirmative card with next‑step guidance. When it does not, I show a different card clarifying that no rain is expected. The language is direct and avoids hedging so the decision is actionable. In a later version I can add calibrated probabilities for richer context.


## `models/rain_pipeline.joblib` (Frozen Pipeline)

The model file is a serialized pipeline that contains preprocessing and the estimator. 
I trained it outside this repository to keep deployment lean. 
The bundle also carries `feature_order` so inference code can align columns deterministically. 
This practice blocks silent errors when new columns appear or when order changes.

Shipping the bundle alone makes maintenance simple. 
If I retrain with more data or a different algorithm, I only replace the joblib file. 
The rest of the application remains the same because the interface to `model.predict()` does not change. 
This detachment between model lifecycle and app lifecycle is the key to fast iteration.


## Design Notes and Trade‑offs

- **Caching vs Memory**: I cache the model for speed, accepting small memory cost. This pays off because prediction calls stay snappy.
- **Strict Feature Contract**: Enforcing column order increases code clarity and reduces debugging time. It trades a tiny bit of verbosity for confidence.
- **Minimal UI, Clear Action**: A single form and a single decision keeps the mental model simple. It trades feature richness for trust and speed.
- **Externalized Training**: Not training in‑app means the repo ships fast. It trades exploration convenience for deployability.


## Testing the App End‑to‑End

I validate three layers before sharing the app. 
First, the environment: I recreate a clean venv, install from `requirements.txt`, and import each library. 
Second, the interface: I launch `streamlit run app.py` and exercise the form with realistic values, including edge cases at input bounds. 
Third, the pipeline: I feed a few known input rows and confirm consistent outputs with the training notebook’s holdout checks. 
These steps keep surprises low and help isolate failures quickly.


## How I Deploy It

This repository is deployment‑friendly because it is small. 
I run it on Streamlit Cloud or any host that supports `streamlit run`. 
The only requirement is to include the `models/` folder with the joblib file. 
When the app starts, it loads the model once and then handles requests without retraining. 
For updates, I replace the model bundle and push a new build.

**File map**

- `app.py` — Streamlit UI and inference code.  
- `requirements.txt` — pinned dependencies.  
- `models/rain_pipeline.joblib` — serialized pipeline and feature order.


## Appendix A — Data Dictionary for Inputs

- **MinTemp**: Today’s minimum air temperature in °C measured near the surface.
- **MaxTemp**: Today’s maximum air temperature in °C measured near the surface.
- **Rainfall**: Precipitation measured today in millimeters.
- **WindGustSpeed**: Maximum wind gust speed recorded today in km/h.
- **WindDir9am / WindDir3pm**: Compass direction of the wind at 9 a.m. and 3 p.m.
- **Humidity9am / Humidity3pm**: Relative humidity in percent at 9 a.m. and 3 p.m.
- **Pressure9am / Pressure3pm**: Atmospheric pressure in hPa at 9 a.m. and 3 p.m.
- **Temp9am / Temp3pm**: Air temperature in °C at 9 a.m. and 3 p.m.
- **RainToday**: Whether any rain fell during the day (categorical).

## Appendix B — User Guide (Practical Steps)

1. Open the app and read the short description.
2. Enter today’s readings in the form fields.
3. Use the select boxes for compass and the rainfall-today flag.
4. Press **Predict** to run the model once.
5. Read the result card and plan accordingly.

## Appendix C — Troubleshooting Playbook

- **Model file missing**: Ensure `models/rain_pipeline.joblib` exists. Place it under the `models/` folder relative to `app.py`.
- **ImportError**: Reinstall using `pip install -r requirements.txt`. Verify versions match the file.
- **Unicode issues in text**: Save files as UTF‑8. Streamlit expects UTF‑8 by default.
- **Strange predictions**: Double‑check that inputs align to realistic ranges. Inspect the feature order reindex step.

## Appendix D — Extending the App Safely

- Add new input fields only after retraining the pipeline with those features.
- Update the `FEATURES` list and include the new columns in the form and the DataFrame.
- Keep the `feature_order` in sync by exporting it with the new bundle.
- Prefer adding a probability output (e.g., `predict_proba`) before altering the binary messaging.

## Block-by-Block Details

### `import`

The import block lists every external tool the script relies on. Streamlit handles the UI, while pandas and numpy manage table and vector operations. Joblib performs the light but critical task of deserializing the pipeline. Putting imports first surfaces missing dependencies early.

### `def load_model`

The loader function isolates file IO and transforms it into a cached resource. The cache keeps the model in memory after the first read, avoiding repeated disk access. The returned tuple includes both the estimator and its required column order. This pairing stops subtle bugs from column position changes.

### `model, feature_order`

Unpacking makes later calls cleaner. There is no need to index into a dict on every reference. If the bundle changes format in the future, this line will fail fast. That makes maintenance safer.

### `st.title(`

The title sets the top-level context with a single line. It tells users what the app does before they interact. A short statement beats a long description at this point. Detail arrives later where it matters.

### `st.write(`

A simple paragraph explains scope and expectations. It sets boundaries so users do not assume capabilities that the app does not have. The copy talks about rain decision only. It leaves temperature and wind forecasts out of scope.

### `FEATURES`

The features list is the contract between training and serving. It is the single source of truth for column names. Any pipeline step that assumes a specific order depends on this contract. Keeping it in code prevents accidental drift.

### `COMPASS`

Compass values are categorical and closed. The UI and encoder both depend on the same set. That guarantees there are no unseen labels at inference time. Users can only choose valid values.

### `with st.form(`

The form is a structured container for data entry. It collects widgets and binds them to a single submit action. That reduces accidental partial execution. It also enables layout primitives like columns.

### `st.form_submit_button(`

Submission flips a boolean that gates the prediction path. This prevents code from running while inputs are incomplete. It also gives a stable checkpoint for logging or metrics. The pattern is reliable in Streamlit apps.

### `pd.DataFrame(`

The single-row DataFrame carries both data and schema. Its columns mirror the FEATURES list so the pipeline sees a familiar shape. DataFrames make debugging easier because columns have names. That helps when inspecting intermediate states.

### `model.predict(`

The pipeline executes preprocessing and the classifier in one call. That hides complexity and keeps the app surface small. The output is numeric or boolean depending on the estimator. The code adapts to either by reading the first item.

### `st.error(`

The negative branch communicates no-rain decisions clearly. It avoids alarmist phrasing. The aim is a calm, decisive message. The user leaves with a plan either way.


## Operational Checklist

- Recreate a clean environment and install from `requirements.txt`.
- Confirm `models/rain_pipeline.joblib` exists and loads once on startup.
- Verify form fields render with expected constraints and categories.
- Enter boundary values to test numeric validation and UI behavior.
- Trigger multiple predictions to confirm caching prevents repeated disk reads.

## Security & Privacy Notes

- No external APIs are called; inputs remain in-session.
- No user data is persisted to disk by default.
- If logging is added later, avoid storing raw sensitive inputs.
- Keep joblib files untrusted; never execute code during load (joblib is data, not code).
- Pin dependencies to reduce the chance of supply-chain surprises.

## FAQ

**Why not include probability?**  
The initial goal was a binary decision for clarity. Adding calibrated probabilities is on the roadmap and requires access to `predict_proba` and proper thresholds.

**What happens if I change a feature name?**  
The reindex step will fail or misalign. Always update the FEATURES list and retrain so the bundle and UI stay in sync.

**Can I swap the estimator?**  
Yes. Retrain the pipeline, export a new joblib, and keep the same interface. The app code does not need to change.

**Will it work without internet?**  
Yes. It has no external network dependencies once the environment is installed.

## My Suggestions

### Modify or Add a Feature Column
1. Retrain your pipeline with the new features so encoders and scalers adapt.
2. Export a new bundle that includes the updated `feature_order`.
3. Update the `FEATURES` list in `app.py` to match the training columns.
4. Add or modify the form inputs so users can provide values for those features.
5. Confirm the `reindex` step still aligns to the updated order.

### Swap to a Probability UI
1. Ensure the estimator supports `predict_proba`.
2. After the form submission, call `model.predict_proba(inference_row)[:, 1]`.
3. Display a percentage with a calibrated threshold and a short explanation of uncertainty.
4. Consider adding a simple chart to visualize confidence bins.

### Add Input Validation Hints
1. Expand widget help texts to describe expected ranges and units.
2. For categorical inputs, keep options tied to constants defined near imports.
3. Add an `st.warning` if a field is left at a default that is atypical for your region.
4. For numeric anomalies, consider a small rules engine that flags outliers for review.

## Performance Notes

- Model loading is the dominant cold-start cost; caching eliminates repeats.
- UI latency is driven by widget count and network; this app is local-only after launch.
- Keep the joblib compact by pruning training artifacts not needed for inference.
- Avoid heavy per-request allocations; reuse structures when possible.

## Accessibility Considerations

- Use clear labels and units for every field.
- Keep contrast high and avoid relying solely on color for meaning.
- Ensure keyboard navigation works across all widgets.
- Provide short error messages that explain how to fix the input.

## Maintenance Plan

- Revisit dependency versions quarterly and update if security patches land.
- Track prediction drift by sampling inputs and outputs, then compare with later observed weather.
- Document every bundle update in the repository releases with hash and training notes.
- Automate a smoke test that loads the model and runs a dummy prediction on CI.
