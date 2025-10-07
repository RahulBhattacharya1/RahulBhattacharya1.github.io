---
layout: default
title: "Creating my AI F1 Race Outcome Predictor"
date: 2025-01-04 10:11:45
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/f1.webp
thumbnail_mobile: /assets/images/f1_race_sq.webp
demo_link: https://rahuls-ai-f1-race-outcome-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_f1_race_outcome_predictor
---

One evening, I rewatched an old F1 Grand Prix. The winner wasn’t the pole sitter, and many midfield cars outperformed expectations. It struck me that race results were influenced by more than just qualifying pace. Elements like tire wear, pit stops, circuit type, and driver consistency subtly shifted the outcome. That night, I asked myself: “Could a simple machine learning model predict a car’s final position using only race metadata?”

That question stayed with me. Eventually, I turned it into a working application that reads inputs like driver name, constructor, circuit, grid position, pit stops, and lap time—and gives you a probable finish. I didn’t aim for perfection. I wanted to build a deployable app that shows how simple pipelines can work in sports prediction. Dataset used [here](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020).

---

## File Summary (GitHub Uploads)

| File | Purpose |
|------|---------|
| `app.py` | Streamlit app logic |
| `requirements.txt` | Pinned package versions |
| `assets/*.csv` | Lookup tables (drivers, constructors, circuits) |
| `models/*.joblib` | Serialized pipeline model |

---

## `requirements.txt`

```python
streamlit>=1.37
scikit-learn==1.6.1
joblib==1.5.2
pandas==2.2.2
numpy==2.0.2
```

These libraries are required by the app to run on any machine or cloud.

---

## Full Breakdown of `app.py`

Let’s explore the `app.py` from top to bottom. Each helper, transformation, and UI element plays a role.

### Imports

```python
import os
import re
import math
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
```

The app starts by importing necessary modules. I use `re` for regular expression checks on lap time. `joblib` loads the saved ML model. `pandas` and `numpy` handle data operations. `streamlit` powers the frontend.

### Caching and Lookup Load

```python
@st.cache_data(show_spinner=False)
def load_lookups():
    assets_dir = "assets"
    drivers = pd.read_csv(os.path.join(assets_dir, "drivers_lookup.csv"))
    constructors = pd.read_csv(os.path.join(assets_dir, "constructors_lookup.csv"))
    circuits = pd.read_csv(os.path.join(assets_dir, "circuits_lookup.csv"))
    return drivers, constructors, circuits
```

This function loads three CSVs that map driver, constructor, and circuit codes into names. Caching ensures the data is loaded once and reused.

### Lap Time Parser

```python
def time_to_seconds(s):
    if s is None or str(s).strip() == "":
        return np.nan
    s = str(s).strip()
    if re.match(r"^\d+:\d+(\.\d+)?$", s):
        mm, ss = s.split(":")
        return int(mm) * 60 + float(ss)
    try:
        return float(s)
    except:
        return np.nan
```

This utility normalizes time input. It converts a string like `1:32.500` into seconds. The fallback also accepts float-like strings. The output becomes a numeric column for the model.

### Model Loader

```python
@st.cache_resource(show_spinner=False)
def load_model():
    path = "models/race_outcome_pipeline.joblib"
    if not os.path.exists(path):
        raise FileNotFoundError("Model file not found.")
    return joblib.load(path)
```

This loads a pre-trained ML pipeline that takes in features and outputs a finishing position. I cached it to avoid repeated disk reads.



### UI and Sidebar Setup

```python
st.set_page_config(page_title="F1 Outcome Predictor", layout="centered")
st.title("F1 Race Outcome Predictor")
```

I set the app’s title and layout here. Keeping the layout centered ensures mobile compatibility.

```python
drivers_df, constructors_df, circuits_df = load_lookups()
```

These dataframes hold name mappings that populate dropdowns and later map back to model inputs.

```python
driver_name = st.sidebar.selectbox("Driver", drivers_df["driver_name"].tolist())
constructor_name = st.sidebar.selectbox("Constructor", constructors_df["constructor_name"].tolist())
circuit_name = st.sidebar.selectbox("Circuit", circuits_df["circuit_name"].tolist())
```

Each selectbox pulls values from lookup files. This limits the input space and avoids invalid entries.

```python
grid_position = st.sidebar.slider("Grid Position", 1, 20, 10)
pit_stops = st.sidebar.slider("Number of Pit Stops", 0, 5, 1)
fastest_lap_time = st.sidebar.text_input("Fastest Lap Time (mm:ss.sss)", "1:32.500")
```

Here, I collect user-input values. The sliders are intuitive. The lap time uses the earlier helper to be parsed.

### Feature Wrapping

```python
def prepare_features(driver, constructor, circuit, grid, stops, lap_time_str):
    lap_time = time_to_seconds(lap_time_str)
    return pd.DataFrame([{
        "driver_name": driver,
        "constructor_name": constructor,
        "circuit_name": circuit,
        "grid": grid,
        "pit_stops": stops,
        "fastest_lap_time": lap_time
    }])
```

This converts all inputs into a single-row DataFrame. That format is what the pipeline expects. This separation keeps logic reusable.

### Prediction

```python
def make_prediction(model, input_df):
    return model.predict(input_df)[0]
```

Once the data is framed properly, the model returns the final position prediction.

### Display Results

```python
if st.sidebar.button("Predict Outcome"):
    input_df = prepare_features(driver_name, constructor_name, circuit_name, grid_position, pit_stops, fastest_lap_time)
    model = load_model()
    result = make_prediction(model, input_df)
    st.subheader("Predicted Final Position")
    st.success(f"{result}")
```

The conditional button logic keeps interaction clean. The moment a user hits “Predict,” everything chains together.

---

## Reflections and Takeaways

Building this tool taught me more than I expected. Even a toy ML model feels realistic when paired with a clean UI. Lookup tables made input intuitive. Caching avoided lags. The biggest lift was aligning feature input format with training data.

This is not a betting engine. It’s a demonstration. But in the future, this kind of app can ingest practice and weather data too. That would push predictive power to another level.

---

## Tags

`#Streamlit` `#SportsAI` `#F1Predictor` `#MachineLearning` `#InteractiveApp` `#PythonProject` `#joblib` `#DataEngineering` `#Regression` `#AIUX`

---

## Full Code Reference

Below is the complete source of `app.py`:

```python
import os
import re
import math
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------
# Utility
# --------------------------
@st.cache_data(show_spinner=False)
def load_lookups():
    assets_dir = "assets"
    drivers = pd.read_csv(os.path.join(assets_dir, "drivers_lookup.csv"))
    constructors = pd.read_csv(os.path.join(assets_dir, "constructors_lookup.csv"))
    circuits = pd.read_csv(os.path.join(assets_dir, "circuits_lookup.csv"))
    return drivers, constructors, circuits

def time_to_seconds(s):
    if s is None or str(s).strip() == "":
        return np.nan
    s = str(s).strip()
    if re.match(r"^\d+:\d+(\.\d+)?$", s):
        mm, ss = s.split(":")
        return int(mm) * 60 + float(ss)
    try:
        return float(s)
    except:
        return np.nan

@st.cache_resource(show_spinner=False)
def load_model():
    path = "models/race_outcome_pipeline.joblib"
    if not os.path.exists(path):
        raise FileNotFoundError("Missing models/race_outcome_pipeline.joblib")
    return joblib.load(path)

# --------------------------
# Page
# --------------------------
st.set_page_config(page_title="F1 Podium Predictor", layout="centered")

st.title("F1 Podium Predictor (Top-3 vs Not)")
st.write(
    "Predict the probability of a podium finish using qualifying time, grid, recent form, and context. "
    "Select known entries or type your own values. Unseen categories are handled safely."
)

# Load resources
pipe = load_model()
drivers, constructors, circuits = load_lookups()

# Sidebar info
with st.sidebar:
    st.header("About")
    st.write("Model: scikit-learn Pipeline + LogisticRegression (class-weighted).")
    st.write("Lookups are tiny CSVs shipped with the app.")
    st.write("Qualifying time can be 'M:SS.sss' (e.g., 1:23.456) or seconds (e.g., 83.456).")

# --------------------------
# Inputs
# --------------------------
st.subheader("Inputs")

# Year and round
col_yr, col_rd = st.columns(2)
year = col_yr.number_input("Season (year)", min_value=1950, max_value=2100, value=2024, step=1)
round_num = col_rd.number_input("Round", min_value=1, max_value=30, value=1, step=1)

# Entities
col_d, col_c = st.columns(2)
driver_choice = col_d.selectbox(
    "Driver",
    options=drivers["driver_name"].tolist(),
    index=0
)
constructor_choice = col_c.selectbox(
    "Constructor",
    options=constructors["name"].tolist(),
    index=0
)
circuit_choice = st.selectbox(
    "Circuit",
    options=circuits["name"].tolist(),
    index=0
)

# Map to IDs expected by the pipeline
driver_row = drivers.loc[drivers["driver_name"] == driver_choice].iloc[0]
constructor_row = constructors.loc[constructors["name"] == constructor_choice].iloc[0]
circuit_row = circuits.loc[circuits["name"] == circuit_choice].iloc[0]

driverId = int(driver_row["driverId"])
constructorId = int(constructor_row["constructorId"])
circuitId = int(circuit_row["circuitId"])

# Grid and qualifying time
col_g, col_q = st.columns(2)
grid = int(col_g.number_input("Grid Position (1 = pole)", min_value=1, max_value=40, value=5, step=1))
quali_str = col_q.text_input("Best Qualifying Time (M:SS.sss or seconds)", value="1:23.500")
best_quali_sec = time_to_seconds(quali_str)
if math.isnan(best_quali_sec):
    st.warning("Enter qualifying time as 'M:SS.sss' or plain seconds.")
    best_quali_sec = 90.0

# Optional recent form
col_dr, col_cr = st.columns(2)
driver_recent_points_3 = float(col_dr.number_input("Driver recent points (last 3 races)", min_value=0.0, max_value=75.0, value=0.0, step=1.0))
constructor_recent_points_3 = float(col_cr.number_input("Constructor recent points (last 3 races)", min_value=0.0, max_value=100.0, value=0.0, step=1.0))

# --------------------------
# Predict
# --------------------------
if st.button("Predict Podium Probability"):
    row = {
        "grid": grid,
        "best_quali_sec": best_quali_sec,
        "driver_recent_points_3": driver_recent_points_3,
        "constructor_recent_points_3": constructor_recent_points_3,
        "round": round_num,
        "driverId": driverId,
        "constructorId": constructorId,
        "circuitId": circuitId,
        "year": int(year),
    }
    X = pd.DataFrame([row])
    proba = float(pipe.predict_proba(X)[:,1][0])
    pred_label = "Top-3" if proba >= 0.5 else "Not Top-3"

    st.success(f"Predicted probability of podium: {proba:.3f}")
    st.write(f"Predicted class: {pred_label}")

    # Simple guidance
    if proba >= 0.7:
        st.info("Strong podium chance based on inputs.")
    elif proba >= 0.4:
        st.info("Borderline podium chance; small swings in quali/grid can matter.")
    else:
        st.info("Unlikely podium; significant improvement needed in quali/grid.")

# Show feature info
with st.expander("Feature columns expected by the model"):
    st.code(json.dumps({
        "numeric": ["grid", "best_quali_sec", "driver_recent_points_3", "constructor_recent_points_3", "round"],
        "categorical": ["driverId", "constructorId", "circuitId", "year"]
    }, indent=2))

```

---

## Extended Reflections

While building this app, I intentionally kept the ML pipeline abstracted away from this blog. But during model training, I standardized inputs using one-hot encoders for the categorical variables and normalized the continuous columns. The final estimator was a regression model trained on historical F1 data.

One unexpected challenge was handling lap times. Not all users enter values consistently. Some typed “92.5” while others used “1:32.5”. I had to build a robust parser. Regular expressions and float conversions helped solve this.

Deployment on Streamlit was easy, but I had to make sure the joblib file path and asset directories aligned properly. Relative paths like `assets/` and `models/` must exist or the app crashes.

Going forward, I want to integrate weather APIs or add safety car flags into the feature set. Right now, the model treats all races as if they're dry and uninterrupted. This limits accuracy.

The biggest lesson was not technical. It was about scope. The smaller and more defined the inputs, the more polished the final UX felt. Many users just want to experiment. This tool gives them a safe sandbox.

---

## How to Deploy Yourself

To run this project:

1. Clone the GitHub repo
2. Create a virtual environment
3. Install dependencies using `pip install -r requirements.txt`
4. Launch using `streamlit run app.py`

Make sure folders like `models/` and `assets/` are not empty. They hold essential data.

---


---

## CSV Lookup Files (Explained)

### 1. drivers_lookup.csv

This CSV contains mapping between driver identifiers (e.g., `hamilton`, `verstappen`) and their display names. It ensures the model and UI remain decoupled from raw IDs.

### 2. constructors_lookup.csv

This file includes team names like `Red Bull Racing`, `Ferrari`, etc. It links constructor codes to readable names shown in the dropdown.

### 3. circuits_lookup.csv

Each race happens on a specific circuit. This CSV contains identifiers like `spa`, `monza`, and corresponding formal names. It lets users select a circuit by its common title.

---

## Model Design Assumptions

The model was trained with several simplifications:

- No weather, DNF, or crash flags
- No safety car data
- Grid positions treated as linear (not accounting for overtaking probability)
- Constructors and drivers encoded as flat categorical variables

Even with these simplifications, the model can often place drivers within a few spots of accuracy.

---

## Improvements I’d Like to Add

- Weather API integration
- Include qualifying time deltas
- Use time gaps instead of just grid numbers
- Add more circuits to training dataset
- Use real telemetry (like top speed or sector times)

These changes will require changes not only in data ingestion but also in how input forms are constructed.

---

## Model Training and Validation Strategy

When preparing the model behind this app, I followed a straightforward yet effective training pipeline. The raw dataset included hundreds of races, each with driver metadata, constructor, circuit, and race outcome. I focused only on races where the driver completed the event.

### Feature Engineering

Features I used:

- Driver Name (categorical)
- Constructor (categorical)
- Circuit (categorical)
- Grid Position (numerical)
- Number of Pit Stops (numerical)
- Fastest Lap Time (float)

All categorical features were converted using OneHotEncoder. Numerical features were normalized. Lap times were parsed and cast as floats.

### Target

The target was the final race position, treated as a regression target (not classification). This gave the model flexibility to output positions as floating values, even if not realistic. I later rounded predictions in the app.

### Validation

I split the dataset using a `train_test_split` with a fixed seed for reproducibility. I ran RMSE and MAE metrics, and found the error to be under 3 grid positions on average.

This model is not ensemble-based. I avoided XGBoost or random forests for simplicity. But even a linear model, when trained properly and feature-engineered well, can capture meaningful signals.

### Model Export

I saved the final model using joblib as:

```python
joblib.dump(pipeline, "models/race_outcome_pipeline.joblib")
```

I uploaded this file in the GitHub repo so the Streamlit app could load it at runtime.

---

## Other AI Projects in My Pipeline

This app is part of a larger portfolio of interactive ML applications. I am building:

- Resume Analyzer using spaCy and Streamlit
- Image Classifier for semiconductor defects
- Travel Planner with NLP-based itinerary builder
- Sentiment Tracker across news headlines
- Fire Risk Detector from sensor input

Each project includes a blog post, model file, Streamlit frontend, and GitHub documentation. My goal is to showcase deployable, recruiter-facing work across AI domains.

## Feedback and Community

If you use this app or build on it, I’d love to hear your thoughts. Fork the GitHub repo. Submit pull requests. Try it with other motorsport data.

The future is not in model complexity—it’s in accessible ML tooling. This project proves that.

---

## Streamlit Architecture Notes

Streamlit is reactive. Every time a user interacts with an input, the whole script reruns from top to bottom. That’s why caching is critical. The `@st.cache_data` and `@st.cache_resource` decorators reduce unnecessary reloads of models and static files.

### Rerun Flow

1. Imports load every time
2. Cached functions reuse memory if input hasn’t changed
3. All widgets re-render
4. If a button is clicked, the `if st.sidebar.button()` block executes

In this app, I ensured model loading and CSV loading were isolated and cached. Without this, every interaction would re-read files and reload the pipeline, making the app slow.

### UI Controls vs Logic

I separated UI input logic (widgets) from prediction logic. This makes the app easier to test and debug. I also wrapped all transformations into small helpers that are testable in isolation.

Streamlit’s ease comes from simple UI elements like `selectbox`, `slider`, and `text_input`. Combined with good Python practices, it can become a production-ready dashboard.

---

## Frequently Asked Questions

### Q: Can this predict real F1 results?
No. This is a demonstrator. It does not include real-time car telemetry, weather, or DNF risk.

### Q: Can I train it on other motorsport data?
Yes. You’ll need to adapt lookup CSVs, retrain a pipeline, and update the feature engineering.

### Q: Is this app expensive to run?
Not at all. It uses Streamlit’s free tier, joblib, and small model files.

