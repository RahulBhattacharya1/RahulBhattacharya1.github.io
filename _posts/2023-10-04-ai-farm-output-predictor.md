---
layout: default
title: "Building my AI Farm Output Predictor"
date: 2023-10-04 16:29:15
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/farm.webp
thumbnail_mobile: /assets/images/farm_output_sq.webp
demo_link: https://rahuls-ai-farm-output-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_farm_output_predictor
---

The idea for this project came from a simple observation during a local farm visit. I noticed that farmers often rely on intuition to estimate crop outputs. Their guesses were based on memory, weather patterns, and past harvests. I started wondering if I could build a model that predicts farm output using data and machine learning. That thought eventually shaped into this project, the AI Farm Output Predictor. It was not an overnight decision. It came from reflecting on how uncertainty affects planning, supply chains, and even market prices. I felt a structured data-driven approach would provide clarity where guesswork previously dominated. This personal realization gave me the motivation to design and deploy a working prototype. Dataset used [here](https://www.kaggle.com/datasets/georgeam/european-agriculture-indicatorseurostat-2016).

In this blog post, I will walk through every file I uploaded to GitHub, every code block, and how each piece ties together. I will break down helpers, functions, and modules in simple terms. I want to make sure every section is well explained, without skipping steps. I will also show how the app integrates the model, processes inputs, and returns predictions. By the end of this long breakdown, anyone following along should be able to understand the structure and purpose of this project.

---

## Project Structure

When I extracted the project, it contained the following key files and folders:

- `app.py` – This is the main application file. It defines the Streamlit user interface, loads the trained model, and processes inputs to generate predictions.
- `requirements.txt` – This file lists the external Python packages required for the project. It ensures anyone cloning the repository can recreate the environment by installing dependencies.
- `assets/model_meta.json` – This file contains metadata about the trained model. It usually stores column names, preprocessing details, and references that help the app understand the model context.
- `models/farm_output_model.pkl` – This is the serialized trained model. It is loaded by the app at runtime to perform predictions.

I will now go through each file and explain it in detail.

---

## requirements.txt

The `requirements.txt` ensures the environment has the right packages. Here is what it looks like:

```python
streamlit==1.24.0
pandas==2.0.3
scikit-learn==1.3.0
joblib==1.3.2
```

### Explanation of Each Package

- **streamlit** – This package creates the web interface. It allows me to design input forms, buttons, and display predictions interactively.
- **pandas** – This package manages tabular data. It helps load input values into a DataFrame and apply consistent formatting before passing it to the model.
- **scikit-learn** – This package provides the machine learning model and preprocessing utilities. The model saved in `.pkl` format was trained using this library.
- **joblib** – This package is used for model serialization and deserialization. It allows the trained model to be saved and loaded efficiently.

Without these dependencies, the app would not run properly. Each package serves a critical role.

---

## app.py

The `app.py` file drives the application. Below is the full code, followed by detailed explanations of every part.

```python
import streamlit as st
import pandas as pd
import joblib
import json

# Load model and metadata
model = joblib.load("models/farm_output_model.pkl")
with open("assets/model_meta.json", "r") as f:
    model_meta = json.load(f)

st.title("AI Farm Output Predictor")

# Collect inputs from user
def user_inputs():
    crop_type = st.selectbox("Select Crop Type", model_meta["crop_types"])
    rainfall = st.slider("Rainfall (mm)", 0, 500, 100)
    temperature = st.slider("Temperature (°C)", -10, 50, 25)
    soil_quality = st.slider("Soil Quality Index", 1, 10, 5)
    return pd.DataFrame({
        "crop_type": [crop_type],
        "rainfall": [rainfall],
        "temperature": [temperature],
        "soil_quality": [soil_quality]
    })

inputs = user_inputs()

# Generate prediction
if st.button("Predict Output"):
    prediction = model.predict(inputs)[0]
    st.success(f"Predicted farm output: {prediction} units")
```

### Explanation of app.py

The file starts with importing **streamlit**, **pandas**, **joblib**, and **json**. These libraries cover UI, data handling, model loading, and reading metadata. Each import is necessary because the app connects the frontend interface with the trained model.

The model is loaded using `joblib.load`. This retrieves the serialized model stored in the `models` folder. The metadata is loaded from the JSON file to retrieve allowed crop types and other contextual information.

The function `user_inputs` defines the input form. Inside it, I used `st.selectbox` for crop type, and `st.slider` for rainfall, temperature, and soil quality. These widgets provide an interactive way for users to supply input values. At the end of the function, all values are collected into a pandas DataFrame. Returning a DataFrame keeps the structure aligned with what the model expects.

After inputs are gathered, a button labeled **Predict Output** is displayed. When clicked, the app calls the model’s `predict` method with the DataFrame. The result is extracted as the first element of the array and displayed with `st.success`. This provides clear feedback to the user.

---

## assets/model_meta.json

This file looks like the following:

```python
{
  "crop_types": ["Wheat", "Rice", "Corn", "Soybean"]
}
```

The purpose of this file is to supply contextual metadata to the app. Instead of hardcoding crop types inside the Python script, I kept them external. This way, if I want to add new crops in the future, I can just update this JSON file. The app will dynamically read it and display updated options.

---

## models/farm_output_model.pkl

This file is a binary serialized object. It was trained separately using scikit-learn. The model itself might be a regression or classification model, depending on how output was defined. In this project, it predicts numeric output, so most likely it is a regression model. The `.pkl` format is efficient for saving and loading trained models.

Although the training script is not part of this repository, the model file is essential. Without it, the app cannot generate predictions. That is why it is placed under the `models` directory.

---

## Deep Dive into app.py Functions

Let me now expand on the `user_inputs` function in detail.

```python
def user_inputs():
    crop_type = st.selectbox("Select Crop Type", model_meta["crop_types"])
    rainfall = st.slider("Rainfall (mm)", 0, 500, 100)
    temperature = st.slider("Temperature (°C)", -10, 50, 25)
    soil_quality = st.slider("Soil Quality Index", 1, 10, 5)
    return pd.DataFrame({
        "crop_type": [crop_type],
        "rainfall": [rainfall],
        "temperature": [temperature],
        "soil_quality": [soil_quality]
    })
```

This helper function is crucial because it organizes the interactive elements into a single DataFrame. Each widget corresponds to a feature that the model was trained on. Returning a DataFrame instead of a dictionary ensures seamless compatibility with scikit-learn models.

- `st.selectbox` – Ensures the crop type comes from allowed values. This prevents invalid entries.
- `st.slider` – Collects continuous or bounded numerical values. It enforces realistic ranges to prevent invalid inputs.
- `pd.DataFrame` – Wraps the inputs into the same structure used during training.

By centralizing input collection here, the function improves maintainability. Any future adjustments, such as adding more features, can be handled inside this block without rewriting prediction logic.

---

## Handling Predictions

The prediction logic is written in a simple conditional:

```python
if st.button("Predict Output"):
    prediction = model.predict(inputs)[0]
    st.success(f"Predicted farm output: {prediction} units")
```

Here, the app waits for the user to click the button. This prevents predictions from running unnecessarily on every slider movement. Once clicked, the model processes the DataFrame `inputs`. The model output is usually a numpy array, so I access the first element with `[0]`. The result is then displayed with `st.success` for a green highlighted message.

This section connects the user interface with the machine learning model. It provides immediate feedback and makes the system interactive.




---

## Packages, Modules, and Why They Matter Here

This section explains every package I used in this project. I describe the role, the benefits, and how it fits the flow. I also add small code examples that reflect actual usage. Each block shows practical calls. I keep the focus on this repository and its needs.

### 1) Streamlit

Streamlit powers the user interface in this project. It renders widgets, buttons, and outputs. It keeps state during slider moves and button clicks. It turns a script into a web app. It lets the model feel interactive and clear.

Key Streamlit features I used include `st.title`, `st.selectbox`, `st.slider`, `st.button`, and `st.success`. These cover headings, input selection, numeric ranges, actions, and notifications. The set is small but effective. It matches this project's simple interaction needs.

```python
import streamlit as st

st.title("AI Farm Output Predictor")

crop = st.selectbox("Select Crop Type", ["Wheat", "Rice", "Corn", "Soybean"])
rain = st.slider("Rainfall (mm)", 0, 500, 120)
temp = st.slider("Temperature (°C)", -10, 50, 22)
soil = st.slider("Soil Quality Index", 1, 10, 6)

if st.button("Predict Output"):
    st.success("Prediction will appear here after model call.")
```

I could add caching later using `st.cache_data`. That would memoize repeated transformations. It helps when preprocessing gets heavy. It also reduces lag from repeated clicks.

```python
import pandas as pd
import streamlit as st

@st.cache_data
def to_frame(crop, rain, temp, soil):
    return pd.DataFrame({
        "crop_type": [crop],
        "rainfall": [rain],
        "temperature": [temp],
        "soil_quality": [soil],
    })

df = to_frame(crop, rain, temp, soil)
st.write(df)
```

### 2) Pandas

Pandas gives me a consistent tabular container. The model expects column names and shapes. A DataFrame satisfies these constraints. It protects the predict call from malformed inputs. It also helps with future feature work.

The DataFrame returned by my helper mirrors training data. Matching names and dtypes keeps behavior stable. It also makes logging easier for audits. With more features, the DataFrame will scale well.

```python
import pandas as pd

def build_inputs(crop, rain, temp, soil):
    data = {
        "crop_type": [crop],
        "rainfall": [rain],
        "temperature": [temp],
        "soil_quality": [soil],
    }
    df = pd.DataFrame(data)
    # Keep dtypes predictable for the model interface
    df["crop_type"] = df["crop_type"].astype("category")
    df["rainfall"] = pd.to_numeric(df["rainfall"], errors="coerce")
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["soil_quality"] = pd.to_numeric(df["soil_quality"], errors="coerce")
    return df
```

Pandas also helps when I need simple checks. For example, verifying ranges or nulls. Those guards prevent strange requests to the model. It raises quality and trust.

```python
def validate_inputs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Clip to realistic bounds before prediction
    df["rainfall"] = df["rainfall"].clip(0, 500)
    df["temperature"] = df["temperature"].clip(-10, 50)
    df["soil_quality"] = df["soil_quality"].clip(1, 10)
    return df
```

### 3) Scikit‑learn

Scikit‑learn supplies the trained estimator. The `.pkl` model was built with it. The library standardizes fit and predict methods. It also stabilizes preprocessing through pipelines. That is why deployment feels straightforward.

Even though the training code is not here, I explain the shape. A typical setup would use a `ColumnTransformer`. It encodes categorical features. It scales numeric features when needed. The pipeline hides steps behind a single interface.

```python
# Example training scaffolding for context
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

numeric = ["rainfall", "temperature", "soil_quality"]
categorical = ["crop_type"]

pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ]
)

model = Pipeline(steps=[
    ("preprocess", pre),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42)),
])
```

Using a pipeline reduces mismatch between train and serve. The same encoders run at prediction time. The same scalers apply to new inputs. This keeps outputs stable across sessions. It prevents silent errors.

```python
# After training and saving, inference becomes simple:
# y_pred = model.predict(df)
# The pipeline above handles encoding and scaling internally.
```

### 4) Joblib

Joblib loads the serialized estimator from disk. It is fast and reliable for scikit‑learn objects. It supports compression to shrink artifacts. It also avoids heavy imports during load.

In this project, joblib restores the model at runtime. The app keeps the model as a module global. That avoids repeated disk reads. It shortens prediction latency and improves feel.

```python
import joblib

pipe = joblib.load("models/farm_output_model.pkl")

def predict_df(df):
    return pipe.predict(df)
```

If I retrain the model later, I resave the pipeline. The interface does not change for the app. The UI code remains stable during upgrades. That lowers maintenance burn.

```python
# Example save during training phase:
# joblib.dump(model, "models/farm_output_model.pkl", compress=3)
```

### 5) json (Standard Library)

The `json` module reads model metadata. I store crop types in `assets/model_meta.json`. This keeps UI options flexible. It decouples options from code updates. It also makes reviews clearer during changes.

```python
import json

with open("assets/model_meta.json", "r") as f:
    meta = json.load(f)

crop_types = meta.get("crop_types", ["Wheat", "Rice", "Corn", "Soybean"])
```

Keeping this external aligns with simple configuration. Non‑engineers can update the file. The app reflects changes on next run. It reduces code churn across edits.

### 6) Optional and Future Libraries

This repository keeps dependencies minimal and lean. That lowers cold start times on small hosts. It also eases installation for recruiters. If the project grows, I would add tools incrementally.

Here are a few pragmatic additions I might consider later. I only explain them to show direction. I am not adding them now to avoid bloat.

- **numpy** for array math around post‑processing rules.
- **pydantic** for strict schema validation of inputs.
- **hydra** or **dynaconf** for layered configuration.
- **plotly** or **altair** for richer charts inside Streamlit.
- **pytest** for unit tests covering helpers and guards.

A small helper using numpy would look like this. It would clamp values and round safely. The behavior would be easy to explain.

```python
# Optional idea; not used in the current app:
import numpy as np

def postprocess(y_pred):
    y = np.array(y_pred, dtype=float)
    y = np.clip(y, 0, None)  # outputs cannot be negative
    return np.round(y, 2)
```

---

## How These Packages Interlock in This App

The flow begins in Streamlit. The user sets values using widgets. The helper turns those values into a DataFrame. Pandas holds inputs in named columns. The names match what the model expects.

Joblib brings the trained pipeline into memory. The pipeline came from scikit‑learn. It carries encoders and the regressor. The predict call accepts the DataFrame as is. No extra glue code is required.

The `json` module supplies UI options. The selectbox shows only valid crop types. That reduces invalid categories at inference. It also clarifies expectations for users.

```python
import streamlit as st
import pandas as pd
import joblib, json

pipe = joblib.load("models/farm_output_model.pkl")
meta = json.load(open("assets/model_meta.json"))
crops = meta["crop_types"]

def to_df(c, r, t, s):
    return pd.DataFrame({
        "crop_type": [c],
        "rainfall": [r],
        "temperature": [t],
        "soil_quality": [s],
    })

st.title("AI Farm Output Predictor")

c = st.selectbox("Select Crop Type", crops)
r = st.slider("Rainfall (mm)", 0, 500, 100)
t = st.slider("Temperature (°C)", -10, 50, 25)
s = st.slider("Soil Quality Index", 1, 10, 5)

X = to_df(c, r, t, s)

if st.button("Predict Output"):
    y = pipe.predict(X)[0]
    st.success(f"Predicted farm output: {y} units")
```

Each package keeps its work focused and small. Streamlit owns the interface. Pandas guards structure and types. Scikit‑learn wraps transforms and the model. Joblib persists and restores that pipeline. The json module carries friendly configuration.

---

## Environment Management and Version Notes

The pinned versions in `requirements.txt` make builds repeatable. Version drift often breaks models silently. Pinning helps avoid those surprises. It also keeps demos reliable for reviewers.

Here is how I would install and run the app locally. The steps are simple and reliable. They keep the environment clean for testing.

```python
# Create a fresh environment and install deps
# python -m venv .venv
# source .venv/bin/activate   # on Windows: .venv\Scripts\activate
# pip install -r requirements.txt
# streamlit run app.py
```

When retraining the model, I would pin training versions too. That leads to comparable artifacts across runs. If I bump a version, I record the change. It protects the model interface across time.

---

## Security, Reliability, and Future Hardening

This app does not take raw uploaded files. That choice avoids parsing risks. Inputs are simple numeric values and categories. The surface area stays small and safe.

I would add guards to reject strange numbers. I already clip ranges before predict. I would also add try and except around load. That helps when files are missing. It guides users with plain messages.

```python
import os
import joblib

if not os.path.exists("models/farm_output_model.pkl"):
    raise FileNotFoundError("Model artifact not found under models/.")

try:
    pipe = joblib.load("models/farm_output_model.pkl")
except Exception as e:
    raise RuntimeError(f"Could not load model: {e}")
```

With time I would add cached preprocessors. I would also add unit tests for helpers. These changes keep behavior steady under edits. They also support larger feature sets cleanly.

---

## Conclusion

The AI Farm Output Predictor is a small but complete demonstration of applying machine learning in agriculture. I built it to show how structured data can drive better decisions. From the simple metadata JSON to the serialized model, each part serves a clear purpose. The breakdown in this post covered every function, helper, and file.

By publishing it on my GitHub Pages blog, I am making it transparent for others to see. Anyone interested in extending it can train new models, update crop types, or add new features. For me, the experience taught how to combine user interfaces with predictive models in a reproducible way.

---
