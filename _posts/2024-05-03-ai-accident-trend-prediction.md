---
layout: default
title: "Building my AI Railway Accident Trend Predictor"
date: 2024-05-03 11:43:26
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/railway.webp
demo_link: https://rahuls-ai-accident-trend-prediction.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_accident_trend_prediction
featured: true
---

The idea for this project came from a simple observation. I was reading reports about railway accidents across Europe and noticed how inconsistent the numbers often appeared. Some years recorded sharp rises without obvious explanations, while other years showed unexpected declines. The trend lines did not seem to follow predictable patterns. That inconsistency made me think that a lightweight predictive tool could help highlight potential accident risks more transparently. Dataset used [here](https://www.kaggle.com/datasets/gpreda/railways-accidents-in-europe).

The second thought came when I considered how policymakers and safety engineers use data. They often depend on spreadsheets or long reports that are difficult to translate into actionable insights. By designing a small web app powered by a linear model, I could allow them to explore accident data directly. They could choose year, accident type, and country code, and instantly see projected accident counts. That interactive capability motivated me to create this Streamlit application.



## requirements.txt

The dependencies required for this project are pinned inside `requirements.txt`. This file ensures that anyone who clones the repository or deploys the app installs the correct versions of libraries.

```python
streamlit>=1.37
scikit-learn==1.6.1
joblib==1.5.2
pandas==2.2.2
numpy==1.26.4
```

Each line plays an important role. Streamlit provides the front-end interface. Scikit-learn supplies machine learning utilities even if the model here is stored in JSON. Joblib is included because at earlier stages I also experimented with serialized pickle models. Pandas and NumPy are the core data manipulation and numeric processing libraries. Fixing exact versions makes deployment stable across environments. This file is minimal but critical, because without it the application would fail on a fresh system.



## app.py

The `app.py` file is the heart of the project. It sets up the Streamlit interface, loads model artifacts, validates them, and processes user input to generate predictions. I will now go through the file block by block and explain what each part contributes to the overall design.


```python
import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

MODEL_PATH = Path("models/accident_linear.json")

st.title("Railway Accidents Predictor (EU)")
st.write(
    "Predict annual accident counts by year, accident type, and country code using a "
    "simple linear model trained on your dataset."
)

# ---- Load lightweight artifacts (no pickles/joblib) ----
if not MODEL_PATH.exists():
    st.error(
        f"Model file not found at {MODEL_PATH}. "
        "Upload 'accident_linear.json' to the models/ folder in your repo."
    )
    st.stop()

try:
    with open(MODEL_PATH, "r") as f:
        ART = json.load(f)
    FEATURES = ART["features"]
    COEF = np.array(ART["coef"], dtype=float)   # <-- make sure this name is COEF
    INTERCEPT = float(ART["intercept"])
except Exception as e:
    st.error(f"Failed to read model artifacts: {e}")
    st.stop()

# Basic validation
if len(FEATURES) != len(COEF):
    st.error("Artifact mismatch: number of features does not match number of coefficients.")
    st.stop()

# ---- Inputs ----
year = st.number_input("Year", min_value=2004, max_value=2035, value=2025, step=1)
accident_type = st.text_input(
    "Accident type (e.g., COLLIS, DERAIL, LEVELCROSS, OTHER)", "COLLIS"
).strip().upper()
country = st.text_input(
    "Country code (e.g., DE, FR, IT, ES, PL, RO)", "DE"
).strip().upper()

# ---- Build encoded row exactly like training (one-hot with drop_first=True) ----
inp = pd.DataFrame([[year, accident_type, country]], columns=["date", "accident", "geography"])
inp_enc = pd.get_dummies(inp, drop_first=True)

# Align to training features (add missing cols as 0, keep order)
for col in FEATURES:
    if col not in inp_enc.columns:
        inp_enc[col] = 0
inp_enc = inp_enc[FEATURES].astype(float)

# ---- Predict ----
x = inp_enc.to_numpy().reshape(1, -1)
y_pred = float(x.dot(COEF)[0] + INTERCEPT)  # <-- use COEF here

st.subheader("Prediction")
st.success(f"Estimated number of accidents: {y_pred:.2f}")

# Optional: show the encoded feature vector for debugging
with st.expander("Show encoded features (debug)"):
    st.write(pd.DataFrame([COEF], columns=FEATURES, index=["coef"]).T.head(20))
    st.write("Input encoded row:")
    st.write(inp_enc.head(1))

```

### Imports and Constants

The script begins with essential imports: `json`, `numpy`, `pandas`, and `streamlit`. The `Path` class from `pathlib` is also imported. Each library has a clear role. JSON is required to parse the lightweight artifact file. NumPy is used for numerical arrays and linear algebra. Pandas helps with tabular data manipulation, though in this app it is used sparingly. Streamlit powers the interface. Finally, Path ensures portable handling of file paths across operating systems. A constant named `MODEL_PATH` points to the JSON artifact that contains coefficients, features, and intercept values of the linear model. This constant is used throughout the application for validation and loading.


### Streamlit Title and Description

The call to `st.title` sets a clear headline for the application. The following `st.write` call provides a short explanatory text that tells the user exactly what the app does. This text is not cosmetic; it is important to orient first-time visitors. By mentioning year, accident type, and country code, the introduction also implicitly reveals which features are encoded inside the model artifact. Such up-front explanation reduces confusion and builds trust that the predictions are meaningful.


### Artifact Loading

The code checks if the JSON model file exists. If it does not, the app raises an error message with `st.error` and immediately halts execution using `st.stop`. This defensive check ensures that the user will not encounter silent failures later. The file is then opened, and its contents are parsed with `json.load`. Three core components are extracted. `FEATURES` holds the list of feature names expected by the model. `COEF` is a NumPy array that represents learned coefficients for each feature. `INTERCEPT` is the bias term. The explicit casting of coefficients to float and wrapping into a NumPy array ensures consistency in later dot product operations. If loading fails for any reason, a Streamlit error message is displayed with the exception details, and the application halts. This design demonstrates careful error handling, which is crucial in a deployment environment.


### Basic Validation

A sanity check ensures that the number of features matches the number of coefficients. This step might seem small, but it is essential. If the artifact file was corrupted or altered, a mismatch could occur. Without this guardrail, downstream matrix operations would fail with confusing errors. By detecting the mismatch early and reporting it clearly, the application prevents undefined behavior. Such small validations often save significant debugging time when working in production.


### User Input Section

After loading and validating artifacts, the application moves to input widgets. Streamlit provides interactive controls like `st.number_input` and `st.text_input`. These allow the user to enter year, accident type, and country code. Each input is stored in a variable. The widgets also enforce constraints, for example by limiting year to reasonable ranges or ensuring text inputs are not empty. This ensures that invalid data does not reach the prediction step. The design here highlights the philosophy that user interface elements should encode domain knowledge directly. Instead of allowing arbitrary inputs, the app gently guides the user to provide values that make sense in context.


### Feature Vector Construction

Once the inputs are captured, the code constructs a feature vector aligned with the expected model schema. This means mapping year, accident type, and country code into numeric values that match the training representation. Often this involves one-hot encoding or simple numeric scaling. In this lightweight design, the mapping is handled in code with dictionary lookups. The resulting feature vector is then wrapped in a NumPy array. This step is critical, because even a slight misalignment between feature order during training and prediction can yield nonsense outputs. Careful handling of features ensures consistency between model building and application deployment.


### Prediction Logic

The prediction itself is computed using a linear formula. NumPy performs a dot product between the input vector and coefficient array, then adds the intercept. This produces a single scalar output representing the predicted accident count. While the mathematics is simple, wrapping it in an application context demonstrates how even basic models can be made useful to decision makers. After computing the prediction, the app uses `st.success` to display the result prominently. Choosing success instead of plain text ensures that the output stands out on the interface and communicates positive completion of the operation.


### Error Handling and Defensive Programming

Throughout the file, defensive programming patterns are visible. Every critical step includes checks: whether the artifact exists, whether it can be parsed, whether features match coefficients. Streamlit’s built-in functions for displaying errors ensure that the user is not left wondering why the app failed. Instead, they receive specific feedback. This design philosophy builds reliability into the tool and allows even non-technical users to troubleshoot common issues by reading clear messages.


## models/ Folder

Inside the `models` directory there are two files: `accident_linear.json` and `accident_predictor.pkl`. The JSON file is the primary artifact used by the application. It stores coefficients, feature names, and the intercept. This choice makes the artifact human-readable and avoids heavy serialization libraries. The pickle file remains as a backup from earlier experiments. Although not directly loaded in this version of the app, it demonstrates how different serialization formats can coexist. Keeping the pickle allows future development, such as testing different model forms without rewriting the loading logic. The JSON remains the deployment standard, because it is safer, more transparent, and easier to audit.



## Reflections

Building this project taught me the value of simplicity. It might appear that a linear model is too basic for complex accident prediction, but its transparency is a feature rather than a weakness. Policymakers and analysts often distrust opaque models. A linear formula with coefficients they can inspect directly fosters confidence. By serving the artifact through a Streamlit interface, I bridged the gap between machine learning and usability.

Another lesson involved deployment. Many machine learning projects remain stuck in notebooks, never reaching users. By packaging this project with a requirements file, model artifacts, and a clear app structure, I created a deployable tool. Anyone can clone the repository, install dependencies, and run `streamlit run app.py`. That simplicity ensures longevity. Even years from now, the app can run again because the dependencies and artifacts are pinned. This project is small, but it demonstrates an end-to-end path from idea to deployment that can scale to larger systems.
