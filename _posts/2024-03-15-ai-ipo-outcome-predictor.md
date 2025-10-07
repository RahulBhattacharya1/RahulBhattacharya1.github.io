---
layout: default
title: "Building my AI IPO Outcome Predictor"
date: 2024-03-15 10:12:33
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/ipo.webp
thumbnail_mobile: /assets/images/ipo_outcome_sq.webp
demo_link: https://rahuls-ai-initial-public-offering.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_initial_public_offering
---

Sometimes inspiration comes from unexpected places. I once found myself reading news articles about companies planning to go public, and I kept wondering which of these would perform well. The market reports offered opinions but lacked structured analysis. That gap gave me the idea of creating a tool where IPO details could be entered and predictions generated instantly.

This idea grew into a full project. I trained a machine learning model on IPO data and deployed it with Streamlit. The app became interactive, easy to test, and shareable on GitHub Pages. In this post I will break down every file, every code block, and every helper that made the project work. Dataset used [here](https://www.kaggle.com/datasets/karanammithul/ipo-data-india-2010-2025).

---

## Files in the Project

- **app.py**: Main Streamlit app that ties together schema, model, and interface.
- **requirements.txt**: Lists Python packages required to run the app.
- **models/ipo_model.joblib**: Trained machine learning model compressed for deployment.
- **models/schema.json**: Describes input features, their ranges, and the prediction task.

Each file is important. Without the model or schema, the app would stop. Without requirements.txt, others could not reproduce the environment. The next sections dive into the code itself.

---

## Imports and Page Setup

```python
import os, json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="IPO Outcome Predictor", page_icon="📈", layout="centered")
st.title("IPO Outcome Predictor")
st.write("Enter IPO attributes to estimate a **return/gain** or a **status** (auto-detected from your dataset).")
```

Here the libraries are imported. Each serves a specific purpose. **os** and **json** handle paths and schema parsing. **joblib** loads the machine learning model efficiently. **pandas** manages structured data once user inputs are collected. **streamlit** builds the UI.

The next lines configure the page, setting its title, icon, and layout. This establishes the identity of the app right from launch.

---

## File Paths and File Requirement Helper

```python
MODEL_PATH = os.path.join("models", "ipo_model.joblib")
SCHEMA_PATH = os.path.join("models", "schema.json")

def require(path, kind):
    if not os.path.exists(path):
        st.error(f"{kind} file missing: `{path}`. Upload it to your repo and restart.")
        st.stop()

require(MODEL_PATH, "Model")
require(SCHEMA_PATH, "Schema")
```

The **require** helper ensures that the necessary files exist. If a file is missing, the app stops gracefully while showing an error message. This prevents cryptic runtime errors and informs the user clearly about what needs fixing. Both the model and schema must be present for the app to function.

---

## Loading Model and Schema

```python
@st.cache_resource
def load_model_and_schema():
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)
    model = joblib.load(MODEL_PATH)
    return model, schema

model, schema = load_model_and_schema()
```

The function **load_model_and_schema** opens the schema JSON and loads the trained model. Decorating with **@st.cache_resource** ensures it runs once and the results are cached, making subsequent predictions faster. Both schema and model are returned together and stored in variables for later use.

---

## Extracting Schema Details

```python
task = schema.get("task", "regression")
target_name = schema.get("target", "listing_gain")
cat_schema = schema.get("categorical", {})
num_schema = schema.get("numeric_ranges", {})
feature_order = schema.get("feature_order", list(cat_schema.keys()) + list(num_schema.keys()))
```

The schema defines how data should be interpreted. The app reads whether the task is regression or classification, what the target column is, and how to handle categorical and numerical features. **feature_order** ensures consistency in how data is structured before prediction.

---

## Building the Input Form

```python
with st.form("ipo_form"):
    inputs = {}
    # categorical controls
    for c, options in cat_schema.items():
        label = c.replace("_"," ").title()
        if options:
            inputs[c] = st.selectbox(label, options, index=0)
        else:
            inputs[c] = st.text_input(label)

    # numeric controls
    for c, rng in num_schema.items():
        mn, mx = float(rng.get("min", 0.0)), float(rng.get("max", 100.0))
        med = float(rng.get("median", (mn+mx)/2))
        if mn == mx:
            mx = mn + 1.0
        step = max(0.01, (mx - mn) / 100.0)
        inputs[c] = st.number_input(c.replace("_"," ").title(), min_value=mn, max_value=mx, value=med, step=step)

    submitted = st.form_submit_button("Predict")
```

This form collects user inputs. Categorical features are displayed as dropdowns or text fields. Numerical features are displayed with number inputs bounded by schema-provided min and max values. Median values serve as defaults. Using **st.form** groups inputs and creates a submission button, ensuring predictions only run after explicit user action.

---

## Handling Predictions

```python
if submitted:
    row = {k: inputs.get(k) for k in feature_order}
    X = pd.DataFrame([row])

    try:
        if task == "classification":
            pred = model.predict(X)[0]
            prob_txt = ""
            try:
                proba = model.predict_proba(X)[0]
                p = float(max(proba))
                prob_txt = f" (confidence: {p*100:.1f}%)"
            except Exception:
                pass
            st.success(f"Predicted {target_name.replace('_',' ').title()}: **{pred}**{prob_txt}")
        else:
            val = float(model.predict(X)[0])
            st.success(f"Predicted {target_name.replace('_',' ').title()}: **{val:,.3f}**")
        st.caption("Model: scikit-learn Pipeline (One-Hot for categoricals + Linear/Logistic Regression).")
    except Exception as e:
        st.error("Prediction failed. Ensure the model & schema match this app.")
        st.exception(e)
```

Once the form is submitted, the app builds a row of inputs ordered by schema. It creates a pandas DataFrame with one row, ready for prediction. If the task is classification, it tries to display both the predicted class and the probability confidence. If it is regression, it outputs a numeric prediction with formatting.

Error handling ensures that mismatched models or schema structures are caught gracefully. A caption explains the underlying model pipeline so users understand how predictions are generated.

---

## Explaining the Model in an Expander

```python
with st.expander("How it works"):
    st.markdown(
        """
        - **Auto-detected task**: classification if a status-like label exists; otherwise regression on a return/gain column.
        - **Preprocessing**: One-Hot for text features; numeric features passed through with median imputation at training time.
        - **Model size**: Compressed joblib to stay below GitHub’s 25 MB limit.
        """
    )
```

The expander provides a concise description of how the model works. It reassures users by explaining that preprocessing and task detection were handled systematically. It also clarifies why the model file size matters on GitHub, noting the 25 MB limit.

---

## requirements.txt

The requirements file lists packages such as streamlit, pandas, scikit-learn, and joblib. Anyone cloning the repo can install them with pip. This guarantees the app runs in the same environment everywhere.

---

## Model and Schema Files

The file **ipo_model.joblib** contains the trained machine learning pipeline. It was saved in compressed format to respect GitHub file limits. The **schema.json** describes input features, their ranges, categorical options, and task type. These two files must match each other, otherwise predictions will fail. The schema also defines feature order, which the app uses to align input with the model.

---

## Conclusion

This project started with a question about IPO outcomes and grew into a deployed prediction tool. By combining a trained model with Streamlit, I made the system interactive and reproducible. Every helper and conditional in the code played a role in making the app robust. This breakdown showed how each part worked together, from imports to schema handling, input forms, prediction logic, and user explanations.

By uploading these files to GitHub, the project became easy to share. Anyone could clone the repo, install dependencies, and start exploring IPO predictions right away. That was the final goal: to turn curiosity into a functioning application.
