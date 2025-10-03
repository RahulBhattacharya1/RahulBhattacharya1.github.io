---
layout: default
title: "Enhancing my AI Student Performance Predictor"
date: 2024-05-19 08:41:33
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/student.webp
demo_link: https://rahuls-ai-student-performance.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_student_performance
featured: true
---

It started with a moment of curiosity while looking at a dataset of student records. I wondered if there was a simple way to estimate how students might perform in mathematics if I only had some details about their study habits, attendance, and classroom engagement. The thought lingered for days until I decided to turn it into a small project. The goal was clear: take raw attributes about a student, feed them into a model, and let the application produce an estimated math score.

This project was not just about predicting a number. It was about building something practical from end to end. I wanted to train a model, save it, describe its schema, and then create a web interface where anyone could enter values and see predictions instantly. The experience taught me how every part of the pipeline fits together: the data definition, the trained model, and the application that serves predictions. What follows is a detailed technical breakdown of every file, code block, and helper that makes the system work. Dataset used [here](https://www.kaggle.com/datasets/nabeelqureshitiii/student-performance-dataset).

---


## Project Structure

The repository contained these files:

- `app.py`: the Streamlit application.
- `requirements.txt`: Python dependencies.
- `models/schema.json`: schema file for features.
- `models/student_model_linear.joblib`: trained model.

---

## app.py Explained

```python
import os, json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Math Score Predictor", page_icon="📘", layout="centered")
st.title("Student Math Score Predictor")
st.write("Enter the student attributes to estimate their **Math score**.")

# ---------- Locate model & schema ----------
MODEL_PATH = os.path.join("models", "student_model_linear.joblib")
SCHEMA_PATH = os.path.join("models", "schema.json")

def require(path, kind):
    if not os.path.exists(path):
        st.error(f"{kind} file missing: `{path}`. Make sure it's uploaded to your repo.")
        st.stop()

require(MODEL_PATH, "Model")
require(SCHEMA_PATH, "Schema")

@st.cache_resource
def load_model_and_schema():
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)
    model = joblib.load(MODEL_PATH)
    return model, schema

model, schema = load_model_and_schema()

cat_schema = schema.get("categorical", {})
num_schema = schema.get("numeric_ranges", {})
feature_order = schema.get("feature_order", [])
target_name = schema.get("target", "math_score")

# ---------- Build form ----------
with st.form("student_form"):
    inputs = {}
    # Categorical inputs (dropdowns from schema)
    for c, options in cat_schema.items():
        if not options:
            # if schema accidentally empty, fall back to text input
            inputs[c] = st.text_input(c.replace("_", " ").title())
        else:
            # Use the first option as default
            default_idx = 0
            inputs[c] = st.selectbox(c.replace("_", " ").title(), options, index=default_idx)

    # Numeric inputs (use median as default)
    for c, rng in num_schema.items():
        mn, mx = rng.get("min", 0.0), rng.get("max", 100.0)
        med = rng.get("median", (mn + mx) / 2)
        # Guard against equal min/max
        if mn == mx:
            mx = mn + 1.0
        step = max(1.0, (mx - mn) / 100.0)
        inputs[c] = st.number_input(c.replace("_", " ").title(), min_value=float(mn), max_value=float(mx), value=float(med), step=float(step))

    submitted = st.form_submit_button("Predict")

# ---------- Predict ----------
if submitted:
    # Keep the exact training feature order
    row = {k: inputs.get(k) for k in feature_order}
    X = pd.DataFrame([row])

    try:
        pred = float(model.predict(X)[0])
        st.success(f"Estimated {target_name.replace('_',' ').title()}: **{pred:,.2f}**")
        st.caption("Estimate from a scikit-learn Pipeline (One-Hot + Linear Regression).")
    except Exception as e:
        st.error("Prediction failed. Ensure model & schema match. Try redeploying with the latest files.")
        st.exception(e)

with st.expander("How it works"):
    st.markdown(
        """
        - **Model**: scikit-learn Pipeline with One-Hot Encoding for categoricals and a Linear Regression.
        - **Features**: Taken from your dataset (built dynamically from `schema.json`).
        - **Target**: Math score (or the closest available math column).
        - **Why small**: Linear Regression keeps the model artifact under the GitHub 25 MB limit (usually <1 MB).
        """
    )

```

### Explanation

The application begins with imports. It pulls in `os` and `json` for file paths and schema loading. `joblib` is used to load the trained model. `pandas` is imported for dataframe operations, though Streamlit primarily handles the user interface. The call to `st.set_page_config` sets the layout and metadata of the page. The title and short description give the user context before they start filling the form.

The code then defines two important file paths: the trained model path and the schema path. These are checked using the helper `require`. This function ensures the files exist, otherwise it raises a clear error and halts execution with `st.stop`. Without this, the app might continue running with missing resources and fail silently.

The cached function `load_model_and_schema` reads the schema JSON and loads the trained model using `joblib`. Caching is important here because Streamlit re-runs scripts on every interaction. Without caching, the model would reload repeatedly, slowing down the app. Once loaded, the model and schema become available globally.

After loading, the schema is unpacked into categorical features, numeric ranges, feature order, and the target variable. This separation allows the form to be generated dynamically. Categorical inputs are built with `st.selectbox`, ensuring only valid schema-defined options can be chosen. Numeric inputs are created with `st.number_input`, which enforces min, max, and median defaults. Step values are computed to make the slider or input box practical.

The form is built under `with st.form("student_form")`. Each input field is generated automatically from the schema. Once the user fills the fields and presses submit, the collected inputs are assembled in a dictionary. These are then aligned to the schema-defined order before being passed to the model for prediction.

The prediction is displayed back to the user using `st.success`, which highlights the result clearly. This loop — schema definition, form generation, model prediction — is the heart of the app.

---

## requirements.txt Explained

```python
streamlit==1.36.0
scikit-learn==1.5.2
pandas==2.2.2
numpy==2.1.3
joblib==1.4.2
```

Each dependency plays a role:

- `streamlit`: builds the web interface.
- `scikit-learn`: provides model training and prediction tools.
- `pandas`: handles tabular data manipulation.
- `numpy`: supports numerical computation.
- `joblib`: saves and loads trained models efficiently.

---

## schema.json Explained

```python
{
  "task": "regression",
  "target": "total_score",
  "categorical": {
    "grade": [
      "A",
      "B",
      "C",
      "D",
      "F"
    ]
  },
  "numeric_ranges": {
    "student_id": {
      "min": 1.0,
      "max": 1000000.0,
      "median": 500000.5
    },
    "weekly_self_study_hours": {
      "min": 0.0,
      "max": 40.0,
      "median": 15.0
    },
    "attendance_percentage": {
      "min": 50.0,
      "max": 100.0,
      "median": 85.0
    },
    "class_participation": {
      "min": 0.0,
      "max": 10.0,
      "median": 6.0
    }
  },
  "feature_order": [
    "grade",
    "student_id",
    "weekly_self_study_hours",
    "attendance_percentage",
    "class_participation"
  ]
}
```

The schema defines the rules for the application:

- `task`: identifies this as regression.
- `target`: specifies the variable to predict.
- `categorical`: lists features with fixed categories.
- `numeric_ranges`: describes numeric fields with min, max, and median.
- `feature_order`: ensures model input order is consistent.

This schema allows the app to remain flexible. Even if the model changes, the schema can be updated independently, keeping the interface consistent.

---

## The Model File

The file `student_model_linear.joblib` contains the regression model. It was trained beforehand using scikit-learn and saved with joblib. This file is binary and cannot be opened like text, but when loaded, it provides the `.predict` method for scoring new student records. By separating training from prediction, the repository stays lightweight, and the Streamlit app focuses only on serving predictions.

---

## Conclusion

This project demonstrates a complete machine learning pipeline exposed through a simple web interface. Every piece has a role: the model encodes statistical learning, the schema describes expected inputs, the requirements ensure reproducibility, and the Streamlit app provides the user experience. Together they form a self-contained project that can be deployed on GitHub and shared with others. Building this taught me not only how models are served, but also how design decisions at each step shape the reliability and clarity of the final tool.
