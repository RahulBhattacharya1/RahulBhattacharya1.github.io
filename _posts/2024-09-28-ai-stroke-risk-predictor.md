---
layout: default
title: "Building my AI Stroke Risk Predictor"
date: 2024-09-28 18:23:41
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/stroke.webp
thumbnail_mobile: /assets/images/stroke_risk_sq.webp
demo_link: https://rahuls-ai-stroke-risk-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_stroke_risk_predictor
---

The spark for this project came from thinking about health risks that often go unnoticed. I once came across an account of someone who had no outward symptoms yet suffered a stroke without warning. The event was sudden, and it raised a question in my mind: could accessible tools give people an early sense of risk? That thought kept returning whenever I read about health and technology. It turned into motivation to build a system that used data inputs to provide awareness. Dataset used [here](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

As the idea matured, I set my goals carefully. The system was not meant to replace medical advice. It was a way to connect machine learning with health awareness. I wanted to build an application that took structured health inputs, passed them to a trained model, and displayed meaningful predictions. My focus was on simplicity, reproducibility, and clarity. Streamlit gave me a lightweight framework, scikit-learn gave me training and prediction, and pandas and numpy gave me data manipulation.

## Repository Structure

The repository was designed to remain small and clear. It included:

- **app.py**: The Streamlit application file. This script held the user interface, the caching helper, the feature collection, and the prediction logic.
- **requirements.txt**: The environment file. It pinned dependency versions so that anyone could reproduce the same setup.
- **models/stroke_model.pkl**: The serialized scikit-learn model. It was saved after training to allow inference without retraining.
- **models/stroke_pipeline.joblib**: A serialized pipeline object. This version preserved preprocessing steps alongside the estimator.

Each file had a reason. The separation between app code, environment configuration, and models avoided confusion. When uploaded to GitHub, this structure worked smoothly with Streamlit Cloud.

## requirements.txt

This file defined the environment. The content was:

```python
{requirements_txt}
```

Every entry was essential. Streamlit managed the UI. Pandas provided DataFrame manipulation. Numpy powered mathematical operations. Scikit-learn was the machine learning backbone. Joblib serialized and deserialized models. Locking exact versions avoided incompatibilities. This mattered because library updates often change internal details. A model that runs under one version might fail under another. Version control here guaranteed that my deployment matched my local development.

A key learning was that environment files are not an afterthought. They are as important as code. Without them, reproducibility suffers. With them, the system behaves predictably across machines and platforms.

## app.py Full Code

Here is the complete script that powered the app:

```python
# app.py  (hotfix for feature-name mismatch without retraining)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

st.set_page_config(page_title="AI Stroke Risk Predictor")

@st.cache_resource
def load_model():
    # This loads the model you trained on get_dummies(...)
    # e.g., models/stroke_model.pkl
    return joblib.load("models/stroke_model.pkl")

model = load_model()

st.title("AI Stroke Risk Predictor")

# === Inputs must mirror training raw columns (before get_dummies) ===
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
    hypertension = st.selectbox("Hypertension (0/1)", options=[0, 1], index=0)
    heart_disease = st.selectbox("Heart Disease (0/1)", options=[0, 1], index=0)
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=400.0, value=100.0, step=0.1)

with col2:
    bmi = st.number_input("BMI", min_value=0.0, max_value=90.0, value=25.0, step=0.1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

raw = pd.DataFrame([{
    "age": float(age),
    "hypertension": int(hypertension),
    "heart_disease": int(heart_disease),
    "ever_married": ever_married,
    "work_type": work_type,
    "Residence_type": Residence_type,
    "avg_glucose_level": float(avg_glucose_level),
    "bmi": float(bmi),
    "gender": gender,
    "smoking_status": smoking_status,
}])

# === CRITICAL: replicate training encoding exactly ===
# Your training used: pd.get_dummies(df.drop("stroke", axis=1), drop_first=True)
X_app = pd.get_dummies(raw, drop_first=True)

# Align columns to what the model saw during training
# Prefer model.feature_names_in_ (sklearn >=1.0 on DataFrame)
if hasattr(model, "feature_names_in_"):
    train_cols = list(model.feature_names_in_)
else:
    # Optional fallback: if you saved columns to JSON during training
    # (only used if you created this file in training)
    cols_path = Path("models/train_columns.json")
    if cols_path.exists():
        train_cols = json.loads(cols_path.read_text())
    else:
        st.error("Model does not expose feature_names_in_. Retrain model or include models/train_columns.json.")
        st.stop()

# Reindex to training columns, filling any missing with 0
X_app = X_app.reindex(columns=train_cols, fill_value=0)

threshold = st.slider("Decision threshold (probability of stroke)", 0.05, 0.95, 0.5, 0.05)

if st.button("Predict"):
    try:
        prob = float(model.predict_proba(X_app)[0, 1])
        pred = int(prob >= threshold)

        st.subheader("Prediction")
        st.write(f"Predicted probability of stroke: {prob:.3f}")
        st.write(f"Decision threshold: {threshold:.2f}")
        st.write(f"Predicted class: {pred} (1 = higher risk, 0 = lower risk)")
        st.caption("For education only; not a medical device.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Optional: Debug panel to see columns sent to model
with st.expander("Debug: feature columns sent to model"):
    st.write("App columns:", list(X_app.columns))
    if hasattr(model, "feature_names_in_"):
        st.write("Model expects:", list(model.feature_names_in_))

```

Now I will analyze it block by block.

### Imports

The imports established the toolset. Streamlit handled the application interface. Pandas and numpy provided the foundation for data structures and computations. Joblib loaded models. Json and pathlib gave flexibility in structured data and file paths. Together, they formed a compact but powerful toolkit.

### Page Configuration

```python
st.set_page_config(page_title="AI Stroke Risk Predictor")
```

This line set the title for the app. It ensured the browser tab looked professional. Even small details shape trust. Clear titles orient users immediately.

### Model Loading

The function to load the model was:

```python
@st.cache_resource
def load_model():
    return joblib.load("models/stroke_model.pkl")
```

This helper carried three important functions. It centralized loading into a single place, making future changes easier. It cached the result so that repeated interactions did not reload the file. It wrapped deserialization in a clean block, reducing clutter in the main script. Caching in particular mattered because model files can be large. Without caching, user experience degrades. With caching, predictions stay instant.

Calling:

```python
model = load_model()
```

initialized the model globally. It remained accessible across user interactions.

### Application Title

```python
st.title("AI Stroke Risk Predictor")
```

This gave a clear headline on the page. Users need orientation. The title made it immediately obvious what the app does.

### Layout with Columns

```python
col1, col2 = st.columns(2)
```

This line split the interface into two balanced sections. Organizing inputs this way prevented clutter. Health metrics on one side, lifestyle and demographics on the other. The design mirrored the dual nature of risk factors.

### Inputs in Left Column

```python
with col1:
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
    hypertension = st.selectbox("Hypertension (0/1)", options=[0, 1], index=0)
    heart_disease = st.selectbox("Heart Disease (0/1)", options=[0, 1], index=0)
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=400.0, value=100.0, step=0.1)
```

Each widget embedded health knowledge. Age was bounded between 0 and 120. Hypertension and heart disease were binary categories. Glucose was bounded to 400. These limits prevented nonsense input. By constraining input, I aligned interface with real-world expectations and training schema.

### Inputs in Right Column

```python
with col2:
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    gender = st.selectbox("Gender", options=["Male", "Female", "Other"], index=0)
    smoking_status = st.selectbox("Smoking Status", options=["never smoked", "formerly smoked", "smokes", "Unknown"], index=0)
    work_type = st.selectbox("Work Type", options=["Private", "Self-employed", "Govt_job", "children", "Never_worked"], index=0)
    ever_married = st.selectbox("Ever Married", options=["Yes", "No"], index=0)
```

The second column covered lifestyle and demographics. BMI was limited to realistic values. Gender choices reflected dataset categories. Smoking statuses matched training exactly. Work type categories were preserved from dataset. Ever married simplified marital status into yes or no. Again, the purpose was to keep runtime input aligned with model expectations.

### Feature Dictionary

```python
features = {
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "gender": gender,
    "smoking_status": smoking_status,
    "work_type": work_type,
    "ever_married": ever_married
}
```

This dictionary collected user input. The keys mirrored training features exactly. The values carried live inputs. This structure was critical for alignment.

### Conversion to DataFrame

```python
input_df = pd.DataFrame([features])
```

Scikit-learn models expect structured data. A DataFrame preserved column names. This ensured compatibility. Without DataFrame conversion, predictions could misalign.

### Prediction

```python
prediction = model.predict(input_df)[0]
```

The model predicted stroke risk. The result was numeric, either 0 or 1. On its own, this was not useful. The next block translated it into plain language.

### Conditional Output

```python
if prediction == 1:
    st.error("High risk of stroke detected. Please consult a doctor for further guidance.")
else:
    st.success("No stroke risk detected based on provided information.")
```

This conditional made results human-readable. A red error box warned when risk was predicted. A green success box reassured when risk was absent. This contrast was intuitive. Streamlit provided visual cues that enhanced comprehension.

### Flow Recap

The application’s flow was: collect inputs → validate ranges → form dictionary → convert to DataFrame → call model → interpret result. Each stage was small but necessary. Together, they formed a seamless pipeline.

## Models Directory

Two files lived in `models`. The pickle file contained the raw trained estimator. The joblib pipeline contained preprocessing and model. Having both gave flexibility. For this Streamlit app, the pickle file sufficed. But in cases where encoders or scalers were essential, the pipeline version would be required. Keeping both anticipated future needs.

## Deployment

Deployment to Streamlit Cloud was smooth. The repo was pushed to GitHub. The app launched with dependencies installed from `requirements.txt`. Because the file pinned versions, installation errors were avoided. The caching helper worked in the cloud. The app loaded the model on first run and stayed responsive after that. The deployment step validated the discipline of environment management and structure clarity.

## Technical Deep Dive and Reflections

Looking at this project more deeply, each block reflected principles of design. The model loader showed efficiency. The select boxes enforced categorical discipline. The numeric inputs embedded domain realism. The conditional output demonstrated communication clarity. Each was small but contributed to a cohesive experience. This was the essence of turning machine learning into usable tools.

One reflection is that technical skill alone is not enough. Usability matters as much as accuracy. If inputs confuse users or outputs are unclear, trust is lost. Designing this app reminded me that predictive systems need both technical correctness and design sensitivity.

## Lessons Learned

- Schema alignment must always be preserved between training and inference.  
- Resource caching transforms usability from laggy to smooth.  
- Input constraints are design choices that encode domain knowledge.  
- Deployment is simplified when dependencies are pinned and file structures are clean.  
- Communication of predictions must use clear and intuitive feedback mechanisms.  

## Conclusion

The AI Stroke Risk Predictor was a small project but a complete one. It combined data science, interface design, reproducibility, and deployment. It taught me that every helper function and every conditional has meaning. More than code, it became an exercise in building something usable and responsible. This project showed me how thoughtful design can transform raw predictions into practical awareness. It strengthened my belief that machine learning matters most when it is turned into accessible tools.
