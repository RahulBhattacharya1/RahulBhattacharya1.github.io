---
layout: default
title: "Creating my AI Diabetes Classifier"
date: 2025-09-15 15:29:32
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/diabetes_classifier.webp
thumbnail_mobile: /assets/images/diabetes_classifier_sq.webp
demo_link: https://rahuls-ai-diabetes-classifier.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_diabetes_classifier
featured: true
synopsis: Here I build a diabetes risk classifier that transforms raw health indicators into clear, educational insights. By simplifying interpretation of values like glucose, blood pressure, and body mass index, it offers an accessible tool aimed at raising awareness and supporting personal understanding.
custom_snippet: true
custom_snippet_text: Transforms indicators into insights for diabetes risk awareness. 
---

There was a moment when I realized how fragile health can be. I had come across a routine medical camp where many individuals were quietly waiting for their turn. What struck me most was the silent anxiety in the air with people holding complex medical reports. I kept thinking how useful it would be if simple tools existed to guide them in real time. That thought stayed with me, and it gradually shaped into an idea. I wanted to create something that could take raw numbers and give back clarity. The motivation did not come from a sudden spark but from a slow build-up of small observations. I knew that technology could step in where words often fail. Dataset used is [here](https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes).

When I explored the problem further, diabetes stood out as a silent condition that many overlook. Numbers on glucose, blood pressure, and body mass index are often difficult for individuals to interpret. I wanted to design a lightweight web application that could simplify this process. My approach was not to replace doctors or medical advice but to provide educational insights. The goal was clarity and accessibility, especially for those who may not always have immediate access to clinical expertise. That is how I started shaping this project into a working classifier with a clear and simple interface. 

---
## Requirements File Explanation

I uploaded a `requirements.txt` file to GitHub to make sure the environment installs the correct dependencies. Below is the content:

```python
streamlit>=1.36.0
pandas>=2.0,<2.3
numpy==1.26.4
scikit-learn==1.4.2
skops==0.9.0

```

This file is important because Streamlit Cloud or any platform hosting the application needs to know which versions of libraries are required. By locking versions, I make sure the behavior of my model is consistent across runs. Streamlit provides the interface, Pandas manages tabular data, and NumPy handles numerical arrays. Scikit-learn is the core machine learning library that powered my logistic regression model. Skops is included because it allows me to safely serialize and load the model artifact. Every dependency here plays a role in stabilizing the environment. If one is missing or mismatched, the app may crash or misbehave.

## Application Code: `app.py`

The main script is `app.py`. This is the file Streamlit runs to create the web interface. I explain it block by block.

### Imports and Configuration

```python
import os
import numpy as np
import pandas as pd
import streamlit as st

# skops-safe load
import skops.io as sio

# -------------------------
# App Config
# -------------------------
st.set_page_config(page_title="AI Diabetes Classifier", layout="centered")

st.title("AI Diabetes Classifier")
st.write(
    "Enter clinical values to estimate diabetes risk. "
    "This app uses a logistic regression model trained on your dataset."
)

MODEL_PATH = "models/diabetes_model.skops"

```

I designed this section to establish both structure and clarity. The imports supply essential tools for handling data, user interaction, and model loading. The configuration defines how the app looks and feels, making navigation intuitive from the start. The title and description set user expectations by explaining purpose in simple terms. A dedicated path points to the stored model, ensuring reproducibility and consistent predictions. Each decision balances technical necessity with usability: clarity in layout, modularity in design, and transparency in function. Together these elements lay a reliable foundation, allowing the rest of the application to focus on transforming inputs into informed, actionable results.

### Model Loading Function

```python
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model file not found at 'models/diabetes_model.skops'. "
            "Please upload it to the GitHub repo under models/."
        )
        st.stop()
    model = sio.load(MODEL_PATH, trusted=True)
    return model

model = load_model()

```

This function is the safeguard for reliability. Before running predictions, it verifies that the model file exists, immediately halting execution if the resource is missing. The explicit error message guides the user toward the correct fix, reducing confusion. Loading is done with a trusted method to prevent corruption or unsafe behavior, ensuring integrity of the prediction pipeline. Encapsulating this in a reusable function improves maintainability and keeps logic consistent. By assigning the return to a variable, the application guarantees that every downstream operation uses a validated and secure model object. This makes the workflow both robust and transparent.

### Feature Schema and Input Form

```python
feature_names = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

with st.form("input-form"):
    col1, col2 = st.columns(2)

    Pregnancies = col1.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    Glucose = col1.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120, step=1)
    BloodPressure = col1.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70, step=1)
    SkinThickness = col1.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1)

    Insulin = col2.number_input("Insulin (pmol/L)", min_value=0, max_value=1000, value=80, step=1)
    BMI = col2.number_input("BMI", min_value=0.0, max_value=80.0, value=28.0, step=0.1, format="%.1f")
    DiabetesPedigreeFunction = col2.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01, format="%.2f")
    Age = col2.number_input("Age (years)", min_value=1, max_value=120, value=35, step=1)

    submitted = st.form_submit_button("Predict Risk")

```

This block defines the structure through which users interact with the model. The feature list ensures consistency by matching inputs with the schema used during training. A two-column layout organizes the form, balancing readability with efficient use of space. Each numeric field enforces realistic bounds, guiding users to enter values within medically plausible ranges. Defaults provide a starting point, reducing friction for first-time use. By grouping related measures like glucose, insulin, and BMI, the interface mirrors real-world health indicators, making it intuitive for non-technical users. The submit button ties everything together, sending validated inputs forward for risk prediction.

### Data Preparation Helper

```python
def as_dataframe():
    data = {
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": Age,
    }
    return pd.DataFrame([data], columns=feature_names)

```

This helper ensures inputs are packaged in a format the model can understand. By mapping each user-entered value to its corresponding feature name, it maintains alignment with the schema used during training. Wrapping the dictionary inside a DataFrame provides compatibility with model expectations, avoiding mismatched structures or missing fields. The design is minimal yet critical: it translates raw interface inputs into structured tabular data ready for prediction. This step bridges the gap between user interaction and machine learning inference, guaranteeing smooth and accurate flow through the pipeline.

### Prediction and Display

```python
if submitted:
    X_input = as_dataframe()

    proba = float(model.predict_proba(X_input)[0, 1])
    pred = int(proba >= 0.5)

    st.subheader("Result")
    st.write(f"Predicted Class: **{pred}** (0 = No Diabetes, 1 = Diabetes)")
    st.write(f"Risk Score: **{proba*100:.1f}%**")

    tips = []
    if Glucose >= 140:
        tips.append("Glucose is elevated; consider medical evaluation and blood sugar control strategies.")
    if BMI >= 30:
        tips.append("BMI is in the obese range; modest weight reduction can lower risk.")
    if BloodPressure >= 90:
        tips.append("Diastolic blood pressure is high; discuss BP management with a clinician.")
    if Insulin == 0 or Insulin >= 200:
        tips.append("Insulin measurement is atypical; ensure proper labs and follow-up.")
    if Age >= 45:
        tips.append("Age is a non-modifiable risk factor; maintain regular screening.")
    if not tips:
        tips.append("Maintain a balanced diet, regular exercise, and periodic screening.")

    st.subheader("Recommendations")
    for t in tips:
        st.write(f"- {t}")

    st.caption(
        "Disclaimer: This tool is for educational purposes and not a medical diagnosis. "
        "Consult a qualified professional for healthcare decisions."
    )

```

This section transforms model output into actionable insights. Once inputs are submitted, they are prepared for inference and passed to the classifier. The probability score is converted into a simple binary label, making results easier to interpret. Both the predicted class and risk percentage are displayed clearly, balancing precision with accessibility. To go beyond numbers, the app generates tailored recommendations by checking individual factors such as glucose, BMI, and blood pressure. These context-specific tips provide guidance without overwhelming the user. A disclaimer reinforces that this is an educational tool, encouraging safe and responsible interpretation of results.

## Model Artifact

The project also includes a serialized model stored as `models/diabetes_model.skops`. This binary file cannot be read directly here, but it contains the logistic regression pipeline trained earlier. It embeds preprocessing steps like scaling and imputation along with the classifier. I uploaded it to GitHub so that the app can load it instantly at runtime without retraining. By separating code from model, I ensured faster startup and consistent predictions. If the model is missing, the app displays an error and stops execution. This prevents misleading behavior.

## Conclusion

This project brought together several skills: data preprocessing, model training, safe serialization, user interface design, and deployment. Each file played a critical role. `requirements.txt` managed the environment, `app.py` powered the interface and logic, and the model artifact ensured reproducibility. I focused on clarity, accessibility, and responsible presentation. Building this classifier was not just a coding exercise but a way to imagine how small tools can give timely insights. The process reaffirmed my belief that even compact projects can make a difference if designed with care.

### Reflections on Design

I spent significant time deciding how to balance transparency with simplicity. A classifier can easily overwhelm a user with technical terminologies. My decision was to hide complexity while exposing meaningful numbers like probability and binary classification. This project also reflects my attention to deployment details. By packaging the model in `skops`, I reduced security concerns while keeping compatibility. Streamlit made the interface development fast but required careful structuring of forms and results. Every part of the application is aligned with one principle: clarity for the end-user.
