---
layout: default
title: "Building my AI Cancer Survival Predictor"
date: 2023-04-12 18:22:51
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-cancer-survival-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_cancer_survival_predictor
featured: true
---

The idea for this project came from a quiet moment of reflection when I was reading about the ways machine learning is used in health care. Articles kept mentioning how predictive models can help guide clinical decision making by providing probabilities, not final answers. I realized that building a simple demonstration could help me practice technical skills while showing how such an idea works in real life. The focus was not to provide medical advice, but to show how technology can assist. Dataset used [here](https://www.kaggle.com/datasets/amankumar094/lung-cancer-dataset).

I wanted to create an application that accepts basic patient details and returns a prediction. The app would demonstrate how a model trained on structured health data could be used in a lightweight, interactive environment. Streamlit was the obvious choice because it allows me to design a web interface without building complex frontend code. From that initial thought, the project grew into a repository containing carefully arranged files, each one serving a role.

---

## Repository Structure
When I uploaded everything to GitHub, my repository had the following files:

```
ai_cancer_survival_predictor/
├── app.py
├── requirements.txt
└── models/
    ├── cancer_survival_model.pkl
    ├── feature_columns.json
    └── raw_input_columns.json
```

Every file in this structure plays a part. The `app.py` file is the Streamlit script that brings everything together. The `requirements.txt` file guarantees that the app installs with the correct dependencies. The `models/` folder is critical because it holds the trained model and the schema references that make predictions possible. If a single piece is missing, the app would either fail to run or would provide incomplete functionality.

---

## Application Code (`app.py`)
The largest file is `app.py`, and it contains the main Streamlit application. Below, I break down each section and explain in detail how it works and why it matters. Understanding this file is key to appreciating how the entire predictor functions.

### Imports and Configuration
```python
import os
import json
import hashlib
import textwrap
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import joblib

MODEL_PATH = "models/cancer_survival_model.pkl"
RAW_COLS_PATH = "models/raw_input_columns.json"
EXPECTED_SHA256 = st.secrets.get("EXPECTED_SHA256", "").strip()

NUM_COLS = ["age", "bmi", "cholesterol_level"]
BIN_COLS = ["hypertension", "asthma", "cirrhosis", "other_cancer"]
```
This code block imports all the dependencies and sets up configuration. Libraries like `numpy` and `pandas` handle numerical processing and data frames. `streamlit` provides the user interface. The configuration variables point to the model file and schema file paths. I also read a secret hash value from Streamlit’s secrets manager to check the integrity of the model. By defining `NUM_COLS` and `BIN_COLS`, I separate numeric features from binary health conditions.

### Helper: SHA256 Hashing
```python
def _sha256(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
```
This helper computes a SHA256 hash for any file. It reads the file in chunks to avoid memory overload on large files. Each chunk is fed into the hash until the file is fully processed. The result is a long string that uniquely identifies the file’s contents. If even a single byte changes, the hash will be completely different. In this project, it allows me to verify that the model file is exactly the one I trained.

### Helper: Safe Model Loader
```python
def safe_load_model(path):
    try:
        if EXPECTED_SHA256:
            observed = _sha256(path)
            if observed != EXPECTED_SHA256:
                st.error("Model hash mismatch. Upload a correct file.")
                return None
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None
```
This function safely loads the model file. First, it checks if a hash was provided in the configuration. If so, it calculates the observed hash and compares it to the expected one. A mismatch produces an error message and prevents the app from running an invalid model. If the hash matches, it proceeds to open the file and load it using `pickle`. Any unexpected error is caught and reported. This design avoids silent failures and reassures users that the model is authentic.

### Helper: Load Raw Columns
```python
def load_raw_columns():
    try:
        with open(RAW_COLS_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return NUM_COLS + BIN_COLS
```
This function reads the list of raw columns from a JSON file. The list defines which features the model expects at prediction time. If the JSON file is missing, I return a combination of numerical and binary columns as a fallback. This ensures that the application still works in demonstration mode. It is an example of graceful degradation: if something is missing, the program continues to function with reasonable defaults.

### Helper: Schema Alignment
```python
def align_schema(df, raw_cols):
    for c in raw_cols:
        if c not in df.columns:
            df[c] = 0
    return df[raw_cols]
```
This helper ensures the user input matches the schema the model expects. It checks if each required column is present in the DataFrame. If a column is missing, it creates it with a default value of zero. Finally, it reorders the DataFrame to follow the exact column order. This alignment is necessary because machine learning models can be sensitive to both the presence and order of input features. By handling this alignment, I avoid shape mismatch errors during prediction.

### Page Layout
```python
st.title("AI Cancer Survival Predictor")
st.write("This tool demonstrates a machine learning model that estimates survival outcomes based on health factors.")
```
This block sets the user interface header. The title introduces the project clearly, while the `st.write` call gives context. These lines are simple but important for usability. They remind the user that this is a demonstration tool meant to showcase machine learning, not to give medical advice.

### Input Form
```python
with st.form("patient_form"):
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
    
    hypertension = st.checkbox("Hypertension")
    asthma = st.checkbox("Asthma")
    cirrhosis = st.checkbox("Cirrhosis")
    other_cancer = st.checkbox("Other Cancer History")

    submitted = st.form_submit_button("Predict Survival")
```
This is the form where users provide input. Age, BMI, and cholesterol are collected as numeric values. Binary health conditions are collected using checkboxes. Wrapping the inputs inside a form groups them logically. Only after filling the form does the user click “Predict Survival,” which ensures cleaner interactions. Each input reflects features the model was trained on, so the app can map user responses directly to model features.

### Preparing Data for Prediction
```python
if submitted:
    data = {
        "age": age,
        "bmi": bmi,
        "cholesterol_level": cholesterol,
        "hypertension": int(hypertension),
        "asthma": int(asthma),
        "cirrhosis": int(cirrhosis),
        "other_cancer": int(other_cancer),
    }
    df = pd.DataFrame([data])
    raw_cols = load_raw_columns()
    df = align_schema(df, raw_cols)
```
This section converts form inputs into a dictionary and then into a pandas DataFrame. Converting checkboxes to integers ensures compatibility with numeric model inputs. Loading the raw columns ensures we are consistent with the schema from training. Aligning the schema then guarantees that every required column exists in the correct order. By the time this block finishes, the input is completely ready for model prediction.

### Running the Model
```python
    model = safe_load_model(MODEL_PATH)
    if model is not None:
        try:
            pred = model.predict(df)[0]
            proba = model.predict_proba(df)[0][1]
            st.success(f"Predicted Survival Outcome: {pred}")
            st.info(f"Survival Probability: {proba:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
```
Here, the model is loaded using the safe loader. If the model is available, predictions are made. Both the predicted class and the probability of survival are displayed. Showing probability provides more depth than a single binary output. Wrapping the prediction in a `try` block ensures that any errors produce a helpful message. This prevents the app from breaking in front of the user. The final output makes the app engaging because users see not just a label but also the underlying probability.

---

## Requirements File (`requirements.txt`)
```text
streamlit
pandas
numpy
scikit-learn
joblib
```
The requirements file is short but critical. It ensures that when deployed on Streamlit Cloud, the right packages are installed. Without it, the deployment environment might not have the correct versions. Listing dependencies explicitly is also good practice when sharing a project with others.

---

## Model Files (`/models/`)
The models folder holds three files that support the application.

1. **cancer_survival_model.pkl**: This is the trained classifier serialized with pickle. It contains all the parameters learned from data. Without it, the app cannot make predictions.  
2. **feature_columns.json**: This file lists transformed features from training. It can be used to verify preprocessing steps. Even though the app uses raw columns, this file is a useful reference for consistency checks.  
3. **raw_input_columns.json**: This defines the raw schema. The app loads it to ensure the user input DataFrame matches training. It provides a contract between training and inference.  

By storing these files in GitHub, I ensure reproducibility and transparency. Anyone can clone the repository and run the application without retraining.

---

## Deployment and Testing
After pushing the repository to GitHub, I deployed it on Streamlit Cloud. The deployment required no special setup beyond connecting the repository. The environment installed dependencies from `requirements.txt`, and then ran `app.py`. Testing was straightforward: I entered values in the form and received predictions in seconds. Deployment made the project real by allowing others to interact with it directly in their browser. This step taught me how important it is to design projects for reproducibility.

---

## Reflections and Lessons
Building this predictor taught me several lessons. First, data preprocessing and schema alignment are just as important as the model itself. A well-trained model is useless if the inputs are not aligned correctly. Second, user experience matters. A confusing form or unclear output would reduce trust in the tool. Third, safety matters. By adding a hash check and exception handling, I protected the app from subtle errors that could undermine confidence.

By expanding each block of the application, I learned how every piece of code connects to the final outcome. Explaining the helpers in detail also clarified my own thinking. What began as a simple idea became a full workflow from training in Colab to deployment on Streamlit Cloud. The finished project not only demonstrates machine learning but also shows engineering practices such as error handling, environment management, and schema validation.

---


## Step by Step Workflow
The workflow of this project follows a simple but strict sequence. First, I trained a model in Google Colab using a synthetic dataset that contained health records. During training, I experimented with different classifiers and eventually settled on a gradient boosting approach. The trained model was then exported as a pickle file and stored in the `models/` folder. I also saved the schema files so that the app could align incoming data with the training schema.

Once the model and schema files were ready, I created the Streamlit application. The application starts by loading the configuration and then defining utility functions. These utilities protect the model loading process and align user input. After utilities, the app sets up a simple layout with a title and description. The input form is the heart of the user interface. Once submitted, the inputs are transformed into a DataFrame, aligned with the schema, and then passed into the model for prediction.

## Why Schema Alignment Matters
Machine learning models trained on structured data are very strict about their input format. If the model was trained with a column called `cholesterol_level`, then predicting with a DataFrame missing that column will fail. Similarly, if the columns are in the wrong order, predictions may be meaningless. Schema alignment ensures consistency between training and inference. By building alignment into the app, I created a robust bridge between the model and the user interface.

## Error Handling and User Safety
I spent time making sure that errors do not crash the application. The `safe_load_model` function and the prediction block both use `try` and `except` clauses. This means that if something goes wrong, the user sees a clear message instead of a blank screen. For example, if the model file is corrupted, the app explains the problem instead of failing silently. This transparency builds trust and allows debugging to be straightforward.

## Possible Improvements
The current version of the application is a strong demonstration, but there are many ways it could be improved. I could add more features to the form, such as lifestyle factors or family history. I could also add visualizations that show how each feature contributes to the prediction. Another improvement would be to allow the user to upload a CSV file with multiple patient records and receive predictions for all of them. These extensions would make the app more powerful and interactive.

## Deployment Notes
Deploying to Streamlit Cloud was simple, but I learned several details that are worth sharing. The app requires that all model and schema files be committed to GitHub. Large models can exceed GitHub’s 25 MB limit, so in this case I kept the model small enough to fit. The `requirements.txt` file must be accurate, or the deployment will fail. During deployment, logs provide useful hints if something goes wrong. I learned to check these logs carefully and fix small issues such as missing dependencies.

## Broader Impact
Even though this project is only a demonstration, it made me think deeply about the role of AI in healthcare. Predictive models can provide support, but they must be used responsibly. Models are limited by the data they are trained on, and they cannot capture the full complexity of human health. This is why every message in the app emphasizes that it is a demonstration tool. The broader lesson is that technology should complement human expertise, not replace it.

## Ethical Reflections
Working on this project also raised ethical questions. What happens if someone mistakes the app for medical advice? To address this, I wrote clear disclaimers in the app description. Another question is about data privacy. Even though the app does not store user inputs, it is important to reassure users of this fact. Thinking about ethics reminded me that building AI systems is not only about technical skills. It also involves responsibility to ensure safe and respectful use.

## Lessons for Future Projects
This project taught me lessons I will apply to future work. First, building utilities for safety and schema alignment should be part of any data-driven app. Second, deployment should be tested early to avoid surprises at the end. Third, writing documentation and blog posts helps clarify design choices for myself and others. Finally, small demonstration projects can still have significant educational impact. They show potential employers or collaborators how I think about both code and responsibility.

## Final Thoughts
When I first imagined this project, I thought it would just be a small coding exercise. By the time I finished, it had become a complete workflow that included model training, schema design, application development, error handling, deployment, and reflection. Each block of code represents more than just instructions for a machine. It represents choices about safety, usability, and clarity. Writing this blog helped me see the connections between those choices.
