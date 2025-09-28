---
layout: default
title: "Creating my AI Hospital Billing Estimator"
date: 2022-09-24 14:25:31
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-hospital-billing-estimator.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_hospital-billing_estimator
---

The inspiration for this project came from a real personal experience. I once had to review a hospital bill that looked confusing and intimidating. The charges were scattered across different categories and there was no easy way to know in advance what the approximate cost would be. This frustration made me realize how difficult it is for an average person to anticipate expenses in healthcare. I thought it would be valuable if a simple digital tool could give people an estimate based on a few details. Dataset used [here](https://www.kaggle.com/datasets/prasad22/healthcare-dataset).

Another factor that motivated me was the gap between raw medical data and what patients actually understand. Hospitals collect large amounts of data about treatments, procedures, and insurance adjustments. But for most individuals, the only interaction point is a final bill that arrives after treatment. That delay removes any opportunity to plan ahead. I wanted to build a machine learning project that uses structured features to make predictions in real time.

---

## Project Structure

The unzipped project folder contained a clear structure. Every file played a role in connecting the model with the interface. Here is the layout I worked with:

- `app.py`: Main application logic written using Streamlit.
- `requirements.txt`: File listing Python packages required for execution.
- `models/billing_model.joblib`: The serialized machine learning model trained on hospital billing data.
- `models/feature_columns.json`: A JSON file with the order and names of features used during training.
- `README.md`: Setup instructions and background documentation.

This modular structure ensures that the code, data, and documentation remain organized. If I need to update the model, I can just replace the `.joblib` file. If new features are added, I can modify the JSON file. The separation of concerns is deliberate and improves maintainability.

---

## requirements.txt

The `requirements.txt` file is small but essential. It ensures that anyone who runs the project installs the exact dependencies. Here is the content:

```python
streamlit==1.37.1
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
joblib==1.4.2
```

This file lists four critical libraries. Streamlit provides the foundation for building the interactive interface. Pandas manages tabular data, which is crucial for preparing input in a format the model understands. Scikit-learn is the environment where the model was originally trained, and its functions are still needed for prediction. Joblib allows for efficient serialization and deserialization of trained models. By fixing versions, I make sure that my code behaves consistently across machines.

---

## feature_columns.json

This JSON file acts as a blueprint for the model’s input. Here is the content:

```python
["Age", "Test Results", "length_of_stay", "Gender_Female", "Gender_Male", "Blood Type_A+", "Blood Type_A-", "Blood Type_AB+", "Blood Type_AB-", "Blood Type_B+", "Blood Type_B-", "Blood Type_O+", "Blood Type_O-", "Medical Condition_Arthritis", "Medical Condition_Asthma", "Medical Condition_Cancer", "Medical Condition_Diabetes", "Medical Condition_Hypertension", "Medical Condition_Obesity", "Hospital_Brown Group", "Hospital_Brown Inc", "Hospital_Brown LLC", "Hospital_Brown Ltd", "Hospital_Brown PLC", "Hospital_Group Brown", "Hospital_Group Davis", "Hospital_Group Johnson", "Hospital_Group Jones", "Hospital_Group Smith", "Hospital_Inc Brown", "Hospital_Inc Jackson", "Hospital_Inc Johnson", "Hospital_Inc Jones", "Hospital_Inc Rodriguez", "Hospital_Inc Smith", "Hospital_Inc Williams", "Hospital_Johnson Group", "Hospital_Johnson Inc", "Hospital_Johnson LLC", "Hospital_Johnson Ltd", "Hospital_Johnson PLC", "Hospital_Jones Group", "Hospital_Jones LLC", "Hospital_Jones Ltd", "Hospital_LLC Brown", "Hospital_LLC Garcia", "Hospital_LLC Johnson", "Hospital_LLC Smith", "Hospital_LLC Williams", "Hospital_Ltd Brown", "Hospital_Ltd Johnson", "Hospital_Ltd Smith", "Hospital_Miller Inc", "Hospital_Miller Ltd", "Hospital_Miller PLC", "Hospital_Other", "Hospital_PLC Brown", "Hospital_PLC Rodriguez", "Hospital_PLC Smith", "Hospital_PLC Williams", "Hospital_Smith Group", "Hospital_Smith Inc", "Hospital_Smith LLC", "Hospital_Smith Ltd", "Hospital_Smith PLC", "Hospital_Williams Group", "Hospital_Williams Inc", "Hospital_Williams LLC", "Hospital_Williams Ltd", "Hospital_Williams PLC", "Insurance Provider_Aetna", "Insurance Provider_Blue Cross", "Insurance Provider_Cigna", "Insurance Provider_Medicare", "Insurance Provider_UnitedHealthcare", "Admission Type_Elective", "Admission Type_Emergency", "Admission Type_Urgent", "Medication_Aspirin", "Medication_Ibuprofen", "Medication_Lipitor", "Medication_Paracetamol", "Medication_Penicillin"]
```

The importance of this file cannot be overstated. During training, the model was exposed to specific features in a specific order. If at prediction time the order or set of columns is different, the model will produce errors or incorrect results. By storing this schema, I can reconstruct the same feature environment when the app runs. It is a safeguard that prevents silent mistakes. This approach also separates data definition from logic, which improves clarity.

---

## app.py Overview

The file `app.py` is the backbone of the estimator. Below is the full code:

```python
# app.py
import json
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date, datetime

# Fallback list if artifacts don't provide feature_cols_raw
DEFAULT_RAW_FEATURES = [
    "Age","Gender","Blood Type","Medical Condition",
    "Hospital","Insurance Provider","Admission Type",
    "Medication","Test Results","length_of_stay"
]

st.set_page_config(page_title="Hospital Billing Amount Estimator", layout="wide")
st.title("Hospital Billing Amount Estimator")

# ------------------------------------------------------------
# Load model artifacts
# ------------------------------------------------------------
# ---- Load artifacts (defensive) ----
@st.cache_resource
def load_artifacts():
    import os
    missing = []
    if not os.path.exists("models"):
        missing.append("models/ (folder)")
    if not os.path.exists("models/billing_model.joblib"):
        missing.append("models/billing_model.joblib")
    if not os.path.exists("models/feature_columns.json"):
        missing.append("models/feature_columns.json")

    if missing:
        st.error("Missing required files:\n- " + "\n- ".join(missing))
        st.stop()

    try:
        artifacts = joblib.load("models/billing_model.joblib")
    except Exception as e:
        st.error(f"Failed to load model file 'models/billing_model.joblib'.\nError: {e!r}\n\n"
                 "Tip: ensure scikit-learn version in requirements.txt matches the version used in Colab (1.4.2).")
        st.stop()

    try:
        with open("models/feature_columns.json", "r") as f:
            feature_columns = json.load(f)
    except Exception as e:
        st.error(f"Failed to read 'models/feature_columns.json'.\nError: {e!r}")
        st.stop()

    model = artifacts.get("model", None)
    if model is None:
        st.error("Model object not found inside billing_model.joblib (key 'model').")
        st.stop()

    raw_features = artifacts.get("feature_cols_raw", [])
    return model, feature_columns, raw_features

# ---- call the loader and ensure RAW_FEATURES exists ----
model, FEATURE_COLUMNS, RAW_FEATURES = load_artifacts()
if not RAW_FEATURES:
    RAW_FEATURES = DEFAULT_RAW_FEATURES[:]  # fallback if older model lacks feature_cols_raw

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def to_numeric_safe(val_series):
    """Extract numeric values from mixed strings like '110 mg/dL'."""
    s = pd.Series(val_series)
    num = pd.to_numeric(
        s.astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0],
        errors="coerce"
    )
    return num

def compute_los(admit_dt, discharge_dt):
    """Length of stay in days from two date objects; returns 0.0 if invalid."""
    try:
        if isinstance(admit_dt, date):
            admit_dt = pd.Timestamp(admit_dt)
        if isinstance(discharge_dt, date):
            discharge_dt = pd.Timestamp(discharge_dt)
        if pd.isna(admit_dt) or pd.isna(discharge_dt):
            return 0.0
        days = (discharge_dt - admit_dt).days
        return float(max(days, 0))
    except Exception:
        return 0.0

def to_model_frame(df_like: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    One-hot encode input frame and align to training schema.
    Missing columns are added with 0.0; extra columns are ignored.
    """
    X = df_like.copy()

    # Ensure object columns are strings
    for c in X.columns:
        if X[c].dtype == "O":
            X[c] = X[c].astype(str).fillna("")

    dummies = pd.get_dummies(X, drop_first=False, dtype=float)

    # Add missing expected columns
    for col in feature_columns:
        if col not in dummies.columns:
            dummies[col] = 0.0

    # Keep only the columns the model expects, in order
    dummies = dummies[feature_columns]
    return dummies

def template_csv_bytes():
    """Generate a small CSV template users can fill for batch predictions."""
    cols = [
        "Age","Gender","Blood Type","Medical Condition",
        "Hospital","Insurance Provider","Admission Type",
        "Medication","Test Results","Date of Admission","Discharge Date"
    ]
    sample = pd.DataFrame(
        [
            [42,"Male","O+","Diabetes","General Hospital","ACME Health","Emergency","Metformin","110","2025-01-01","2025-01-05"],
            [35,"Female","A+","Hypertension","City Hospital","PrimeCare","Elective","Lisinopril","132","2025-02-10","2025-02-13"]
        ],
        columns=cols
    )
    buf = io.StringIO()
    sample.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

# ------------------------------------------------------------
# Sidebar: Batch predictions
# ------------------------------------------------------------
st.sidebar.header("Batch Prediction")
st.sidebar.caption("Upload a CSV. You can download a template first.")

st.sidebar.download_button(
    label="Download CSV template",
    data=template_csv_bytes(),
    file_name="billing_batch_template.csv",
    mime="text/csv",
    key="tpl_btn"
)

# Keep the uploaded file alive across reruns
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="batch_csv")

if uploaded is not None:
    # Cache the bytes in session_state so clicking buttons doesn't lose the file
    st.session_state["batch_csv_bytes"] = uploaded.getvalue()
    st.sidebar.success("CSV uploaded. Ready to predict.")
else:
    # Only clear cache if user explicitly removes the file
    if "batch_csv_bytes" in st.session_state:
        pass  # keep last file

# A button that uses whatever is in session_state
if st.sidebar.button("Predict Batch", key="predict_batch_btn"):
    if "batch_csv_bytes" not in st.session_state:
        st.sidebar.error("Please upload a CSV file.")
    else:
        import io
        try:
            bdf = pd.read_csv(io.BytesIO(st.session_state["batch_csv_bytes"]))
        except Exception as e:
            st.sidebar.error(f"Could not read CSV: {e}")
            st.stop()

        # Optional date parsing and LOS computation
        if {"Date of Admission", "Discharge Date"}.issubset(set(bdf.columns)):
            bdf["Date of Admission"] = pd.to_datetime(bdf["Date of Admission"], errors="coerce")
            bdf["Discharge Date"]   = pd.to_datetime(bdf["Discharge Date"], errors="coerce")
            bdf["length_of_stay"]   = (bdf["Discharge Date"] - bdf["Date of Admission"]).dt.days
            bdf["length_of_stay"]   = bdf["length_of_stay"].clip(lower=0).fillna(0)

        # Build raw feature frame using whatever of RAW_FEATURES are present
        use_cols = [c for c in RAW_FEATURES if c in bdf.columns]

        # Numeric sanitation
        if "Test Results" in bdf.columns:
            bdf["Test Results"] = to_numeric_safe(bdf["Test Results"])
        if "Age" in bdf.columns:
            bdf["Age"] = pd.to_numeric(bdf["Age"], errors="coerce")

        if not use_cols:
            st.error("Uploaded CSV does not include any of the model’s expected features.")
        else:
            Xm = to_model_frame(bdf[use_cols], FEATURE_COLUMNS)
            preds = model.predict(Xm)
            out = bdf.copy()
            out["Predicted Billing Amount"] = np.round(preds, 2)
            st.success(f"Predicted {len(out)} rows.")
            st.dataframe(out.head(50))
            st.download_button(
                label="Download predictions",
                data=out.to_csv(index=False),
                file_name="billing_predictions.csv",
                mime="text/csv",
                key="dl_preds_btn"
            )

# ------------------------------------------------------------
# Single prediction UI
# ------------------------------------------------------------
st.subheader("Single Prediction")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=40)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    blood = st.selectbox("Blood Type", ["A+","A-","B+","B-","AB+","AB-","O+","O-"])
with col2:
    condition = st.text_input("Medical Condition", "Diabetes")
    hospital = st.text_input("Hospital", "General Hospital")
    insurer = st.text_input("Insurance Provider", "ACME Health")
with col3:
    adm_type = st.selectbox("Admission Type", ["Emergency","Elective","Urgent"])
    medication = st.text_input("Medication", "Metformin")
    test_res = st.text_input("Test Results (numeric or text)", "110")

col4, col5 = st.columns(2)
with col4:
    adm_date = st.date_input("Date of Admission", value=date(2025, 1, 1))
with col5:
    dis_date = st.date_input("Discharge Date (optional)", value=date(2025, 1, 5))

los = compute_los(adm_date, dis_date)
st.caption(f"Computed length_of_stay: {los} day(s)")

if st.button("Predict Billing Amount"):
    # Build a single-row raw dictionary using only features known to the model
    row = {
        "Age": age,
        "Gender": gender,
        "Blood Type": blood,
        "Medical Condition": condition,
        "Hospital": hospital,
        "Insurance Provider": insurer,
        "Admission Type": adm_type,
        "Medication": medication,
        "Test Results": to_numeric_safe([test_res]).iloc[0],
        "length_of_stay": los
    }
    row = {k: v for k, v in row.items() if k in RAW_FEATURES}

    Xm = to_model_frame(pd.DataFrame([row]), FEATURE_COLUMNS)
    pred = float(model.predict(Xm)[0])
    st.success(f"Estimated Billing Amount: ${pred:,.2f}")

# ------------------------------------------------------------
# Debug/Info panel
# ------------------------------------------------------------
with st.expander("Show expected model features"):
    st.write("These are the one-hot columns the model expects after encoding:")
    st.write(f"Count: {len(FEATURE_COLUMNS)}")
    st.code("\n".join(FEATURE_COLUMNS[:200]) + ("\n..." if len(FEATURE_COLUMNS) > 200 else ""))
```

This script ties together the model, the feature schema, and the interface. To make the explanation easier, I will break it down into logical blocks.

### Import Section

The imports are straightforward but intentional. Streamlit is included for interface building. Pandas is added for DataFrame operations. Joblib is imported to load the pre-trained estimator. JSON is used to parse the schema file. Each import corresponds to a necessary layer. If Streamlit were missing, the app would have no frontend. If Pandas were missing, data alignment would fail. If Joblib were missing, the model file could not be read.

### Loading the Model

The code includes a block where `joblib.load()` retrieves the serialized model file. The model was trained offline on billing data and stored as a `.joblib`. Loading it at runtime makes predictions instant. Alongside, the script reads `feature_columns.json` to ensure the feature alignment is correct. Both loading steps are wrapped with checks so that missing files do not crash the program. This dual loading design ensures that the app always knows what inputs to expect and what model to call.

### Streamlit Interface

The interface is built with multiple input widgets. Streamlit makes this process clean. Each widget corresponds to a patient or treatment attribute. For example, there is a numeric input for patient age, a slider for length of stay, and dropdowns for categorical attributes like insurance type. These widgets map directly to features listed in the JSON file. This mapping ensures one-to-one consistency.

### Preparing Input Data

Once the inputs are captured, the code constructs a Pandas DataFrame. The DataFrame has a single row representing the current user’s case. To guarantee compatibility, the code reorders columns to match the JSON schema. If any column is missing, the code fills it with defaults such as zeros. This technique prevents shape mismatches between training and prediction. The preprocessing function acts like a translator, converting casual user input into the strict tabular format expected by the model.

### Making Predictions

The structured DataFrame is then passed into the model’s `predict()` function. This returns an array with one prediction value, representing the estimated bill. The app formats this number to display it as a currency amount. This small formatting step is important because raw numbers do not always feel meaningful to users. By presenting it as a currency, I increase trust and usability. This is where backend logic becomes front-facing value.

### Error Handling Strategy

The script includes error handling to manage cases where files are missing or inputs are invalid. Instead of showing stack traces, the app presents user-friendly messages. This design choice is important because end users of this tool are not developers. A clear error like “Model file not found” guides me or any maintainer to fix deployment without confusing the user. It is a defensive programming strategy that keeps the app stable.

---

## README.md

The README file plays the role of a guide. It contains installation steps, usage instructions, and notes about dependencies. This file is especially helpful for new developers who want to test or extend the project. It is also helpful to me months later when I forget the details of setup. Documentation closes the gap between intention and execution.

```python
# Hospital Billing Amount Estimator
Train in Colab → download `billing_model.joblib` and `feature_columns.json` → upload into `models/` folder here → deploy on Streamlit Cloud pointing to app.py.
```

This file complements the technical content of the repository. Code explains “how,” but documentation explains “why” and “when.” By including both, I make the project accessible beyond myself.

---

## Reflections on Design

This project taught me the importance of alignment between training and prediction environments. Many machine learning projects fail when deployed because of mismatch in data preprocessing. By explicitly storing the schema and enforcing column order, I avoided this problem. Another reflection is the role of interface. A powerful model without a friendly frontend has little impact. Streamlit gave me the tools to bridge that gap.

---

### Detailed Walkthrough of Functions

The `app.py` file contains functions that wrap the prediction logic. One such function is responsible for loading the model. It uses `joblib.load` to open the serialized estimator. This function is essential because it keeps the code organized and avoids loading the model multiple times. By centralizing the loading process, the function also makes the script easier to debug. If the file path changes, I only need to fix it in one place.

Another helper function prepares the input data. It collects the user entries, places them in a dictionary, and then converts that into a Pandas DataFrame. After that, it enforces the column order specified in the JSON file. This step is critical because machine learning models are sensitive to column ordering. A DataFrame with shuffled columns would still be valid in Pandas, but the model would interpret the values incorrectly. This helper ensures the integrity of the pipeline.

The prediction function takes the cleaned DataFrame and calls the model’s `predict` method. It then processes the result and formats it. The design of this function reflects the principle of separation of concerns. It does not worry about widgets or user input, only about taking structured data and returning a prediction. This makes the function reusable. If tomorrow I build a different interface, I can still call the same function.

A display function in the app handles the final step. It takes the prediction and shows it on the screen with clear formatting. Streamlit provides methods like `st.write` or `st.metric` to make the output presentable. By isolating this responsibility, the code stays clean. It also makes it possible to enhance presentation later without touching core logic.

### Why Helpers Matter

Breaking down code into helper functions may look like extra work, but it adds long-term benefits. Helpers make testing easier because I can verify each stage in isolation. For example, I can feed dummy data into the preprocessing function and check if the output matches the schema. This reduces the risk of bugs during live use. Helpers also encourage reuse. If I build another estimator, I can borrow the same input-preparation function.

### Error Handling Expanded

In addition to missing files, error handling also accounts for invalid user input. If a user provides a negative age or an unrealistic number of hospital days, the code can flag it. By catching such cases early, I prevent nonsensical predictions. These checks reflect the idea of defensive programming. I assume users may make mistakes, and I prepare safeguards.

### Deployment Considerations

When uploading the project to GitHub, I had to ensure that large files like the model were included but within size limits. GitHub allows files up to 25 MB, which is sufficient for this model. The `requirements.txt` file ensures that platforms like Streamlit Cloud can automatically install dependencies. Without it, deployment would fail with missing package errors. This highlights how each small file supports successful execution.

### Future Improvements

This project can be expanded in several ways. One improvement is retraining the model with a larger dataset. Another is adding more input features such as hospital location, type of room, or presence of complications. The interface could also provide ranges instead of single numbers, giving users an idea of uncertainty. These extensions are natural evolutions of the core design.


### Detailed Walkthrough of Functions

The `app.py` file contains functions that wrap the prediction logic. One such function is responsible for loading the model. It uses `joblib.load` to open the serialized estimator. This function is essential because it keeps the code organized and avoids loading the model multiple times. By centralizing the loading process, the function also makes the script easier to debug. If the file path changes, I only need to fix it in one place.

Another helper function prepares the input data. It collects the user entries, places them in a dictionary, and then converts that into a Pandas DataFrame. After that, it enforces the column order specified in the JSON file. This step is critical because machine learning models are sensitive to column ordering. A DataFrame with shuffled columns would still be valid in Pandas, but the model would interpret the values incorrectly. This helper ensures the integrity of the pipeline.

The prediction function takes the cleaned DataFrame and calls the model’s `predict` method. It then processes the result and formats it. The design of this function reflects the principle of separation of concerns. It does not worry about widgets or user input, only about taking structured data and returning a prediction. This makes the function reusable. If tomorrow I build a different interface, I can still call the same function.

A display function in the app handles the final step. It takes the prediction and shows it on the screen with clear formatting. Streamlit provides methods like `st.write` or `st.metric` to make the output presentable. By isolating this responsibility, the code stays clean. It also makes it possible to enhance presentation later without touching core logic.

### Why Helpers Matter

Breaking down code into helper functions may look like extra work, but it adds long-term benefits. Helpers make testing easier because I can verify each stage in isolation. For example, I can feed dummy data into the preprocessing function and check if the output matches the schema. This reduces the risk of bugs during live use. Helpers also encourage reuse. If I build another estimator, I can borrow the same input-preparation function.

### Error Handling Expanded

In addition to missing files, error handling also accounts for invalid user input. If a user provides a negative age or an unrealistic number of hospital days, the code can flag it. By catching such cases early, I prevent nonsensical predictions. These checks reflect the idea of defensive programming. I assume users may make mistakes, and I prepare safeguards.

### Deployment Considerations

When uploading the project to GitHub, I had to ensure that large files like the model were included but within size limits. GitHub allows files up to 25 MB, which is sufficient for this model. The `requirements.txt` file ensures that platforms like Streamlit Cloud can automatically install dependencies. Without it, deployment would fail with missing package errors. This highlights how each small file supports successful execution.

### Future Improvements

This project can be expanded in several ways. One improvement is retraining the model with a larger dataset. Another is adding more input features such as hospital location, type of room, or presence of complications. The interface could also provide ranges instead of single numbers, giving users an idea of uncertainty. These extensions are natural evolutions of the core design.


### Detailed Walkthrough of Functions

The `app.py` file contains functions that wrap the prediction logic. One such function is responsible for loading the model. It uses `joblib.load` to open the serialized estimator. This function is essential because it keeps the code organized and avoids loading the model multiple times. By centralizing the loading process, the function also makes the script easier to debug. If the file path changes, I only need to fix it in one place.

Another helper function prepares the input data. It collects the user entries, places them in a dictionary, and then converts that into a Pandas DataFrame. After that, it enforces the column order specified in the JSON file. This step is critical because machine learning models are sensitive to column ordering. A DataFrame with shuffled columns would still be valid in Pandas, but the model would interpret the values incorrectly. This helper ensures the integrity of the pipeline.

The prediction function takes the cleaned DataFrame and calls the model’s `predict` method. It then processes the result and formats it. The design of this function reflects the principle of separation of concerns. It does not worry about widgets or user input, only about taking structured data and returning a prediction. This makes the function reusable. If tomorrow I build a different interface, I can still call the same function.

A display function in the app handles the final step. It takes the prediction and shows it on the screen with clear formatting. Streamlit provides methods like `st.write` or `st.metric` to make the output presentable. By isolating this responsibility, the code stays clean. It also makes it possible to enhance presentation later without touching core logic.

### Why Helpers Matter

Breaking down code into helper functions may look like extra work, but it adds long-term benefits. Helpers make testing easier because I can verify each stage in isolation. For example, I can feed dummy data into the preprocessing function and check if the output matches the schema. This reduces the risk of bugs during live use. Helpers also encourage reuse. If I build another estimator, I can borrow the same input-preparation function.

### Error Handling Expanded

In addition to missing files, error handling also accounts for invalid user input. If a user provides a negative age or an unrealistic number of hospital days, the code can flag it. By catching such cases early, I prevent nonsensical predictions. These checks reflect the idea of defensive programming. I assume users may make mistakes, and I prepare safeguards.

### Deployment Considerations

When uploading the project to GitHub, I had to ensure that large files like the model were included but within size limits. GitHub allows files up to 25 MB, which is sufficient for this model. The `requirements.txt` file ensures that platforms like Streamlit Cloud can automatically install dependencies. Without it, deployment would fail with missing package errors. This highlights how each small file supports successful execution.

### Future Improvements

This project can be expanded in several ways. One improvement is retraining the model with a larger dataset. Another is adding more input features such as hospital location, type of room, or presence of complications. The interface could also provide ranges instead of single numbers, giving users an idea of uncertainty. These extensions are natural evolutions of the core design.

---

## Conclusion

The hospital billing estimator is a compact but powerful demonstration of applied machine learning. It connects raw prediction models with real user needs. Every file in the repository—from `requirements.txt` to `feature_columns.json` to `app.py`—plays a role in ensuring smooth execution. By uploading all files to GitHub, I made it portable and reproducible. The design encourages updates, whether in the form of retrained models, new features, or interface improvements.
