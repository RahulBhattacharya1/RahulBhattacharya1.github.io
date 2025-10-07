---
layout: default
title: "Creating my AI Sleep Disorder Classifier"
date: 2022-07-28 12:31:29
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/sleep_disorder.webp
thumbnail_mobile: /assets/images/sleep_disorder_sq.webp
demo_link: https://rahuls-ai-sleep-disorder-classifier.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_sleep_disorder_classifier
---

The idea for this project came from moments when I struggled with fatigue despite what seemed like reasonable sleep hours. I often wondered whether the problem was lifestyle, stress, or something more clinical. This uncertainty sparked the thought that a simple machine learning application might give insights by classifying the likelihood of common sleep disorders. It would not replace professional diagnosis, but it could encourage further attention. Dataset used [here](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset).

The second motivation arose while working with structured datasets in a professional setting. Observing how predictive models in healthcare can uncover hidden signals made me realize that a lightweight prototype could be both educational and practical. I wanted an interactive tool that let users input their own daily metrics and instantly see predictions. This mix of personal experience and professional observation laid the foundation for building a Streamlit‑based classifier backed by a trained model.

---

## Repository Structure

The repository was organized clearly so that each file had a defined purpose:

- **`app.py`** – the main Streamlit application handling layout, forms, and predictions.
- **`requirements.txt`** – declared dependencies to ensure consistent setup across machines.
- **`feature_columns.json`** – stored the feature schema expected by the model.
- **`sleep_disorder_pipeline.joblib`** – contained the serialized pipeline trained with scikit‑learn.

Keeping this separation meant that the application code remained simple, while the heavy lifting of the trained model was hidden inside the joblib artifact.

---

## Breaking Down `app.py`

The file `app.py` is the entry point【27†source】. Below is a block‑by‑block exploration.

### Import Statements

```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
from pathlib import Path
```

This import block prepares the libraries. Streamlit drives the UI. Pandas and NumPy support data transformations. Joblib handles model persistence. JSON is used to parse the schema file. Path objects simplify filesystem navigation. Each library was chosen deliberately for reliability and simplicity.

### Streamlit Configuration

```python
st.set_page_config(page_title="Sleep Disorder Classifier", layout="centered")
st.title("Sleep Disorder Classifier")
st.caption("Predict Insomnia / Sleep Apnea / None")
```

These lines control page settings. The `page_title` parameter influences the browser tab label. The centered layout ensures a neat presentation across devices. The title conveys purpose immediately, while the caption clarifies the scope of predictions. Small touches like these improve clarity and user trust.

### File Paths

```python
MODEL_PATH = Path("models/sleep_disorder_pipeline.joblib")
COLS_PATH = Path("models/feature_columns.json")
```

This block standardizes file references. By using `Path`, the code avoids string concatenation issues. The variables point to the exact artifacts required for inference. Encapsulating them in constants makes maintenance easier if folder structures change.

### Guarding Against Missing Files

```python
if not MODEL_PATH.exists():
    st.error("Missing model file: models/sleep_disorder_pipeline.joblib")
    st.stop()
if not COLS_PATH.exists():
    st.error("Missing columns file: models/feature_columns.json")
    st.stop()
```

Here the app checks if required resources exist. Instead of allowing a traceback, it gracefully stops with a clear error message. This defensive pattern improves robustness. It also avoids confusing users who might otherwise assume the app was broken. These checks highlight the importance of validating external dependencies early.

### Loading the Model and Schema

```python
pipe = joblib.load(MODEL_PATH)
FEATURE_COLUMNS = json.loads(COLS_PATH.read_text())
```

This section initializes the heart of the application. The pipeline object contains preprocessing and the classifier. The feature columns list ensures the DataFrame matches training expectations. Without these two, prediction cannot proceed. It demonstrates how model artifacts and schema definitions must always travel together.

### Categorical Choices

```python
gender_choices = ["Male", "Female"]
bmi_choices = ["Underweight", "Normal", "Overweight", "Obese"]
```

These arrays supply options for form widgets. Constraining gender and BMI to predefined categories reduces risk of invalid values. It also aligns user inputs with categories known during training. Handling unseen categories could be possible, but limiting options is safer for demonstration.

---

## Designing the Form

I used a two‑column layout to balance inputs.

```python
with st.form("form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        gender = st.selectbox("Gender", gender_choices)
        occupation = st.text_input("Occupation", value="Software Engineer")
        sleep_duration = st.number_input("Sleep Duration (hours)", 3.0, 12.0, 6.5, 0.1)
        quality_of_sleep = st.slider("Quality of Sleep (1–10)", 1, 10, 7)
        physical_activity = st.slider("Physical Activity Level (0–100)", 0, 100, 50)
    with col2:
        stress_level = st.slider("Stress Level (1–10)", 1, 10, 5)
        bmi_category = st.selectbox("BMI Category", bmi_choices)
        heart_rate = st.number_input("Heart Rate (bpm)", 40, 200, 75)
        daily_steps = st.number_input("Daily Steps", 0, 50000, 6000, 100)
        bp_sys = st.number_input("Systolic BP", 70, 250, 120)
        bp_dia = st.number_input("Diastolic BP", 40, 150, 80)

    submitted = st.form_submit_button("Predict")
```

This structure places half the fields on each side. Column one covers demographics and lifestyle. Column two covers physiological data. Widgets like sliders make it intuitive to select values. Defaults provide immediate usability. The submit button consolidates all values. This design promotes balance and reduces clutter.

---

## Handling Input After Submission

When the form is submitted, the app builds a dictionary.

```python
if submitted:
    row = {
        "Gender": gender,
        "Age": age,
        "Occupation": occupation,
        "Sleep Duration": float(sleep_duration),
        "Quality of Sleep": int(quality_of_sleep),
        "Physical Activity Level": int(physical_activity),
        "Stress Level": int(stress_level),
        "BMI Category": bmi_category,
        "Heart Rate": int(heart_rate),
        "Daily Steps": int(daily_steps),
        "BP_Systolic": float(bp_sys),
        "BP_Diastolic": float(bp_dia),
    }

    input_df = pd.DataFrame([row])
```

This ensures that every key matches the schema. Casting values to `int` or `float` avoids type mismatches. Pandas DataFrame encapsulates the row in the correct structure. Even though there is only one input row, using a DataFrame ensures compatibility with scikit‑learn expectations.

### Reindexing and Alignment

```python
    input_df = input_df.reindex(columns=FEATURE_COLUMNS)
```

This step aligns the user input to the schema. It enforces correct order. If a column is missing, it appears as NaN. If extras exist, they are dropped. Alignment guarantees that the pipeline interprets each column correctly. Without it, predictions could be meaningless.

### Diagnosing Mismatches

```python
    missing = [c for c in FEATURE_COLUMNS if c not in row]
    extra = [c for c in row if c not in FEATURE_COLUMNS]
    if missing or extra:
        st.warning(f"Adjusted columns. Missing: {missing} | Extra: {extra}")
```

This diagnostic message provides transparency. It helps detect when schema drift occurs. If the training set changes in future, this alert will guide updates. Transparency in preprocessing is critical when building explainable systems.

### Predicting and Displaying Results

```python
    try:
        pred = pipe.predict(input_df)[0]
        st.success(f"Predicted Sleep Disorder: {pred}")

        if hasattr(pipe.named_steps["model"], "predict_proba"):
            classes = pipe.named_steps["model"].classes_
            probs = pipe.predict_proba(input_df)[0]
            st.write("Probabilities:")
            st.dataframe(pd.DataFrame({"Class": classes, "Probability": probs}).set_index("Class"))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
```

This code executes the pipeline prediction. Wrapping in a try‑except prevents the app from crashing. The success message provides immediate feedback. If probability estimates exist, they are displayed in a neat DataFrame. Showing probabilities is important because it communicates uncertainty. Users can see relative likelihoods instead of only a hard label.

---

## Supporting Files

### Requirements

```text
streamlit>=1.37
scikit-learn==1.6.1
joblib==1.5.2
pandas==2.2.2
numpy==2.0.2
```

The requirements file【28†source】 specifies exact versions. Pinning avoids subtle bugs. This ensures the training environment and deployment environment stay aligned. It is often overlooked, but it is vital for reproducibility.

### Feature Columns

```json
["Gender", "Age", "Occupation", "Sleep Duration", "Quality of Sleep", "Physical Activity Level", "Stress Level", "BMI Category", "Heart Rate", "Daily Steps", "BP_Systolic", "BP_Diastolic"]
```

The JSON schema【29†source】 prevents errors caused by column misalignment. Saving the schema as a file means the model is portable. Without it, prediction code would need to hardcode column orders. Explicit schema storage is a best practice.

### Model Pipeline

The `.joblib` file preserves the fitted preprocessing and classifier. It likely contains encoders for categorical features, scalers for numerical features, and a supervised model such as RandomForest or GradientBoosting. By serializing it, the training step does not need to be rerun. This reduces deployment time and ensures predictions are consistent.

---

## Deployment Considerations

Deploying this application required more than just running `streamlit run app.py`. I had to ensure the environment installed the exact dependencies. Hosting platforms like Streamlit Cloud or Hugging Face Spaces require small adjustments in folder structure. For example, placing model artifacts in a `models/` directory made it easier to reference. Clear folder naming prevented path errors. Testing deployment in a fresh environment helped catch missing files early.

---

## Testing the Application

I tested the app by entering edge values. For example, I set age to 100 and daily steps to zero. The pipeline still responded correctly, which confirmed input ranges were handled. I also tried non‑default occupations to verify unseen categories did not break predictions. These tests increased confidence that the app could handle a variety of cases gracefully.

---

## Possible Enhancements

Future improvements could include:

- Adding data validation layers to catch unrealistic combinations such as extremely high heart rate with low blood pressure.
- Incorporating visualization of probability distributions for better interpretability.
- Extending the schema to include sleep start and end times for circadian rhythm analysis.
- Logging inputs anonymously for research and improving model generalization.
- Deploying with containerization to ensure consistent environments across cloud platforms.

---

## Reflections

This project reinforced the principle that simplicity in design leads to usability. The separation between model artifacts, schema, and application logic kept responsibilities clear. Explaining each block here has shown how even small pieces like reindexing columns carry importance. It is often the supporting details, not only the model, that decide whether a project works reliably.

---

## Conclusion

I began with curiosity about my own sleep patterns and finished with a fully functioning Streamlit application. Along the way I integrated scikit‑learn pipelines, JSON schemas, and Streamlit forms. Every part, from defensive checks to probability outputs, contributed to robustness. This blog post has captured the technical depth behind what at first looks like a simple app. The project demonstrates how combining health insights with machine learning can yield engaging tools that educate and spark further decision making.

---

