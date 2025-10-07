---
layout: default
title: "Building my AI College Placement Predictor"
date: 2023-11-24 17:31:22
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/college.webp
thumbnail_mobile: /assets/images/college_placement_sq.webp
demo_link: https://rahuls-ai-college-placement.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai-college-placement
---

There are moments when uncertainty shapes the mind more than clarity. I once watched several students struggling to understand how their academic record would impact their placement chances. The doubt on their faces stayed with me for long. It made me think about building a tool that could bring numbers into the discussion. Instead of vague assumptions, a student could see probabilities, and maybe even salary ranges. That personal encounter sparked this project. Dataset used [here](https://www.kaggle.com/datasets/vrajesh0sharma7/college-student-placement).

The more I reflected, the more I felt the importance of translating raw academic data into insights. Employers look at grades, skills, and experiences in many ways. By designing a system that could combine all these signals, I realized it was possible to produce something transparent. The project became my way of simplifying an otherwise stressful process. The idea was not only to predict placement probability, but also to recommend a realistic salary band. That is how the journey of this application started.

---

## Files Uploaded to GitHub

For the project to run correctly, three main files were uploaded along with one model file. Each file has its role in the overall design. These files are:

1. **app.py** – The Streamlit application that powers the user interface and ties all logic together.
2. **band_rules.json** – A simple JSON file that holds the rules for salary bands and thresholds.
3. **requirements.txt** – A dependency list so that the right versions of libraries can be installed.
4. **models/placement_pipeline.pkl** – A serialized machine learning model that predicts placement probabilities.

Every file has a unique role. The application cannot run if one of them is missing. Now let me walk step by step through each file and code block.

---

## The Streamlit Application (app.py)

The `app.py` file is the heart of the project. It loads the trained model, applies the band rules, collects user inputs, and finally shows the output in a web interface. I will explain every block of code and why it is important.

### Imports and Page Configuration

```python
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Placement Probability + Salary Band", page_icon="🎓", layout="centered")
```

This section imports all the libraries needed. `json` is used to read the salary band rules. `joblib` loads the trained machine learning model. `numpy` and `pandas` are used for handling data arrays and tabular inputs. `streamlit` is the web framework used to display the application. `Path` from the standard library makes file handling easier. Finally, `set_page_config` defines the title, icon, and layout of the Streamlit page.

### Loading the Pipeline

```python
@st.cache_resource
def load_pipeline():
    path = Path("models/placement_pipeline.pkl")
    if not path.exists():
        st.error("Model file not found at models/placement_pipeline.pkl")
        st.stop()
    return joblib.load(path)
```

This function loads the trained pipeline from disk. The decorator `@st.cache_resource` tells Streamlit to store the loaded object so it does not reload every time the page refreshes. The code checks if the file exists first. If not, it stops the app and shows an error. Once the path is verified, the model is loaded into memory. This step ensures predictions can be made on user inputs.

### Loading Salary Band Rules

```python
@st.cache_data
def load_band_rules():
    path = Path("band_rules.json")
    if not path.exists():
        st.error("band_rules.json not found.")
        st.stop()
    with open(path, "r") as f:
        return json.load(f)
```

This function loads the JSON file that contains band thresholds. It also uses caching so the file is not re-read repeatedly. The conditional ensures that missing files do not cause silent failures. If the file is present, it opens the JSON file and parses it into a Python dictionary. These rules are later used to assign salary bands.

### Pipeline and Rules Initialization

```python
pipe = load_pipeline()
rules = load_band_rules()
```

These lines call the two helper functions to bring both the model and rules into memory. The variables `pipe` and `rules` are used later in the app when computing results. Without this step, no prediction or salary recommendation could be generated.

### Title and Caption

```python
st.title("Placement Probability + Salary Band Recommender")
st.caption("Trained on CollegePlacement.csv (no salary column). Bands are rule-based and configurable.")
```

This code displays the title and a caption on the Streamlit page. It clarifies what the tool does and mentions that the salary bands are not learned from the dataset, but come from external rules.

### Explaining the Band Logic

```python
with st.expander("How bands are computed", expanded=False):
    st.write("""
    1. We compute **placement probability** using a trained classifier.  
    2. We read thresholds from **band_rules.json**.  
    3. We pick the first band whose minimum criteria are satisfied (probability, CGPA, IQ, Projects).
    """)
```

This block uses an expander widget in Streamlit. The content explains how the salary bands are determined. It highlights that probability comes from the model, thresholds come from the JSON file, and selection follows the first rule matched. This transparency helps users trust the system.

---

## Input Form

The application collects user data through a form. Every variable matches a column used in training.

```python
with st.form("input_form", clear_on_submit=False):
    st.subheader("Enter Profile")

    college_id = st.text_input("College ID", value="C001")
    iq = st.number_input("IQ", min_value=50, max_value=200, value=110, step=1)
    prev_sem = st.number_input("Previous Semester Result (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.2, step=0.1)
    academic_perf = st.number_input("Academic Performance (score)", min_value=0, max_value=100, value=75, step=1)
    internship = st.selectbox("Internship Experience", options=["Yes", "No"], index=0)
    extra_curr = st.number_input("Extra Curricular Score", min_value=0, max_value=100, value=60, step=1)
    comms = st.number_input("Communication Skills (score)", min_value=0, max_value=100, value=70, step=1)
    projects = st.number_input("Projects Completed", min_value=0, max_value=50, value=2, step=1)

    submitted = st.form_submit_button("Predict")
```

The form defines inputs for student information. Each field specifies limits and default values. `College_ID` is a string, while most others are numeric. Internship experience is handled with a dropdown. When the "Predict" button is pressed, the application captures all inputs for processing.

---

## Prediction Logic

```python
if submitted:
    input_row = pd.DataFrame([{
        "College_ID": college_id,
        "IQ": int(iq),
        "Prev_Sem_Result": float(prev_sem),
        "CGPA": float(cgpa),
        "Academic_Performance": int(academic_perf),
        "Internship_Experience": internship,
        "Extra_Curricular_Score": int(extra_curr),
        "Communication_Skills": int(comms),
        "Projects_Completed": int(projects)
    }])

    try:
        proba = pipe.predict_proba(input_row)[:, 1][0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()
```

Once the form is submitted, the data is placed inside a pandas DataFrame. This structure matches the model input format. The pipeline then computes prediction probabilities. The second column (index 1) corresponds to the positive class, meaning successful placement. If an error occurs, the system stops gracefully and shows the error message.

---

## Displaying Results

```python
    st.markdown("### Results")
    st.metric("Placement Probability", f"{proba*100:.1f}%")
```

This code displays the prediction as a percentage. The `st.metric` widget gives a clean way to highlight important values, here the probability of placement.

---

## Salary Band Recommendation

```python
    band_name = "Low"
    for band in rules.get("bands", []):
        ok_prob = proba >= band.get("min_prob", 0.0)
        ok_cgpa = cgpa >= band.get("min_cgpa", 0.0)
        ok_iq = iq >= band.get("min_iq", 0)
        ok_proj = projects >= band.get("min_projects", 0)

        if ok_prob and ok_cgpa and ok_iq and ok_proj:
            band_name = band["name"]
            break

    st.success(f"Recommended Salary Band: **{band_name}**")
```

This block checks each band rule in sequence. It verifies probability, CGPA, IQ, and project requirements. As soon as all conditions are met, that band is selected. If none are met, the band remains "Low". The final result is displayed in green as a success message.

---

## Explaining the Recommendation

```python
    with st.expander("Why this band?"):
        st.write({
            "probability": round(float(proba), 4),
            "CGPA": cgpa,
            "IQ": iq,
            "Projects_Completed": projects,
            "applied_rule": band_name
        })
```

This expander explains the reasoning behind the chosen band. It prints the values checked and the rule that was applied. Users can see exactly what influenced the outcome.

---

## Band Rules (band_rules.json)

The JSON file defines thresholds for salary bands.

```python
{
  "bands": [
    { "name": "High", "min_prob": 0.75, "min_cgpa": 7.5, "min_iq": 115, "min_projects": 3 },
    { "name": "Mid",  "min_prob": 0.45, "min_cgpa": 6.5, "min_iq": 105, "min_projects": 1 },
    { "name": "Low",  "min_prob": 0.0 }
  ],
  "notes": "Tune these thresholds freely."
}
```

Each dictionary in the list corresponds to a salary band. The fields define minimum thresholds that must be satisfied. "High" is the most demanding, "Mid" requires moderate performance, and "Low" accepts any case. These rules can be adjusted without retraining the model, giving flexibility.

---

## Requirements (requirements.txt)

The dependencies are written in a simple text file.

```python
streamlit==1.33.0
pandas==2.1.4
numpy==1.26.4
scikit-learn==1.6.1
joblib==1.3.2
```

Each line specifies a library and version. Installing them ensures reproducibility. Streamlit powers the interface, pandas and numpy handle data, scikit-learn provides the pipeline framework, and joblib manages serialization. This list is essential when deploying to services like Streamlit Cloud.

---

## Model File

The file `models/placement_pipeline.pkl` contains the trained machine learning model. It was built using scikit-learn and saved with joblib. This file is not directly human-readable. Instead, it is loaded through the function `load_pipeline`. The model was trained on historical student data. It transforms raw inputs into features, applies a classifier, and outputs a probability. Without this model file, predictions cannot be computed.

---

## Conclusion

The project demonstrates how simple rules and a trained model can work together. The model brings predictive power while the JSON file keeps recommendations flexible. The Streamlit interface makes it accessible to anyone. Each file, from requirements to the pipeline, has its role. Taken together, they create a transparent, interactive tool for college placement insights.

---
