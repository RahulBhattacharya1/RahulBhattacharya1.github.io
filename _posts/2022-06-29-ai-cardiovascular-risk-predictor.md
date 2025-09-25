---
layout: default
title: "Creating my AI Cardiovascular Risk Predictor"
date: 2022-06-29 17:31:44
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-cardiovascular-risk-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_cardiovascular_risk_predictor
featured: true
---

Sometimes ideas are born from the most ordinary of moments. For me, it was a routine health assessment. The process felt mechanical and generalized. I noticed how the medical staff relied on a handful of questions and quick calculations to provide risk guidance. It struck me that these assessments, while useful, could be more tailored. That realization made me curious. Could machine learning capture subtle interactions between risk factors and provide richer insights? Dataset used [here](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset).

I did not set out to replace medical expertise. Instead, I imagined creating a lightweight tool for awareness. Something that could let a user enter values and instantly see a risk estimate. A simple but informative experience. That vision pushed me to design this application. I decided to use **Streamlit** because of its simplicity and focus on rapid prototyping. This blog captures the journey of building the cardiovascular risk predictor, explaining every file, every code block, and the reasoning behind each.

---

## Repository Layout and Components

The extracted repository contained the following files:

```
ai_cardiovascular_risk_predictor-main/
│
├── app.py
├── requirements.txt
└── models/
    ├── cardio_pipeline.pkl
    └── cardio_schema.json
```

Each of these files plays an important role. The `app.py` drives the web interface and prediction pipeline. The `requirements.txt` ensures that dependencies are installed consistently. The `models/` folder carries the serialized model and its schema. None of these files is optional. Without them, the project would fail to run correctly. Understanding their roles is the first step toward grasping how the system fits together.

---

## requirements.txt: Ensuring Dependencies

The project cannot run unless dependencies are installed. The `requirements.txt` file specifies them clearly:

```python
streamlit
pandas
numpy
scikit-learn
joblib
```

This list may look short, but it carries weight. Streamlit builds the user interface. Pandas and NumPy provide data manipulation tools. Scikit-learn delivers the model framework. Joblib enables saving and loading models. By committing this file to the repository, I made it possible for others to replicate my environment with a single installation command. It prevents version mismatches and keeps experiments reproducible.

---

## app.py: The Heart of the Application

The **app.py** script is the centerpiece. Streamlit executes it to generate the web page. I will go through its code block by block and explain what each part contributes.

### Import Statements

```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
```

Imports establish the capabilities of the script. Streamlit manages user interaction. Pandas and NumPy structure data. Joblib provides serialization. OS manages file system checks. JSON allows reading structured metadata. By listing them at the top, the script makes dependencies explicit. Without these libraries, the later code would not function.

### Configuring the Page Layout

```python
st.set_page_config(page_title="Cardio Risk Predictor", page_icon="❤️", layout="centered")
```

This single line shapes the user’s first impression. It controls how the page appears in the browser. The title “Cardio Risk Predictor” reassures the user they are in the right place. The heart icon adds a subtle symbolic touch. The centered layout keeps elements aligned for readability. While it is not part of the model, such configuration enhances the professional feel of the tool.

### Constants for File Paths

```python
MODEL_PATH = "models/cardio_pipeline.pkl"
SCHEMA_PATH = "models/cardio_schema.json"
```

Hardcoding paths multiple times leads to brittle code. I created constants for the model and schema locations. This small choice improves maintainability. If the directory changes, I only need to update it once. Constants also make code easier to read. Anyone scanning the file knows immediately where key assets are expected.

### Displaying Title and Instructions

```python
st.title("Cardiovascular Risk Prediction")
st.write("Provide the details below and click Predict.")
```

Titles and text matter. They orient the user. A tool without instructions can confuse or intimidate. These lines greet the user with clarity. They set expectations about what the app does and what the user needs to provide. The tone is neutral, informative, and professional.

### Checking Model File Existence

```python
if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Please upload 'models/cardio_pipeline.pkl' to the repository."
    )
    st.stop()
```

This block prevents disaster before it starts. If the model file is missing, the script halts immediately. The error message guides the user toward the fix. Without this safeguard, the program would crash unpredictably later when trying to use the absent model. Defensive coding ensures smoother user experience. It anticipates what could go wrong and addresses it early.

### Loading the Model Pipeline

```python
try:
    pipe = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model pipeline: {e}")
    st.stop()
```

Here the model pipeline is loaded. Wrapping the call in a try‑except block is crucial. If the pipeline is corrupted or incompatible, the error is caught. A message appears instead of a cryptic crash. The variable `pipe` becomes the central engine of predictions. Once loaded, it remains in memory for reuse. This design avoids reloading the model repeatedly, saving time and resources.

### Loading Schema Metadata

```python
expected_columns = None
if os.path.exists(SCHEMA_PATH):
    try:
        with open(SCHEMA_PATH, "r") as f:
            schema = json.load(f)
            expected_columns = schema.get("expected_columns", None)
    except Exception:
        expected_columns = None
```

This block handles schema reading. If the schema file exists, it is parsed into JSON. The `expected_columns` are extracted. This list defines the precise order of inputs expected by the pipeline. If parsing fails, the script defaults to `None`. The presence of schema adds a safety net. It guarantees that inputs align correctly with the model. Misaligned columns could lead to silent, dangerous mispredictions. Schema awareness removes that risk.

---

## cardio_schema.json: The Role of Schema

The schema file is simple but powerful. An example structure is shown below:

```python
{
    "expected_columns": [
        "age",
        "gender",
        "systolic_bp",
        "diastolic_bp",
        "cholesterol",
        "glucose",
        "smoking",
        "alcohol",
        "physical_activity"
    ]
}
```

This file enforces discipline. It reminds the developer what features were used in training. It ensures that when new data comes in, it is shaped identically. Schema management is often overlooked in small projects. Here it prevents subtle bugs and makes the pipeline portable. Anyone using the model knows exactly what to provide.

---

## Building the User Input Form

An essential part of the app is collecting user inputs. Streamlit widgets make this possible.

```python
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
systolic = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
diastolic = st.number_input("Diastolic BP", min_value=60, max_value=140, value=80)
cholesterol = st.selectbox("Cholesterol", ["Normal", "Above Normal", "Well Above Normal"])
glucose = st.selectbox("Glucose", ["Normal", "Above Normal", "Well Above Normal"])
smoking = st.radio("Smoking", ["Yes", "No"])
alcohol = st.radio("Alcohol Intake", ["Yes", "No"])
activity = st.radio("Physical Activity", ["Yes", "No"])
```

Each line introduces a widget. They standardize how data is captured. Numeric ranges prevent outlandish inputs. Dropdowns ensure consistent categories. Binary radios map lifestyle habits clearly. This collection of widgets transforms vague user self‑reports into structured, machine‑readable data. The form is intuitive, yet precise.

---

## Constructing the Feature Vector

The next step is preparing data for the model.

```python
data = pd.DataFrame([{
    "age": age,
    "gender": 1 if gender == "Male" else 0,
    "systolic_bp": systolic,
    "diastolic_bp": diastolic,
    "cholesterol": cholesterol,
    "glucose": glucose,
    "smoking": 1 if smoking == "Yes" else 0,
    "alcohol": 1 if alcohol == "Yes" else 0,
    "physical_activity": 1 if activity == "Yes" else 0
}])
```

This block builds a Pandas DataFrame with one row. Each feature is assigned explicitly. Categories are encoded into numbers. The mapping mirrors the training process. Consistency is crucial. If the model learned with numeric encodings, the same encodings must be applied at prediction. This transformation stage ensures fidelity between training and inference.

---

## Aligning Columns with Schema

```python
if expected_columns:
    data = data[expected_columns]
```

This is a small line with big impact. It forces the DataFrame to follow the schema’s order. Without it, columns might shuffle. A shuffled order would feed wrong values to the wrong features. That could break prediction accuracy silently. This single line eliminates that entire class of errors.

---

## Predicting and Displaying Results

```python
if st.button("Predict"):
    try:
        prediction = pipe.predict_proba(data)[0][1]
        st.success(f"Predicted cardiovascular risk: {prediction:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
```

This is where the application delivers its core function. When the user clicks the button, the pipeline’s `predict_proba` method is invoked. It returns probability estimates for each class. Indexing selects the probability of risk. The result is presented cleanly. Users receive feedback in seconds. Errors, if they occur, are shown clearly. This balance of transparency and usability defines the experience.

---

## cardio_pipeline.pkl: The Trained Pipeline

The `.pkl` file is a serialized object. Inside it, preprocessing and classification steps are bundled. It may include scaling, encoding, and a classifier like logistic regression. By persisting the pipeline as one object, reproducibility is guaranteed. The app does not need to retrain or rebuild. Loading it restores the trained state instantly. Although unreadable to humans, this file embodies the intelligence of the system.

---

## Reflections on Defensive Design

Throughout the script, careful checks are built in. Model file presence is verified. Loading errors are caught. Schema is validated. Prediction errors are surfaced. Each of these guards shields the user from confusion. This approach reflects a philosophy: anticipate problems before they reach the user. In predictive healthcare tools, reliability is not optional. It is essential for trust.

---

## Potential Applications

The cardiovascular risk predictor is not limited to personal use. Potential scenarios include:

1. **Clinical Awareness Tools**: Clinics could offer patients a simple app for self‑assessment before appointments.  
2. **Educational Platforms**: Medical students could explore how risk factors interact through hands‑on inputs.  
3. **Research Prototypes**: Scientists could adapt the pipeline to test hypotheses about different populations.  
4. **Public Campaigns**: Health organizations could deploy a version for awareness campaigns.  
5. **Integration in Portals**: Insurers or wellness platforms could embed the predictor into dashboards.

Each scenario demonstrates adaptability. The design is simple enough for wide adoption, yet rigorous enough for credible results.

---

## Lessons Learned

Building this project taught me several lessons. Schema alignment is as important as model accuracy. Defensive coding saves hours of debugging. User experience can determine whether a tool is adopted. Most importantly, simplicity has power. A clear, minimal design can still deliver meaningful insights. These insights will guide my future work.

---

## Final Thoughts

This project began with curiosity and ended as a functional web application. It demonstrated how personal observations can lead to technical creations. While it does not replace medical consultation, it empowers users with awareness. That blend of human concern and machine intelligence is the essence of applied data science. The cardiovascular risk predictor is just one example, but it symbolizes the broader future where machine learning augments everyday decisions.

---


## Detailed Breakdown of Helpers and Conditionals

Every function and conditional in this script serves a purpose. Streamlit scripts do not usually define many standalone functions, but conditionals act as critical flow controllers. Let me elaborate on them in detail.

### Conditional: Checking Model File
This conditional prevents runtime issues. By stopping execution early, it saves the user from encountering confusing stack traces. It demonstrates defensive programming where the developer assumes that something might go wrong and prepares for it.

### Conditional: Schema Loading
This block shows resilience. If the schema exists, the program uses it. If it fails to load, the program still continues. This flexibility ensures that the app never fails solely because of schema absence. It gracefully degrades functionality rather than breaking.

### Conditional: Prediction Button
This conditional is central to interactivity. Predictions are only computed when the user explicitly asks for them. It prevents unnecessary computation and aligns with how people expect a button to behave. Wrapping the prediction in a try‑except also enhances robustness.

---

## Why I Chose Streamlit

There are many frameworks for building web applications. I chose Streamlit for several reasons. First, it is lightweight. Second, it integrates seamlessly with Python, removing the need to manage backend and frontend separately. Third, it emphasizes rapid prototyping. I could test features instantly and iterate quickly. These strengths outweighed limitations such as less control over low‑level HTML. For a data‑driven tool like this, Streamlit offered the best balance between speed and usability.

---

## Broader Healthcare Perspective

Cardiovascular disease remains one of the leading causes of mortality worldwide. Awareness and early detection play a vital role in reducing risk. Tools like this project align with a larger healthcare trend: democratizing access to predictive insights. While they cannot replace formal diagnosis, they can encourage proactive behavior. For example, a user seeing a high risk score might schedule a checkup earlier than planned. That simple action could make a difference.

---

## Possible Enhancements

Several improvements could extend this project:

1. **Expanded Features**: Adding more lifestyle or genetic data could improve prediction accuracy.  
2. **Explainability**: Integrating SHAP values or feature importance charts would help users understand why the model predicted a certain risk.  
3. **Mobile Optimization**: Adapting the layout for phones would expand accessibility.  
4. **Data Logging**: With user consent, storing anonymized inputs could help refine the model.  
5. **API Integration**: Exposing the model as an API could allow other applications to connect.

Each enhancement requires careful design, especially when dealing with sensitive health data. Privacy and transparency must remain central.

---

## Reflections on Error Handling

In predictive healthcare applications, silent failures are unacceptable. If a model prediction fails, the user must be informed. That is why I wrapped loading and prediction in try‑except blocks. The alternative would be a blank screen or broken output, which could mislead users. Clear error messages, though sometimes frustrating, protect trust. They communicate honesty and allow troubleshooting.

---

## Ethical Considerations

Machine learning in healthcare raises ethical questions. This project is a prototype, not a diagnostic tool. It should not be used for medical decision‑making without professional validation. However, even as a prototype, it teaches important lessons. Transparency, proper disclaimers, and accuracy monitoring are essential. Developers must remind users about the limits of such tools. Ethics in design is not an afterthought but an integral requirement.

---

## Final Case Study Example

Imagine two individuals use the app. One enters values that indicate high blood pressure and lack of physical activity. The model predicts elevated risk. This user might decide to adopt healthier habits or consult a doctor. Another user enters values showing normal ranges. The model predicts low risk. This reassurance can encourage continuation of healthy behavior. These examples demonstrate how the tool provides actionable awareness without making clinical claims.

---

## Looking Forward

This project is not an endpoint. It is a foundation. Future work could involve integrating real clinical datasets, expanding feature sets, and deploying at scale. The skills developed here—schema alignment, defensive coding, intuitive design—apply to many domains. Whether in healthcare, finance, or education, the principle remains: machine learning must be accessible and trustworthy.

---


## Extended Technical Walkthrough

### Data Encoding Choices
Binary encoding for gender, smoking, alcohol, and activity was deliberate. It keeps the model simple and avoids one‑hot expansion that might not be necessary for binary categories. Encoding aligns training with inference. Any mismatch here would undermine validity.

### Probabilistic Output
Using `predict_proba` instead of `predict` was another choice worth noting. A binary prediction (0 or 1) can feel too absolute. Probabilities communicate nuance. A score of 0.72 means higher risk but not certainty. This probabilistic framing is more aligned with how risk is perceived in real life.

### Defensive Defaults
By setting default values for number inputs and select boxes, I reduced the chance of empty submissions. Defaults also provide an educational baseline. A new user can immediately click predict without entering anything and still see how the app behaves.

---

## Broader Reflections

What excites me most about this project is not the specific prediction but the broader message. Data science can transform personal awareness. With only a few files and a trained model, I created a working prototype. That accessibility is powerful. It lowers the barrier for experimentation and invites more people to build tools for good. Even a modest project can spark meaningful conversations about health, design, and responsibility.

---

## Concluding Note

This cardiovascular risk predictor shows how a personal observation can evolve into a practical application. It demonstrates technical rigor, thoughtful design, and awareness of ethical boundaries. The project is both a technical achievement and a reminder: technology should serve people by making knowledge more accessible. I look forward to building on these lessons in future projects.

---
