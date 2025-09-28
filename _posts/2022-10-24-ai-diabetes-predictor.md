---
layout: default
title: "Building my AI Diabetes Risk Classifier"
date: 2022-10-24 18:32:43
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-diabetes-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai-diabetes-predictor
---

I often think about how small signals in daily life hint at larger health risks. During a regular health check, I realized how several numbers like glucose, BMI, and blood pressure hold predictive power. This made me imagine a system that could take routine data and provide a clear risk assessment. It inspired me to design a simple but functional machine learning project that predicts diabetes risk. Dataset used [here](https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes).

This project became more than code. It became a way for me to blend healthcare data, machine learning, and web deployment into a single working product. I chose to use Streamlit as the interface, scikit-learn for training, and joblib for persistence. Hosting it on GitHub Pages was not possible since this is a dynamic application, but placing the code in my GitHub repository let me share it and deploy on platforms like Streamlit Cloud. This blog post explains every file, every function, and every block of logic I used to make this application.



## Understanding the README.md

The `README.md` file is the starting point for anyone who lands on the repository. In this file, I wrote a short description of what the project does, how to install the requirements, and how to run the application. It may look simple, but it provides guidance on cloning the repository, installing dependencies, and launching the app with the `streamlit run app.py` command.

Including this file ensures that someone without prior knowledge can still set up the environment and reproduce the results. It also acts as a record of the purpose of the repository, reminding me later why I built it and how it should be used. The Markdown format makes it readable both on GitHub and in any editor.



## The Role of requirements.txt

The `requirements.txt` file holds all the Python libraries that my project depends on. In this file I listed `streamlit`, `pandas`, `numpy`, `scikit-learn`, and `joblib`. These packages power the interface, handle data operations, and provide model persistence. Without this file, someone cloning the repository might guess the libraries, but they would likely run into errors.

This file standardizes the environment. By running `pip install -r requirements.txt`, the correct versions of packages get installed, reducing incompatibility issues. Including it also makes deployment on cloud services seamless, since most platforms read the requirements file and set up dependencies automatically.



## The Heart of the Project: app.py

The `app.py` script is the main application file. It ties together the interface, the model, and the prediction workflow. Every function and every input element here was designed to provide a smooth user experience.



In addition to the explanation above, it is important to note how this block integrates into the wider flow of the application. The imports ensure that external libraries are accessible in memory before any code execution begins. The Streamlit configuration aligns the user experience with a clear title and centered layout, preventing visual clutter. The model loading function is wrapped with caching because loading large models repeatedly can slow down response times. Each input widget is designed not just for data entry but also for user trust, since proper ranges avoid nonsensical inputs. Finally, the prediction block links inputs to outputs in a single clear flow, which is the essence of an interactive machine learning application.



### Importing Libraries

```python
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
```

In this section I import the core libraries. `os` helps me manage file paths. `joblib` is used to load the trained machine learning model. `numpy` and `pandas` are essential for handling numeric arrays and tabular data. `streamlit` provides the framework for creating the interactive web application. Each import is essential. Leaving any of these out would either break the model loading, data processing, or user interface.



In addition to the explanation above, it is important to note how this block integrates into the wider flow of the application. The imports ensure that external libraries are accessible in memory before any code execution begins. The Streamlit configuration aligns the user experience with a clear title and centered layout, preventing visual clutter. The model loading function is wrapped with caching because loading large models repeatedly can slow down response times. Each input widget is designed not just for data entry but also for user trust, since proper ranges avoid nonsensical inputs. Finally, the prediction block links inputs to outputs in a single clear flow, which is the essence of an interactive machine learning application.



### Streamlit Configuration and Title

```python
st.set_page_config(page_title="Diabetes Risk Classifier", layout="centered")

st.title("Diabetes Risk Classifier")
st.write(
    "Enter health metrics to estimate diabetes risk. "
    "This app uses a trained logistic regression model."
)
```

This block sets up the appearance of the app. The page title and layout are specified using `set_page_config`. Then I display a title at the top and a short description below it. These lines may look cosmetic, but they guide the user immediately, telling them the app's purpose and how it works.



In addition to the explanation above, it is important to note how this block integrates into the wider flow of the application. The imports ensure that external libraries are accessible in memory before any code execution begins. The Streamlit configuration aligns the user experience with a clear title and centered layout, preventing visual clutter. The model loading function is wrapped with caching because loading large models repeatedly can slow down response times. Each input widget is designed not just for data entry but also for user trust, since proper ranges avoid nonsensical inputs. Finally, the prediction block links inputs to outputs in a single clear flow, which is the essence of an interactive machine learning application.



### Loading the Model

```python
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = os.path.join("models", "diabetes_model.joblib")
    if not os.path.exists(model_path):
        st.error("Model file not found at models/diabetes_model.joblib")
        st.stop()
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["features"]

model, feature_cols = load_model()
```

The `load_model` function is critical. I used the `@st.cache_resource` decorator so the model is loaded once and reused across app reruns, improving speed. Inside the function I build the model path using `os.path.join`. I check if the file exists. If not, the app stops with an error message, ensuring I do not proceed without a model.

Then I use `joblib.load` to read the serialized bundle. I stored both the model and its expected feature columns in this bundle. Returning both ensures predictions are made with the right input structure. Finally I assign them to `model` and `feature_cols` so they are available throughout the app.



In addition to the explanation above, it is important to note how this block integrates into the wider flow of the application. The imports ensure that external libraries are accessible in memory before any code execution begins. The Streamlit configuration aligns the user experience with a clear title and centered layout, preventing visual clutter. The model loading function is wrapped with caching because loading large models repeatedly can slow down response times. Each input widget is designed not just for data entry but also for user trust, since proper ranges avoid nonsensical inputs. Finally, the prediction block links inputs to outputs in a single clear flow, which is the essence of an interactive machine learning application.



### Building the Input Form

```python
with st.form("risk_form"):
    st.subheader("Enter Health Metrics")

    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        Glucose     = st.number_input("Glucose", min_value=0, max_value=300, value=120, step=1)
        BloodPressure = st.number_input("BloodPressure", min_value=0, max_value=200, value=70, step=1)
        SkinThickness = st.number_input("SkinThickness", min_value=0, max_value=100, value=20, step=1)
    with col2:
        Insulin    = st.number_input("Insulin", min_value=0, max_value=900, value=80, step=1)
        BMI        = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
        Age        = st.number_input("Age", min_value=1, max_value=120, value=33, step=1)

    submitted = st.form_submit_button("Predict Risk")
```

Here I created a form where the user can input their health metrics. I divided the form into two columns for better readability. Each field uses `st.number_input` with sensible min and max values. For example, pregnancies cannot be negative, glucose cannot exceed 300, and age has an upper bound of 120. This validation ensures cleaner inputs.

At the end of the form, I added a submit button labeled "Predict Risk". When the user clicks this button, all input values get processed for prediction.



In addition to the explanation above, it is important to note how this block integrates into the wider flow of the application. The imports ensure that external libraries are accessible in memory before any code execution begins. The Streamlit configuration aligns the user experience with a clear title and centered layout, preventing visual clutter. The model loading function is wrapped with caching because loading large models repeatedly can slow down response times. Each input widget is designed not just for data entry but also for user trust, since proper ranges avoid nonsensical inputs. Finally, the prediction block links inputs to outputs in a single clear flow, which is the essence of an interactive machine learning application.



### Preparing the Input and Making Predictions

```python
if submitted:
    input_data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]], columns=feature_cols)
    proba = model.predict_proba(input_data)[0][1]
    risk = "High Risk" if proba >= 0.5 else "Low Risk"

    st.subheader("Prediction Result")
    st.write(f"Predicted Probability of Diabetes: {proba:.2f}")
    st.success(f"Risk Category: {risk}")
```

This block activates only if the user presses the submit button. I create a `DataFrame` with the entered values, ensuring the column names match `feature_cols`. Then I call `predict_proba` on the model to get probabilities. I use the second element `[0][1]` because logistic regression returns two probabilities: one for no diabetes and one for diabetes.

I apply a threshold of 0.5 to categorize risk as high or low. The result is displayed back to the user with the probability and a label. Using `st.success` makes the result stand out clearly.



In addition to the explanation above, it is important to note how this block integrates into the wider flow of the application. The imports ensure that external libraries are accessible in memory before any code execution begins. The Streamlit configuration aligns the user experience with a clear title and centered layout, preventing visual clutter. The model loading function is wrapped with caching because loading large models repeatedly can slow down response times. Each input widget is designed not just for data entry but also for user trust, since proper ranges avoid nonsensical inputs. Finally, the prediction block links inputs to outputs in a single clear flow, which is the essence of an interactive machine learning application.



## The Data and Model Files

The `data/Healthcare-Diabetes.csv` file contains the dataset used for training the model. It includes columns such as pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, pedigree function, and age. These attributes are widely recognized in medical literature as important predictors of diabetes risk.

The `models/diabetes_model.joblib` file stores the trained logistic regression model and its associated feature columns. By saving the model with joblib, I avoid retraining every time I run the app. This file is loaded directly in `app.py`, ensuring predictions are fast and consistent.



## Conclusion

This project tied together multiple aspects of applied machine learning. I managed data, trained a model, saved it with joblib, and built a simple but effective interface with Streamlit. Every file in the repository plays a role, from the requirements file to the model itself. By explaining each code block, I also reminded myself that clarity in design matters as much as correctness in results.

What started from a personal reflection on health metrics turned into a working classifier I could share. While it is not meant for medical advice, it demonstrates how technical skills can create meaningful prototypes. Deploying such projects makes machine learning tangible and relatable.


### Reflections on the Input Form

The input form represents more than just data entry. It reflects the process of transforming user knowledge into structured variables suitable for machine learning. When a user enters a value like BMI or glucose level, they may not realize that these values map directly to numerical features that a model can interpret. By constraining the ranges in `st.number_input`, I ensure that values remain realistic and meaningful. For example, preventing a negative number of pregnancies or an unrealistic glucose level.

Another important detail is the division of inputs into two columns. Streamlit allows flexible layout, and by choosing two columns, I made the interface less intimidating. Seeing all eight health metrics in one long list would overwhelm the user. Instead, placing them side by side provides balance and makes the task of filling them easier. This is an example of user experience design applied in code, where even small layout choices affect how users engage with the app.

Finally, the submit button anchors the interaction. The label "Predict Risk" is not accidental. I could have written "Submit" or "Run", but those words are generic. By choosing "Predict Risk", I guide the user toward understanding what will happen. The system is not storing their data, not training a new model, but producing a prediction based on existing knowledge. This choice of wording reduces uncertainty and strengthens trust in the system.

