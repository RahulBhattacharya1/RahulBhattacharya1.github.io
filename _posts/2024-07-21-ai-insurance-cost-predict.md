---
layout: default
title: "Building my Insurance Cost Predictor App"
date: 2024-07-21 11:35:27
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/insurance.webp
demo_link: https://rahuls-ai-insurance-model.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_insurance_model
featured: true
---


The idea for this project came to me after seeing how unpredictable medical costs can be. I once compared bills from two similar cases and noticed the charges were very different. That led me to think about how a prediction system could help people plan better. Instead of relying on vague estimates, I wanted to design something that gave direct predictions based on real factors. The more I thought about it, the clearer it became that a small web application could make this possible. I decided to use Streamlit to bring this idea to life.

This project became both a technical experiment and a personal journey. I had never tried to turn a trained model into a full interactive tool before. The goal was to keep things simple, so anyone could try it without needing special setup. All the user had to do was open a browser and fill out a form. Behind the scenes, the model would do the heavy lifting. What follows is a complete breakdown of how I built this project, the files I uploaded to GitHub, and how each block of code works together. Dataset used [here](https://www.kaggle.com/datasets/mirichoi0218/insurance).

---

## Project Structure

When I uploaded the project to GitHub, the folder looked like this:

```
ai_insurance_model-main/
│
├── app.py
├── requirements.txt
└── models/
    └── insurance_model.joblib
```

Each part has a clear purpose:

- **`app.py`** is the main Streamlit script. It runs the web application and connects everything together.  
- **`requirements.txt`** lists the libraries that must be installed. Without this file, the project would not run consistently.  
- **`models/insurance_model.joblib`** is the saved machine learning model. It contains the training knowledge that produces predictions.  

---

## The Requirements File

Before looking at the application code, I defined the dependencies. Here is the content of `requirements.txt`:

```python
streamlit==1.36.0
scikit-learn==1.5.2
pandas==2.2.2
numpy==2.1.3
joblib==1.4.2
```

This file ensures that the correct versions are installed when someone clones the repository. Streamlit is used for the user interface. Scikit-learn is the library where the model was trained. Pandas and NumPy help with data handling. Joblib allows saving and loading of the trained model. Without this file, a new environment might break due to version mismatches.

---

## The Application Code

Now let’s open `app.py`. This file is the heart of the project. I will go through it block by block.

### Import Statements

```python
import os
import joblib
import pandas as pd
import streamlit as st
```

This block pulls in the libraries. I used `os` to handle file paths in a portable way. The `joblib` library loads the trained model from disk. Pandas is used to build a small DataFrame before making predictions. Streamlit powers the user interface and manages everything that appears in the browser.

### Page Configuration

```python
st.set_page_config(page_title="Insurance Charge Predictor", page_icon=":bar_chart:", layout="centered")
```

This sets up the look of the application. I gave it a title, an icon, and defined the layout. The centered layout keeps the form easy to read. This simple line ensures the app looks neat on different screens.

### Title and Description

```python
st.title("Insurance Charge Predictor")
st.write("Enter details and get an estimated insurance **charges** prediction.")
```

This block creates the top banner of the page. The title explains the purpose of the app. The description adds a short instruction so users know what to do. Without this, the page would feel empty.

### Model Loading Helper

```python
MODEL_PATH = os.path.join("models", "insurance_model.joblib")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
```

Here I defined how the model is loaded. First, I created a path to the `joblib` file. Then I built a helper function called `load_model`. This function uses joblib to load the saved model from disk. The decorator `@st.cache_resource` is important. It tells Streamlit to cache the model after loading it once. That means the model does not reload every time the user interacts with the page. This saves time and keeps the app smooth.

### Input Form

```python
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=35, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=28.5, step=0.1, format="%.1f")
        children = st.number_input("Children", min_value=0, max_value=10, value=1, step=1)
    with col2:
        sex = st.selectbox("Sex", options=["male","female"], index=0)
        smoker = st.selectbox("Smoker", options=["yes","no"], index=1)
        region = st.selectbox("Region", options=["northeast","northwest","southeast","southwest"], index=0)
    submitted = st.form_submit_button("Predict")
```

This block builds the form where the user enters their details. I created two columns to keep the design balanced. On the left side, the user inputs numeric values like age, body mass index, and number of children. On the right side, the user chooses categorical values like sex, smoker status, and region. At the bottom, a button triggers the prediction. Streamlit’s `form` keeps all values grouped so they are only submitted when the button is clicked.

### Preparing Input Data

```python
if submitted:
    input_df = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })
```

This block checks if the form was submitted. If true, it creates a new Pandas DataFrame with the input values. Each feature matches the columns the model was trained on. Wrapping them in a DataFrame makes it easy to pass them into the prediction function. Without this step, the model would not accept the raw inputs.

### Prediction

```python
    prediction = model.predict(input_df)[0]
    st.subheader("Estimated Charges")
    st.write(f"${prediction:,.2f}")
```

This part generates the prediction. The model takes the DataFrame and outputs an estimated charge. I only take the first element because the model returns an array. Then I display the result with a subheader and format it as currency. This turns raw numbers into something meaningful for the user.

---

## Summary of Code Helpers

- **`load_model`** is the only custom helper. It encapsulates the loading of the model so I don’t repeat joblib calls. It also benefits from caching.  
- **Form inputs** use built-in Streamlit helpers like `number_input`, `selectbox`, and `form_submit_button`. These handle validation and make sure the data is collected cleanly.  
- **Conditionals** like `if submitted:` control the flow. Without them, predictions would run even without user input.  

---

## Conclusion

This project taught me how to combine machine learning with interactive web tools. Training the model was one part, but delivering it in a usable form was another challenge. By saving the model, writing a Streamlit application, and uploading everything to GitHub, I created something practical. Now anyone can try it in a browser without special setup. This project might be small, but it shows how models can leave the notebook and enter real life applications.
