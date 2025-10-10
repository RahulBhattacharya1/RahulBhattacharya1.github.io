---
layout: default
title: "Building my AI Asthma Risk Classifier"
date: 2022-12-16 15:26:37
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/asthma_risk.webp
thumbnail_mobile: /assets/images/asthma_risk_sq.webp
demo_link: https://rahuls-ai-asthma-risk-classifier.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_asthma_risk_classifier
featured: true
synopsis: In this project I introduce an asthma risk classifier that transforms health data into a prediction. It emphasizes awareness and early detection by identifying hidden patterns behind common symptoms, providing a supportive tool for education and personal health decision-making.
custom_snippet: true
custom_snippet_text: Classifies asthma risk, promoting awareness through data-driven health insights. 
---

It started with a memory of a difficult evening when breathing became heavier after a simple walk outside. That moment stayed in my thoughts for weeks. I realized how fragile health can feel when air itself turns into an obstacle. This led me to imagine a small digital companion, something that could tell me in advance if I might be at risk. I knew I wanted to build more than a model. I wanted to create a tool that felt practical and personal at the same time. That became the seed for my asthma risk classifier project. Dataset used [here](https://www.kaggle.com/datasets/jatinthakur706/copd-asthma-patient-dataset).

Later, while reviewing simple health trackers and prediction tools, I found myself asking whether a small AI app could make a difference in awareness. Many conditions rely on early detection and pattern recognition. Asthma is a condition where warning signs can hide behind common symptoms like coughs or fatigue. My goal became clear. I wanted to design a classifier that uses health data and outputs a simple yes or no for asthma risk. The project was not about complex medical accuracy. It was about awareness and education through technology.

---

## Project Files Overview

I created this project with three important components. Each one had its role in making the final Streamlit app work.

1. **app.py** – This is the Streamlit script where the entire user interface and logic live. It collects input, processes it, and shows prediction results.
2. **requirements.txt** – This file holds all the Python libraries needed to run the project in Streamlit Cloud or any other platform. Without it, deployment would fail because the environment would miss key packages.
3. **models/asthma_model.pkl** – This is the saved machine learning model trained earlier. It carries the logic learned from data and is loaded by the Streamlit app.

In the following sections, I will explain each file, every function, and every helper used. I will also give detailed context for why each step is necessary and how it ties into the overall picture.

---

## requirements.txt

This file ensures that the environment has the exact versions of libraries that the code depends on. Without it, different environments might produce errors or inconsistent results. My requirements file looked like this:

```python
streamlit>=1.36.0
pandas>=2.0,<2.3
numpy==1.26.4
scikit-learn==1.4.2
skops==0.9.0
```

I included `streamlit` because the application runs as a web app using this library. I specified `pandas` since I used DataFrame structures to store input. I pinned `numpy` to a fixed version to avoid conflicts with scikit-learn. The `scikit-learn` library was required to train and use the machine learning model. Finally, I included `skops` because it helps in saving and loading models in a format that integrates well with scikit-learn. Each package was chosen carefully, not only to make the app run but also to guarantee stability when deployed online.

---

## app.py

The `app.py` file is the heart of the project. It handles everything from loading the model to creating the user interface for predictions. Let me go through the script in detail.

### Imports and Model Loading

```python
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/asthma_model.pkl")
```

In the first part of the script, I imported the essential libraries. `streamlit` powers the interface, `pandas` manages data in tabular form, and `joblib` loads the pre-trained model. I stored the model file in a `models` directory. The `joblib.load` function reads the file and makes the trained model available inside the script. This block of code is simple but foundational. Without these imports and the model loading, the app cannot work.

### Title and Description

```python
st.title("AI Asthma Risk Classifier")
st.write("Predict asthma risk based on patient information.")
```

Here, I created a title for the app and wrote a short description. The title makes the app look polished when someone opens it. The description gives a simple statement of purpose. It tells the user what they can expect from the tool. This section is small but creates the first impression.

### User Input Section

```python
# User inputs
age = st.number_input("Age", 1, 100, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
smoking = st.selectbox("Smoking Status", ["Non-Smoker", "Ex-Smoker", "Current Smoker"])
peak = st.number_input("Peak Flow (L/min)", 50, 800, 200)
```

This block builds the input form. I used `st.number_input` for numeric fields like age and peak flow. It lets the user type values or use arrows to increase or decrease numbers. I gave ranges so the inputs remain valid. For gender and smoking status, I used `st.selectbox`. It limits the options and avoids typing mistakes. Together, these widgets make sure the data collected is clean and reliable. The default values are set so that the app starts with reasonable inputs without forcing the user to type everything.

### Converting Inputs into DataFrame

```python
# Convert inputs
df = pd.DataFrame([[age, gender, smoking, peak]], 
                  columns=["Age", "Gender", "Smoking_Status", "Peak_Flow"])
df = pd.get_dummies(df)
```

After the inputs are collected, they must be prepared for the model. The machine learning model does not work directly with text like "Male" or "Current Smoker". It needs numeric representation. I used `pd.get_dummies` to convert categorical variables into binary columns. For example, if the user selects "Male", the column `Gender_Male` becomes 1 and `Gender_Female` becomes 0. This ensures that the input matches the structure used during training. This transformation is critical because any mismatch will break the prediction step.

### Matching Training Columns

```python
# Match training columns
train_cols = model.feature_names_in_
for col in train_cols:
    if col not in df.columns:
        df[col] = 0
df = df[train_cols]
```

This block aligns the input DataFrame with the structure expected by the model. During training, the model saw specific feature columns. If the current input does not have all those columns, prediction would fail. To handle this, I looped through each column name in `train_cols`. If a column is missing in the input, I added it with a default value of zero. Finally, I reordered the DataFrame to match the training order. This careful step ensures stability and prevents errors caused by mismatched column order.

### Making Prediction

```python
# Prediction
pred = model.predict(df)[0]
st.subheader("Prediction Result:")
st.write("Asthma Risk: **Yes**" if pred==1 else "Asthma Risk: **No**")
```

This section calls the model’s `predict` method and uses the prepared input DataFrame. The prediction result is either 1 or 0. A result of 1 means asthma risk is detected, and 0 means no risk. I displayed the result clearly using `st.subheader` and `st.write`. The conditional expression adds clarity by showing "Yes" or "No" instead of just numbers. This block completes the user experience. The user gets immediate feedback in a way that is easy to read.

---

## models/asthma_model.pkl

This file stores the trained model in serialized form. It was built separately using scikit-learn. The details of training are not part of this Streamlit app, but the result is critical. The pickle file acts like memory. It carries all the coefficients, learned parameters, and encoded mappings from training. By loading it inside `app.py`, I avoided retraining every time the app runs. This made the application fast and practical.

---

## Conclusion

This project was small but meaningful. I connected a simple health motivation with the power of machine learning. The three files—`app.py`, `requirements.txt`, and the model file—worked together to create a working classifier. I did not aim for clinical precision. Instead, I aimed for awareness and demonstration of how AI can inform health. By breaking down each code block, I also deepened my own understanding of why every detail matters. What seemed like a short script at first grew into a complete application that could reach users online.

This project shows how careful design and explanation can turn a simple script into a full technical blog post. I wanted to provide every detail so that someone reading it could reproduce the app without guessing steps. Each repetition of this point also reinforces how the project balances clarity with technical depth.
