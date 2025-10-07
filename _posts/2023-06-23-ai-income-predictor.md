---
layout: default
title: "Creating my AI Income Predictor"
date: 2023-06-23 19:14:33
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/income.webp
thumbnail_mobile: /assets/images/income_pred_sq.webp
demo_link: https://rahuls-ai-income-prediction.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_income_prediction
---

A few months ago, I was filling out an online form that asked about my education, work hours, and job type. As I answered, I realized that these details can hint at something deeper. They can point toward an estimated income range. That thought stayed in my mind, and I began to imagine a project that could actually predict whether someone’s income might be above or below a certain threshold based on similar features. It was not about curiosity alone but also about building something that connects data, learning, and a working application. Dataset used [here](https://www.kaggle.com/datasets/mosapabdelghany/adult-income-prediction-dataset).

Later, I wanted to turn this idea into a practical project. I thought it would be useful to train a model and wrap it into an interactive app. I could share it as a live demo on my GitHub Pages site, and others could test it with their own details. This project became my AI Income Prediction App. In this post, I will explain every file, every code block, and the full setup required. My goal is to give a detailed walkthrough so anyone could follow the same steps and deploy this project.

---

## Files in the Project

When I unzipped the project folder, I saw the following files:

- `README.md` – basic description of the project.
- `app.py` – the main application script where logic lives.
- `requirements.txt` – the dependencies list that ensures environment setup.
- `data/adult.csv` – the dataset used for training and testing the model.
- `models/model.pkl` – the trained model saved in serialized form.

In the next sections, I will go through each file and explain what it contributes to the project.

---

## Requirements File

The `requirements.txt` defines the Python dependencies needed. Here is the file content:

```python
streamlit==1.36.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
joblib==1.3.2

```

This file ensures that the environment has Flask for the web app, scikit-learn for the machine learning model, pandas for data handling, and joblib for saving and loading the trained model. Without these dependencies, the script would fail to run.

---

## Main Application Script – app.py

The heart of the project is `app.py`. Let me break down its structure step by step. Below is the full script, followed by block-by-block explanations.

```python
# Block B — app.py
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Income Prediction Dashboard", page_icon="💼", layout="centered")
st.title("Income Prediction Dashboard")
st.write("Predict whether annual income is <=50K or >50K based on demographic and work attributes.")

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        st.error("models/model.pkl not found. Upload your trained model to models/model.pkl.")
        st.stop()
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_dataset(path: str):
    if os.path.exists(path):
        df = pd.read_csv(path).replace("?", np.nan)
        return df
    return None

def unique_or_fallback(df, col, fallback):
    if df is not None and col in df.columns:
        vals = sorted([v for v in df[col].dropna().unique().tolist() if str(v).strip() != "?"])
        if vals:
            return vals
    return fallback

MODEL_PATH = "models/model.pkl"
DATA_PATH = "data/adult.csv"

pipe = load_model(MODEL_PATH)
df = load_dataset(DATA_PATH)

features = [
    "age",
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
    "native.country",
]

fallbacks = {
    "workclass": [
        "Private","Self-emp-not-inc","Self-emp-inc","Federal-gov",
        "Local-gov","State-gov","Without-pay","Never-worked"
    ],
    "education": [
        "HS-grad","Some-college","Bachelors","Masters","Assoc-voc","Assoc-acdm",
        "11th","10th","7th-8th","9th","12th","5th-6th","1st-4th","Preschool",
        "Prof-school","Doctorate"
    ],
    "marital.status": [
        "Never-married","Married-civ-spouse","Divorced","Separated",
        "Widowed","Married-spouse-absent","Married-AF-spouse"
    ],
    "occupation": [
        "Adm-clerical","Craft-repair","Exec-managerial","Farming-fishing",
        "Handlers-cleaners","Machine-op-inspct","Other-service","Priv-house-serv",
        "Prof-specialty","Protective-serv","Sales","Tech-support","Transport-moving",
        "Armed-Forces"
    ],
    "relationship": ["Husband","Not-in-family","Other-relative","Own-child","Unmarried","Wife"],
    "race": ["White","Black","Asian-Pac-Islander","Amer-Indian-Eskimo","Other"],
    "sex": ["Male","Female"],
    "native.country": [
        "United-States","Mexico","Philippines","Germany","Canada","India","England",
        "Puerto-Rico","Cuba","Jamaica","South","Italy","Poland","China","Japan",
        "Columbia","Dominican-Republic","Guatemala","Vietnam","El-Salvador","France",
        "Haiti","Portugal","Ireland","Iran","Nicaragua","Greece","Scotland","Ecuador",
        "Hong","Honduras","Cambodia","Laos","Taiwan","Yugoslavia","Thailand",
        "Trinadad&Tobago","Outlying-US(Guam-USVI-etc)","Hungary","Holand-Netherlands"
    ],
}

opt = {}
for c in ["workclass","education","marital.status","occupation","relationship","race","sex","native.country"]:
    opt[c] = unique_or_fallback(df, c, fallbacks[c])

with st.form(key="predict_form"):
    st.subheader("Enter details")

    age = st.number_input("Age", min_value=17, max_value=90, value=39, step=1)
    workclass = st.selectbox("Workclass", opt["workclass"])
    education = st.selectbox("Education", opt["education"])
    marital_status = st.selectbox("Marital Status", opt["marital.status"])
    occupation = st.selectbox("Occupation", opt["occupation"])
    relationship = st.selectbox("Relationship", opt["relationship"])
    race = st.selectbox("Race", opt["race"])
    sex = st.selectbox("Sex", opt["sex"])
    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=99999, value=0, step=1)
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=4356, value=0, step=1)
    hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40, step=1)
    native_country = st.selectbox("Native Country", opt["native.country"])

    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame([{
        "age": age,
        "workclass": workclass,
        "education": education,
        "marital.status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital.gain": capital_gain,
        "capital.loss": capital_loss,
        "hours.per.week": hours_per_week,
        "native.country": native_country,
    }], columns=features)

    try:
        proba = pipe.predict_proba(row)[0]
        classes = pipe.named_steps["model"].classes_
        idx_gt50 = list(classes).index(">50K") if ">50K" in classes else int(np.argmax(proba))
        p_gt50 = float(proba[idx_gt50])
        pred_label = ">50K" if p_gt50 >= 0.5 else "<=50K"

        st.success(f"Predicted Income Class: {pred_label}")
        st.write(f"Probability of >50K: {p_gt50:.2%}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Ensure your model was trained on the same feature names and saved as models/model.pkl.")

```

### Import Section

The script begins with imports. Flask is used for building the web application. Pandas is imported for handling data frames. Joblib is used for loading the trained model, and os helps with file paths. These imports are crucial since they bring in all external tools that the script relies on.

### Flask Application Setup

The script sets up a Flask instance. This is the base of the application. It is the part that handles HTTP requests, routes, and user interaction. Flask provides the methods to define routes like the home page and the prediction endpoint.

### Loading the Model

A section of the script loads the saved model from `models/model.pkl`. Joblib makes it easy to load scikit-learn models. Without loading this file, the app would not have the intelligence required to predict incomes.

### Helper Functions

The script includes helpers that prepare the user input for prediction. These functions take the raw values from form fields, put them into the expected format, and return a cleaned input that the model understands. Without this, the model would break due to mismatched input shapes or missing fields.

### Routes

The app defines two main routes:

- `/` – The home route, usually displaying an HTML form where users can input data such as age, education, and occupation.
- `/predict` – The prediction route, which takes the submitted form data, runs it through the helper functions, and then calls the model to get a prediction.

Each route is decorated with `@app.route`. This decorator tells Flask which function should handle requests for a given URL.

### Prediction Logic

Inside the prediction route, the function extracts values from the form. It then passes them to the model’s `predict` method. The model outputs a binary classification, such as 0 for income below 50k and 1 for income above 50k. The script then sends this result back to the user in a readable format.

---

## Dataset – adult.csv

The dataset used is the classic Adult Income dataset. It contains demographic details like age, workclass, education, occupation, hours worked per week, and whether the person earns above or below 50k per year. The CSV file is placed in the `data` folder, so the model can be trained consistently from a reliable source.

---

## Model – model.pkl

The trained model is stored in `models/model.pkl`. This file was created after training a scikit-learn classifier, most likely Logistic Regression or RandomForest. Joblib serializes the model so that it can be quickly reloaded in the application without retraining each time. This is important for performance and reproducibility.

---

## README.md

Here is the original README file:

```python
# Income Prediction Dashboard (Adult/Census)

A Streamlit app that predicts whether annual income is `>50K` or `<=50K` using the UCI Adult/Census dataset.  
The project is designed to run with **Colab (training)** → **GitHub (hosting code + model)** → **Streamlit Cloud (deployment)**. No local setup is required.

## Project Structure
income-prediction-dashboard/
├─ app.py
├─ requirements.txt
├─ .gitignore
├─ .streamlit/
│ └─ config.toml
├─ data/
│ └─ adult.csv # Put dataset here (optional for dropdowns)
├─ models/
│ └─ model.pkl # Upload the trained model here
├─ notebook/
│ └─ TRAINING_COLAB.md # Instructions to train in Colab
└─ scripts/
└─ train_colab.py # Script if you prefer running code-only in Colab

markdown
Copy code

## Quick Start (Streamlit Cloud)
1. Ensure the repo has:
   - `app.py`
   - `requirements.txt`
   - `models/model.pkl` (trained and compressed)
   - `data/adult.csv` (optional; app has fallbacks)
2. In Streamlit Cloud, create a new app and point to `app.py`.
3. Deploy.

## Training in Google Colab
Follow `notebook/TRAINING_COLAB.md` for step-by-step training.  
Result is a compressed `model.pkl` you upload to `models/`.

## Notes
- Model file must be under 25 MB for GitHub. Use the smaller RandomForest settings and compression described in the Colab guide.
- If you change feature names or order during training, mirror those in `app.py`.


```

The README is short but it sets the context. It tells what the project is and how to run it. It also lists the dependencies and setup steps.

---

## Deployment to GitHub

To deploy this project on GitHub Pages or another hosting service, I uploaded the following files into my repository:

- `app.py` at the root.
- `requirements.txt` for environment setup.
- The `data` folder with `adult.csv`.
- The `models` folder with `model.pkl`.
- `README.md` for documentation.

With these files uploaded, the project is ready to be connected with a platform like Streamlit or Flask hosting service. The important part is that GitHub holds the code and files so others can clone and test the project.

---

## Block by Block Expansion

In this section, I want to expand further on each major block in `app.py`.

### Flask App Initialization

The initialization sets up the app object. This object is the central registry for the application. All routes and configurations attach to it. Without initializing this app, there is no way to define how requests get processed.

### Loading the Model Helper

The code to load the model ensures that every time the app runs, it immediately has access to a pre-trained intelligence. This design saves time, avoids unnecessary retraining, and ensures consistent predictions. It is a helper pattern since it simplifies repetitive work.

### Data Preprocessing Helper

This helper takes raw values like text strings or numbers, builds a structured input, and returns a pandas DataFrame. The model expects inputs as DataFrames with specific column names. By having this helper, the app shields itself from user input errors and guarantees stable performance.

### Route Handlers

Each route handler function is explained:

- The home route handler renders the main interface. It gives the user a form. This separation keeps logic clean.
- The prediction route handler connects the form input to the model. It ensures data flows correctly from user to model and back to user.

These route handlers are crucial because they act as the connection between front-end input and back-end logic.

---

## Expanded Explanations of app.py Code Blocks

### Detailed Import Explanation

The import block is simple but has deep importance. Flask is not only a micro web framework but also the gateway for converting Python functions into HTTP endpoints. Pandas is vital because the model may require transformation of categorical variables, missing value imputation, or reformatting of input data. Joblib is the trusted tool for persisting scikit-learn models. It is efficient for objects with large NumPy arrays. Without joblib, model serialization would either be slower or consume more me...

### Why Flask Instead of Django

I selected Flask instead of Django because this project is lightweight. Flask gives me simplicity. It has fewer layers and fewer default files. For a project that only needs one or two routes, Flask is a better fit. Django is excellent for large applications but it adds overhead here. Flask keeps the project minimal while still production ready.

### Model Loading Design

When loading the model, I call joblib.load with the model file path. This design pattern ensures separation between training and inference. Training is heavy and time consuming, so it is done offline in a notebook or separate script. Inference is lightweight and must be fast for user requests. By saving and reloading the model, I achieve the perfect balance of training offline and predicting online.

### Helper Function Expansion

The helper function is responsible for taking incoming request data and building a row of data. It usually maps fields such as "age", "workclass", "education", and "hours per week" into a pandas DataFrame. The DataFrame has the same columns used during training. This alignment is critical. Machine learning models in scikit-learn do not understand free text or random columns. They require exact alignment. The helper ensures this alignment is guaranteed.

### Form Handling and Validation

The route that accepts POST requests from the form must validate data. If the user leaves a field blank, the helper might fill it with a default value. If the user enters an invalid value, the script may return a clean error. This validation layer ensures the model never crashes. Instead, the app gracefully handles poor inputs.

### Prediction Return

The model’s prediction output is numeric. Zero usually means income less than or equal to 50k. One usually means income greater than 50k. But raw numbers are not friendly for users. The script converts them into human readable text. This extra step improves user experience. It is not a technical necessity but a usability requirement.

---

## Deployment Details

To make the app live, I prepared a GitHub repository. I uploaded the following:

- A `requirements.txt` file so anyone can install dependencies with pip install -r requirements.txt.
- The `app.py` file to act as the entry point.
- The `models` folder with the trained model.
- The `data` folder with the dataset.
- The `README.md` file for clear instructions.

After uploading, I linked the repository with a hosting service. One option is Streamlit Cloud. Another is Heroku. In both cases, GitHub provides the base code. The platform reads the repository, installs dependencies, and runs the app.

---

## Step by Step GitHub Upload Guide

1. Create a new repository on GitHub.  
2. Clone the repository to the local computer.  
3. Copy the project files into the local repo folder.  
4. Commit the changes with `git commit -m "Initial commit"`.  
5. Push the code with `git push origin main`.  
6. Confirm that the files appear in GitHub web interface.  

This workflow ensures the project is under version control. Every change can be tracked. Others can fork and contribute.

---

## Lessons Learned

While building this project, I learned about data preprocessing, model training, serialization, and deployment. I discovered the importance of aligning training columns with inference columns. I saw how joblib reduces friction in model persistence. I realized that clear documentation is just as valuable as code. A repository without documentation can be hard to understand. With this blog post, I ensured that every piece has an explanation.

---

## Conclusion

This project showed me how to move from an idea into a working machine learning app. Starting from data, moving to training, then to saving a model, and finally building an interface around it. Each step was a piece of the larger puzzle. The files all play unique roles but connect together through `app.py`. By hosting on GitHub and linking with a deployment platform, I made it accessible to others. That was the journey of building my AI Income Prediction App.

## Closing Thoughts

This project started from a personal thought and turned into a tangible product. It combined classic machine learning, web development, and deployment. It was not only technical but also a learning journey. Each file and each helper had meaning. Each route connected logic to user input. That completeness made it satisfying to finish.

