---
layout: default
title: "Creating my AI IPL Player Predictor"
date: 2024-02-17 11:34:52
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/ipl_player.webp
thumbnail_mobile: /assets/images/ipl_player_sq.webp
demo_link: https://rahuls-ai-ipl-players.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_ipl_players
---

There are times when an idea refuses to fade until it finds an outlet. Mine began when I was reading through cricket player statistics late one night. The rows of averages, strike rates, and boundary counts made me think about patterns hidden in plain sight. It struck me that these numbers, if combined with the right tools, could tell me much more than surface-level performance. I wanted to know if a machine learning model could interpret those values and provide predictions in a structured way.

I also knew that keeping such an idea locked in a notebook would not be enough. I wanted it to be visible, interactive, and accessible. Streamlit provided exactly that. With minimal effort, I could build a browser app to input player details and receive predictions instantly. This blog is a record of how I turned that thought into a working web app. The post is detailed, with breakdowns of every file and every block of code. My aim is to make the path clear enough for anyone to follow and replicate. Dataset used [here](https://www.kaggle.com/datasets/prashantsharma3006/ipl-players-dataset).


## Project Structure

The repository looked like this after extraction:

```
ai_ipl_players-main/
│── app.py
│── requirements.txt
│── models/
│     ├── final_model.joblib
│     ├── schema.json
```

- **app.py**: The Streamlit app that defines layout, loads the model, collects user input, and makes predictions.  
- **requirements.txt**: Specifies dependencies with version pins to ensure consistent behavior.  
- **models/final_model.joblib**: The pre-trained scikit-learn model stored using joblib.  
- **models/schema.json**: A schema that defines expected features, target column, and type of task (classification or regression).  

In the following sections, I will explain each file. Most of the detail is in app.py, where I break down each block and explain how it fits into the larger picture.


## Requirements File

The `requirements.txt` file ensures reproducibility. It contains:

```python
streamlit==1.36.0
scikit-learn==1.5.2
pandas==2.2.2
numpy==2.1.3
joblib==1.4.2

```

- **streamlit**: Provides the web interface.  
- **scikit-learn**: Required since the model was trained with it.  
- **pandas**: Makes input data easier to handle in DataFrame form.  
- **numpy**: Supports underlying numerical operations.  
- **joblib**: Loads the pre-trained model.  

Without these exact versions, the app may break or behave differently. Fixing versions avoids surprises when deploying.


## Application Code: app.py

### Imports
```python
import os, json
import joblib
import pandas as pd
import streamlit as st
```
This group brings in all required libraries.  
- **os, json**: Handle file paths and JSON reading for schema.  
- **joblib**: Loads the trained machine learning model.  
- **pandas**: Converts input into tabular form so the model can accept it.  
- **streamlit**: Powers the interactive web application.  
Together, they establish the foundation for the rest of the app.

### Page Setup
```python
st.set_page_config(page_title="Final Data Predictor", page_icon="🔮", layout="centered")
st.title("Final Data Predictor")
st.write("Fill the fields to get a prediction. The app auto-detects whether the model is regression or classification.")
```
This section configures the Streamlit page and sets the tone for the interface.  
- The page has a title, an icon, and a centered layout.  
- A main heading and short description are displayed to guide the user.  
It ensures that when someone opens the app, they know immediately what it does and how to start.

### Paths and File Validation
```python
MODEL_PATH = os.path.join("models", "final_model.joblib")
SCHEMA_PATH = os.path.join("models", "schema.json")

def require(path, kind):
    if not os.path.exists(path):
        st.error(f"{kind} file missing: `{path}`. Upload it to your repo and restart.")
        st.stop()

require(MODEL_PATH, "Model")
require(SCHEMA_PATH, "Schema")
```
This block defines where the model and schema files live and introduces a `require` helper.  
The helper verifies that essential files exist before proceeding. If missing, it stops execution with a clear error message.  
This approach prevents unexpected crashes later and gives immediate feedback about missing dependencies.

### Model Loading
```python
@st.cache_resource
def load_model_and_schema():
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)
    model = joblib.load(MODEL_PATH)
    return model, schema

model, schema = load_model_and_schema()
```
Here the model and schema are loaded together.  
The `@st.cache_resource` decorator ensures this happens only once, avoiding slow reloads.  
The schema describes the dataset structure, and the model is the predictor.  
Returning both together makes them available throughout the app with a single call.

### Schema Setup
```python
task = schema.get("task", "regression")
target_name = schema.get("target", "target")
cat_schema = schema.get("categorical", {})
```
This part extracts important details from the schema.  
- `task` decides whether the app runs in classification or regression mode.  
- `target_name` identifies the target column so it is not included as an input.  
- `cat_schema` provides category lists for categorical features.  
These values guide how the input form and predictions are handled later.

### Input Form
```python
def input_form():
    inputs = {}
    with st.form("prediction_form"):
        st.subheader("Enter Feature Values")
        for col, dtype in schema["columns"].items():
            if col == target_name:
                continue
            if dtype == "number":
                val = st.number_input(f"{col}", value=0.0)
            elif dtype == "integer":
                val = st.number_input(f"{col}", value=0)
            elif dtype == "category":
                options = cat_schema.get(col, [])
                val = st.selectbox(f"{col}", options) if options else st.text_input(f"{col}")
            else:
                val = st.text_input(f"{col}")
            inputs[col] = val
        submitted = st.form_submit_button("Submit")
    return inputs, submitted
```
This function builds the input form dynamically.  
- It loops over all columns defined in the schema.  
- It chooses widgets based on column type: numeric input, integer input, select box, or text field.  
- The target column is skipped because it is not supposed to be provided by the user.  
The function returns both the dictionary of inputs and a boolean indicating whether the form was submitted.  
This design makes the input process flexible and schema-driven.

### Prediction Helper
```python
def predict(inputs: dict):
    df = pd.DataFrame([inputs])
    pred = model.predict(df)[0]
    return pred
```
This helper wraps the model prediction process.  
- It accepts a dictionary of inputs.  
- Converts it into a pandas DataFrame because scikit-learn models expect tabular structures.  
- Calls `model.predict` and retrieves the first element of the result.  
It isolates prediction logic, keeping the main script cleaner and more readable.

### Button and Output Handling
```python
inputs, submitted = input_form()

if submitted:
    result = predict(inputs)
    if task == "classification":
        st.success(f"Predicted class: {result}")
    else:
        st.success(f"Predicted value: {result}")
```
This section connects the form submission to prediction and output.  
- It first retrieves user inputs and checks if the form was submitted.  
- If so, it calls the `predict` helper and displays the result.  
- The output message changes depending on whether the task is classification or regression.  
This ensures predictions only appear when the user explicitly triggers them and are formatted appropriately.
