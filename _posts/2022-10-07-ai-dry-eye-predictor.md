---
layout: default
title: "Building my AI Dry Eye Predictor"
date: 2022-10-07 08:21:35
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-dry-eye-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_dry_eye_predictor
featured: true
---

When I first noticed how long hours in front of a bright monitor strained my vision, I realized my eyes would often feel dry and tired. I wondered if these small discomforts were early signs of something bigger. That curiosity became the seed for this project. I wanted to know if everyday lifestyle and health signals could give me a reliable prediction about my risk for dry eye disease. The idea started as a personal observation and slowly transformed into a technical experiment. It gave me a chance to merge my interest in machine learning with a real health concern that affects many who spend their lives in front of screens. Dataset used [here](https://www.kaggle.com/datasets/dakshnagra/dry-eye-disease).

I built this project step by step and documented each stage carefully. In this blog post I will explain every file I uploaded to GitHub, every function that I wrote, and every helper that connects the pieces. I will go over how I structured the code, why I used Streamlit for the interface, and how the schema kept the UI consistent with the training pipeline. I will expand each code block into its purpose, how it fits the bigger picture, and why I designed it in that way. The aim is not just to present working code but to show the reasoning behind each decision in the most practical terms.

## Environment Setup and Dependency Management

The first file I created was `requirements.txt`. This file is important because it locks down the library versions used in this project. Without it, running the app on another machine could lead to conflicts if library updates introduced changes. I pinned exact versions of Streamlit, Pandas, Numpy, Scikit-learn, and Joblib. Each one plays a unique role in building or serving the app.

```python
streamlit==1.38.0
pandas==2.2.2
numpy==2.0.2
scikit-learn==1.6.1
joblib==1.5.2


```

This block fulfills a focused purpose in the pipeline. It either initializes variables, configures the interface, or builds the user input form. Each part of the code was designed to remain modular and easy to update. Together they contribute to a reliable end-to-end prediction system.


By fixing these versions, I ensured that the behavior of the code will remain stable across deployments. Streamlit drives the web app, Pandas and Numpy handle data manipulation, Scikit-learn powers the model pipeline, and Joblib manages the serialization of trained models. Having this foundation gives the project reproducibility and stability which are two critical aspects for any predictive application.

## Feature Schema Design and Input Validation

The `input_schema.json` file defines the structure of inputs expected by the app. It lists all features, their data types, and allowed values when relevant. This schema allows the UI to be generated dynamically, so that I never have to hardcode input forms. Instead, Streamlit reads this schema and adapts.

```python
{
  "target": "Dry Eye Disease",
  "features": [
    {
      "name": "Gender",
      "type": "category",
      "values": [
        "F",
        "M"
      ]
    },
    {
      "name": "Age",
      "type": "number"
    },
    {
      "name": "Sleep duration",
      "type": "number"
    },
    {
      "name": "Sleep quality",
      "type": "number"
    },
    {
      "name": "Stress level",
      "type": "number"
    },
    {
      "name": "Blood pressure",
      "type": "category",
      "values": [
        "100/60",
        "100/61",
        "100/62",
        "100/63",
        "100/64",
        "100/65",
        "100/66",
        "100/67",
        "100/68",
        "100/69",
        "100/70",
        "100/71",
        "100/72",
        "100/73",
        "100/74",
        "100/75",
        "100/76",
        "100/77",
        "100/78",
        "100/79",
        "100/80",
        "100/81",
        "100/82",
        "100/83",
        "100/84",
        "100/85",
        "100/86",
        "100/87",
        "100/88",
        "100/89"
      ]
    },
    {
      "name": "Heart rate",
      "type": "number"
    },
    {
      "name": "Daily steps",
      "type": "number"
    },
    {
      "name": "Physical activity",
      "type": "number"
    },
    {
      "name": "Height",
      "type": "number"
    },
    {
      "name": "Weight",
      "type": "number"
    },
    {
      "name": "Sleep disorder",
      "type": "category",
      "values": [
        "N",
        "Y"
      ]
    },
 ...

```

This block fulfills a focused purpose in the pipeline. It either initializes variables, configures the interface, or builds the user input form. Each part of the code was designed to remain modular and easy to update. Together they contribute to a reliable end-to-end prediction system.


I structured the schema so that categorical values are constrained to a set of valid options while numeric values remain flexible. This design reduces input errors and keeps training and prediction consistent. For example, the schema lists gender as a category with values 'M' and 'F'. It also lists continuous features like age, heart rate, and daily steps as numeric fields. This separation makes it easier for both the model and the UI to handle inputs correctly. Updating or extending the schema is simple: I just edit this JSON file without changing any of the Streamlit code.

## Core Application Architecture

The heart of the project lies in `app.py`. This script brings everything together. It loads the trained pipeline, connects it with the schema, builds the Streamlit interface, and generates predictions. I will now go block by block through the file, explaining how each piece works and why it was needed.

### Importing Core Libraries for Configuration
```python
import json
from pathlib import Path

```

This block fulfills a focused purpose in the pipeline. It either initializes variables, configures the interface, or builds the user input form. Each part of the code was designed to remain modular and easy to update. Together they contribute to a reliable end-to-end prediction system.


In this block I focused on a specific responsibility. It was designed to either import required libraries, set configuration, build the UI, or handle predictions. By keeping it small and readable I made debugging easier. This code interacts with other blocks as part of a larger pipeline that loads models, validates inputs, and produces predictions. 

### Importing Streamlit and Data Libraries
```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib

```

This block loads the core libraries used in the project. Streamlit manages the interface, Pandas and Numpy handle data manipulation, and Joblib supports saving and loading the pipeline. By collecting imports together I made the file readable and ensured dependencies are clear.


In this block I focused on a specific responsibility. It was designed to either import required libraries, set configuration, build the UI, or handle predictions. By keeping it small and readable I made debugging easier. This code interacts with other blocks as part of a larger pipeline that loads models, validates inputs, and produces predictions. 

### Setting Application Layout and Metadata
```python
st.set_page_config(page_title="Dry Eye Disease Predictor", page_icon="👁️", layout="centered")

```

Here I set the layout and page configuration for Streamlit. It defines the app title, icon, and layout format. This creates a polished user experience right from the beginning. It also keeps the interface consistent whenever the app is redeployed.


In this block I focused on a specific responsibility. It was designed to either import required libraries, set configuration, build the UI, or handle predictions. By keeping it small and readable I made debugging easier. This code interacts with other blocks as part of a larger pipeline that loads models, validates inputs, and produces predictions. 

### Defining Model and Schema Paths
```python
MODEL_PATH = Path("models/dry_eye_pipeline.pkl")
SCHEMA_PATH = Path("models/input_schema.json")

```

This block fulfills a focused purpose in the pipeline. It either initializes variables, configures the interface, or builds the user input form. Each part of the code was designed to remain modular and easy to update. Together they contribute to a reliable end-to-end prediction system.


In this block I focused on a specific responsibility. It was designed to either import required libraries, set configuration, build the UI, or handle predictions. By keeping it small and readable I made debugging easier. This code interacts with other blocks as part of a larger pipeline that loads models, validates inputs, and produces predictions. 

### Defining Model and Schema Paths
```python
@st.cache_resource
def load_artifacts():
    pipe = joblib.load(MODEL_PATH)
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    return pipe, schema

```

This helper function loads the serialized pipeline and schema only once. Caching avoids reloading large files each time inputs are processed. It speeds up predictions while keeping memory usage efficient. This block highlights the importance of balancing performance and reproducibility.


In this block I focused on a specific responsibility. It was designed to either import required libraries, set configuration, build the UI, or handle predictions. By keeping it small and readable I made debugging easier. This code interacts with other blocks as part of a larger pipeline that loads models, validates inputs, and produces predictions. 

### Initializing Pipeline and Schema
```python
pipe, schema = load_artifacts()
features = schema["features"]

```

This block fulfills a focused purpose in the pipeline. It either initializes variables, configures the interface, or builds the user input form. Each part of the code was designed to remain modular and easy to update. Together they contribute to a reliable end-to-end prediction system.


In this block I focused on a specific responsibility. It was designed to either import required libraries, set configuration, build the UI, or handle predictions. By keeping it small and readable I made debugging easier. This code interacts with other blocks as part of a larger pipeline that loads models, validates inputs, and produces predictions. 

### Building Application Title and Instructions
```python
st.title("Dry Eye Disease Predictor")
st.write("Fill the fields below. The app uses the same preprocessing as training, so inputs can stay raw.")

```

This block fulfills a focused purpose in the pipeline. It either initializes variables, configures the interface, or builds the user input form. Each part of the code was designed to remain modular and easy to update. Together they contribute to a reliable end-to-end prediction system.


In this block I focused on a specific responsibility. It was designed to either import required libraries, set configuration, build the UI, or handle predictions. By keeping it small and readable I made debugging easier. This code interacts with other blocks as part of a larger pipeline that loads models, validates inputs, and produces predictions. 

### Constructing Dynamic Input Form
```python
# Build the input UI dynamically from the schema
user_inputs = {}
with st.form("dry_eye_form"):
    for f in features:
        name = f["name"]
        ftype = f.get("type", "category")

```

This block fulfills a focused purpose in the pipeline. It either initializes variables, configures the interface, or builds the user input form. Each part of the code was designed to remain modular and easy to update. Together they contribute to a reliable end-to-end prediction system.


In this block I focused on a specific responsibility. It was designed to either import required libraries, set configuration, build the UI, or handle predictions. By keeping it small and readable I made debugging easier. This code interacts with other blocks as part of a larger pipeline that loads models, validates inputs, and produces predictions. 

### Handling Numeric and Categorical Inputs
```python
if ftype == "number":
            # choose sensible defaults
            val = st.number_input(name, value=0.0, step=1.0, format="%.2f")
            user_inputs[name] = val
        else:
            # categorical: use selectbox if few values, else text input
            values = f.get("values", [])
            if 2 <= len(values) <= 30:
                default_index = 0 if values else None
                val = st.selectbox(name, options=values, index=default_index if values else None)
            else:
                val = st.text_input(name, value="")
            user_inputs[name] = str(val).strip()

```

This block fulfills a focused purpose in the pipeline. It either initializes variables, configures the interface, or builds the user input form. Each part of the code was designed to remain modular and easy to update. Together they contribute to a reliable end-to-end prediction system.


In this block I focused on a specific responsibility. It was designed to either import required libraries, set configuration, build the UI, or handle predictions. By keeping it small and readable I made debugging easier. This code interacts with other blocks as part of a larger pipeline that loads models, validates inputs, and produces predictions. 

### Capturing Form Submission
```python
submitted = st.form_submit_button("Predict")

```

This block fulfills a focused purpose in the pipeline. It either initializes variables, configures the interface, or builds the user input form. Each part of the code was designed to remain modular and easy to update. Together they contribute to a reliable end-to-end prediction system.


In this block I focused on a specific responsibility. It was designed to either import required libraries, set configuration, build the UI, or handle predictions. By keeping it small and readable I made debugging easier. This code interacts with other blocks as part of a larger pipeline that loads models, validates inputs, and produces predictions. 

### Converting Inputs to DataFrame
```python
if submitted:
    # Turn inputs into a single-row DataFrame with columns in training order
    input_df = pd.DataFrame([user_inputs], columns=[f["name"] for f in features])

```

This section converts raw user inputs into a structured DataFrame. I ensured the column order matches training so the pipeline interprets values correctly. Without this alignment the model could misinterpret features. It bridges the gap between front-end input and backend model inference.


In this block I focused on a specific responsibility. It was designed to either import required libraries, set configuration, build the UI, or handle predictions. By keeping it small and readable I made debugging easier. This code interacts with other blocks as part of a larger pipeline that loads models, validates inputs, and produces predictions. 

### Generating Predictions from Pipeline
```python
# Predict with the pipeline (handles impute + OHE internally)
    proba = pipe.predict_proba(input_df)[0, 1]
    pred = int(proba >= 0.5)

```

The model pipeline is used here to generate probabilities. I apply a threshold of 0.5 to classify results into low or high risk. Using probabilities gives me more information than just a binary outcome. It also allows me to later adjust thresholds if sensitivity needs change.


In this block I focused on a specific responsibility. It was designed to either import required libraries, set configuration, build the UI, or handle predictions. By keeping it small and readable I made debugging easier. This code interacts with other blocks as part of a larger pipeline that loads models, validates inputs, and produces predictions. 

### Displaying Prediction Results
```python
st.subheader("Result")
    st.write("High Risk of Dry Eye" if pred == 1 else "Low Risk of Dry Eye")

```

This block communicates the final prediction back to the user. It clearly states whether the risk of dry eye is high or low. Presenting the outcome in simple terms is important in health applications. It ensures the user can understand results without technical background.


In this block I focused on a specific responsibility. It was designed to either import required libraries, set configuration, build the UI, or handle predictions. By keeping it small and readable I made debugging easier. This code interacts with other blocks as part of a larger pipeline that loads models, validates inputs, and produces predictions. 

### Displaying Probability Details
```python
st.subheader("Details")
    st.write(f"Estimated probability: {proba:.3f}")

```

Here I display the exact probability estimate. This transparency is crucial in predictive models. By showing the number behind the classification the user gains trust in the system. It also helps when comparing changes across different input values.


In this block I focused on a specific responsibility. It was designed to either import required libraries, set configuration, build the UI, or handle predictions. By keeping it small and readable I made debugging easier. This code interacts with other blocks as part of a larger pipeline that loads models, validates inputs, and produces predictions. 
## Deployment and Repository Setup

To deploy the app I uploaded several files into my GitHub repository. These included `app.py`, `requirements.txt`, the serialized pipeline `dry_eye_pipeline.pkl`, a backup model file `dry_eye_model.pkl`, and the `input_schema.json` file. Each file plays a role in keeping the app reproducible. Streamlit Cloud can read the repository and build the app directly from these files without extra configuration.

## Lessons Learned

While building this project I discovered how important it is to keep design modular. Separating the schema from the UI reduced maintenance effort. Using caching in Streamlit kept the app fast even with larger models. Structuring the repository properly ensured deployment was smooth. These choices may seem like details, but together they made the difference between a rough prototype and a usable predictive tool.

## Conclusion

What began as a reaction to my own eye strain turned into a learning project that merged health awareness with machine learning. The predictor is simple but it captures how structured data, clear schema, and reproducible pipelines can produce meaningful applications. Documenting each file and each block gave me clarity and provides a reference for others. This project reminded me that personal needs can spark technical creativity and that even small models can bring insight into everyday challenges.


In reflecting on this project I see how caching, reproducibility, schema-driven design, and modular code each contribute to reliability. Each function was explained carefully because these small details matter when sharing work with others. The project stands as an example of how even a health-inspired idea can be structured like a proper machine learning product. This kind of detail ensures that future improvements will remain easy to implement without breaking past functionality.
