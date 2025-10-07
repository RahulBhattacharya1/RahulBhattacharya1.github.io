---
layout: default
title: "Building my AI Power Plant Capacity Predictor"
date: 2024-06-02 10:27:41
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/plant.webp
thumbnail_mobile: /assets/images/power_plant_sq.webp
demo_link: https://rahuls-ai-power-plant-capacity-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_power_plant_capacity_predictor
---

The motivation for this project started when I realized how difficult it can be to make sense of raw capacity data in energy reports. Many reports describe plants by fuel type, technology, or commissioning year, but then they jump straight to presenting final capacities. I found myself wondering how those capacities were estimated. This inspired me to create an application that could act as a bridge. Dataset used [here](https://www.kaggle.com/datasets/trolukovich/conventional-power-plants-in-europe).

Another part of the motivation came from my desire to move beyond static notebooks. I wanted to show that machine learning could be packaged into a tool with an interface that anyone could use. I saw the potential in Streamlit to act as that interface. It provided the right balance of simplicity and power. By combining it with a trained pipeline, I could create an interactive app that predicted capacity in megawatts.

## Repository File Structure

```
ai_power_plant_capacity_predictor-main/
├── app.py
├── requirements.txt
└── models/
    ├── capacity_pipeline.pkl
    ├── capacity_model.pkl
    └── meta.json
```

This layout is compact and purposeful. The `app.py` script contains all of the user interface and prediction logic. The `requirements.txt` file ensures that anyone reproducing the project can install exactly the same versions of packages. The `models` folder stores artifacts produced after training, including the serialized pipeline, the underlying model, and supporting metadata. By adopting this structure, I avoided clutter and allowed someone new to the project to immediately know where to look.

## requirements.txt File

```python
streamlit==1.37.1
pandas==2.2.2
numpy==2.0.2
scikit-learn==1.6.1
joblib==1.5.2
```

The requirements file lists all dependencies with pinned versions. `streamlit` provides the interactive web framework. `pandas` and `numpy` manage structured data. `scikit-learn` is the backbone of the machine learning workflow. `joblib` serializes the model pipeline. Pinning versions reduces uncertainty and ensures the model can be loaded consistently in the future. Without pinning, a future version of scikit-learn might change how serialization works, causing the model to fail when loaded.

## app.py File Detailed Breakdown

The core of the application resides inside `app.py`. This file is intentionally small, but every line matters. I will explain each function, helper, and conditional.

### Import Statements and Page Setup

```python
import os, joblib
import streamlit as st
import pandas as pd

st.set_page_config(page_title="EU Power Plant Capacity Predictor", page_icon="⚡")
```

This block imports all required libraries. `os` is a standard library module for path management. While it is not heavily used in the script, having it imported makes it easy to adjust file paths if needed. `joblib` is the helper for loading serialized scikit-learn objects. `streamlit` is imported as `st` because this is the common convention and makes calls concise. `pandas` is imported because the prediction inputs will be wrapped into a DataFrame.

### Constant for Model Path

```python
MODEL_PATH = "models/capacity_pipeline.pkl"
```

This line defines a constant string that points to the serialized pipeline. By placing it at the top, I avoid hardcoding paths throughout the script. If I ever change the file location, I only need to update it here. Defining constants in this way improves maintainability.

### load_model Function

```python
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)  # this is a scikit-learn Pipeline
```

This is the first defined function in the script. It serves a single purpose: load the pipeline from disk. The decorator `st.cache_resource` is a Streamlit helper that ensures the function result is cached. This means that even if the function is called multiple times, the underlying model is only loaded once per session. This drastically improves efficiency. If caching were absent, the pipeline would be reloaded from disk every time a user clicked the prediction button, introducing unnecessary latency.

### Assignment of Pipeline Variable

```python
pipe = load_model()
```

This line calls the helper function and stores the returned pipeline into a variable named `pipe`. From this point forward, the variable `pipe` is the gateway to making predictions. Abstracting it this way makes the later code more readable. Instead of repeatedly writing out the loading logic, I can just use `pipe.predict` directly.

### Title and Column Layout

```python
st.title("EU Power Plant Capacity Predictor")

col1, col2 = st.columns(2)
```

The title function call displays a large heading at the top of the app. It clearly tells the user what the app does. The call to `st.columns(2)` creates a two-column layout, which I then use to separate input fields. This improves readability and creates balance. Without this, the form would stack vertically, consuming more space and appearing less professional.

### Input Widgets in Columns

```python
with col1:
    energy_source = st.text_input("Energy source", "Natural gas")
    technology    = st.text_input("Technology", "Combined cycle")
with col2:
    commissioned  = st.number_input("Commissioning year", min_value=1900, max_value=2030, value=2000, step=1)
    country       = st.text_input("Country", "DE")
```

This block defines four input widgets. In the first column, I create text inputs for energy source and technology. Defaults are provided so that even if users are unfamiliar, they see valid examples. In the second column, I add a number input for commissioning year. The `min_value` and `max_value` parameters restrict possible inputs, reducing the risk of invalid values. The default year is 2000, which acts as a reasonable mid-point. Finally, a text input is provided for country code, defaulting to "DE".

### Prediction Conditional and Logic

```python
if st.button("Predict capacity (MW)"):
    X_new = pd.DataFrame([{
        "energy_source": energy_source,
        "technology": technology,
        "commissioned": int(commissioned),
        "country": country
    }])
    pred = float(pipe.predict(X_new)[0])
    st.success(f"Predicted capacity: {pred:.2f} MW")
```

This conditional is the heart of the user interaction. The helper `st.button` renders a button on the page. The `if` condition ensures that the logic runs only when the button is pressed. Inside, I construct a pandas DataFrame with one row. Each field corresponds to the model features. Casting the commissioning year to integer ensures type correctness. The pipeline’s `predict` function is then called. The result is typically returned as a numpy array.

## The Models Directory and Artifacts

The models directory contains the serialized pipeline, the standalone model, and metadata. Each has a purpose. The `capacity_pipeline.pkl` file is the main artifact. It is a scikit-learn pipeline that bundles preprocessing with the estimator. By serializing the pipeline, I do not need to manually re-implement preprocessing in the app. This makes the code cleaner and reduces the chance of mismatches. The `capacity_model.pkl` file contains the estimator without preprocessing.

The `meta.json` file is not loaded by the app but provides context. It might contain the dataset used, training parameters, or feature importance scores. Having metadata stored separately improves transparency. Anyone maintaining the project later can trace back what training decisions were made. This contributes to reproducibility.

## Deployment Strategy

Once all files were ready, I deployed the app to Streamlit Cloud. This required no extra configuration beyond uploading the repository. Streamlit Cloud detects `requirements.txt`, installs dependencies, and runs `app.py`. Since the models folder is included, the pipeline loads immediately. I did not need environment variables because no secrets were required. This kind of deployment is both simple and powerful. It shows that a well-structured repository can make hosting effortless.

## Extended Technical Reflections

One major reflection is about caching. Without caching, model loading would dominate the runtime. Users might experience several seconds of lag with every prediction. By caching, the app becomes responsive. Another reflection is about separating code into functions. Even though `load_model` is a small function, it improves readability and provides a single place to manage loading. These practices may seem minor, but they scale. As projects grow, clarity of code structure makes or breaks maintainability.

Another reflection concerns input validation. I used numeric bounds for commissioning year, but text inputs remain free form. This gives flexibility but risks invalid entries. If I were to extend the project, I might add dropdowns or regex validation. However, in this version, simplicity was prioritized. The balance between flexibility and validation is always delicate. I leaned toward trust in the user.

## Lessons Learned

I learned that deployment should not be an afterthought. By thinking early about how to load the model, manage inputs, and keep dependencies minimal, I avoided problems later. I also learned that even small helpers like caching decorators can change user experience dramatically. This reinforced the lesson that details matter. Another lesson is that clarity in file structure reduces friction. A minimal repository with three elements—script, requirements, models—was enough.

## Possible Extensions and Future Work

Future extensions could include visualizations of capacity predictions, integration with real datasets, or enhancements to input validation. For example, once the user inputs data, the app could generate comparative charts showing how different countries or technologies compare. Another extension could involve uncertainty estimates, providing not only a single capacity prediction but also a range. These features would enhance the educational value of the app.

## Conclusion

The EU Power Plant Capacity Predictor began with a simple motivation: explain capacity predictions through an accessible interface. It grew into a fully deployed application that demonstrates caching, reproducibility, and design simplicity. Every function, every helper, and every conditional serves a role. Nothing is redundant. By documenting it thoroughly here, I ensure transparency and share lessons for others. The themes of clarity and reproducibility will continue to guide my projects.
