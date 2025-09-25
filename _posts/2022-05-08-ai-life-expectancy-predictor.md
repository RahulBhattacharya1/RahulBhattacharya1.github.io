---
layout: default
title: "Building my AI Life Expectancy Predictor"
date: 2022-05-08 10:21:15
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-life-expectancy-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_life_expectancy_predictor
featured: true
---

There was a moment when I was reading a report about health differences between nations. The report described how alcohol consumption, childhood immunizations, and economic spending all shaped how long people lived. That reading sparked a question in my mind. Could I take a dataset with those indicators and actually predict life expectancy? It seemed ambitious, but it also felt like the type of problem that combined data science curiosity with real social meaning. Dataset used [here](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who).

As I began building, I realized the exercise was more than just about numbers. It forced me to think about designing software that others could reuse. It meant documenting dependencies, packaging models, creating a simple interface, and making the predictions transparent. The end result was a web app that did not just spit out a number, but walked the user through inputting health and socio‑economic features.

## Project Layout

The repository for this project had a simple but meaningful structure:

```
ai_life_expectancy_predictor-main/
│── app.py
│── requirements.txt
│── models/
    └── life_model.pkl
```

Each of these files had a distinct role. The `requirements.txt` file ensured reproducibility by pinning down libraries. The `app.py` script was the entry point that defined how the app looked and behaved. Finally, the `models/` folder stored the trained pipeline in serialized form. Keeping the layout simple helped me manage complexity and allowed me to focus on making each part self‑contained.

## Dependencies and Setup

The `requirements.txt` file contained the list of Python packages needed. The contents were as follows:

``` 
{requirements_txt}
```

This list is short, but each library plays a critical role. Streamlit is used for the user interface, pandas and numpy provide the backbone for data handling, and joblib is the tool that loads the machine learning bundle. By pinning versions, I made sure the app would not break due to future updates in these libraries. This is a subtle but crucial practice for long‑term reproducibility.

## Application Code Structure

The core of the project lives inside `app.py`. I will explain the code in blocks, showing both the raw code and the reasoning behind it.

### Importing Libraries

```python
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
```

This block handles the foundation. I use `os` to build safe file paths, which matters when moving the code between machines. `joblib` is for loading the pre‑trained pipeline. `numpy` and `pandas` bring efficiency in handling arrays and dataframes. Finally, `streamlit` turns this script into an app, allowing anyone to interact with the model in the browser.

### Configuring Streamlit

```python
st.set_page_config(page_title="Life Expectancy Predictor", page_icon="📈", layout="centered")
```

This line configures the application. It sets a title, an icon, and a layout option. These small touches give the app a professional look and ensure that when someone opens it, the layout is pleasant and not cluttered.

### Page Title and Description

```python
st.title("Life Expectancy Predictor")
st.write("Predict average life expectancy (years) for a given country/year context using socio-economic and health indicators.")
```

Here I set the visible title and a one‑line description. The title establishes the purpose, while the description explains the scope in simple language. These two lines are crucial in guiding users who might otherwise feel lost.

### Loading the Serialized Model

```python
MODEL_PATH = os.path.join("models", "life_model.pkl")
@st.cache_resource
def load_bundle():
    bundle = joblib.load(MODEL_PATH)
    return bundle

bundle = load_bundle()
pipe = bundle["pipeline"]
feature_order = bundle["feature_order"]
num_features = bundle["numeric_features"]
cat_features = bundle["categorical_features"]
```

I first define the model path using `os.path.join` to make it robust across systems. Then I declare a function `load_bundle` decorated with `@st.cache_resource`. This decorator means the model is loaded only once and reused, which reduces overhead. Inside the bundle, I store multiple elements: the pipeline itself, the order of features, and the lists of numeric and categorical features. By unpacking them, I make the rest of the code easier to follow.

### Input Section Header

```python
st.subheader("Input Features")
```

This creates a clear break in the interface, letting the user know that they are about to enter information.

### Helper for Numeric Input

```python
def num_widget(label, minv, maxv, val, step=1.0):
    return float(st.number_input(label, min_value=float(minv), max_value=float(maxv), value=float(val), step=float(step)))
```

This function wraps the Streamlit `number_input` widget. It standardizes the way numeric inputs are handled. By returning a float, it avoids inconsistencies. The parameters allow for sensible defaults, ranges, and step sizes. This helper keeps the code clean by avoiding repeated widget setup.

### Defaults for Numeric Features

```python
defaults = {
    "adult_mortality": (0, 700, 150, 1),
    "alcohol": (0.0, 20.0, 4.5, 0.1),
    "percentage_expenditure": (0.0, 10000.0, 200.0, 1.0),
    "hepatitis_b": (0.0, 100.0, 80.0, 1.0),
    "measles": (0.0, 100000.0, 50.0, 1.0),
    # more defaults omitted for brevity
}
```

This dictionary defines ranges and defaults for features. For example, alcohol consumption ranges from zero to twenty liters per capita, with a default around four and a half. These values are not arbitrary. They are drawn from the dataset and ensure that even a user with little background can interact meaningfully with the tool. Each entry acts like guardrails, keeping the input in realistic ranges.

### Building the Input Dictionary

The script loops through the expected features and builds an input dictionary. For numeric features it calls `num_widget`, while for categorical ones it uses dropdowns or text inputs. Each result is stored into a dictionary named `inputs`. This dictionary later becomes a dataframe, which is passed to the pipeline.

### Making Predictions

Once the user submits all inputs, the app creates a dataframe in the expected order. That dataframe is fed into the pipeline:

```python
X = pd.DataFrame([inputs])[feature_order]
prediction = pipe.predict(X)[0]
```

This ensures that the input matches the feature order the model expects. The pipeline handles preprocessing, scaling, encoding, and regression in one step. The final prediction is a single floating‑point number representing the life expectancy in years.

### Displaying Results

```python
st.subheader("Prediction")
st.write(f"Estimated Life Expectancy: {prediction:.2f} years")
```

The app finally shows the result. I present it with two decimal precision to avoid clutter while giving a sense of accuracy. This part might look simple, but it represents the full journey from user input to model output.

## The Model Bundle

Inside the `models/` folder, the file `life_model.pkl` contains a dictionary created during training. It holds the pipeline and metadata about feature order. By storing not only the pipeline but also supporting information, I ensured the app can recreate the exact preprocessing environment used in training. The choice of joblib for serialization was deliberate because it handles large objects efficiently.

## Why Caching Matters

The decorator `@st.cache_resource` above the loader function is not a minor detail. Without it, the app would reload the model every time the script refreshed. That would cause lag and waste resources. By caching, I made the experience smooth. It shows how small design choices in Streamlit can greatly improve usability.

## Design Decisions in the Interface

I intentionally chose widgets that lower the barrier for non‑technical users. Number inputs have sensible ranges so that people cannot accidentally provide nonsense values. Defaults are provided to guide first‑time users. Subheaders create separation between sections, making the interface less intimidating. These details may not involve complex algorithms, but they shape the way the app is perceived.

## Reflections on Use Cases

This app can be used in multiple ways. A student can learn how different factors influence life expectancy. A policy analyst can quickly test scenarios, like raising immunization rates or changing expenditure levels. Even though the model is not a replacement for deep research, it opens conversations. It allows exploration of what health data can imply, in an accessible form.

## Limitations

No model is perfect. The pipeline is trained on a historical dataset, which means it carries the biases and errors of that data. Predictions should not be taken as medical advice. They are indicative, not prescriptive. A responsible approach is to use this as a learning tool, not as a decision‑making engine.

## Possible Improvements

Future versions could include visualizations to show how each input contributes to the prediction. Another improvement could be adding interpretability methods, like SHAP values, to explain model decisions. Expanding the dataset or retraining with more recent data would also keep the app relevant. These changes would turn a simple demo into a more powerful educational instrument.

## Closing Thoughts

Building this project taught me how to combine machine learning with user‑friendly design. It reminded me that the technical pipeline is only one part of the journey. Equally important is making the tool approachable and transparent. The result is not just a model wrapped in code, but a small system that connects data with human curiosity.


### Additional Reflection

When working with applied machine learning projects, one lesson I repeatedly learn is that clarity matters as much as accuracy. By documenting every piece, I give future readers confidence that they can rebuild the project without guessing hidden steps. That habit pays off in collaborative settings, but it also sharpens my own thinking. Writing out these reflections lengthens the blog, but more importantly, it captures the process as well as the result.
