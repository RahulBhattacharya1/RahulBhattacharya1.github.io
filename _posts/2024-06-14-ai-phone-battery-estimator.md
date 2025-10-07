---
layout: default
title: "Creating my Phone Battery Estimator"
date: 2024-06-14 09:22:06
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/battery.webp
thumbnail_mobile: /assets/images/phone_battery_sq.webp
demo_link: https://rahuls-ai-phone-battery.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_phone_battery
---

It started from a moment of frustration. While researching phones online, I realized that battery capacity was often hidden in long specification sheets. Many times, the listed features like RAM or screen refresh rate were clear, but the actual battery capacity was not always highlighted. I imagined a tool that could predict the battery capacity from other known specifications. This thought gave birth to the project I am describing here.

The experience made me think about how specifications we see every day could tell a deeper story. A screen size is not just a number, it drives energy consumption. A chipset family does not just shape performance, it influences efficiency. These connections between hardware details inspired me to build a model that could estimate the unseen number: the battery capacity. That is the central idea behind this project. Dataset used [here](https://www.kaggle.com/datasets/sulthonaqthoris/mobile-phone-specifications-dataset).

---

## Files in the Project

The project has three important files that make everything work:

1. **requirements.txt** – This file contains the dependencies. Without it, the app cannot install the correct packages when deployed.  
2. **app.py** – The Streamlit application where the interface is built and the prediction logic is written.  
3. **models/phone_battery_model.joblib** – The trained machine learning model that is loaded at runtime for predictions.

I will now go through each file in order, explain what it does, and expand every block of code into detailed reasoning. This is the process that helped me understand my own project better.

---

## requirements.txt

The file looks like this:

```python
streamlit
pandas
scikit-learn
joblib
```

This small file plays a large role. It lists every dependency needed by the app. Streamlit is used to build the user interface. Pandas is used to handle tabular data and structure the inputs before they are sent to the model. Scikit-learn is not directly imported in the app, but it is the library from which the model was trained, and so it is required for joblib to load the model correctly. Joblib is the tool that serializes and deserializes the model file.

Without this requirements file, deploying on GitHub Pages or Streamlit Cloud would fail. The environment would not know which versions to install. This is why such a small file is always present in Python projects.

---

## app.py

This is the main program of the project. It is where the interface is defined and the machine learning model is used to make predictions. I will explain each section in blocks.

### Imports and Setup

```python
import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Phone Battery Estimator", page_icon="🔋", layout="centered")
st.title("Phone Battery Estimator")
st.write("Enter key specs to estimate **battery capacity (mAh)**.")
```

This block imports the required libraries. The `os` module helps with file paths. The `joblib` library loads the model. Pandas is required to create structured data frames from the form inputs. Streamlit is the framework for the web interface. The call to `set_page_config` defines the title, icon, and layout of the page. Then the title and description are displayed. This sets the stage for the user.

### Defining Model Path and Loading Helper

```python
MODEL_PATH = os.path.join("models", "phone_battery_model.joblib")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
```

Here the location of the saved model file is stored in a constant. The `load_model` function loads it using joblib. The decorator `@st.cache_resource` makes sure that the model is only loaded once. If it were not cached, every user action would reload the file, wasting time. This helper function ensures smooth experience. The model variable then holds the trained estimator.

### User Input Form

```python
with st.form("specs"):
    col1, col2 = st.columns(2)

    with col1:
        os_family = st.selectbox("OS", ["android","ios","windows","other"], index=0)
        chipset_brand = st.selectbox(
            "Chipset Brand",
            ["qualcomm","mediatek","exynos","apple","unisoc","other"],
            index=0
        )
        screen_in = st.number_input("Screen Size (inches)", min_value=3.5, max_value=8.5, value=6.5, step=0.1, format="%.1f")

    with col2:
        ram_gb = st.number_input("RAM (GB)", min_value=1.0, max_value=24.0, value=8.0, step=1.0)
        storage_gb = st.number_input("Storage (GB)", min_value=8.0, max_value=1024.0, value=128.0, step=16.0)
        refresh_hz = st.number_input("Refresh Rate (Hz)", min_value=30.0, max_value=240.0, value=120.0, step=10.0)

    supports_5g = st.selectbox("5G Support", ["no","yes"], index=1)

    submitted = st.form_submit_button("Predict")
```

This block builds the user form. It creates two columns for layout balance. On the left side, inputs for operating system, chipset brand, and screen size are taken. On the right side, inputs for RAM, storage, and refresh rate are entered. Below them, the option for 5G support is provided. The button at the bottom submits the form. Streamlit handles the rendering in a clean format. This form is critical because it collects all the parameters that the model expects.

### Handling Form Submission

```python
if submitted:
    x = pd.DataFrame([{
        "os_family": os_family,
        "chipset_brand": chipset_brand,
        "ram_gb": float(ram_gb),
        "storage_gb": float(storage_gb),
        "refresh_hz": float(refresh_hz),
        "screen_in": float(screen_in),
        "supports_5g": supports_5g
    }])
    y_pred = model.predict(x)[0]
    st.success(f"Estimated Battery Capacity: {int(y_pred)} mAh")
```

This conditional checks whether the form has been submitted. If so, a new data frame is created from the values. Each field from the form is included as a key in the dictionary. This ensures the model receives the correct feature names. The prediction is then made using the model. The result is displayed as a success message on the page. This final block brings together the whole workflow.

---

## The Model File

The third file is **phone_battery_model.joblib**. It is a serialized machine learning model. While I cannot open it as plain text, its purpose is clear. It was trained using scikit-learn. It learned the patterns between specifications and battery capacity. The training was done outside this repository, and the final estimator was saved here for inference.

This separation is useful. Training can be heavy, but inference is light. Users interact with the small Streamlit app, while the large computations remain invisible. Joblib ensures that the trained estimator is reloaded in the same state each time.

---

## Closing Thoughts

This project taught me that user interface and machine learning model must meet halfway. The interface must collect the right features. The model must understand them. When connected correctly, the result feels natural. A person enters specifications, and the app provides a prediction that feels like it belongs in the spec sheet itself.

Such projects are not only technical exercises. They show how unseen values can be estimated from visible ones. This idea has many applications in other fields as well. The lesson I carried forward is that a small model, when presented with clarity, can provide real value.

---
