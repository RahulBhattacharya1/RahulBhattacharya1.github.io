---
layout: default
title: "Enhancing my AI Real Estate Price Estimator"
date: 2023-10-18 14:26:51
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/real_estate.webp
demo_link: https://rahuls-ai-property-price-estimator.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_property_price_estimator
featured: true
---

I once spent a long time browsing real estate listings, trying to understand why two houses with similar size and features were priced so differently. It was not only confusing but also frustrating, since I wanted to get a sense of whether a place was overvalued or fairly listed. That experience made me wonder how I could use machine learning to quickly generate an estimate of property prices, based only on the main details that I could collect. Dataset used [here](https://www.kaggle.com/datasets/hassankhaled21/egyptian-real-estate-listings).

That idea stayed with me for a while until I decided to turn it into a project. I wanted something simple enough to run in a browser, but powerful enough to make predictions from past data. I also wanted it to be a learning exercise. I combined my interest in coding with my need for practical answers, and the result was this property price estimator. It predicts prices based on property features and shows the user a clear output inside a Streamlit web app.



## Files I Uploaded to GitHub

To make this work on GitHub Pages with Streamlit Cloud, I uploaded a few essential files. These files are:

1. `app.py` - The main application file that runs the Streamlit interface and handles predictions.
2. `requirements.txt` - This lists all dependencies needed to run the app in the cloud environment.
3. `artifacts/model.joblib` - The trained machine learning model that performs the actual price estimation.

Each file has a purpose, and I will explain the content and logic inside them step by step. My focus is to expand every code block and explain how it fits into the whole picture.


```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# --- shims for any FunctionTransformer helpers you used in Colab (safe no-ops) ---
def text_fill_1d(s):
    import pandas as _pd
    if isinstance(s, _pd.DataFrame):
        s = s.squeeze(axis=1)
    return s.fillna("")

def _to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X

densify = _to_dense

# Let pickles that refer to __main__/ipykernel resolve here
sys.modules['__main__'] = sys.modules[__name__]
for alias in ("ipykernel_launcher", "ipykernel", "notebook"):
    sys.modules[alias] = sys.modules[__name__]

# --- NEW: monkey-patch sklearn to provide the missing internal class -----------
try:
    from sklearn.compose import _column_transformer as _ct_mod
    if not hasattr(_ct_mod, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        _ct_mod._RemainderColsList = _RemainderColsList
except Exception:
    # If sklearn import fails, we’ll see it when loading the model anyway
    pass

st.set_page_config(page_title="Egypt Property Price Estimator", layout="centered")

@st.cache_resource
def load_model():
    here = Path(__file__).resolve().parent
    model_path = here / "artifacts" / "model.joblib"
    if not model_path.is_file():
        st.error(f"model.joblib not found at: {model_path}")
        st.stop()
    return joblib.load(model_path)

model = load_model()

st.title("Egypt Real Estate Price Estimator")
st.caption("Predict price based on property details. Model trained on scraped listings.")

with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        prop_type = st.selectbox("Property Type", ["Apartment","Villa","Chalet","Townhouse","Duplex","Studio","Other"])
        size_sqm  = st.number_input("Size (sqm)", min_value=10.0, max_value=3000.0, value=150.0, step=10.0)
        bedrooms  = st.number_input("Bedrooms", min_value=0.0, max_value=20.0, value=3.0, step=1.0)

    with col2:
        bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=20.0, value=2.0, step=1.0)
        payment   = st.selectbox("Payment Method", ["Cash","Installments","Mortgage","Other"])
        down_pay  = st.number_input("Down Payment (EGP)", min_value=0.0, max_value=1e9, value=0.0, step=10000.0)

    location_text    = st.text_input("Location (area, city)", "New Cairo, Cairo")
    description_text = st.text_area("Short Description", "Modern apartment with balcony and parking.")

    available_year  = st.number_input("Available Year", min_value=2000, max_value=2100, value=2025, step=1)
    available_month = st.number_input("Available Month", min_value=1, max_value=12, value=9, step=1)

    submitted = st.form_submit_button("Estimate Price")

if submitted:
    # Build a single-row DataFrame matching training columns
    row = pd.DataFrame([{
        "type": prop_type,
        "payment_method": payment,
        "size_sqm": float(size_sqm),
        "bedrooms_num": float(bedrooms),
        "bathrooms_num": float(bathrooms),
        "down_payment_num": float(down_pay),
        "available_year": int(available_year),
        "available_month": int(available_month),
        "location": location_text,
        "description": description_text
    }])

    try:
        yhat = model.predict(row)[0]

        # Clip unrealistic predictions
        if np.isnan(yhat) or np.isinf(yhat):
            yhat = 0.0
        else:
            # Cap between 0 and, say, 200 million EGP
            yhat = float(np.clip(yhat, 0, 2e8))

        st.subheader(f"Estimated Price: {yhat:,.0f} EGP")
        st.caption("This estimate is based on historical listings and provided features.")

        # Simple sensitivity: +/-10% band (not a statistical CI, just a communication band)
        low, high = yhat*0.9, yhat*1.1
        st.write(f"Range: {low:,.0f} – {high:,.0f} EGP")

        st.divider()
        st.markdown("**Inputs used:**")
        st.json(row.to_dict(orient="records")[0])

    except Exception as e:
        st.error(f"Prediction failed: {e}")

```

## Understanding `app.py`


The `app.py` file is the backbone of this project. It defines the Streamlit web interface, loads the model, and makes predictions. I will now go through the important sections and explain them in detail.

### Initial Imports and Setup
```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# --- shims for any FunctionTransformer helpers you used in Colab (safe no-ops) ---
```

This section imports Streamlit, pandas, numpy, and joblib. These libraries are essential. Streamlit powers the web interface. Pandas manages structured data. Numpy provides array operations. Joblib is used to load the trained machine learning model. The pathlib and sys modules help with file paths and module patching. Together, these imports prepare the environment so the rest of the code can work smoothly.

### Function `text_fill_1d`
```python
def text_fill_1d(s):
    import pandas as _pd
    if isinstance(s, _pd.DataFrame):
        s = s.squeeze(axis=1)
    return s.fillna("")
```

This function, `text_fill_1d`, has a specific role in the pipeline. It either transforms input data, fills missing values, or ensures compatibility between objects. I explain each function separately below.

### Function `_to_dense`
```python
def _to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X

densify = _to_dense

# Let pickles that refer to __main__/ipykernel resolve here
sys.modules['__main__'] = sys.modules[__name__]
for alias in ("ipykernel_launcher", "ipykernel", "notebook"):
    sys.modules[alias] = sys.modules[__name__]

# --- NEW: monkey-patch sklearn to provide the missing internal class -----------
try:
    from sklearn.compose import _column_transformer as _ct_mod
    if not hasattr(_ct_mod, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        _ct_mod._RemainderColsList = _RemainderColsList
except Exception:
    # If sklearn import fails, we’ll see it when loading the model anyway
    pass

st.set_page_config(page_title="Egypt Property Price Estimator", layout="centered")

@st.cache_resource
```

This function, `_to_dense`, has a specific role in the pipeline. It either transforms input data, fills missing values, or ensures compatibility between objects. I explain each function separately below.

### Function `load_model`
```python
def load_model():
    here = Path(__file__).resolve().parent
    model_path = here / "artifacts" / "model.joblib"
    if not model_path.is_file():
        st.error(f"model.joblib not found at: {model_path}")
        st.stop()
    return joblib.load(model_path)

model = load_model()

st.title("Egypt Real Estate Price Estimator")
st.caption("Predict price based on property details. Model trained on scraped listings.")

with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        prop_type = st.selectbox("Property Type", ["Apartment","Villa","Chalet","Townhouse","Duplex","Studio","Other"])
        size_sqm  = st.number_input("Size (sqm)", min_value=10.0, max_value=3000.0, value=150.0, step=10.0)
        bedrooms  = st.number_input("Bedrooms", min_value=0.0, max_value=20.0, value=3.0, step=1.0)

    with col2:
        bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=20.0, value=2.0, step=1.0)
        payment   = st.selectbox("Payment Method", ["Cash","Installments","Mortgage","Other"])
        down_pay  = st.number_input("Down Payment (EGP)", min_value=0.0, max_value=1e9, value=0.0, step=10000.0)

    location_text    = st.text_input("Location (area, city)", "New Cairo, Cairo")
    description_text = st.text_area("Short Description", "Modern apartment with balcony and parking.")

    available_year  = st.number_input("Available Year", min_value=2000, max_value=2100, value=2025, step=1)
    available_month = st.number_input("Available Month", min_value=1, max_value=12, value=9, step=1)

    submitted = st.form_submit_button("Estimate Price")

if submitted:
    # Build a single-row DataFrame matching training columns
    row = pd.DataFrame([{
        "type": prop_type,
        "payment_method": payment,
        "size_sqm": float(size_sqm),
        "bedrooms_num": float(bedrooms),
        "bathrooms_num": float(bathrooms),
        "down_payment_num": float(down_pay),
        "available_year": int(available_year),
        "available_month": int(available_month),
        "location": location_text,
        "description": description_text
    }])

    try:
        yhat = model.predict(row)[0]

        # Clip unrealistic predictions
        if np.isnan(yhat) or np.isinf(yhat):
            yhat = 0.0
        else:
            # Cap between 0 and, say, 200 million EGP
            yhat = float(np.clip(yhat, 0, 2e8))

        st.subheader(f"Estimated Price: {yhat:,.0f} EGP")
        st.caption("This estimate is based on historical listings and provided features.")

        # Simple sensitivity: +/-10% band (not a statistical CI, just a communication band)
        low, high = yhat*0.9, yhat*1.1
        st.write(f"Range: {low:,.0f} – {high:,.0f} EGP")

        st.divider()
        st.markdown("**Inputs used:**")
        st.json(row.to_dict(orient="records")[0])

    except Exception as e:
        st.error(f"Prediction failed: {e}")
```

This function, `load_model`, has a specific role in the pipeline. It either transforms input data, fills missing values, or ensures compatibility between objects. I explain each function separately below.


## The `requirements.txt` File

The `requirements.txt` file ensures that the correct library versions are installed when running this project on Streamlit Cloud. Without this, the environment might have different versions of libraries, leading to errors. The file lists libraries like Streamlit, Pandas, Numpy, scikit-learn, and Joblib. Each has a minimum version specified so the code can run reliably.

Below is the full content of the file:


```python
streamlit>=1.36
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
joblib>=1.3

```


## The Trained Model (`artifacts/model.joblib`)

The `model.joblib` file is the trained scikit-learn model. I trained this earlier in Google Colab using historical property listing data. After training, I exported it into a Joblib file. The app loads this model and calls its `predict` method whenever the user provides property details. The heavy lifting of prediction is done inside this artifact. It is not human-readable, but it is essential for the system to generate results.



## Conclusion

This project turned my initial frustration with property prices into a useful learning exercise. By building a Streamlit web app, uploading a trained model, and structuring the files clearly, I was able to create a working price estimator. Every function and helper inside `app.py` contributes to making the app robust and user-friendly. With just three files uploaded to GitHub, the system is fully operational in the cloud. This shows how accessible machine learning deployment can be when broken down into manageable steps.

### Deep Dive into `text_fill_1d`

The `text_fill_1d` function is a helper used during the preprocessing stage. Its job may look small, but it is critical to keeping the pipeline stable. When data is passed into a machine learning model, it often comes in different shapes or formats. This function makes sure that the data is reshaped, cleaned, or filled in such a way that the model can understand it.

For example, if there are missing text values, `text_fill_1d` ensures that they are replaced with empty strings rather than leaving them as NaN. This avoids crashes later when the model expects consistent input. Another case is when arrays are sparse; `text_fill_1d` converts them into dense arrays so that other parts of the pipeline can handle them. Without this helper, predictions could fail due to shape mismatches or unexpected nulls.

By isolating these fixes in a single helper function, the rest of the code remains clean. The model loading and the Streamlit app do not need to worry about these small details. Instead, they can focus on their primary tasks, while `text_fill_1d` silently handles edge cases. This separation of duties makes the project modular and easier to maintain.



### Deep Dive into `_to_dense`

The `_to_dense` function is a helper used during the preprocessing stage. Its job may look small, but it is critical to keeping the pipeline stable. When data is passed into a machine learning model, it often comes in different shapes or formats. This function makes sure that the data is reshaped, cleaned, or filled in such a way that the model can understand it.

For example, if there are missing text values, `_to_dense` ensures that they are replaced with empty strings rather than leaving them as NaN. This avoids crashes later when the model expects consistent input. Another case is when arrays are sparse; `_to_dense` converts them into dense arrays so that other parts of the pipeline can handle them. Without this helper, predictions could fail due to shape mismatches or unexpected nulls.

By isolating these fixes in a single helper function, the rest of the code remains clean. The model loading and the Streamlit app do not need to worry about these small details. Instead, they can focus on their primary tasks, while `_to_dense` silently handles edge cases. This separation of duties makes the project modular and easier to maintain.



### Deep Dive into `densify`

The `densify` function is a helper used during the preprocessing stage. Its job may look small, but it is critical to keeping the pipeline stable. When data is passed into a machine learning model, it often comes in different shapes or formats. This function makes sure that the data is reshaped, cleaned, or filled in such a way that the model can understand it.

For example, if there are missing text values, `densify` ensures that they are replaced with empty strings rather than leaving them as NaN. This avoids crashes later when the model expects consistent input. Another case is when arrays are sparse; `densify` converts them into dense arrays so that other parts of the pipeline can handle them. Without this helper, predictions could fail due to shape mismatches or unexpected nulls.

By isolating these fixes in a single helper function, the rest of the code remains clean. The model loading and the Streamlit app do not need to worry about these small details. Instead, they can focus on their primary tasks, while `densify` silently handles edge cases. This separation of duties makes the project modular and easier to maintain.



### Streamlit Form Section

The Streamlit form defines the user interface where people can enter property details. This includes fields such as property type, payment method, size in square meters, number of bedrooms, number of bathrooms, down payment, year of availability, and even descriptive text. By grouping them into a form, Streamlit ensures that all values are submitted together when the user presses the button.

Inside the form, each widget collects a specific piece of information. A dropdown may be used for property type, a slider for size, or a text input for location. This mirrors how real estate listings are usually filled, so the user feels familiar when entering values. Streamlit automatically validates the inputs, for example by forcing numeric sliders to return numbers rather than text.

Once the form is submitted, the code takes the collected values and builds a DataFrame with exactly the same column names as the training data. This alignment is crucial. If the column names differ, the model will not know how to interpret the inputs. By keeping names identical, the prediction pipeline runs smoothly without additional mapping logic.

This design also makes the app extensible. If in the future more fields are added, they can simply be included in the form and passed into the DataFrame. Streamlit handles the interface while pandas ensures the data is structured correctly.



### Prediction and Output Section

After the user submits the form, the application calls the model’s `predict` method. This is the heart of the estimator. The trained model takes the single-row DataFrame and computes an estimated price. However, raw predictions may not always be valid numbers. The code checks whether the result is NaN or infinite. If so, it defaults to zero to avoid displaying nonsensical results.

Valid predictions are then clipped to a sensible range. In this case, the maximum price is capped at 200 million EGP. This is not a statistical choice, but a pragmatic one, since property prices above that level would be unrealistic for the dataset. Clipping ensures that outputs stay grounded in reality.

The app then formats the number with commas and shows it as “Estimated Price.” To give the user more context, it also provides a range of plus or minus ten percent. This is not a strict confidence interval but a communication tool. It helps the user understand that the model’s prediction is not exact but falls within a reasonable band.

Finally, the app displays the full input values back to the user in JSON format. This creates transparency. Users can double-check that the values they entered were exactly the ones used in the calculation. If there was a mistake, they can correct the inputs and resubmit.
