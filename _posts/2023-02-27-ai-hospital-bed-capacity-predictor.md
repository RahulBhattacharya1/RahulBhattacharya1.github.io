---
layout: default
title: "Building my AI Hospital Bed Capacity Predictor"
date: 2023-02-27 14:27:43
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-hospital-bed-capacity-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_hospital_bed_capacity_predictor
---

The idea for this project came from a personal moment when I read about the rising demand for hospital resources during health emergencies. I wondered how hospitals could better forecast their bed capacity needs and prepare before it was too late. That thought slowly turned into an idea of creating a simple predictive tool where anyone could estimate bed requirements with just a few inputs. I wanted to capture the urgency of resource planning while keeping the solution approachable. Dataset used [here](https://www.kaggle.com/datasets/carlosaguayo/usa-hospitals).

Another motivation came when I saw data about local hospitals being strained even outside of crisis events. That observation made me realize that a predictor is not just useful for pandemics but also for regular planning. With the right data, a model could give administrators a way to anticipate if their facilities might operate above or below average capacity. I set out to build this project, which I am now documenting here in detail.

---

## Project Files Overview

To make this project run on GitHub Pages and Streamlit Cloud, I had to upload the following files into my repository:

- **app.py**: This is the main application script written with Streamlit. It handles the entire logic from user interface to prediction output.
- **requirements.txt**: This file lists all Python dependencies required to run the application. Streamlit Cloud automatically reads it to install packages.
- **models/bed_capacity_model.joblib**: This file contains the pre-trained machine learning model serialized with joblib. It is loaded by the application at runtime.

In the following sections, I will explain each of these files in detail, focusing on `app.py` since it contains the complete logic. I will go block by block, show the code, and describe the purpose of each piece. My goal is to show not just what the code does but why each function and conditional exists and how it supports the overall predictor.

---

## Code Block 1

```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Hospital Bed Capacity Predictor", layout="centered")
st.title("Hospital Bed Capacity Predictor")
```

### Explanation

This first block brings in the external libraries that the application depends on. Streamlit is the framework that lets me write simple Python code and automatically get a clean web interface. It avoids the need for HTML, CSS, or JavaScript, which makes experimentation very fast. Pandas and numpy work together for handling structured numerical data. Pandas gives me dataframes with rows and columns, while numpy supports efficient calculations and array operations under the hood. The joblib library is specifically designed to save and load trained models quickly, especially those created with scikit-learn. Pathlib gives me an object-oriented way to handle file system paths, which is safer than writing string-based paths manually. When these libraries are combined, they create the base environment for my application: a user-facing web app backed by a trained model, with tools to load, manipulate, and prepare data in exactly the same format as during training. It is important to bring in these libraries at the top, since they will be used in almost every other part of the code that follows.

---

## Code Block 2

```python
# --- Load model ---
MODEL_PATH = Path("models/bed_capacity_model.joblib")
if not MODEL_PATH.exists():
    st.error("models/bed_capacity_model.joblib not found. Upload your model to models/ and restart.")
    st.stop()

model = joblib.load(MODEL_PATH)
```

### Explanation

Here the application defines the model path and makes sure it exists before proceeding. This block is about fail-fast behavior: instead of letting the app continue and later break with a vague error, it checks upfront. The use of Path from pathlib makes the code more portable across operating systems, since the library handles slashes and path formats. If the model file is missing, the app not only displays an error to the user but also calls `st.stop()`, which halts execution. This is necessary because otherwise Streamlit would try to continue rendering, leading to further errors. Once the file is confirmed, joblib loads it into memory exactly as it was trained. This restoration step is critical: the logic ensures the loaded model is identical to the one that was evaluated earlier. The block essentially forms the safety net and the loading mechanism, so the rest of the app can trust the model object is available.

---

## Code Block 3

```python
# --- Derive schema from trained model ---
# We assume the model was trained on one-hot encoded columns built from:
#   - numeric: POPULATION
#   - categoricals: OWNER_*, TRAUMA_*
# If you trained on more fields, this still works; only the 3 inputs are exposed in the UI here.
feature_cols = list(getattr(model, "feature_names_in_", []))
if not feature_cols:
    st.error("The trained model does not expose feature_names_in_. Retrain with a scikit-learn estimator that sets it.")
    st.stop()

owner_values = sorted([c.replace("OWNER_", "") for c in feature_cols if c.startswith("OWNER_")])
trauma_values = sorted([c.replace("TRAUMA_", "") for c in feature_cols if c.startswith("TRAUMA_")])

# If training created an 'Unknown' bucket, keep it available.
if not owner_values:
    owner_values = ["Unknown"]
if not trauma_values:
    trauma_values = ["Unknown"]
```

### Explanation

After loading the model, the app needs to confirm what features the model was trained on. Many scikit-learn models keep track of their feature names using the `feature_names_in_` attribute. By extracting this list, I can align user inputs to exactly those features, in the right order. If this attribute does not exist, it usually means the model was not trained with scikit-learn or was not stored with proper metadata. In that case, the app halts and tells the user to retrain in a compatible way. The presence of this check prevents subtle mismatches later where inputs could be misaligned, leading to wrong predictions. This block guarantees that the contract between the app and the model is clear: the model tells us what inputs it expects, and the app makes sure to provide them in that format. This is a crucial safeguard.

---

## Code Block 4

```python
# --- Inputs (driven by the model's own categories) ---
pop_min = 100
pop_max = 2_000_000
population = st.number_input("County Population", min_value=pop_min, max_value=pop_max, value=50_000, step=1_000)

owner = st.selectbox("Owner Type", owner_values, index=0)
trauma = st.selectbox("Trauma Level", trauma_values, index=0)
```

### Explanation

Categorical features like hospital ownership and trauma status need special handling. During training, these were converted into one-hot encoded dummy variables. That means each possible category became its own binary column. This block reverse-engineers those dummy column names to rebuild the actual categories for the user to pick. For example, if the model has columns OWNER_Private and OWNER_Public, then the app presents 'Private' and 'Public' as options. The code strips off the prefix so the dropdown is cleaner. Sorting ensures a stable and predictable display order. This is not just cosmetic: if categories were shown in random order, it could confuse users or create inconsistency between runs. The key contribution of this block is bridging the training-time encoding with the runtime interface, so the user can select categories in plain language while the model still gets the numeric form it expects.

---

## Code Block 5

```python
# --- Build one-hot row aligned to model columns ---
raw_row = pd.DataFrame(
    [[population, owner, trauma]],
    columns=["POPULATION", "OWNER", "TRAUMA"]
)

# One-hot only the exposed categoricals; keep everything else at 0 then set POPULATION.
# Start with all-zero frame in the trained feature order for perfect alignment.
x = pd.DataFrame([np.zeros(len(feature_cols))], columns=feature_cols, dtype=float)

# Set numeric feature(s) if present
if "POPULATION" in x.columns:
    x.loc[0, "POPULATION"] = float(population)

# Set categorical dummies if present
owner_col = f"OWNER_{owner}"
if owner_col in x.columns:
    x.loc[0, owner_col] = 1.0
elif any(c.startswith("OWNER_") for c in x.columns):
    # If user-picked owner isn't in training columns, fall back to an 'Unknown' bucket if it exists.
    unk = "OWNER_Unknown"
    if unk in x.columns:
        x.loc[0, unk] = 1.0

trauma_col = f"TRAUMA_{trauma}"
if trauma_col in x.columns:
    x.loc[0, trauma_col] = 1.0
elif any(c.startswith("TRAUMA_") for c in x.columns):
    unk = "TRAUMA_Unknown"
    if unk in x.columns:
        x.loc[0, unk] = 1.0
```

### Explanation

After loading the model, the app needs to confirm what features the model was trained on. Many scikit-learn models keep track of their feature names using the `feature_names_in_` attribute. By extracting this list, I can align user inputs to exactly those features, in the right order. If this attribute does not exist, it usually means the model was not trained with scikit-learn or was not stored with proper metadata. In that case, the app halts and tells the user to retrain in a compatible way. The presence of this check prevents subtle mismatches later where inputs could be misaligned, leading to wrong predictions. This block guarantees that the contract between the app and the model is clear: the model tells us what inputs it expects, and the app makes sure to provide them in that format. This is a crucial safeguard.

---

## Code Block 6

```python
# --- Predict ---
# If the model supports predict_proba, show probability; otherwise show class only.
try:
    proba = model.predict_proba(x)[0][1]
except Exception:
    proba = None
pred_class = int(model.predict(x)[0])

label = "High Capacity (Above Average Beds)" if pred_class == 1 else "Low Capacity (Below Average Beds)"
if proba is not None:
    st.success(f"Prediction: {label} | Probability: {proba:.2%}")
else:
    st.success(f"Prediction: {label}")
```

### Explanation

With the feature DataFrame prepared, the model is asked to predict. The code first tries to call predict_proba, which is available for classifiers that support probability outputs. If available, it gives not just the class but also the likelihood of that class. If not supported, the app just retrieves the predicted class. The predicted class is then mapped into a human-readable label: High Capacity or Low Capacity. If probability is available, the label is augmented with the percentage. This design ensures the app remains flexible with different types of classifiers. It gives the richest possible feedback when available but still works in simpler cases.

---

## Code Block 7

```python
# --- Diagnostics (helps you confirm the input is changing) ---
with st.expander("Diagnostics (for troubleshooting)"):
    st.write("Non-zero features being sent to the model:")
    nz = x.T[x.T[0] != 0.0]
    st.dataframe(nz.rename(columns={0: "value"}))
    st.caption("If changing the dropdowns or population does not change the non-zero features above, "
               "the selected category likely wasn't present in training. Use the provided options.")
```

### Explanation

The final block adds diagnostics for transparency. It uses an expander widget, which keeps details hidden by default but available on demand. Inside, it shows which features in the input vector are non-zero. This helps the user confirm that their selections are actually reflected in the model input. For example, if they pick OWNER_Private, they should see that column set to 1.0. If they do not, it indicates a mismatch between user input and training features. The caption further explains this behavior, pointing out that not all categories may have been present during training. This feature is critical for debugging and user trust, since predictions without visibility can otherwise feel like a black box.

---

## requirements.txt

```text
streamlit
pandas
scikit-learn
joblib
plotly

```

This file defines the exact dependencies needed for the app. Each entry is essential: Streamlit powers the interface, pandas and numpy process numerical and tabular data, scikit-learn represents the ecosystem in which the model was trained, joblib ensures loading and saving models is consistent, and plotly allows optional visualization if needed. This minimal but precise list is what lets Streamlit Cloud automatically build the environment. Without this file, deployments would fail or produce inconsistent results.

---

## Why Streamlit Was Chosen

I chose Streamlit because it allows me to create interactive dashboards with minimal overhead. In other frameworks, I would need to design HTML pages, write JavaScript for interactivity, and style everything with CSS. Streamlit handles these automatically, which means I can focus purely on the logic of the model and the data. This decision reduced development time and made the project easier to share. Streamlit is also highly compatible with cloud deployment, which is why I could publish the app with almost no extra work.

## Handling Categorical Data

The biggest challenge in real-world prediction systems is managing categorical data. When categories are encoded as dummy variables, any mismatch between training and prediction time can cause errors. This is why I added explicit checks for unknown categories and built fallbacks. Without this, the app could crash or silently misclassify inputs. Handling categories with care not only stabilizes the system but also ensures fairness, because it avoids completely rejecting unseen groups.

## Importance of requirements.txt

Listing dependencies in requirements.txt may seem minor, but it is the backbone of reproducibility. If one library is missing or uses a different version, the behavior of the app can change. For example, if the wrong version of scikit-learn is used, the model may not load. By pinning or at least listing exact packages, I reduce the risk of such failures. This file is what makes the difference between an app that only works on my laptop and one that works anywhere.

## Future Improvements

While the current version works well, I see room for improvements. One direction is adding visualization of predicted bed usage trends with plotly, so administrators can see graphs instead of just labels. Another idea is to extend the input form with more features, like hospital location or seasonal demand patterns. Finally, I could retrain the model on larger datasets to improve accuracy. These changes would make the predictor even more practical in real-world settings.

## Reflection on Model Robustness

Machine learning models are only as good as the data they are trained on. I designed this application to be transparent because no prediction should be accepted blindly. The diagnostic panel shows which features are being used, which builds trust. This also encourages retraining when users notice categories missing. By exposing the inner mechanics slightly, the app reminds us that predictive systems must be maintained and monitored, not just deployed once.

---



## Role of Each Dependency

- **Streamlit**: Beyond just building forms, Streamlit provides caching, layout customization, and session state management. In this project, I used only the basics, but these advanced features would let me scale the app to larger use cases. Streamlit also simplifies deployment, which makes it an excellent choice for sharing prototypes publicly.
- **Pandas**: This library provides a dataframe structure, which is the closest thing Python has to a spreadsheet. It lets me keep columns named, so they align naturally with the model’s feature names. That alignment is the reason predictions are consistent. Pandas also integrates well with Streamlit, so dataframes can be displayed directly.
- **NumPy**: Pandas relies heavily on numpy internally, but I also use it directly for quick array manipulations. In predictive modeling, numerical stability matters, and numpy ensures efficient operations on large arrays.
- **Joblib**: Without joblib, loading a scikit-learn model could be unreliable. Joblib serializes the model with all its internal structure, so when I reload it, I get the same coefficients, feature encodings, and pipeline structure as during training. This consistency is crucial for production-like environments.
- **Plotly**: While not heavily used in this version, plotly provides interactive charts. I included it in requirements to allow for easy extension. With plotly, users could get time-series plots, probability histograms, or interactive exploration of predictions.

## Example User Scenarios

Scenario 1: A hospital administrator wants to know if their planned expansion will move them into high capacity. They input a larger population figure, select their ownership type, and review the output. The app gives an immediate prediction, letting them test assumptions interactively.

Scenario 2: A policy analyst wants to compare public versus private hospitals under the same conditions. They keep population constant and just switch the dropdown for ownership. The app updates predictions instantly, showing how capacity expectations differ by ownership type.

Scenario 3: A researcher wants to validate how trauma designation impacts capacity. By toggling between trauma levels, they can see whether the model predicts systematic differences. This helps evaluate whether trauma status is a significant predictor in the dataset.

## Deployment Considerations

When deploying on Streamlit Cloud, the platform automatically reads requirements.txt and sets up the environment. However, large model files can be an issue, which is why I kept the joblib model lightweight. Another consideration is that the repository must contain all files in the right structure: models in a subfolder, app.py at the root, and requirements.txt alongside it. These structural details matter because missing files cause runtime errors. By documenting this carefully, I ensure that anyone can clone and deploy without surprises.

## Conclusion

The Hospital Bed Capacity Predictor demonstrates how a compact but complete pipeline can be implemented. From importing libraries to carefully handling user input, from validating model compatibility to providing predictions with transparency, every section of code has a purpose. By writing the application with Streamlit, I ensured the interface was approachable, and by focusing on defensive programming, I ensured the predictions were reliable. This blog post has walked through every code block, explained its role, and shown how the pieces connect to form a coherent application. The design is not just about predicting a label but about showing the pathway from input to output clearly. That clarity is what makes the project both technically sound and practically useful.
