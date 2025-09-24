---
layout: default
title: "Creating my AI Cancer Severity Predictor"
date: 2022-11-16 14:27:32
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-cancer-severity-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_cancer_severity_predictor
featured: true
---

There was a day when I was reading about how people face uncertainty
during medical treatment. The numbers that are given to them often sound
abstract. I thought about how difficult it must be to get even a rough
estimate of how severe a condition may turn out. This personal thought
made me imagine that if I could build a simple tool, it would let people
experiment with factors and see how the risks could change. That was the
spark for me to try creating a cancer severity predictor. It started
from a curiosity about how machine learning could bring a sense of
clarity to complicated medical problems. Dataset used [here](https://www.kaggle.com/datasets/zahidmughal2343/global-cancer-patients-2015-2024).

Another reason I wanted to try this project was my growing practice of
turning every learning into a working demo. It was not enough for me to
read about prediction models in theory. I wanted to package it into
something interactive that could live on a webpage. This way, anyone
could enter values and get predictions. That small interaction makes the
concept far more real. Building such an app also made me think through
the entire cycle of machine learning deployment. From training, to
saving artifacts, to presenting them in an application, the steps became
clear and practical once I did them myself.

I also wanted to test how well a machine learning model could generalize
from simple features such as lifestyle risks and cancer stages. While
the model is not intended for medical advice, it shows how predictive
systems can be structured. By walking through the full workflow, I not
only learned about technical challenges but also about the importance of
transparency and clear communication in presenting outputs to end users.

------------------------------------------------------------------------

## Project Files Overview

The project has four important components that I had to upload into my
GitHub repository. Each of these files plays a distinct role in ensuring
that the app runs smoothly from end to end.

1.  **app.py**: This is the Streamlit application script. It handles the
    user interface, input collection, feature alignment, and prediction
    process.\
2.  **requirements.txt**: This file contains all the Python dependencies
    needed for the app to run. Streamlit Cloud uses this file to install
    packages.\
3.  **models/severity_model.joblib**: This is the trained machine
    learning model saved using joblib. It stores the weights and
    structure learned during training.\
4.  **models/feature_list.json**: This JSON file stores the exact list
    of features the model expects. It ensures that the input data aligns
    correctly with the model schema.

### Why These Files Matter

Each file represents a layer of the deployment process. The application
file defines the logic and user flow. The requirements file guarantees
consistency across environments. The trained model acts as the
intelligence powering the predictions. The feature list locks the
structure, making sure the input always matches training expectations.
Without one of these files, the system would either fail to run or
produce incorrect results.

------------------------------------------------------------------------

## requirements.txt

This file tells Streamlit Cloud and any other environment which exact
versions of libraries are required.

``` python
numpy==2.0.2
pandas==2.2.2
scikit-learn==1.6.1
scipy==1.16.2
joblib==1.5.2
streamlit==1.38.0
```

Each package plays a role. Numpy and pandas handle data operations.
Scikit-learn and scipy support the model training and math functions.
Joblib allows saving and loading models. Streamlit is the web framework.
Fixing versions avoids errors. If one of these versions drifts, then
functions may change subtly and cause incompatibility with saved model
artifacts.

I learned that pinning exact versions is important in machine learning
projects because small changes in libraries can affect serialization
formats or default behaviors. For example, a model saved with one
version of scikit-learn may not load correctly with another. By locking
the versions, I can ensure reproducibility.

------------------------------------------------------------------------

## feature_list.json

This file lists all the features expected by the model. Keeping this
list ensures alignment.

``` python
["Age", "Gender", "Country_Region", "Year", "Genetic_Risk", "Air_Pollution", "Alcohol_Use", "Smoking", "Obesity_Level", "Cancer_Type", "Cancer_Stage"]
```

The file shows fields like Age, Gender, Country_Region, and
Cancer_Stage. Each of them was present during model training. When I
build the app, I must make sure input matches this list. Otherwise, the
model would reject the data or give wrong results.

I realized how fragile models can be when feature alignment is not
enforced. Even if the training set included the right columns, if the
app sends them in a different order or misses one, the model might
either crash or silently misinterpret values. This JSON file acts like a
contract that the app must follow, protecting the model from bad inputs.

------------------------------------------------------------------------

## app.py Full Script

``` python
import streamlit as st
import pandas as pd
import joblib, json
from pathlib import Path

st.set_page_config(page_title="Cancer Severity Predictor", layout="centered")
st.title("Cancer Severity Predictor")

MODEL_PATH = Path("models/severity_model.joblib")
FEATS_PATH = Path("models/feature_list.json")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(FEATS_PATH) as f:
        features = json.load(f)
    return model, features

try:
    model, FEATURE_LIST = load_artifacts()
except Exception as e:
    st.error("Failed to load model/artifacts. Ensure requirements.txt matches training versions and both files exist in /models.")
    st.exception(e)
    st.stop()

# ---- UI ----
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 1, 110, 50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    country = st.text_input("Country_Region", value="USA")
    year = st.number_input("Year", min_value=2015, max_value=2024, value=2020)
    cancer_type = st.selectbox("Cancer_Type", ["Lung", "Breast", "Colon", "Skin", "Leukemia", "Prostate", "Ovary", "Other"])
with col2:
    cancer_stage = st.selectbox("Cancer_Stage", ["Stage 0", "Stage I", "Stage II", "Stage III", "Stage IV"])
    genetic_risk = st.slider("Genetic_Risk", 0.0, 10.0, 5.0, 0.1)
    air_pollution = st.slider("Air_Pollution", 0.0, 10.0, 3.0, 0.1)
    alcohol_use = st.slider("Alcohol_Use", 0.0, 10.0, 2.0, 0.1)
    smoking = st.slider("Smoking", 0.0, 10.0, 2.0, 0.1)
    obesity_level = st.slider("Obesity_Level", 0.0, 10.0, 3.0, 0.1)

# Only predictor columns (no outcomes)
input_row = {
    "Age": age,
    "Gender": gender,
    "Country_Region": country,
    "Year": int(year),
    "Genetic_Risk": float(genetic_risk),
    "Air_Pollution": float(air_pollution),
    "Alcohol_Use": float(alcohol_use),
    "Smoking": float(smoking),
    "Obesity_Level": float(obesity_level),
    "Cancer_Type": cancer_type,
    "Cancer_Stage": cancer_stage,
}

raw_df = pd.DataFrame([input_row])

# Align to training features exactly
def align_columns(df, feature_list):
    df = df.copy()
    # Add missing with sensible defaults
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0
    # Drop extras not seen in training
    df = df[feature_list]
    return df

X_input = align_columns(raw_df, FEATURE_LIST)

st.subheader("Features sent to model")
st.dataframe(X_input)

if st.button("Predict Severity Score"):
    try:
        pred = model.predict(X_input)[0]
        st.success(f"Predicted Target Severity Score: {pred:.2f}")
    except Exception as e:
        st.error("Prediction failed. Verify the model and feature schema.")
        st.exception(e)
```

------------------------------------------------------------------------

## Explanation of app.py Sections

### Setup and Configuration

The script starts with importing all the required libraries. Streamlit
provides the web interface, pandas handles the tabular input, and joblib
loads the trained model. I also used the `Path` object to keep paths
clean and reliable across environments.

### Model Loading

The function `load_artifacts` is wrapped with caching so that the model
and feature list are loaded only once. This is important in a web app
where the script may rerun frequently. Without caching, every refresh
would reload the model from disk, slowing down performance.

### Error Handling

I wrapped the loading process in a try-except block. If the model file
or the feature list is missing, the app stops and shows a clear error
message. This ensures users are not left with silent failures.

### User Interface

I created two columns using Streamlit so the input widgets appear neatly
side by side. Inputs like age, gender, and country are collected in the
first column, while cancer stage and lifestyle risk factors are in the
second. This layout makes the app more user friendly.

### Input Preparation

The helper function `align_columns` ensures that the input DataFrame
exactly matches the feature schema used in training. It adds missing
columns with safe defaults and removes any extra fields. This prevents
schema mismatch errors during prediction.

### Prediction

When the "Predict Severity Score" button is clicked, the aligned input
is passed to the model. The prediction result is displayed with
formatting. If an error occurs during prediction, it is caught and shown
to the user along with the exception details.

------------------------------------------------------------------------

## Detailed Breakdown of Code Blocks

### Streamlit Configuration

The very first part of the script sets the Streamlit page configuration
and title. By using `st.set_page_config`, I defined the title of the
browser tab and also chose a layout style. These small details improve
user experience because they make the app look polished. Without this
step, the page would use a default title and a narrow layout, which may
feel unprofessional.

### Path Definitions

I declared constants for the model path and feature list path. This step
ensures that the rest of the script does not rely on hardcoded strings
scattered around. If I need to move the files into another folder, I
only update the constant. This separation of configuration from logic
makes the script easier to maintain.

### The load_artifacts Function

This function loads both the model and the feature list from disk. I
added `@st.cache_resource` so that Streamlit keeps them in memory once
they are loaded. If I did not cache them, every change in input would
reload the model again, causing delays. This helper acts like a
gatekeeper, ensuring that the heavy resources are fetched once and then
reused.

Inside the function, joblib loads the machine learning model. Then, the
feature list is read from JSON. The two are returned as a pair. I liked
this design because it groups related artifacts, and I can easily expand
it later if I want to add more metadata.

### Error Handling During Load

The try-except block captures any errors that occur while loading
artifacts. If the files are missing, Streamlit shows a descriptive error
message and the script stops. This is critical for debugging. Instead of
guessing why predictions are failing, I can see exactly which file
caused the problem.

### User Input Collection

I structured the user interface into two columns. In the first column,
demographic information and general context are captured: age, gender,
country, year, and cancer type. In the second column, I focused on
medical stage and lifestyle risk factors. This separation was
intentional. It makes the UI balanced and reduces clutter.

Each widget corresponds to a feature. For example, sliders are used for
continuous variables such as genetic risk and air pollution. Drop-down
select boxes are used for categorical fields such as cancer type or
stage. Text input allows free-form entry of country. This mapping of
widget type to data type ensures that the collected values are valid.

### Constructing the Input Row

After collecting inputs, I combined them into a dictionary called
`input_row`. This dictionary maps feature names to values. Then I
converted it into a pandas DataFrame. Using pandas here gives me
flexibility to manipulate data further if needed. It also matches the
data structure expected by scikit-learn models.

### The align_columns Helper

This helper is one of the most important parts. Its job is to make sure
the input DataFrame matches the training schema. It loops through each
feature in the stored list. If a column is missing, it creates one with
a default value of zero. If extra columns are present, they are dropped.
Finally, the DataFrame is reordered to exactly match the order of
features used in training.

This function prevents subtle bugs. Models in scikit-learn assume that
columns are in the same order as during training. If they are
misaligned, the model may treat gender as age or smoking as obesity.
Such mismatches would completely ruin predictions. By aligning columns
strictly, I enforce consistency.

### Displaying the Features Sent to the Model

I chose to display the aligned DataFrame back to the user using
`st.dataframe`. This gives transparency. Users can see exactly what is
being passed into the model. If they made a mistake in input, they can
correct it before running prediction. Transparency in machine learning
is vital, especially when building trust with end users.

### Prediction Button and Inference

Finally, the app checks if the user clicked the "Predict Severity Score"
button. If clicked, the model processes the aligned DataFrame and
returns a prediction. I format the output to two decimal places and
display it as a success message. If any error occurs, Streamlit shows an
error message along with the exception stack trace.

This design choice gives clear feedback. The user knows whether the
prediction was successful and, if not, what went wrong.

------------------------------------------------------------------------

## Reflection on Deployment

Deploying the app taught me about environment consistency, debugging
deployment errors, and designing a user interface. The simplicity of
Streamlit made the project possible in a short time, but the hidden
complexity was in aligning the machine learning artifacts. I had to
think carefully about what files to include, how to structure folders,
and how to ensure reproducibility.

By walking through the project in this detailed way, I realized that
deployment is just as important as training a model. A model that cannot
be served reliably is of little use. The careful error handling,
alignment checks, and explicit requirements file all worked together to
create a smooth deployment.

------------------------------------------------------------------------

## Future Improvements

1.  **Better Visualizations**: Instead of only showing numbers, I could
    plot severity scores on a chart or provide comparisons.\
2.  **Expanded Feature Set**: Adding more lifestyle or genetic factors
    could improve accuracy.\
3.  **Model Monitoring**: In a real-world setting, monitoring
    predictions for drift would be necessary.\
4.  **Internationalization**: Supporting multiple languages for the
    interface would expand accessibility.\
5.  **Security and Privacy**: A medical-related app must take care of
    how data is handled, even if it is only for demonstration.

## Lessons Learned

1.  **Importance of Feature Alignment**: Even a single mismatched
    feature can break predictions. Having a stored feature list protects
    against this.\
2.  **Version Pinning**: Library versions must be pinned to guarantee
    consistent behavior between training and deployment.\
3.  **Interactive Interfaces**: Allowing users to input values directly
    makes machine learning more approachable.\
4.  **Error Transparency**: Showing detailed error messages helps debug
    issues quickly when deploying models.\
5.  **End-to-End Thinking**: Building the app forced me to think not
    only about training models but also about packaging, serving, and
    presenting them.

------------------------------------------------------------------------

## Closing Thoughts

This project was not about building a production-grade medical tool but
about practicing the entire lifecycle of a machine learning project. I
took an idea, trained a model, saved it, built a front-end interface,
and deployed it as a working demo. The process made concepts concrete
and gave me a repeatable template for other projects. Each file in the
repository plays a role, and together they create an experience that
users can interact with.

By cleaning up this workflow and documenting it thoroughly, I not only
reinforced my own understanding but also created a guide I can share
publicly. It shows recruiters, collaborators, or anyone curious how I
translate abstract concepts into working systems.

------------------------------------------------------------------------

## Conclusion

This cancer severity predictor was my way of converting theory into
practice. It demonstrates the lifecycle of a machine learning project in
a contained and reproducible format. From data preparation to model
training, from saving artifacts to aligning features, and finally from
deploying an interface to testing user inputs, each step contributed to
the final outcome. The process was challenging, but by breaking it down
into files, functions, and helpers, I created a tool that can both
educate and inspire.
