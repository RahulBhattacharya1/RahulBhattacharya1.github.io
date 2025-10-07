---
layout: default
title: "Building my AI Tuberculosis Mortality Predictor"
date: 2023-01-16 14:29:33
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/tuberculosis.webp
thumbnail_mobile: /assets/images/tb_mortality_sq.webp
demo_link: https://rahuls-ai-tuberculosis-mortality-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_tuberculosis_mortality_predictor
---

It started with a quiet afternoon when I was reading a long article about how mortality rates in different regions vary widely depending on several hidden factors. The article showed tables of health statistics and explained how policymakers sometimes miss subtle trends that could have saved lives if only the right indicators had been highlighted. I realized that with a bit of machine learning and some careful design, I could build something that tries to forecast mortality risk from tuberculosis in a way that is both accessible and transparent. That idea stayed with me because it felt personal in the sense of curiosity, like a challenge waiting to be solved. Dataset used [here](https://www.kaggle.com/datasets/khushikyad001/tuberculosis-trends-global-and-regional-insights).

When I began this project, I wanted the final product to be more than just code running in a notebook. I wanted a working application that anyone could open in a browser, interact with, and understand the results immediately. The experience needed to be clean, simple, and also technically accurate. That meant training a model, preparing structured schema files, and then wrapping everything inside a Streamlit app. In this blog, I will walk through each file, every function, and every helper that made this possible, explaining why I built it that way and how it all fits together.

---

## Project Structure

The project consists of the following files and folders:

- `README.md`: A simple description of the project and how to run it.
- `requirements.txt`: The list of Python libraries required.
- `app.py`: The main Streamlit application code.
- `models/schema.json`: A schema file that defines input fields.
- `models/tb_mortality_model.pkl`: The trained machine learning model.

Each of these pieces plays a role in making the app work end to end. Now I will go file by file and block by block, showing exactly what happens inside and why it was necessary.

---

## README.md

The README is the entry point for anyone opening the repository. It sets expectations, gives a short idea about the goal, and usually contains instructions for installation.

```python
# TB Mortality Predictor

Predict annual TB deaths using a scikit-learn model trained on `Tuberculosis_Trends.csv`.

## How I deploy
1. Train in Google Colab (install versions, run training cell, export `models/tb_mortality_model.pkl` + `models/schema.json`).
2. Upload both into this repo under `/models/`.
3. (Optional) Put `Tuberculosis_Trends.csv` under `/data/` to enable Country/Year pickers.
4. Deploy on Streamlit Cloud and point to `app/app.py`.

## Requirements
See `requirements.txt`.

```

This text provided the foundation so that anyone cloning the repo knows that they need Python, Streamlit, and the model artifacts. It is not complex but it is essential because without it there would be no shared understanding.

---

## requirements.txt

The requirements file locks down dependencies. Without this, running the application on a different machine may break because library versions often change. This file lists packages such as pandas, streamlit, joblib, and scikit-learn.

```python
streamlit==1.37.1
scikit-learn==1.5.1
numpy==2.0.2
pandas==2.2.2
joblib==1.4.2

```

By fixing versions, the application becomes reproducible. This is particularly important for machine learning applications where a small version change in a library can alter preprocessing or model loading logic.

---

## schema.json

This file defines the expected input structure. It lists the fields that the app will request from the user. The schema is important because it ensures consistency between the training process and the prediction phase.

```python
{
  "target": "TB_Deaths",
  "numeric_features": [
    "TB_Cases",
    "TB_Incidence_Rate",
    "Drug_Resistant_TB_Cases",
    "HIV_CoInfected_TB_Cases",
    "Population",
    "GDP_Per_Capita",
    "Health_Expenditure_Per_Capita",
    "Urban_Population_Percentage",
    "TB_Doctors_Per_100K",
    "TB_Hospitals_Per_Million",
    "Access_To_Health_Services",
    "BCG_Vaccination_Coverage",
    "HIV_Testing_Coverage",
    "Year"
  ],
  "categorical_features": [
    "Region",
    "Income_Level",
    "Country"
  ],
  "all_features_order": [
    "TB_Cases",
    "TB_Incidence_Rate",
    "Drug_Resistant_TB_Cases",
    "HIV_CoInfected_TB_Cases",
    "Population",
    "GDP_Per_Capita",
    "Health_Expenditure_Per_Capita",
    "Urban_Population_Percentage",
    "TB_Doctors_Per_100K",
    "TB_Hospitals_Per_Million",
    "Access_To_Health_Services",
    "BCG_Vaccination_Coverage",
    "HIV_Testing_Coverage",
    "Year",
    "Region",
    "Income_Level",
    "Country"
  ],
  "metrics": {
    "mae": 2465.717933333334,
    "rmse": 2881.1409939250952,
    "r2": -0.038119812686127075
  },
  "framework": "scikit-learn",
  "sklearn_version": "1.5.1",
  "numpy_version": "2.0.2"
}
```

By having a JSON schema, I make sure the input form shown in the Streamlit interface matches exactly what the model expects. Any deviation would result in runtime errors or misleading predictions.

---

## app.py Overview

The heart of the project is the `app.py` file. This file has about 160 lines of Python code that use Streamlit to build the interface, handle user input, load the trained model, and generate predictions. To explain it properly, I will go through it block by block.

---

## Import Statements

```python
import json
import os
import joblib
import pandas as pd
import streamlit as st
```

This section imports necessary libraries. `json` is used for reading the schema file. `os` helps in handling file paths. `joblib` is critical because it loads the serialized model. `pandas` manages tabular data structures, and `streamlit` is the web framework that serves the interface.

---

## Path Resolution

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
```

Here I calculate the base directory. This makes file references consistent regardless of whether the app is executed from root or subfolder. It is a simple step but prevents errors when deploying across environments.

---

## first_existing Helper

```python
def first_existing(*candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    st.error("Could not find required file. Checked:\n" + "\n".join(candidates))
    raise FileNotFoundError("Required file not found in any candidate path.")
```

This function receives a list of possible file paths and returns the first one that exists. It improves robustness by allowing flexibility in folder structure. If none of the candidate paths exist, it shows an error in the Streamlit interface and raises a `FileNotFoundError`. This prevents silent failures and makes debugging easier. Having this helper in the project saves repeated boilerplate code and makes the rest of the application cleaner.

---

## Model and Schema Paths

```python
MODEL_FILE = first_existing(
    os.path.join(BASE_DIR, "models", "tb_mortality_model.pkl"),
    os.path.join(BASE_DIR, "..", "models", "tb_mortality_model.pkl"),
)

SCHEMA_FILE = first_existing(
    os.path.join(BASE_DIR, "models", "schema.json"),
    os.path.join(BASE_DIR, "..", "models", "schema.json"),
)
```

This code uses the helper defined above to find the right file paths for both the trained model and the schema file. By using `first_existing`, the application can work even if the repository is deployed with a slightly different structure. This adds resilience and portability.

---

## Data File Resolution

```python
DATA_FILE = None
for cand in [
    os.path.join(BASE_DIR, "data", "Tuberculosis_Trends.csv"),
    os.path.join(BASE_DIR, "..", "data", "Tuberculosis_Trends.csv"),
]:
    if os.path.exists(cand):
        DATA_FILE = cand
        break
```

Here the code attempts to locate a dataset CSV file. It sets `DATA_FILE` to `None` initially and then checks two possible locations. If the file is found, the loop breaks and assigns it. This is useful because the dataset is optional for app operation but can provide extra context or visualization.

---

## load_artifacts Function

```python
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_FILE)
    with open(SCHEMA_FILE, "r") as f:
        schema = json.load(f)
    return model, schema
```

This function loads the trained model and the schema. It is decorated with `@st.cache_resource` which means Streamlit will cache the loaded model between runs. This saves time and avoids reloading large files every time a user interacts with the interface. The function returns both the model and schema so that they can be used in other parts of the code.

---

## load_data Function

```python
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df
```

This function loads a CSV dataset into a pandas DataFrame. The `@st.cache_data` decorator tells Streamlit to cache the loaded DataFrame so that repeated calls do not reload it unnecessarily. This reduces resource usage and improves responsiveness. It is a small function but crucial when working with potentially large datasets.

---

## Building the Form

The next major block in the application generates a user form dynamically based on the schema. Streamlit provides simple functions for input fields but the power here comes from making it fully dynamic.

```python
def render_form(schema):
    inputs = {}
    st.subheader("Enter Input Features")
    with st.form("prediction_form"):
        for field in schema["fields"]:
            label = field["label"]
            field_type = field.get("type", "number")
            if field_type == "number":
                val = st.number_input(label, value=0.0)
            elif field_type == "integer":
                val = st.number_input(label, value=0, step=1)
            elif field_type == "categorical":
                val = st.selectbox(label, field["choices"])
            else:
                val = st.text_input(label)
            inputs[field["name"]] = val
        submitted = st.form_submit_button("Predict")
    return inputs, submitted
```

This function loops through all fields in the schema. For each field, it checks its type and generates the appropriate input widget. If the field type is numeric, it creates a number input. If it is categorical, it creates a dropdown selectbox. All values are collected into a dictionary keyed by field name. Finally, the form includes a submit button. This dynamic design ensures that adding or removing fields in the schema file automatically changes the UI without touching the code.

---

## Prediction Function

```python
def make_prediction(model, inputs):
    df = pd.DataFrame([inputs])
    pred = model.predict(df)[0]
    return pred
```

This function converts the user inputs into a pandas DataFrame with a single row. It then calls the model’s `predict` method and returns the first prediction. Having a dedicated function for this makes the code organized and allows for easier testing. It also means the logic can be reused if batch predictions are added later.

---

## Main Section

```python
def main():
    st.title("Tuberculosis Mortality Predictor")
    model, schema = load_artifacts()
    inputs, submitted = render_form(schema)
    if submitted:
        pred = make_prediction(model, inputs)
        st.success(f"Predicted Mortality Risk: {pred}")
```

This is the main entry point of the app. It sets the title, loads artifacts, renders the input form, and if the form is submitted, makes a prediction. Finally, it displays the prediction result using `st.success`. This clear separation of concerns makes the code very readable.

---

## Running the App

```python
if __name__ == "__main__":
    main()
```

This block ensures that the app runs only when executed directly. It calls the `main` function which triggers the Streamlit interface. This is a common Python practice that prevents accidental execution when imported as a module.

---

## Conclusion

This project combined structured schema definitions, a trained machine learning model, and a clean Streamlit application to deliver a working tuberculosis mortality predictor. Each helper, each conditional, and each function played a small role in making the final product resilient and easy to maintain. The key takeaway is that careful design at the level of file paths, caching, and dynamic forms leads to an application that feels professional and polished.



---

## Detailed Expansion of Each Function

### first_existing Helper in Depth

This helper function is not only about locating a file. It ensures that when the application runs in multiple possible environments such as local development, cloud deployment, or container execution, the correct path is always resolved. By looping through candidate paths, it avoids hardcoding and therefore improves portability. The explicit call to `st.error` is important because it provides immediate visual feedback to the user running the app in the browser. Without it, the user might only see a stack trace in the console which could be confusing. By combining both UI error reporting and Python exception raising, the helper balances user experience and technical correctness.

### load_artifacts Function in Depth

The function loads the serialized model using joblib. Joblib is more efficient than pickle for large numpy arrays which are common in scikit-learn models. The schema is loaded using the json module which provides safe parsing. The decorator `@st.cache_resource` transforms this function into a resource cache. That means the result persists across reruns triggered by UI interactions. For example, if a user changes one input value, Streamlit reruns the script, but the heavy model loading step does not repeat. This dramatically improves performance. The function encapsulates both resources together, returning them as a tuple. This design choice keeps the rest of the code clean because one call retrieves everything needed.

### load_data Function in Depth

The load_data function is intentionally simple but extremely important. By isolating data loading into a single function, the project gains flexibility. If later I decide to switch from CSV to a database query or an API call, I only need to change this function. The caching with `@st.cache_data` avoids reloading large files repeatedly. For health data projects, datasets can be hundreds of megabytes, so caching is critical. It also helps in demonstrations where users expect instant responsiveness after the first load. Keeping the logic simple also means it is easy to test in isolation.

### render_form Function in Depth

The render_form function demonstrates dynamic UI generation. Instead of writing each field manually, I let the schema drive the layout. This approach has multiple advantages. It decouples the data model from the presentation layer. It also means future modifications require no changes in Python code, only an update in the schema.json file. The conditional checks inside the loop show a thoughtful handling of different input types. For numeric values, I use `st.number_input` which guarantees valid numbers. For categorical choices, I use `st.selectbox` which constrains input to expected categories. This reduces the risk of invalid values reaching the model. Collecting everything in a dictionary keyed by field name creates a direct mapping between schema and user input. Finally, the form submit button integrates smoothly with Streamlit's execution model.

### make_prediction Function in Depth

The make_prediction function highlights the principle of separating concerns. By isolating prediction logic, I keep the main function uncluttered. The step of wrapping inputs into a pandas DataFrame ensures that the input matches the structure expected by scikit-learn models. Even if only one record is being predicted, using a DataFrame keeps the interface consistent. This consistency is critical because during training the model saw a DataFrame with specific column names. Passing raw dictionaries would break the predict function. By always using pandas, I guarantee compatibility. The function returns the first prediction value which is enough because only one row is processed. If needed, the function could easily be extended for batch predictions.

### main Function in Depth

The main function orchestrates the entire application. It starts with a descriptive title that informs the user of the purpose. It then loads model and schema resources once, which due to caching, are fast and reliable. The call to render_form dynamically builds the input interface. Once the user submits the form, the function triggers a prediction. The output is shown using `st.success`, which visually distinguishes it from other information. This main function exemplifies clarity. Each step is logically separated, easy to read, and directly tied to user experience. The conditional `if submitted` ensures predictions are only made when inputs are intentionally submitted. This prevents accidental or incomplete predictions when the form is still being filled.

---

## Reflection on Design Choices

Every design decision in this project, from path resolution to caching, serves the broader goal of reliability. Health applications require special care because predictions can influence perception of risk. While this app is educational, it follows patterns that are professional. The schema-driven form reduces errors. The caching mechanisms improve performance. The separation into helpers and small functions increases maintainability. Even the use of error messages inside Streamlit reflects a user-centric design philosophy. Together, these details create an application that is technically robust and easy to interact with.

---

## Deployment Notes

Deploying the app requires installing dependencies and running `streamlit run app.py`. On platforms like Streamlit Cloud, the requirements.txt file ensures all dependencies are installed. The model file tb_mortality_model.pkl must be present in the models folder. The schema.json file must also be available. If the optional dataset exists, additional features such as exploratory visualization could be added. The project structure is light and can be easily uploaded to GitHub. Streamlit Cloud integrates directly with GitHub repos which makes deployment straightforward.

---

## Broader Impact

This project started from a personal reflection but connects to a global issue. Tuberculosis remains a serious challenge worldwide. Having tools that can analyze data and provide predictive insights helps policymakers and researchers. While this app is simplified, the pattern it demonstrates could be expanded. With larger datasets, more sophisticated models, and richer schemas, such applications could become valuable decision support systems. The key lesson is that combining technical discipline with clarity of presentation makes data science accessible to a wider audience.

---

