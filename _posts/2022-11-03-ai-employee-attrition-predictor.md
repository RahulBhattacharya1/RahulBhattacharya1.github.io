---
layout: default
title: "Creating my Employee Attrition Predictor"
date: 2022-11-03 21:46:33
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/attrition.webp
demo_link: https://rahuls-ai-employee-attrition-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_employee_attrition_predictor
---

The idea for this project came during a moment of reflection about workplace stability. I once observed how sudden staff turnover in a department disrupted ongoing projects. That experience made me think about how valuable it would be if managers could anticipate which employees might leave. I realized a data-driven tool could help predict attrition and give leaders insights into workforce planning. That personal observation planted the seed for this application. Dataset used [here](https://www.kaggle.com/datasets/jpmiller/employee-attrition-for-healthcare).

Later I revisited that thought when working with machine learning models in practice. Predicting customer churn had always been a common example, but applying a similar concept to employee attrition seemed both practical and impactful. I imagined an interactive tool where HR or leadership teams could input employee data and instantly see a probability of attrition. This project is the result of that imagination, combining a trained machine learning pipeline with a Streamlit interface that makes predictions accessible.

---

## Files I Uploaded

When preparing this project for deployment on GitHub Pages and Streamlit Cloud, I had to upload several key files. Each file plays an important role in making the system work end-to-end.

- **app.py**: This is the main entry point of the application. It handles user inputs, loads the trained model pipeline, performs predictions, and renders results on the Streamlit interface. It contains helper functions, model artifact paths, loading logic, and the interface code.
- **requirements.txt**: This file specifies the external Python libraries required to run the project. Without it, the deployment platform would not know which packages to install. It includes Streamlit for the user interface, pandas for handling data, and joblib for loading the trained model pipeline.
- **models/attrition_pipeline.joblib**: This file contains the serialized machine learning pipeline that I trained earlier in Colab. It includes preprocessing steps and the final estimator. It is the heart of the prediction process.
- **models/defaults_row.csv**: This file stores a row of default feature values. It acts as a template for constructing new prediction inputs, ensuring all required features are present.
- **models/features.json**: This JSON file lists the features expected by the pipeline. It ensures consistency between the model training environment and the deployed app.

With these files in place, the app can read defaults, accept user inputs, align them to the correct feature set, and pass the structured row into the machine learning pipeline.

---

## The Main Application File: app.py

Below is the code that powers the interface and prediction logic.

```python
import os
import joblib
import pandas as pd
import streamlit as st

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")
st.title("Employee Attrition Predictor")
```

This first block imports the essential libraries. I use `os` for file path checks, `joblib` for loading the trained pipeline, `pandas` for handling tabular input data, and `streamlit` for building the interactive web application. Then I configure the page with a title and centered layout. The `st.set_page_config` call ensures that the application window is properly labeled when opened in the browser. Setting the title at the top gives a clear context to the user right away.

---

```python
# ----------------------------
# Helper functions
# ----------------------------
def _safe_float(x, fallback=None):
    try:
        return float(x)
    except Exception:
        return fallback

def _safe_int(x, fallback=None):
    v = _safe_float(x, fallback)
    try:
        return int(v) if v is not None else fallback
    except Exception:
        return fallback
```

These are two helper functions that make input handling more reliable. `_safe_float` attempts to convert a value to a float. If the conversion fails, it returns the fallback instead of raising an exception. This ensures that unexpected input does not break the app. `_safe_int` builds on `_safe_float`. It first tries to convert to float, then to integer. If either step fails, it again returns the fallback. These helpers are important because user inputs can sometimes be empty or invalid, and graceful handling avoids runtime crashes. They contribute to robustness of the application.

---

```python
# ----------------------------
# Artifact paths
# ----------------------------
PIPE_PATH = "models/attrition_pipeline.joblib"
DEFAULTS_PATH = "models/defaults_row.csv"
```

Here I define the paths for the model artifacts. `PIPE_PATH` points to the serialized machine learning pipeline file. `DEFAULTS_PATH` points to the CSV containing default feature values. Defining them as constants at the top makes the code easier to maintain, since I can update paths in one place if needed. These paths are later used when checking if files exist and when loading them into memory.

---

```python
# ----------------------------
# Load artifacts
# ----------------------------
if not os.path.exists(PIPE_PATH):
    st.error("Missing models/attrition_pipeline.joblib. Upload the pipeline file from Colab.")
    st.stop()

pipeline = joblib.load(PIPE_PATH)

if not os.path.exists(DEFAULTS_PATH):
    st.error("Missing models/defaults_row.csv. Upload the defaults file from Colab.")
    st.stop()

defaults_df = pd.read_csv(DEFAULTS_PATH)
```

This section ensures that critical artifacts are present before the application proceeds. The code first checks if the pipeline file exists. If it is missing, it displays an error message on the Streamlit interface and stops execution using `st.stop()`. This is necessary to prevent further steps from failing silently. If the file exists, it loads the pipeline with `joblib.load`. The same structure follows for the defaults CSV. After confirming its existence, the file is read into a pandas DataFrame. This defensive programming style guarantees that the user is informed immediately if required files are absent, rather than encountering cryptic errors later.

---

```python
# ----------------------------
# User input form
# ----------------------------
with st.form("attrition_form"):
    st.subheader("Employee Details")
    input_data = {}
    for col in defaults_df.columns:
        val = defaults_df.iloc[0][col]
        if isinstance(val, (int, float)):
            input_data[col] = st.number_input(col, value=float(val))
        else:
            input_data[col] = st.text_input(col, value=str(val))
    submitted = st.form_submit_button("Predict Attrition")
```

This block builds the input form for the user. I use a Streamlit form to group all inputs together with a submit button. Inside the form, I loop through the columns of the defaults DataFrame. For each column, I fetch the default value from the first row. If the value is numeric, I display a numeric input box initialized with that value. Otherwise, I display a text input box. This approach ensures that all required features are presented to the user in a structured way, consistent with the trained model’s expectations. The form submission button at the end triggers prediction only when the user has finished filling the inputs. This design prevents premature predictions and improves user experience.

---

```python
# ----------------------------
# Prediction
# ----------------------------
if submitted:
    row = pd.DataFrame([input_data])
    try:
        prediction = pipeline.predict(row)[0]
        prob = pipeline.predict_proba(row)[0][1]
        st.success(f"Prediction: {'Attrition' if prediction==1 else 'No Attrition'} (Probability {prob:.2f})")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
```

This final block executes when the form is submitted. A new pandas DataFrame is created from the collected input dictionary, ensuring it has the correct structure. The code then attempts to run the prediction using the loaded pipeline. It calls `predict` to get the class label, and `predict_proba` to get the probability for attrition. The result is displayed as a success message, showing whether attrition is predicted and the associated probability. If any error occurs during prediction, it is caught and displayed as an error message. This pattern of try-except makes the app resilient and informative for the user, rather than crashing silently.

---

## requirements.txt

The `requirements.txt` file is short but critical. It lists the packages that must be installed on the deployment environment.

```python
streamlit
pandas
joblib
scikit-learn
```

Each line ensures the runtime has the right library. `streamlit` is required for the interface, `pandas` for data manipulation, `joblib` for loading the model, and `scikit-learn` because the pipeline was built using its transformers and estimators. Without this file, the deployment platform would not know which versions of these packages to bring in.

---

## defaults_row.csv

This file contains one row of data representing defaults. For example:

```python
Age,JobRole,MonthlyIncome,OverTime
35,Sales Executive,5000,Yes
```

Having this template row ensures that when the app constructs the input form, every feature expected by the model has a default value. It also guides me in designing the input interface, since it reflects the data schema used during training. Without it, I might miss features or initialize them incorrectly.

---

## features.json

This file lists the model features. An example structure is:

```python
["Age", "JobRole", "MonthlyIncome", "OverTime"]
```

It enforces the ordering and completeness of features. When aligning user input with the pipeline, I can rely on this list to confirm consistency. This prevents subtle bugs that might arise if the DataFrame columns are mismatched.

---

## The Trained Pipeline File

The `attrition_pipeline.joblib` file is the serialized model pipeline. It includes preprocessing steps such as encoding categorical variables, scaling numeric variables, and the final estimator. Loading this pipeline is efficient because it encapsulates all the logic needed to transform raw input into a prediction. I trained this pipeline separately in Colab and exported it using joblib. By deploying it here, I separate the concerns of model training and model serving, which is a good practice.

---

## Deep Dive Into Helper Functions

The `_safe_float` function is more than just a utility. It reflects a defensive programming approach. In real-world data entry, values may be missing, malformed, or entered in unexpected formats. For instance, an empty text box might pass an empty string, or a user might accidentally type letters instead of numbers. Normally, calling `float()` on such input would raise a ValueError. That would interrupt execution and confuse the user.
This pattern ensures that the application behaves predictably even under imperfect conditions. The fallback value can be defined based on context, such as `None` or a default number, making it flexible. This allows downstream logic to either skip the field, substitute a mean value, or provide a user-facing warning. The idea is not only to prevent crashes but also to maintain data integrity in the pipeline.

The `_safe_int` function expands this principle by chaining two conversions. Often, values like "12.0" or "45.7" might appear. If I attempted to directly convert them to integers, they might raise errors depending on formatting. By first converting to float, I ensure a broader range of valid inputs. Then, I attempt an integer conversion. If either step fails, the fallback is again returned. This makes the function resilient to both malformed strings and floating-point values.

---

## Detailed Explanation of Artifact Paths

Defining artifact paths at the top of the code achieves several goals. First, it centralizes configuration. Instead of scattering file paths throughout the code, I keep them in one place. This reduces the chance of errors if paths need to be updated later. Second, it communicates clearly to anyone reading the code that these are core resources the app depends on. It separates the notion of configuration from logic. This is a hallmark of good coding practice because it improves maintainability.

The paths point to files inside the `models` directory. This directory is a convention in many projects for storing machine learning artifacts. Keeping model files out of the main source directory prevents clutter and makes the structure more predictable. If I ever need to replace the pipeline with a retrained version, I simply overwrite the `.joblib` file in the models folder without touching the rest of the code. Similarly, defaults and feature lists can be updated independently.

---

## Error Handling While Loading Artifacts

The section that checks for the existence of files before loading them deserves close attention. Many machine learning deployment errors occur because expected artifacts are missing. By calling `os.path.exists`, I ensure the app verifies availability before attempting to use them. If they are absent, I show a Streamlit error message with `st.error`. This not only informs the user but also logs the issue in the app session. Immediately after, I call `st.stop` which halts execution.

For the defaults file, the logic is similar. Missing defaults could lead to a form that lacks fields, which would break the alignment with the model. By stopping early, the app avoids presenting a broken interface. Once confirmed present, I use `pandas.read_csv` to load the defaults into a DataFrame. Pandas is efficient and offers easy access to column names, which I later use when constructing input widgets.

---

## Building the Input Form in Depth

The form construction step uses Streamlit’s `st.form`. This feature groups input widgets and ties them to a single submission action. Without a form, every widget change would trigger reruns of the script, which can be inefficient. By grouping, I ensure all values are collected at once and processed only when the user hits "Predict Attrition". This improves the user experience and makes the workflow smoother.

Inside the form, I iterate over columns of the defaults DataFrame. This technique creates dynamic widgets. Instead of hardcoding field names, the code adapts automatically to whatever columns the defaults file contains. This makes the interface flexible. If the model is retrained with new features, updating the defaults CSV automatically updates the interface. Each field respects the type of its default value.

This dynamic approach not only saves coding effort but also reduces the risk of inconsistencies. It ensures that the interface is always aligned with the model expectations. Using the first row of defaults as a baseline is efficient, because I do not need to define schemas manually. This integration between defaults and UI design is one of the clever structural choices in the project.

---

## Prediction Logic in Detail

The prediction step transforms user input into a DataFrame. This is necessary because most machine learning pipelines built with scikit-learn expect inputs in the form of DataFrames or arrays with feature names. By constructing a one-row DataFrame, I mimic the structure used during training. This preserves column ordering and data types, which are essential for consistency.

Once the row is built, I call `pipeline.predict`. This produces the predicted class label. Because the attrition problem is binary, the result is either 0 for no attrition or 1 for attrition. To provide more context, I also call `pipeline.predict_proba`. This returns probabilities for both classes. I select the probability corresponding to attrition, which is index 1. Displaying this value alongside the prediction increases transparency. A probability of 0.95 indicates high confidence.

I wrap this logic in a try-except block. This ensures that if the pipeline fails due to mismatched features or unexpected data, the error is caught gracefully. Instead of crashing, the app displays a Streamlit error message. This gives useful feedback and preserves usability. This structure embodies the principle of fail-safe design, which is critical for user-facing machine learning tools.

---

## Packages and Why They Matter

Each package in requirements.txt carries significance. Streamlit is the foundation of the user interface. Without it, the project would only be a command-line tool, far less accessible to non-technical users. Pandas is essential for structuring tabular data, aligning input with features, and ensuring that the pipeline receives data in the right shape. Joblib is optimized for serializing large objects such as trained pipelines. It is faster and more efficient than pickle for numerical data.

Scikit-learn underlies the entire modeling process. Even though it is not directly visible in app.py, the pipeline depends on transformers and estimators defined by scikit-learn. Without it, joblib would not be able to deserialize the pipeline. Thus, including scikit-learn in the requirements ensures that the deployment environment matches the training environment. This prevents version mismatch errors that are otherwise very common in machine learning deployment.

---

## Broader Significance of defaults_row.csv

Beyond being a template, the defaults row serves another purpose. It encodes assumptions about typical employee values. For instance, if the default age is 35 and the default role is Sales Executive, this reflects the training dataset’s distribution. Presenting these defaults as initial values reduces the effort for the user. Instead of typing everything from scratch, they can adjust only what is relevant. This speeds up interaction and makes the tool more approachable.

The defaults row also ensures that categorical values are aligned with the training set. If the model expects categories like "Yes" and "No" for overtime, initializing with "Yes" prevents users from mistakenly entering "Y" or "True". This reduces the risk of category mismatch errors. In essence, defaults_row.csv encodes not just data but also domain knowledge from the training process.

---

## Role of features.json in Consistency

The features.json file guarantees that column alignment is preserved. Machine learning pipelines are sensitive to column order and naming. A misalignment can silently degrade accuracy or completely break predictions. By maintaining a list of expected features, I enforce strict input validation. If user inputs or defaults deviate, the app can detect the mismatch.

This file is especially valuable during retraining cycles. When retraining the model with new data, I can update the features.json file to reflect the revised schema. This separates schema management from code logic. By doing so, the app remains stable even as the underlying model evolves.

---

## Why Joblib is Critical for the Pipeline

The attrition_pipeline.joblib file contains both preprocessing and the final estimator. This is important because preprocessing steps are often as significant as the model itself. If I only saved the model, I would need to manually replicate preprocessing at inference time. By saving the entire pipeline, I ensure that raw inputs are transformed in the same way as during training. This prevents subtle data leakage or mis-scaling errors.

Joblib is efficient at compressing large numpy arrays, which are common in machine learning models. Loading the pipeline at runtime is nearly instantaneous compared to rebuilding it. This makes deployment practical. It also makes sharing the model easier, since the entire preprocessing and training logic is encapsulated in one file.

---

## Reflections and Lessons Learned

Developing this project taught me the importance of modular design. Each file plays a distinct role, and together they form a coherent system. Streamlit code builds the interface, pandas structures the data, joblib loads the model, and scikit-learn ensures compatibility. The defaults and features files bridge the gap between training and deployment. This modular approach made the system easier to debug, maintain, and extend.

Another lesson was the value of defensive programming. By checking for file existence, using safe converters, and wrapping predictions in try-except blocks, I created an app that fails gracefully. This is essential when deploying to platforms where I cannot directly control user inputs or file states. It builds trust in the tool and reduces downtime.

Finally, this project reinforced the importance of transparency in machine learning. By exposing not just predictions but also probabilities, I made the system more informative. Users can interpret the output with nuance rather than treating it as a black box. This aligns with best practices in responsible AI deployment.

---

## Conclusion

This project demonstrates how to take a trained machine learning model and deploy it as a user-friendly web application. Each helper function, artifact file, and Streamlit block works together to create a seamless prediction tool. From my initial observation about workforce stability to this fully functional application, the journey shows how data science can address practical workplace challenges. By explaining every code block and file, I ensured the logic is transparent and maintainable. This detailed understanding is also valuable when presenting the project to others, whether recruiters, colleagues, or leadership teams.

---

