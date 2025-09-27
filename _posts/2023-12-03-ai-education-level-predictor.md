---
layout: default
title: "Building my AI Education Level Predictor"
date: 2023-12-03 10:19:25
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-education-level-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_education_level_predictor
featured: true
---

When I first thought about creating this project, it came from an observation about how demographic information often relates to education outcomes. I noticed that data such as sex, age group, and geography can influence the distribution of education levels in a population. That made me curious about whether I could create a small predictor that demonstrates this relationship in practice. This curiosity was enough to push me into building a compact Streamlit app that loads a trained pipeline and makes predictions. Dataset used [here](https://www.kaggle.com/datasets/gpreda/population-by-education-level-in-europe).

The inspiration also came from seeing how many projects stop at training a model but never show how to deploy it for interactive use. I wanted a working demonstration where someone could enter basic values and immediately get back a prediction. The experience of taking a trained pipeline and packaging it inside a simple user interface taught me many lessons about structuring a project. This blog explains each part in detail, covering the requirements file, the application script, and the serialized pipeline.

## Requirements File

This file defines the exact Python packages needed.

```python
streamlit==1.38.0
scikit-learn==1.6.1
pandas==2.2.2
numpy==2.0.2
scipy==1.16.1
joblib==1.5.2
packaging
```


- **Streamlit** builds the web UI.
- **Scikit-learn** supports the pipeline.
- **Pandas** and **Numpy** handle tabular inputs.
- **SciPy** is a dependency for algorithms.
- **Joblib** loads the trained model.
- **Packaging** helps manage library versions.


## Application Script: app.py


```python
import streamlit as st
import pandas as pd
import joblib
```


The imports bring in Streamlit for UI, Pandas for data manipulation, and Joblib for model loading.


```python
st.title("Education Level Predictor (Compact Model)")
```


This line sets the title displayed at the top of the web page.


```python
# Choose which model file you uploaded
MODEL_PATH = "models/edu_pipeline.joblib"  # or "edu_pipeline_sgd.joblib"
```


This variable defines where the serialized model is stored. It centralizes the configuration.


```python
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)
```


This variable defines where the serialized model is stored. It centralizes the configuration.


```python
pipe = load_model()
```


This initializes the pipeline once and stores it for later predictions.


```python
st.subheader("Input Features")
sex = st.selectbox("Sex", ["M", "F", "T"])
age = st.text_input("Age group (e.g., Y20-24)", "Y20-24")
country = st.text_input("Country code (e.g., AT)", "AT")
year = st.number_input("Year", value=2020, step=1)
value = st.number_input("Population (THS)", value=100.0, step=1.0)
```


This adds a subheader to organize the UI, guiding users to the input section.


```python
if st.button("Predict"):
    X_new = pd.DataFrame(
        [[sex, age, country, year, value]],
        columns=["sex","age","geography","date","value"]
    )
    pred = pipe.predict(X_new)[0]
    st.success(f"Predicted Education Level (ISCED 2011 band): {pred}")
```


This block responds to a button press. It builds a DataFrame from inputs, calls predict, and displays the result.


## Model File

The `edu_pipeline.joblib` contains the trained scikit-learn pipeline. It was produced offline and includes preprocessing and estimator steps.


## Design of the Input Schema


The reasoning behind design of the input schema is critical. The schema ensures the DataFrame columns align with what the pipeline expects. Streamlit makes the app lightweight to build and share. Caching avoids repeated model loads, improving performance. Joblib handles scikit-learn pipelines better than pickle. Error handling must anticipate invalid inputs from users. Extensions could include new models or additional features. The structure allows scaling by replacing the pipeline file. Testing locally verifies that the app behaves correctly. UI choices reduce user mistakes and clarify usage. Future improvements could bring richer outputs like charts or feature importances.


## Choice of Streamlit for Deployment


The reasoning behind choice of streamlit for deployment is critical. The schema ensures the DataFrame columns align with what the pipeline expects. Streamlit makes the app lightweight to build and share. Caching avoids repeated model loads, improving performance. Joblib handles scikit-learn pipelines better than pickle. Error handling must anticipate invalid inputs from users. Extensions could include new models or additional features. The structure allows scaling by replacing the pipeline file. Testing locally verifies that the app behaves correctly. UI choices reduce user mistakes and clarify usage. Future improvements could bring richer outputs like charts or feature importances.


## Caching and Performance


The reasoning behind caching and performance is critical. The schema ensures the DataFrame columns align with what the pipeline expects. Streamlit makes the app lightweight to build and share. Caching avoids repeated model loads, improving performance. Joblib handles scikit-learn pipelines better than pickle. Error handling must anticipate invalid inputs from users. Extensions could include new models or additional features. The structure allows scaling by replacing the pipeline file. Testing locally verifies that the app behaves correctly. UI choices reduce user mistakes and clarify usage. Future improvements could bring richer outputs like charts or feature importances.


## Error Handling Considerations


The reasoning behind error handling considerations is critical. The schema ensures the DataFrame columns align with what the pipeline expects. Streamlit makes the app lightweight to build and share. Caching avoids repeated model loads, improving performance. Joblib handles scikit-learn pipelines better than pickle. Error handling must anticipate invalid inputs from users. Extensions could include new models or additional features. The structure allows scaling by replacing the pipeline file. Testing locally verifies that the app behaves correctly. UI choices reduce user mistakes and clarify usage. Future improvements could bring richer outputs like charts or feature importances.


## Why Use Joblib Serialization


The reasoning behind why use joblib serialization is critical. The schema ensures the DataFrame columns align with what the pipeline expects. Streamlit makes the app lightweight to build and share. Caching avoids repeated model loads, improving performance. Joblib handles scikit-learn pipelines better than pickle. Error handling must anticipate invalid inputs from users. Extensions could include new models or additional features. The structure allows scaling by replacing the pipeline file. Testing locally verifies that the app behaves correctly. UI choices reduce user mistakes and clarify usage. Future improvements could bring richer outputs like charts or feature importances.


## Potential Extensions of the App


The reasoning behind potential extensions of the app is critical. The schema ensures the DataFrame columns align with what the pipeline expects. Streamlit makes the app lightweight to build and share. Caching avoids repeated model loads, improving performance. Joblib handles scikit-learn pipelines better than pickle. Error handling must anticipate invalid inputs from users. Extensions could include new models or additional features. The structure allows scaling by replacing the pipeline file. Testing locally verifies that the app behaves correctly. UI choices reduce user mistakes and clarify usage. Future improvements could bring richer outputs like charts or feature importances.


## Scalability and Portability


The reasoning behind scalability and portability is critical. The schema ensures the DataFrame columns align with what the pipeline expects. Streamlit makes the app lightweight to build and share. Caching avoids repeated model loads, improving performance. Joblib handles scikit-learn pipelines better than pickle. Error handling must anticipate invalid inputs from users. Extensions could include new models or additional features. The structure allows scaling by replacing the pipeline file. Testing locally verifies that the app behaves correctly. UI choices reduce user mistakes and clarify usage. Future improvements could bring richer outputs like charts or feature importances.


## Testing the Application


The reasoning behind testing the application is critical. The schema ensures the DataFrame columns align with what the pipeline expects. Streamlit makes the app lightweight to build and share. Caching avoids repeated model loads, improving performance. Joblib handles scikit-learn pipelines better than pickle. Error handling must anticipate invalid inputs from users. Extensions could include new models or additional features. The structure allows scaling by replacing the pipeline file. Testing locally verifies that the app behaves correctly. UI choices reduce user mistakes and clarify usage. Future improvements could bring richer outputs like charts or feature importances.


## User Interface Choices


The reasoning behind user interface choices is critical. The schema ensures the DataFrame columns align with what the pipeline expects. Streamlit makes the app lightweight to build and share. Caching avoids repeated model loads, improving performance. Joblib handles scikit-learn pipelines better than pickle. Error handling must anticipate invalid inputs from users. Extensions could include new models or additional features. The structure allows scaling by replacing the pipeline file. Testing locally verifies that the app behaves correctly. UI choices reduce user mistakes and clarify usage. Future improvements could bring richer outputs like charts or feature importances.


## Future Improvements


The reasoning behind future improvements is critical. The schema ensures the DataFrame columns align with what the pipeline expects. Streamlit makes the app lightweight to build and share. Caching avoids repeated model loads, improving performance. Joblib handles scikit-learn pipelines better than pickle. Error handling must anticipate invalid inputs from users. Extensions could include new models or additional features. The structure allows scaling by replacing the pipeline file. Testing locally verifies that the app behaves correctly. UI choices reduce user mistakes and clarify usage. Future improvements could bring richer outputs like charts or feature importances.




## Purpose of Each Imported Package

### Streamlit
Streamlit is the framework that powers the interface of this project. It provides functions like st.title, st.selectbox, and st.button which make it possible to build a web interface without writing HTML or JavaScript. Its simplicity is what allowed me to focus on the predictive logic instead of worrying about frontend code.

### Pandas
Pandas is responsible for handling structured data. In this project, it creates the DataFrame that holds user inputs. By building the DataFrame with the right columns, I ensure the pipeline receives data in the same format it saw during training. Pandas also makes it easier to validate the shape of inputs.

### Joblib
Joblib loads the serialized pipeline stored in a .joblib file. Unlike pickle, joblib is optimized for objects containing large numpy arrays, which are common in scikit-learn models. It is faster and more memory-efficient in such cases. Using joblib reduces the risk of incompatibility when working with scikit-learn.

### Scikit-learn
While scikit-learn is not directly imported in app.py, it is required by the pipeline file. The estimator and preprocessing steps inside the pipeline depend on it. Without having scikit-learn installed at the right version, the deserialization of the pipeline would fail.

### Numpy
Numpy underlies both pandas and scikit-learn. It provides efficient array operations that support the calculations behind the scenes. In this project, it is not imported directly in app.py but is critical for the model to function.

### SciPy
SciPy is also not imported in app.py but is needed as a dependency for certain algorithms in scikit-learn. It provides optimized implementations of linear algebra, optimization, and statistics routines. The pipeline may rely on it indirectly depending on the estimator used.

### Packaging
Packaging is a utility library that helps Python tools check versions of dependencies. Some frameworks import it to ensure compatibility. I included it to satisfy indirect dependencies that may be triggered during deployment.



## Streamlit Caching Explained
The decorator @st.cache_resource ensures that functions marked with it are executed once and then cached. In this project, the function load_model is cached so that joblib.load is only run the first time. Without caching, every time a user pressed the Predict button, the model would reload, slowing the app. Caching improves responsiveness and makes the user experience smoother.

## Joblib vs Pickle
Pickle is the standard Python serialization library, but it is not efficient with large numpy arrays. Joblib was designed to handle these cases more efficiently. When working with scikit-learn pipelines, joblib is the recommended choice because it stores array data in a way that is both compact and fast to reload. This difference can be crucial for real-time apps.

## Deployment Considerations
When deploying on Streamlit Cloud, the app is executed in a container where dependencies are installed from requirements.txt. Any mismatch between the pipeline version and the installed libraries can cause errors. By pinning library versions carefully, I ensured that the pipeline would deserialize successfully in the cloud environment.

## User Interface and Data Validation
The widgets in Streamlit act as the bridge between users and the model. Each widget type was chosen to align with the type of input required. For instance, using a number input for year avoids users typing text like 'two thousand twenty'. This design reduces invalid input and lowers the chance of runtime errors. Good UI design is not just aesthetic but also functional.

## Model Lifecycle Management
Separating training and deployment is an important practice. Training requires resources and may take time, but prediction should be fast. By exporting the trained pipeline to a joblib file, I decoupled these stages. If I retrain the model later, I only need to replace the joblib file. The app code remains unchanged. This modularity simplifies maintenance and ensures long-term usability.

## Scalability Thoughts
Although this project is small, the design allows for scaling. For example, multiple models could be supported by adding more joblib files and extending the interface. The caching mechanism ensures that each model loads only once. For heavier usage, the app could be connected to a backend service, but for demonstration purposes Streamlit alone is sufficient.

## Testing Strategy
Before deployment, I tested the app locally. I tried various combinations of inputs to ensure that the DataFrame creation step worked and that predictions returned without error. This testing confirmed that the schema matched expectations. It also gave me confidence that deployment would run smoothly.

## Future Enhancements
Future versions of this app could show not just the predicted class but also probabilities. Visualizations like bar charts could make the output more informative. Feature importance could be displayed to explain the model’s reasoning. Adding these would increase transparency and make the app more educational.
