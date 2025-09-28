---
layout: default
title: "Creating my AI Bank Customer Subscription Predictor"
date: 2023-04-25 11:53:41
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-bank-customer-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_bank_customer_predictor
---

A few years ago, I was reflecting on how financial institutions decide whether a customer will subscribe to a new service or product. The idea stayed in my mind after noticing how banks often run campaigns that do not always convert as expected. I imagined that if I had a way to analyze the customer data with machine learning, I could predict the likelihood of subscription. That personal thought became the seed of this project. It was less about coding at first, more about curiosity of how data tells a story. From that curiosity came this application, which I will now explain in detail. Dataset used [here](https://www.kaggle.com/datasets/adilshamim8/binary-classification-with-a-bank-dataset).

The project turned into a Streamlit app that loads a trained pipeline and predicts whether a bank customer will subscribe. I structured the project to be small but complete, making it easy to run locally or host online. Each file plays a role in keeping the project clean. I wanted the breakdown to be transparent, so that anyone could see not just the result, but the reasoning behind the code. In this blog post, I will go through every file and explain every code block in the app. The purpose is clarity, not brevity, so this will be long and detailed.



## The `requirements.txt` File

This file defines the Python packages needed to run the app. In my project, it includes core libraries such as:

```python
streamlit
pandas
numpy
joblib
scikit-learn
```

Each package has a purpose. Streamlit provides the web interface so I can interact with the model in a browser. Pandas is used to manage tabular data that flows into the model. NumPy provides numerical support and array handling for computations. Joblib is required because the trained pipeline is serialized into a `.joblib` file. Scikit-learn is the base library that defines the pipeline, transformations, and model training.

By defining these in `requirements.txt`, anyone can install the same environment by running `pip install -r requirements.txt`. That ensures reproducibility of the project. The file is short but it carries significant importance, because without it the app may not run correctly on another system.



## The Trained Model File

The model file is stored in the path:

```
models/subscription_pipeline.joblib
```

This file contains the entire scikit-learn pipeline, trained earlier on bank marketing data. The pipeline likely includes preprocessing steps like encoding categorical variables, scaling numerical features, and then a classifier such as logistic regression or random forest. By saving the whole pipeline, I avoid repeating the preprocessing code in the app. The app can directly load the pipeline and run predictions on user inputs.

The model file is binary and cannot be opened as text. But its presence makes the application functional. Without this file, the app will fail at prediction time. That is why I also added code in the app to check if the file exists and provide debugging help if it does not.



## The `app.py` File

The main file of the project is `app.py`. It defines the Streamlit application, loads the model, and builds the interface for predictions. Below I will show code blocks and explain them in depth.

### Importing Packages

The first section imports all required packages:

```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
import joblib
import streamlit as st
```

This section starts with importing Streamlit as `st`, which is the core of the web interface. Pandas and NumPy are included because customer data may be managed as DataFrames or arrays. Joblib is imported for loading the serialized pipeline. The `Path` class from pathlib is used for file path handling. Some imports are repeated, but they do not break the app. Repeated imports simply mean the module is loaded once, and the second line is redundant but harmless.



### Loading the Pipeline

```python
@st.cache_resource
def load_pipeline():
    root = Path(__file__).parent
    candidates = [
        root / "models" / "subscription_pipeline.joblib",  # preferred
        root / "subscription_pipeline.joblib",             # fallback (root)
    ]
    for p in candidates:
        if p.exists():
            try:
                return joblib.load(p)
            except Exception as e:
                st.error(f"Found model at {p}, but failed to load: {e}")
                st.stop()

    # If we get here, nothing was found. Show what *is* in the workspace to help debug.
    found_joblibs = list(root.rglob("*.joblib"))
    st.error(
        "Model file not found. Expected `models/subscription_pipeline.joblib`.\n\n"
        f"Detected joblib files: {found_joblibs if found_joblibs else 'None'}"
    )
    st.info("Fix: Upload `subscription_pipeline.joblib` ")
```



The `load_pipeline` function is decorated with `@st.cache_resource`, which ensures the model is only loaded once and cached. That improves performance because reloading a large joblib file repeatedly would slow down the app. The function defines `root` as the parent directory of the app file, so that file paths can be built relative to it.

It then defines two candidate paths for the pipeline: one in the `models/` subfolder and one in the root folder. The reason is flexibility, so the app can still work even if the model is placed in a slightly different location. The function loops through each candidate path, checks if it exists, and then tries to load it with joblib.

If loading succeeds, the pipeline object is returned. If loading fails due to corruption or version mismatch, the error is displayed and the app stops. If no file is found, the function searches the workspace for any `.joblib` files and displays them, so that the user knows what is available. This debugging feature is practical in deployed environments where files may not always be in the right folder.



### Building the Input Interface

The next step in the app is to create input fields for the user. In Streamlit, this is usually done with functions like `st.text_input`, `st.selectbox`, and `st.number_input`. For example, we may define:

```python
def user_input_features():
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job", ["admin.", "blue-collar", "technician", "services"])
    balance = st.number_input("Balance", value=1000)
    housing = st.selectbox("Housing Loan", ["yes", "no"])
    loan = st.selectbox("Personal Loan", ["yes", "no"])
    return pd.DataFrame({
        "age": [age],
        "job": [job],
        "balance": [balance],
        "housing": [housing],
        "loan": [loan],
    })
```

This function builds a small DataFrame with the same structure as the training data. Each widget allows the user to supply a value, and the returned DataFrame will later be used for prediction. The function is simple, but it abstracts away all the UI logic into a helper that returns clean data.



### Making Predictions

Once inputs are collected, the app can use the pipeline to make predictions:

```python
pipeline = load_pipeline()
inputs = user_input_features()
if st.button("Predict Subscription"):
    prediction = pipeline.predict(inputs)
    st.write("Prediction:", "Will Subscribe" if prediction[0] == 1 else "Will Not Subscribe")
```

Here, the pipeline is loaded once using the helper function. The inputs are collected from the form function. When the user clicks the button, the pipeline makes a prediction. The result is shown as plain text, telling the user whether the customer is likely to subscribe. This design keeps the flow simple, from input to prediction to output.



### Why Streamlit was Chosen

Streamlit is central to this project. It allows me to turn a Python script into a web app with minimal code. I did not need to learn HTML, CSS, or JavaScript. Instead, Streamlit widgets such as text inputs, select boxes, and buttons are enough to build an interactive UI. This was important because my focus was on the machine learning pipeline rather than building a frontend from scratch. Streamlit also provides caching and session management features that make repeated interactions smooth.

The framework is reactive. That means whenever an input changes, the app reruns from the top and updates the output. In this project, that behavior ensures predictions always reflect the most recent values. The cached pipeline loader avoids performance issues during these reruns. By combining simplicity and reactivity, Streamlit became the natural fit for this project.



### The Role of Pandas and NumPy

Pandas and NumPy form the backbone of data manipulation. In the `user_input_features` function, I used pandas to create a DataFrame from user inputs. This is crucial because the scikit-learn pipeline expects data in tabular format with named columns. A DataFrame provides exactly that. Without pandas, I would have to manually create dictionaries or arrays, which would be less transparent.

NumPy supports operations that may be embedded inside the pipeline. Scaling, encoding, or reshaping are usually powered by NumPy arrays. Even if the app does not directly use NumPy arrays, the pipeline likely depends on them internally. Having NumPy imported ensures compatibility and avoids runtime errors when the model expects its structures.



### Why Joblib and Path are Needed

Joblib is the serialization library used by scikit-learn for saving and loading models. It compresses objects efficiently, which is helpful when pipelines contain large arrays or parameter sets. In this project, the pipeline was trained outside this repository, then saved to disk as `subscription_pipeline.joblib`. The app needs joblib to load this object back into memory. Without it, there would be no way to reconstruct the trained model.

The Path class is used for file navigation. It gives me a clean way to handle operating system paths without hardcoding string separators. In the `load_pipeline` function, I used Path to locate the model file in both `models/` and the root directory. That abstraction makes the code portable between Windows, macOS, and Linux environments. Using Path is therefore a small but important design choice that adds reliability.



### Understanding Caching with `@st.cache_resource`

The decorator `@st.cache_resource` tells Streamlit to cache the return value of the function. Here, the return value is the loaded pipeline. Without caching, the pipeline would reload every time the script reran. That would slow down the app, especially if the file were large or located on a slower disk.

Caching works by storing the pipeline object in memory. On the next run, Streamlit checks if the function inputs or the code have changed. If not, it reuses the cached object instead of calling the function again. This design pattern is efficient and keeps the user experience responsive. It also shows how Streamlit handles state, because normally each run is stateless. With caching, I bridge the gap between a reactive rerun model and persistent resources like models.



### Error Handling in the Pipeline Loader

The `load_pipeline` function includes several layers of error handling. First, it checks whether the candidate path exists. If a file exists, it tries to load it. The try-except block is essential here because joblib loading can fail if the file is corrupted or if it was trained with a very different version of scikit-learn.

If loading fails, the function reports the error directly in the app using `st.error`. This makes the issue visible to the user without crashing the application. The call to `st.stop()` ensures the rest of the app does not run with a missing pipeline. If no file is found at all, the function searches recursively for `.joblib` files and shows them to the user. This debugging step prevents silent failure and guides the user on how to fix the issue. It is an example of defensive programming within a simple helper.



### User Input Helper in Detail

The `user_input_features` function is not just a wrapper around Streamlit widgets. It also enforces structure on the input data. Each widget corresponds to a feature that was used during model training. This ensures the DataFrame has the same schema as the training set. That schema consistency is vital, because machine learning models cannot handle mismatched columns.

Each widget also has validation built in. For example, the age widget restricts values between 18 and 100. That prevents invalid values from being passed to the model. The select boxes constrain values to categories seen during training, which avoids encoding errors. By combining constraints with structure, this helper acts as both a UI builder and a data validator. It ensures smooth operation downstream in the pipeline.



### Prediction Logic in Depth

The prediction section glues together the inputs and the pipeline. The key line is `pipeline.predict(inputs)`. This call triggers the full scikit-learn pipeline, which may include preprocessing and classification. The preprocessing steps transform raw inputs into numerical arrays, and the classifier then produces a label. By using the pipeline, I avoid duplicating transformation code. The pipeline guarantees the same processing steps as training.

The logic then interprets the numeric label into human-readable text. A value of 1 means the customer will subscribe, while 0 means they will not. That mapping is simple but important for user understanding. Finally, the prediction is displayed directly in the Streamlit interface with `st.write`. This closes the loop from user input to machine prediction to user feedback. It demonstrates the power of a minimal interface combined with a trained model.



## Reflections on the Project

This project taught me the value of structuring machine learning applications clearly. By separating the model file, requirements, and application logic, I kept the repository organized. Each part has its role, and together they form a complete product. The use of Streamlit lowered the barrier to presenting results, allowing focus on data and logic rather than web frameworks.

The project also highlighted the importance of reproducibility. With `requirements.txt` and the cached pipeline loader, I ensured the app runs predictably on any system. The debugging messages in `load_pipeline` show that robustness matters even in small projects. A prediction app without clear error messages is frustrating to use. By anticipating failure cases, I made the app more reliable for myself and for others.



### Preprocessing in the Pipeline

Even though the pipeline file itself is not visible in this repository, I know what steps are typically included. Preprocessing often involves encoding categorical variables such as job, housing, and loan. Scikit-learn provides `OneHotEncoder` or `OrdinalEncoder` for this purpose. These encoders convert categories into numerical arrays that models can interpret.

Numerical variables like age and balance may be standardized or normalized. This step ensures that features with larger ranges do not dominate the learning process. In many marketing datasets, balance can vary widely while age is constrained, so scaling is helpful. Missing values, if present, may be imputed with strategies like median replacement. By encapsulating these transformations into a pipeline, I preserved the training logic and avoided repeating it during prediction.



### Choice of Model

The final estimator in the pipeline could be logistic regression, decision tree, or random forest. Logistic regression is common in subscription prediction because the outcome is binary. It provides interpretable coefficients that show which features influence the outcome. Decision trees and random forests add non-linear power, capturing complex interactions between variables.

The model was trained on historical marketing data. Each row represented a customer, with features describing their demographics, account balance, and loan status. The target variable indicated whether the customer subscribed. By fitting a model on this data, the pipeline learned patterns that generalize to new customers. The app reuses that model for predictions in real time.



### Deployment Considerations

The app is lightweight enough to run on local machines or free hosting platforms. Streamlit Cloud is a natural choice for deployment because it supports repositories with `requirements.txt` and Streamlit apps. By placing the repository on GitHub and linking it to Streamlit Cloud, I can deploy the app without additional infrastructure. This makes sharing predictions with others easy.

Another option is to deploy on platforms like Hugging Face Spaces or Heroku. Each platform requires some adjustments but relies on the same structure. The small set of dependencies ensures portability. I designed the repository with this in mind, so that deployment would not be a barrier.



### Scalability and Future Enhancements

While the app works for single-customer predictions, scalability requires adjustments. For batch predictions, I could extend the UI to allow file uploads, where users submit a CSV of customers. The app would then load the file, process all rows, and output predictions in a table. This extension leverages the same pipeline but requires a different interface.

Another enhancement is model monitoring. Predictions should be tracked over time to ensure accuracy does not degrade. If customer behavior changes, the model may need retraining. Adding logging or storing predictions in a database would support this. With these steps, the app could evolve from a demonstration into a production-level system.



## Conclusion

This project began as a thought about how banks decide whether customers will subscribe. It became a concrete application that loads a trained pipeline and makes predictions interactively. Every file has its role: requirements define dependencies, the joblib file holds the model, and app.py glues everything together. By breaking down each block, I showed how the parts form a complete and functional app.

The lesson here is that even small projects benefit from structure and clarity. Error handling, caching, and schema validation improve usability. Streamlit provides a friendly interface while scikit-learn delivers reliable predictions. Together they create a system that demonstrates the practical value of machine learning in finance. The experience taught me not only about prediction but also about building shareable tools that others can run and learn from.
