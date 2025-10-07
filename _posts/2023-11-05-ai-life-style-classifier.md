---
layout: default
title: "Creating my AI Lifestyle Classifier"
date: 2023-11-05 07:36:44
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/lifestyle.webp
thumbnail_mobile: /assets/images/lifestyle_classifier_sq.webp
demo_link: https://rahuls-ai-life-style-classifier.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_lifestyle_classifier
---

The idea for this project came when I thought about how daily routines shape identity. Every country has its unique cultural rhythm, and much of that rhythm is visible in how people spend their time. Some societies dedicate longer hours to work, while others emphasize leisure or family. I wanted to test if these daily choices could be used as signals for classification. The concept was simple but fascinating: can a machine learning model predict a country code just from how many minutes are spent on common daily activities?

I approached this question not as an academic researcher but as an engineer who enjoys building small but functional prototypes. I wanted to build something interactive that anyone could try. By taking input from sliders and pushing it through a trained model, the app could show predictions instantly. This kind of feedback loop is engaging and easy to understand. That was the inspiration, and Streamlit provided the perfect platform to turn this idea into reality. Dataset used [here](https://www.kaggle.com/datasets/yuchendai/european-time-use).

## Repository Structure

The repository is minimal but purposeful. It contains three essential files and one directory that houses the trained model.

- **app.py**: The main Streamlit application. It defines the user interface, loads the trained model, prepares the input, and displays predictions. This is the heart of the project.
- **requirements.txt**: The dependency list. It ensures the right versions of libraries are installed when the app runs on a new machine or in the cloud.
- **models/lifestyle_model.pkl**: The serialized trained model. It contains all the learned parameters from training. This file enables the app to make predictions without retraining each time.

This structure kept the repository lean and easy to maintain. Every file had a clear role, and there were no extras. It also simplified deployment because the hosting platform only needed to copy these files.

## Purpose of Each Package

The `requirements.txt` file lists four packages, and each plays a vital role. I will explain why each one is necessary and what problems it solves.

```python
streamlit
scikit-learn
pandas
joblib
```

### Streamlit

Streamlit provides the interactive web interface. Without it, this project would remain a script with no user-facing component. The `st.slider` elements make it easy to collect numeric input without requiring typing. The `st.write` and `st.error` functions handle communication with the user. Streamlit abstracts away HTML, CSS, and JavaScript, so I could focus purely on Python logic. This package transforms the model from a backend artifact into a working demo.

### Scikit-learn

Scikit-learn is the library used to train the model. The model was serialized with joblib, but the scikit-learn codebase defines the class structure. When loading the model, scikit-learn provides the methods like `predict`. Without it, the unpickling process would fail because Python would not recognize the model type. In addition, scikit-learn ensures compatibility with the attribute `feature_names_in_`, which I used in the inference stage to align columns.

### Pandas

Pandas is indispensable for working with tabular data. In this project, I used it to package slider inputs into a DataFrame. A DataFrame is the native input structure for scikit-learn models trained on tabular data. Pandas also provides utilities for casting types, filling missing values, and reordering columns. These small operations are essential when preparing data for prediction. Pandas bridges the gap between raw inputs and the strict requirements of machine learning models.

### Joblib

Joblib handles the persistence of trained models. Unlike the generic pickle module, joblib is optimized for large numpy arrays and efficient serialization. I used it to load the trained lifestyle model from disk. This package ensures the model can be shared across environments without retraining. It is reliable, lightweight, and directly compatible with scikit-learn.

## Walkthrough of app.py

Now I will explain the entire script step by step. Each block will be expanded to show its purpose in the bigger picture.

### Importing Libraries

```python
import streamlit as st
import pandas as pd
import joblib
```

This block initializes the environment. Streamlit creates the interface, Pandas prepares tabular input, and Joblib loads the trained model. Each import corresponds directly to a task the application must perform. Without these, the script would not run.

### Loading the Model

```python
model = joblib.load("models/lifestyle_model.pkl")
```

Here the trained model is loaded from the models directory. This step is critical because all downstream logic depends on having a ready model object. Joblib deserializes the binary file and reconstructs the scikit-learn object. Storing the model separately allows training and serving to be decoupled. This also makes the app lightweight, since the heavy training process does not need to run on the cloud.

### Setting Title

```python
st.title("AI Country Lifestyle Classifier")
```

A title is added to the page. This simple addition improves usability because users know immediately what the app does. It is good practice to always provide a descriptive heading at the start of a Streamlit application.

### Collecting Input with Sliders

```python
personal_care = st.slider("Personal care (minutes)", 0, 600, 120)
sleep = st.slider("Sleep (minutes)", 0, 1200, 480)
eating = st.slider("Eating (minutes)", 0, 300, 90)
work = st.slider("Employment (minutes)", 0, 600, 240)
leisure = st.slider("Leisure (minutes)", 0, 600, 180)
```

Each slider gathers numeric input from the user. The parameters specify minimum, maximum, and default values. This ensures that inputs remain within realistic ranges. For example, sleep cannot exceed 1200 minutes (20 hours). These constraints improve the model’s reliability and avoid nonsensical predictions. Sliders are user-friendly and prevent invalid entries, making the interface robust.

### Preparing Input DataFrame

```python
X_new = pd.DataFrame([[0, personal_care, sleep, eating, work, leisure]],
                     columns=["SEX", "Personal care", "Sleep", "Eating",
                              "Employment, related activities and travel as part of/during main and second job",
                              "Leisure, social and associative life"])
```

This block converts raw slider values into a structured DataFrame. Each column matches the schema the model was trained on. The column `SEX` is included because the training data contained it. I set it to zero as a placeholder since I did not collect gender in the app. This ensures compatibility with the model while keeping the interface simple. The helper role of this block is aligning live user input with the training schema.

### Ensuring Column Alignment

```python
import numpy as np

expected_cols = getattr(model, "feature_names_in_", None)

if expected_cols is None:
    st.error("Model missing feature_names_in_. Retrain with a DataFrame or use Option B below.")
else:
    for c in expected_cols:
        if c not in X_new.columns:
            X_new[c] = 0
    X_new = X_new.reindex(columns=expected_cols, fill_value=0)
    X_new = X_new.apply(pd.to_numeric, errors="coerce").fillna(0)
```

This block ensures that the DataFrame columns exactly match what the model expects. First, it checks if the model has the attribute `feature_names_in_`. If not, an error is displayed telling the user to retrain. If the attribute exists, the code iterates through all expected columns. Any missing column is added with zero values. The DataFrame is then reordered to match the original sequence. Finally, values are cast to numeric and missing entries are filled with zeros. This is a defensive strategy to prevent runtime errors. It safeguards against schema mismatches and guarantees consistent predictions.

### Generating Prediction

```python
prediction = model.predict(X_new)[0]
st.write("Predicted Country Code:", prediction)
```

This final block performs the prediction. The model’s `predict` method is called with the prepared DataFrame. The output is an array, so I take the first element. Then I use `st.write` to display the result on the page. The result is shown immediately under the sliders. This creates an interactive experience where adjusting input values changes the prediction.

## The Model File

The file `models/lifestyle_model.pkl` is the trained machine learning model. It was built using scikit-learn and saved with joblib. The model encodes the patterns it learned during training. Without it, the app would not function. Keeping the model inside the repository ensures the app runs consistently on any platform. This file is large compared to the scripts, but it is the most important asset.

## Deployment Process

To deploy the project, I uploaded the repository to GitHub. Streamlit Cloud was then connected to the repository. During deployment, the platform installed the dependencies from `requirements.txt` and executed `app.py`. Because the model file was present, everything worked seamlessly. The minimal setup was an advantage because there was little that could go wrong. Each file played a direct role in deployment.

## Lessons Learned

This project reinforced the importance of schema consistency. Models are fragile when it comes to input features, and small mismatches cause failures. I also learned how useful Streamlit is for creating shareable demos. By focusing on the essentials, I avoided complexity and built a working prototype quickly. The separation of training and inference workflows was another lesson. Serving a pre-trained model is more efficient than retraining each time.

## Future Improvements

There are several ways I could extend this project. I could enrich the model with more features like commuting, family care, or exercise. I could also improve the labels so that the output shows country names instead of numeric codes. Adding charts to visualize activity distribution would make the app more engaging. Finally, retraining the model with newer data could improve accuracy. Each of these extensions would add depth and make the tool more useful.

---

This completes the detailed breakdown of my AI Lifestyle Classifier. Every code block, package, and file has been explained in depth. The project is small but demonstrates how structured data and machine learning can be combined into an accessible interactive tool.
