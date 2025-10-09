---
layout: default
title: "Building my AI Wine Quality Classifier"
date: 2024-10-25 11:14:23
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/wine.webp
thumbnail_mobile: /assets/images/wine_quality_sq.webp
demo_link: https://rahuls-ai-wine-quality-classifier.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_wine_quality_classifier
---

I once read an article that described how people struggle to judge wine quality without years of tasting experience. I imagined how difficult it must be for someone who enjoys wine but does not have the vocabulary or training of a sommelier. That personal thought became the seed for this project. Instead of depending on subjective judgment, I wanted to create a model that could classify wine quality based on measurable attributes. The goal was not to replace human expertise but to complement it with data. Dataset used [here](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)..

This project became a journey in turning a machine learning model into a deployable application. I learned that writing the model is only the first step. The real challenge is packaging it into something others can use. I had to prepare code files, manage dependencies, save the trained model, and write a user interface that guided people to input their own values. This blog post is my attempt to document that complete journey, breaking down each file, each function, and each design decision.
## requirements.txt

The first file I created was `requirements.txt`. It lists the exact versions of the libraries that my app depends on. Pinning versions is critical for reproducibility. Without it, the app might behave differently on another machine or in a cloud deployment. The content of the file was straightforward:

```python
streamlit>=1.37
scikit-learn==1.6.1
joblib==1.5.2
pandas==2.2.2
numpy==2.0.2
```

Each line declares a library and version. Streamlit is for the user interface. Scikit-learn is for training and model utilities. Joblib is for saving and loading the trained model. Pandas handles the structured input data. Numpy supports array operations behind the scenes. This small file ensures that anyone who runs the app will install the same environment and avoid mismatches.

I also learned that a small oversight in dependency versions can break an entire deployment. At one stage I had installed the latest scikit-learn, but it had a change in how models were serialized. The model I saved could not be loaded. Pinning the version to 1.6.1 fixed the issue permanently. That small lesson showed me how software stability depends on such subtle but important details.
## app.py

The main application file is `app.py`. It wires together the user interface, model loading, input collection, and prediction display. In this section I will expand every block in detail. The file begins with imports.
### Imports

```python
import streamlit as st
import pandas as pd
import joblib
```

Here I imported three core libraries. Streamlit provides the web application framework. Pandas supports the creation of tabular input data. Joblib is the tool I used to load the serialized model. Each import connects to a specific stage in the pipeline, from interface to data preparation to prediction.

By explicitly listing these imports at the top, I also documented what the file depends on. Anyone reading the file understands immediately that the app uses Streamlit for interface, Pandas for data handling, and Joblib for model management. Such clarity helps both my future self and anyone else who wants to reuse or extend the code.
### Model Loading

```python
# Load model
model = joblib.load("models/wine_model.pkl")
```

This line loads the pre-trained classifier from the models folder. The file `wine_model.pkl` contains the scikit-learn model that I had trained earlier. Joblib handles binary serialization efficiently. Loading the model at the top of the script ensures that it is ready before any prediction is requested. This avoids repeated disk access and speeds up inference.

I discovered that loading the model once outside the button press is far more efficient. Initially, I placed the load call inside the prediction block. That caused the model to reload each time, adding noticeable delay. By restructuring, I gained faster responses. It was a small performance optimization, but it mattered for user experience. Such practical changes often define whether a project feels polished or amateur.
### Application Title

```python
st.title("Wine Quality Classifier")
st.write("Predict whether a wine is 'Good' (quality ≥ 7) or 'Not Good'.")
```

These commands define the heading and description visible to users. The title sets the purpose clearly. The supporting text explains the threshold logic. Wines rated 7 or higher are considered good. This framing helps the user know what outcome to expect when they supply values.

At first I debated whether to show the threshold explicitly. I considered simply labeling the result as "Good" or "Not Good". But by including the condition (quality ≥ 7), I provided transparency. The user understands that the classifier uses a numeric cutoff. That choice aligns with good design practice, where clarity is preferred over mystery.
### User Input Widgets

The application then defines a sequence of numeric inputs. Each input maps to a chemical attribute of wine. For clarity, here is the code:

```python
fixed_acidity = st.number_input("Fixed Acidity", 4.0, 16.0, 8.0)
volatile_acidity = st.number_input("Volatile Acidity", 0.1, 1.6, 0.5)
citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.3)
residual_sugar = st.number_input("Residual Sugar", 0.5, 15.5, 2.5)
chlorides = st.number_input("Chlorides", 0.01, 0.61, 0.08)
free_so2 = st.number_input("Free Sulfur Dioxide", 1, 72, 15)
total_so2 = st.number_input("Total Sulfur Dioxide", 6, 289, 46)
density = st.number_input("Density", 0.99, 1.01, 1.0)
pH = st.number_input("pH", 2.9, 4.0, 3.3)
sulphates = st.number_input("Sulphates", 0.3, 2.0, 0.6)
alcohol = st.number_input("Alcohol", 8.0, 15.0, 10.0)
```

Each widget restricts the valid range. For example, fixed acidity cannot go below 4.0 or above 16.0. These ranges encode domain knowledge from the dataset. The default values act as starting points. Streamlit automatically renders sliders or numeric boxes depending on the configuration. By defining each parameter clearly, the app ensures structured input that the model can handle consistently.

I saw this section as more than just collecting numbers. It was a way of embedding safety nets into the interface. If a user entered a pH of 10, the chemistry would make no sense. The model would produce a result, but it would be meaningless. By limiting ranges, I prevented such mistakes. It felt like a form of silent guidance, keeping the predictions realistic and trustworthy.
### Prediction Logic

```python
if st.button("Predict"):
    input_data = pd.DataFrame([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_so2, total_so2, density, pH, sulphates, alcohol
    ]], columns=[
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol"
    ])
    
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("Good Quality Wine!")
    else:
        st.error("Not a Good Quality Wine")
```

The conditional checks if the predict button is pressed. When activated, it builds a pandas DataFrame containing one row of values from all user inputs. The column names must exactly match what the model expects. This alignment prevents errors at runtime. The model is then called to predict. The output is a numeric label. If the label is one, it represents good quality. If zero, it represents not good. Streamlit messages communicate the result with distinct visual styles.

I found that even this small block involved multiple layers of design. Constructing the DataFrame required careful ordering of values. If I swapped alcohol and sulphates, the model would misinterpret the meaning. Naming the columns explicitly was my safeguard. Then came the decision of how to present the result. I could have shown a raw number. But the user expects a plain message. Choosing `st.success` and `st.error` was deliberate. They add color and clarity, turning abstract output into immediate feedback.
## Reflections and Design Choices

While the app file is short, every block carries meaning. The imports establish dependencies. The model load ensures readiness. The title guides the user. The input widgets encode scientific ranges. The prediction logic converts abstract numbers into a simple classification. The key lesson for me was that usability matters as much as accuracy. Without a friendly interface, even the best model stays hidden.

In building this, I also learned about deployment tradeoffs. Loading the model once instead of each time saved seconds. Pinning dependencies avoided runtime crashes. Naming columns carefully avoided mismatches. These details are not glamorous but they are what make an app reliable. I realized that machine learning is as much about software engineering as it is about statistics.

The final reflection I carry is that a project gains life when others can use it. Before deployment, my classifier was just a Python object. After building this app, it became an experience that others could try. That transformation from code to interaction was what made the project meaningful. It reminded me that data science is never just about numbers, but about creating systems that people can trust and use.

During the making of this project, I thought carefully about robustness. For instance, I reflected on what would happen if someone tried to deploy this in a limited environment like Streamlit Cloud. I recognized that keeping the model file small enough to upload was as important as accuracy. This led me to experiment with compression and pruning of the trained classifier. Such tradeoffs made me realize that performance, memory, and usability are interlinked. The classifier is not only an algorithm but a deployed service where each constraint matters.
