---
layout: default
title: "Building my AI USA Housing Price Predictor"
date: 2023-01-03 10:43:19
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/housing_price.webp
thumbnail_mobile: /assets/images/usa_housing_price_sq.webp
demo_link: https://rahuls-ai-house-price-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_house_price_predictor
featured: true
---

I wanted to create something that combined my interest in data and my curiosity about real estate. 
It started when I was looking through housing listings and noticed how varied prices could be. 
The same number of bedrooms did not always mean the same price. 
Square footage mattered, but location and condition made differences too. 
That was when I thought, why not try to build a model that takes structured housing data and gives a price estimate. 
This project grew from that idea into a working web app I can share with others. Dataset used [here](https://www.kaggle.com/datasets/fratzcan/usa-house-prices).

In this blog I will explain every file and every block of code that I uploaded into my GitHub repository. 
I want to make it clear how the project is set up and how it works. 
I will show how each helper, each function, and each input block plays a role. 
By the end of this post the entire picture of the application will be transparent. 
I will walk through the code step by step and explain the design decisions. 
This way any reader can not only use the app but also understand how it was built. 

## The requirements.txt File

This file lists the dependencies needed to run my project. Without this file, the app would not know which external packages to install. I uploaded this to GitHub so that Streamlit Cloud or any other environment can recreate the same environment.

```python
streamlit
scikit-learn
pandas
joblib

```

Here I listed four packages. Streamlit is the framework I used to build the interactive app. Pandas is useful for handling structured data if I want to extend the app later. Scikit-learn is required because the model was trained with it, so the model object expects that package when being loaded. Joblib is needed to load the trained model that was saved earlier.

## The app.py File

This is the main entry point of the project. It defines the user interface, collects the input values, and calls the trained model to make predictions. I uploaded this file as the core of the web app.

```python
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/house_price_model.joblib")

st.title("🏠 USA Housing Price Predictor")

st.write("Enter home details to predict the price.")

# Inputs
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1, 10, 2)
sqft_living = st.number_input("Sqft Living", 500, 10000, 1500)
sqft_lot = st.number_input("Sqft Lot", 500, 50000, 2000)
floors = st.number_input("Floors", 1, 3, 1)
waterfront = st.selectbox("Waterfront", [0, 1])
view = st.selectbox("View", [0, 1, 2, 3, 4])
condition = st.selectbox("Condition", [1, 2, 3, 4, 5])
sqft_above = st.number_input("Sqft Above", 500, 10000, 1200)
sqft_basement = st.number_input("Sqft Basement", 0, 5000, 300)
yr_built = st.number_input("Year Built", 1900, 2025, 2000)
yr_renovated = st.number_input("Year Renovated", 0, 2025, 0)

# Predict
if st.button("Predict Price"):
    input_data = [[bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,sqft_above,sqft_basement,yr_built,yr_renovated]]
    pred = model.predict(input_data)[0]
    st.success(f"Predicted House Price: ${pred:,.2f}")

```

Now I will explain this file block by block, focusing on each section.

### Importing the Libraries

The first lines of the file bring in the required libraries. Streamlit is used for the interface, pandas is imported in case structured data handling is needed, and joblib is required to load the model file.

```python
import streamlit as st
import pandas as pd
import joblib
```

These imports are critical because without them the code would not be able to show widgets or load the model. Even if pandas is not heavily used here, it is included to allow extensions later.

### Loading the Model

The model is loaded from the models folder using joblib. This line makes sure the trained model is available in memory before we ask it to predict anything.

```python
model = joblib.load("models/house_price_model.joblib")
```

This step is important because the predictor relies on a pre-trained regression model. Without this line, the app would not know how to estimate prices.

### Title and Description

The app uses Streamlit commands to display a title and a short instruction.

```python
st.title("🏠 USA Housing Price Predictor")

st.write("Enter home details to predict the price.")
```

This part gives the app a clear headline and tells users what they should do. The title uses an emoji in the original code, but the important part is that it sets the theme of the app.

### Input Widgets

The app collects user input through a series of widgets. Each line defines one input such as bedrooms or bathrooms. Streamlit provides number_input for numeric values and selectbox for categorical choices.

```python
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1, 10, 2)
sqft_living = st.number_input("Sqft Living", 500, 10000, 1500)
sqft_lot = st.number_input("Sqft Lot", 500, 50000, 2000)
floors = st.number_input("Floors", 1, 3, 1)
waterfront = st.selectbox("Waterfront", [0, 1])
view = st.selectbox("View", [0, 1, 2, 3, 4])
condition = st.selectbox("Condition", [1, 2, 3, 4, 5])
sqft_above = st.number_input("Sqft Above", 500, 10000, 1200)
sqft_basement = st.number_input("Sqft Basement", 0, 5000, 300)
yr_built = st.number_input("Year Built", 1900, 2025, 2000)
yr_renovated = st.number_input("Year Renovated", 0, 2025, 0)
```

Each widget ensures values stay in a realistic range. For example, bedrooms are limited between 1 and 10, while square footage is set within common limits. This avoids invalid input that could break predictions.

### Prediction Logic

At the end of the file the prediction is triggered when the user clicks the button. A list of all collected inputs is passed into the model, which then outputs a predicted price.

```python
if st.button("Predict Price"):
    input_data = [[bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,sqft_above,sqft_basement,yr_built,yr_renovated]]
    pred = model.predict(input_data)[0]
    st.success(f"Predicted House Price: ${pred:,.2f}")
```

The conditional checks if the button is pressed. Then it creates a nested list containing all input features. The model predicts based on this list. Finally the result is shown in a success box formatted as currency.

## Reflections on Design Choices

I chose to keep this app simple so that anyone could interact with it without confusion. The inputs are clearly labeled and the ranges are restricted to sensible values. This prevents errors and also guides the user toward realistic scenarios. The model itself was trained earlier using scikit-learn. By separating the training from the app, I made the deployment cleaner and more stable.

Another design choice was saving the model with joblib. This format is efficient for scikit-learn estimators and can be loaded quickly. It avoids the need to retrain the model every time the app runs. This makes the app faster and reduces resource use. I wanted to be able to redeploy it without worrying about training time.

The use of Streamlit made the app visually accessible. The layout is not cluttered and the widgets are familiar to most users. Number inputs and select boxes are intuitive. The prediction appears instantly when the button is clicked. This gives the app a sense of responsiveness. These choices may appear minor, but together they shape the user experience in an important way.

### Deep Dive

In this part I reflect on how the code structure supports maintainability. Each block is independent yet contributes to the overall function. By keeping the user interface in one file, I can easily update it without touching the model. The model can also be retrained and replaced in the models folder without changes to app.py. This separation of concerns ensures that the workflow remains clean. If new features are needed, I can add them as additional inputs and extend the input_data list. The predict function will still work as long as the model expects the same order of features. This makes the design flexible for growth and stable for current use.
