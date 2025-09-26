---
layout: default
title: "Creating my AI NYC Airbnb Price Predictor"
date: 2024-11-29 10:25:31
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-airbnb-price-prediction.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_airbnb_price_prediction
featured: true
---

It all started with a simple experience. I was browsing through different listings on a travel platform and noticed that two very similar apartments had very different prices. Both were located in Brooklyn, had similar reviews, and looked almost identical in photos. Yet one was priced almost double the other. That inconsistency made me curious. Was there some hidden logic that determined these prices? Could data explain why some listings were valued higher than others? This thought became the seed of the project. Dataset used [here](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data).

Over time I realized that pricing on rental platforms is not only about the property itself. It also reflects neighborhood popularity, room types, the number of reviews, and even host activity. I wanted to capture this complexity in a structured way and use machine learning to estimate what a fair nightly price could be. This blog post tells the entire story of how I transformed that curiosity into a deployed application. I explain each code block, every file I uploaded to GitHub, and all the steps that connect them. The post is intentionally long and technical because I want to document how everything works under the hood.

---

## Requirements File

The first file that I needed in the repository was the `requirements.txt`. This file ensures that anyone running the app installs the same versions of the packages. Reproducibility matters because model behavior can change when library versions differ.

```python
streamlit>=1.37
scikit-learn==1.6.1
joblib==1.5.2
pandas==2.2.2
numpy==2.0.2
```

Each line specifies a dependency. Streamlit powers the user interface. Scikit-learn provides the model training and transformation tools. Joblib is responsible for model serialization and loading. Pandas helps with data manipulation. Numpy is the numerical backbone. Locking these versions prevents silent changes that could break the app.

---

## The Main Application File

The heart of the project is the `app.py` file. This is the script that Streamlit runs to create the interactive website. Let us walk through it block by block.

```python
import streamlit as st
import pandas as pd
import joblib
```

This first block imports the required libraries. Streamlit is shortened to `st` so that all UI calls are concise. Pandas is imported as `pd` which is the community convention. Joblib is pulled in because it allows loading the pre-trained model from disk. Without these imports nothing else in the script would function.

```python
st.title("NYC Airbnb Price Predictor")
```

Here I define the title that appears on top of the application. Streamlit makes it easy to render titles directly from Python code. This one line gives the app a professional header so users know the exact purpose when they open it.

```python
@st.cache_resource
def load_model():
    return joblib.load("models/price_model.joblib.xz")
```

This block defines a helper function. The `@st.cache_resource` decorator is critical. It ensures that the model file is loaded only once per session. Without caching, the app would reload the compressed model every time a user interacts with a widget, which would waste time. The function itself is very simple. It loads the serialized pipeline from the `models` folder and returns it. This allows the rest of the code to call the model without worrying about file operations.

```python
pipe = load_model()
```

This line executes the function and stores the loaded pipeline object into a variable named `pipe`. Now I have a ready-to-use machine learning pipeline in memory. This object contains both preprocessing steps and the trained estimator.

### Input Widgets

The next section of the code builds the interactive inputs. Streamlit allows each widget to directly feed values into variables.

```python
neighbourhood_group = st.selectbox("Neighbourhood Group", ["Manhattan","Brooklyn","Queens","Bronx","Staten Island"])
room_type = st.selectbox("Room Type", ["Entire home/apt","Private room","Shared room"])
minimum_nights = st.number_input("Minimum Nights", 1, 365, 3)
number_of_reviews = st.number_input("Number of Reviews", 0, 1000, 10)
reviews_per_month = st.number_input("Reviews per Month", 0.0, 30.0, 1.0)
calculated_host_listings_count = st.number_input("Host Listings Count", 0, 500, 1)
availability_365 = st.number_input("Availability (days/year)", 0, 365, 180)
```

Each call here creates a widget. The first two are dropdowns (select boxes) for categorical inputs. The others are numeric inputs with explicit minimums, maximums, and default values. The structure ensures that users cannot enter impossible data. For example, `minimum_nights` cannot be less than one and cannot exceed 365. This form design enforces domain rules and keeps the model inputs valid.

```python
df_in = pd.DataFrame([{
    "neighbourhood_group": neighbourhood_group,
    "room_type": room_type,
    "minimum_nights": float(minimum_nights),
    "number_of_reviews": float(number_of_reviews),
    "reviews_per_month": float(reviews_per_month),
    "calculated_host_listings_count": float(calculated_host_listings_count),
    "availability_365": float(availability_365)
}])
```

This block transforms the raw widget inputs into a pandas DataFrame. The model pipeline expects a structured table with the same column names used during training. Each value is explicitly cast to float where appropriate. By creating a DataFrame with one row, I ensure compatibility with scikit-learn’s API.

```python
if st.button("Predict Price"):
    prediction = pipe.predict(df_in)[0]
    st.success(f"Predicted Price: ${prediction:.2f} per night")
```

This conditional defines the prediction logic. The `st.button` call creates a clickable button. When pressed, the code inside the `if` block runs. It calls the pipeline’s `predict` method with the user DataFrame, extracts the first prediction, and displays it with a success message. The formatting rounds the value to two decimal places for clarity. This short block closes the loop between user input and model output.

---

## Model File

The third important piece in the repository is the serialized model stored at `models/price_model.joblib.xz`. This file contains a trained pipeline. The pipeline likely includes preprocessing steps such as encoding categorical variables, scaling numerical variables, and then a regression estimator. I chose to compress the file with `.xz` to reduce repository size. Streamlit loads it in real time whenever a user visits the app.

---

## Why the Structure Matters

Each file has a distinct role. The `requirements.txt` guarantees that the same environment can be rebuilt by anyone. The `app.py` script defines both the user interface and the backend logic. The model file captures the statistical learning from historical Airbnb data. Together these three elements allow the project to run seamlessly on any machine or platform that supports Streamlit.

---

## Extended Technical Deep Dive

The `load_model` function, while small, is central. By separating model loading into its own function, I created a clear boundary. Any changes to the model file path or loading strategy can be made here without affecting the rest of the code. The caching mechanism also highlights a subtle design principle: interactive apps need efficiency. Streamlit reruns the script on every interaction, so without caching the user experience would degrade quickly.

The input widget design is another area worth reflection. Every boundary chosen for numeric inputs is not random. For example, maximum nights at 365 reflects a real-world assumption that no one rents for more than a year. Setting upper bounds for reviews and host listings prevents unrealistic values that could destabilize the model. Each constraint encodes domain understanding into the interface.

The prediction block uses a simple `if` conditional. This is important because it ensures that predictions are not computed until the user explicitly requests them. Without the button, the app would predict continuously every time a widget changed. That would waste computation and also create confusion for the user. The explicit button click makes the workflow natural: adjust values, then confirm prediction.

---

## Lessons Learned

This project reinforced several key lessons. Machine learning models are only as useful as their deployment interface. A well-trained model sitting in a notebook has little impact. Wrapping it in a Streamlit app makes it accessible. Second, small helpers like caching or data validation make a huge difference in usability. Finally, documenting the process is as important as building it. By breaking down each block of code, I can revisit the project months later and still understand every choice.

---

## Closing Reflections

When I first saw that confusing pricing discrepancy, I only had a vague curiosity. Through this project I learned how to channel that curiosity into a structured technical artifact. I created an application that others can actually use. It takes raw inputs that anyone can provide and gives back a clear prediction. The experience taught me that data projects reach their true potential when they move beyond notebooks and become tools. That realization will continue to guide how I approach future projects.

---


## Additional Reflections on Model Deployment

One interesting challenge was balancing complexity with accessibility. The model itself could be made more advanced, but every additional preprocessing step adds risk during deployment. By saving the model pipeline directly, I avoided the need to manually replicate encoders or scalers. This made the application more robust. It also meant that the pipeline encapsulated every assumption made during training. Anyone loading the model would inherit those exact transformations without ambiguity.

Another subtle design choice was the directory layout. Placing the model file in a `models` folder separates it from the app code. This makes the repository cleaner and more extensible. If I train another version in the future, I can drop it in the same folder without touching the rest of the structure. Similarly, having a dedicated `requirements.txt` at the root ensures compatibility across environments. These small organizational details save hours of debugging later.

The decision to use Streamlit was deliberate. Alternatives like Flask or Django offer more customization but at the cost of complexity. Streamlit provides rapid development while still delivering a professional interface. For machine learning demos this tradeoff is ideal. It allows focusing on the model and logic rather than web infrastructure. The entire app fits within a few dozen lines of Python, yet it delivers real predictive power.

---
