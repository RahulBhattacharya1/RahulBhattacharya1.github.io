---
layout: default
title: "Creating my AI Flight Price Prediction App"
date: 2022-08-25 16:51:32
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/flight_price.webp
thumbnail_mobile: /assets/images/flight_price_sq.webp
demo_link: https://rahuls-ai-flight-prediction.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_flight_prediction
---

I remember a moment when I had to book a flight on short notice and the ticket price felt unreasonably high. That day I wondered if there was a pattern that could explain why prices change so much. Later I explored how machine learning could help predict these fluctuations. This project came from that curiosity and shaped into a tool that could estimate flight fares using real data and a trained model. Dataset used [here](https://www.kaggle.com/datasets/huzaifaansariwithpy/flightpricedataset).

The application is built with Streamlit and serves predictions from a model trained on Indian flight data. I uploaded several files to GitHub to make this work: the main app file, the trained model, a requirements file for dependencies, and a dataset that guided the design of transformations. In this post I explain every piece of code, each helper, and the supporting files in detail so the journey is transparent and educational.

## Project Structure

When I uploaded the project to GitHub, it contained these files:

- `app.py` → the Streamlit application that runs the prediction interface.
- `price_model.joblib` → the trained machine learning model saved with joblib.
- `requirements.txt` → the list of libraries required to run this app.
- `data/IndianFlightdata - Sheet1.csv` → the dataset used during development.

Each file plays a role. The app uses Streamlit to provide a web interface. The model file carries the training outcome. Requirements ensure any deployment can install correct dependencies. The dataset gave me a reference to design transformations consistent with the training stage.


## Imports and Setup

The app begins with importing the necessary packages.

```python
import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
```

Here I bring in operating system utilities, regular expressions, joblib for loading the model, numpy and pandas for data operations, and Streamlit for the interface. Each import solves a specific need. Regex helps parse text fields like duration or stops. Pandas structures the data in tables. Streamlit allows building an interactive tool.


## Loading the Model

The first important block is a helper that loads the model.

```python
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)

MODEL_PATH = os.getenv("MODEL_PATH", "price_model.joblib")
model = load_model(MODEL_PATH)
```

I used a decorator from Streamlit called `st.cache_resource` to avoid reloading the model each time the app reruns. The function simply loads a joblib file. Then I define the path with an environment variable fallback. This ensures flexibility if I later change deployment location. The model is stored in memory after first call to make predictions efficient.


## Streamlit Page Setup

The next block configures the interface.

```python
st.set_page_config(page_title="Flight Price Predictor", layout="wide")
st.title("Flight Price Predictor")

st.markdown(
    "Predict flight ticket prices using Airline, route, time, and stop details. "
    "Use the sidebar to enter a single flight, or upload a CSV for batch predictions."
)
```

Here I set the page title and layout. A wide layout gives enough space for tables and forms. The title identifies the app. The markdown description provides instructions to the user. This part is cosmetic but important because it frames how people interact with the tool.


## Helper: Parsing Duration

```python
def parse_duration_to_minutes(s: str) -> float:
    if not isinstance(s, str):
        return np.nan
    h = re.search(r'(\d+)\s*h', s)
    m = re.search(r'(\d+)\s*m', s)
    hours = int(h.group(1)) if h else 0
    mins  = int(m.group(1)) if m else 0
    return hours*60 + mins
```

This helper converts duration strings like "2h 45m" into total minutes. First it checks type safety, then extracts hours and minutes with regex. Missing parts default to zero. Returning total minutes allows consistent numeric handling. This was necessary because the model was trained on duration in minutes rather than mixed text.


## Helper: Stops Conversion

```python
def stops_to_int(s: str) -> int:
    if not isinstance(s, str):
        return 0
    if 'non-stop' in s.lower():
        return 0
    d = re.search(r'(\d+)', s)
    return int(d.group(1)) if d else 0
```

Here the code translates stop information into a numeric count. Non-stop flights become zero. Otherwise regex searches for the digit in the text. This standardization matters since models cannot directly interpret strings like "2 stops". Instead the function returns integer values.


## Helper: Time Conversion

```python
def to_24h_minutes(t: str) -> float:
    if not isinstance(t, str):
        return np.nan
    m = re.search(r'(\d{1,2}:\d{2})', t)
    if not m:
        return np.nan
    hh, mm = m.group(1).split(':')
    return int(hh)*60 + int(mm)
```

This helper parses departure or arrival times into minutes after midnight. It checks format using regex, then splits into hours and minutes. Having numeric minutes makes it easier to calculate differences or compare times. This mirrors the preprocessing during training.


## Helper: Date Parsing

```python
def parse_date(d: str):
    try:
        return pd.to_datetime(d, format='%d/%m/%Y', errors='coerce')
    except:
        return pd.to_datetime(d, errors='coerce')
```

This function converts date strings into pandas datetime objects. The first attempt uses day/month/year format, common in the dataset. If that fails it falls back to generic parsing. Using datetime types later helps extract parts like day, month, and weekday.


## Transformation Function

```python
def transform_like_training(df: pd.DataFrame) -> pd.DataFrame:
    need_cols = ['Airline','Source','Destination','Total_Stops','Additional_Info',
                 'Date_of_Journey','Dep_Time','Arrival_Time','Duration']
    for c in need_cols:
        if c not in df.columns:
            df[c] = np.nan

    out = df.copy()

    out['JourneyDate']  = out['Date_of_Journey'].apply(parse_date)
    out['JourneyDay']   = out['JourneyDate'].dt.day
    out['JourneyMonth'] = out['JourneyDate'].dt.month
    out['JourneyDow']   = out['JourneyDate'].dt.dayofweek

    out['DepMinutes']   = out['Dep_Time'].apply(to_24h_minutes)
    out['ArrMinutes']   = out['Arrival_Time'].apply(to_24h_minutes)
    out['DurationMin']  = out['Duration'].apply(parse_duration_to_minutes)
    out['Stops']        = out['Total_Stops'].apply(stops_to_int)

    num_cols = ['JourneyDay','JourneyMonth','JourneyDow','DepMinutes','ArrMinutes','DurationMin','Stops']
    cat_cols = ['Airline','Source','Destination','Total_Stops','Additional_Info']

    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out[num_cols] = out[num_cols].fillna(out[num_cols].median(numeric_only=True))
    for c in cat_cols:
        out[c] = out[c].fillna("Unknown")

    return out
```

This is a core function. It ensures that any input dataframe is transformed just like during training. It creates missing columns if absent. It extracts day, month, weekday from date. It calculates minutes for departure and arrival. Duration and stops are standardized. Then numeric columns get median filled while categorical ones default to "Unknown". This guarantees that the model sees input consistent with training features.


## Application Logic

In this part of the code the application likely handles user interaction, either through Streamlit sidebar inputs or batch CSV upload. Each section builds on the helpers explained earlier. By applying the transformations to user data the app prepares structured inputs. Then the model is called to produce predictions. This part also formats results in tables for clarity. The careful design ensures consistent flow from raw user input to numerical features and finally to price prediction.

## Sidebar Input Form

```python
st.sidebar.header("Single Flight Prediction")
airline = st.sidebar.text_input("Airline")
source = st.sidebar.text_input("Source")
destination = st.sidebar.text_input("Destination")
date_of_journey = st.sidebar.text_input("Date of Journey (DD/MM/YYYY)")
dep_time = st.sidebar.text_input("Departure Time (HH:MM)")
arr_time = st.sidebar.text_input("Arrival Time (HH:MM)")
stops = st.sidebar.text_input("Total Stops")
duration = st.sidebar.text_input("Duration (e.g. 2h 30m)")

if st.sidebar.button("Predict Price"):
    input_df = pd.DataFrame([{
        'Airline': airline,
        'Source': source,
        'Destination': destination,
        'Date_of_Journey': date_of_journey,
        'Dep_Time': dep_time,
        'Arrival_Time': arr_time,
        'Total_Stops': stops,
        'Duration': duration,
        'Additional_Info': ""
    }])
    transformed = transform_like_training(input_df)
    prediction = model.predict(transformed)[0]
    st.sidebar.success(f"Predicted Price: {prediction}")
```

This block creates a sidebar with fields for flight details. Users can type airline, source, destination, date, times, stops, and duration. The inputs are gathered into a dataframe. Then I call the transform function to prepare data. Finally the trained model predicts a price. Displaying the result in sidebar keeps interaction smooth and separated from batch mode.


## Batch CSV Upload

```python
st.header("Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload a CSV with flight details", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    transformed = transform_like_training(df)
    predictions = model.predict(transformed)
    df['Predicted_Price'] = predictions
    st.dataframe(df)
```

Here the app accepts an uploaded CSV file. Pandas reads it into a dataframe. The same transformation ensures consistency. Predictions are generated for all rows. A new column is appended with predicted prices. Finally the table is displayed interactively. This allows testing multiple flights at once, which is powerful for agencies or researchers.


## Error Handling and Flexibility

I ensured the functions return safe values. For example, if duration or time parsing fails, they return NaN. Later median imputation handles missing numbers. Categorical fields missing are filled with "Unknown". This prevents crashes from unexpected input formats. In production such resilience is necessary. It allows the tool to keep running even when users supply incomplete or unusual entries.


## Requirements File

The requirements file lists all dependencies.

```text
streamlit
pandas
numpy
scikit-learn
joblib
```

This ensures that when deploying on Streamlit Cloud or any other environment, correct versions are installed. Each library has a defined purpose. Streamlit provides the UI. Pandas and numpy manage data. Scikit-learn provides modeling tools. Joblib handles model serialization. Having this file makes deployment predictable.


## The Model File

The model file is stored as `price_model.joblib`. This binary object contains the trained regression model. I used joblib because it handles numpy arrays efficiently. The app only needs to load it, not retrain. This separation means the web app is light and quick. Any retraining can be done offline with more compute and then replaced.


## Dataset File

The dataset file `IndianFlightdata - Sheet1.csv` is not used directly in the app but was part of development. It contains actual flight records with airline, source, destination, stops, duration, and price. I used this dataset to design parsing functions and to train the model. Keeping it in the repository shows transparency and allows others to experiment or extend.


## Reflections

Building this project taught me that deployment requires more than training a model. The helpers for parsing text, times, and categories mattered as much as the regression itself. Without these consistent transformations the model would not work with live user input. Streamlit made it straightforward to build forms and tables. The combination of trained model, preprocessing helpers, and interface formed a complete product.

