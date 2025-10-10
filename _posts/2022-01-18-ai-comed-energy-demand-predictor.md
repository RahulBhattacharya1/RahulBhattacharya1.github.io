---
layout: default
title: "Building my AI Comed Energy Demand Predictor"
date: 2022-01-18 11:43:26
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/comed.webp
thumbnail_mobile: /assets/images/comed_predict_sq.webp
demo_link: https://rahuls-ai-comed-energy-demand-forecasting.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_comed_energy_demand_forecasting
featured: true
synopsis: Here I analyze seasonal electricity consumption patterns and develops a forecasting tool using historical demand data. It produces an interactive app for visualizing and predicting future usage, offering practical insights into demand spikes, efficiency, and planning for different weather-driven consumption trends.
custom_snippet: true
custom_snippet_text: Forecasts electricity demand patterns with interactive visualizations.
---

A few months ago I noticed how unpredictable our local electricity usage felt during different seasons. In summer, air conditioning made demand spike in late afternoons. In winter, heating created another curve that felt less predictable. I started wondering if I could design something that helped me visualize and forecast this demand better. That curiosity eventually grew into a small project where I took historical demand data and built an interactive app to forecast usage for the future. This blog post is my complete breakdown of how I built it step by step. Dataset used [here](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption).

When I began this journey I did not expect to get so deeply into the process of building a forecasting pipeline. But I wanted something practical that I could host easily and share with others. I decided to use Streamlit for the app, since it let me create interactive dashboards without much overhead. I chose Python libraries that I already worked with like pandas, NumPy, and joblib for handling models. What follows is a complete explanation of the code, the helpers, and the files I uploaded to GitHub for this to run smoothly on GitHub Pages or Streamlit Cloud.

---

## Project Structure

When I unzipped my project it had the following structure:

```
ai_comed_energy_demand_forecasting-main/
│
├── app.py
├── COMED_hourly.csv
├── requirements.txt
└── models/
    └── comed_forecaster.joblib
```

Each file plays an important role. The `app.py` file is the main Streamlit application. The dataset `COMED_hourly.csv` contains historical hourly electricity demand for the COMED region. The `requirements.txt` file lists the Python dependencies so the app can install them automatically. Finally, the `comed_forecaster.joblib` file is my pre-trained forecasting model that gets loaded at runtime.

---

## The Main Application (app.py)

The file `app.py` is the heart of the project. It controls the user interface, loads the model, and runs the forecasting functions.

```python
# app.py - Streamlit app for COMED demand forecasting
import os
import io
import math
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import timedelta

st.set_page_config(page_title="COMED Demand Forecaster", layout="wide")
```

I start with imports. I use standard libraries like `os`, `io`, `math`, and `time` for file handling and simple utilities. I rely on `joblib` for loading the trained model, `numpy` and `pandas` for data processing, and `matplotlib` for plots. Streamlit is imported as `st` because it is the framework that runs the app. I also import `timedelta` to manage time offsets in forecasts.

The `set_page_config` line defines the title and layout for the Streamlit app. I chose a wide layout to give space for charts.

---

## Loading the Model

```python
@st.cache_data
def load_artifacts(path="models/comed_forecaster.joblib"):
    artifacts = joblib.load(path)
    return artifacts
```

This function loads the forecasting model. I decorate it with `st.cache_data` so that Streamlit only loads it once and then reuses it across user sessions. This prevents repeated disk access and speeds up the app. The function returns the model object, which I stored earlier in `comed_forecaster.joblib`. The caching layer ensures smoother performance when multiple users interact with the app.

---

## Building Features

```python
def build_features(dt_index, history_series):
    df_f = pd.DataFrame({"Datetime": dt_index})
    df_f['hour'] = df_f['Datetime'].dt.hour
    df_f['dow'] = df_f['Datetime'].dt.dayofweek
    df_f['month'] = df_f['Datetime'].dt.month
    df_f['is_weekend'] = (df_f['dow'] >= 5).astype(int)
    try:
        import holidays as pyholidays
        us_holidays = pyholidays.US()
        df_f['is_holiday'] = df_f['Datetime'].dt.date.astype('datetime64').isin(us_holidays).astype(int)
    except Exception:
        df_f['is_holiday'] = 0

    df_f['lag_1'] = np.nan
    df_f['lag_24'] = np.nan
    df_f['lag_168'] = np.nan
    for i in range(len(dt_index)):
        if i > 0:
            df_f.loc[i, 'lag_1'] = history_series[i-1]
        if i >= 24:
            df_f.loc[i, 'lag_24'] = history_series[i-24]
        if i >= 168:
            df_f.loc[i, 'lag_168'] = history_series[i-168]
    return df_f
```

This function is one of the most important helpers. It takes a datetime index and a historical demand series and builds a feature matrix. First it extracts standard calendar features like hour, day of week, and month. It flags weekends and holidays since demand often shifts on those days. Then it adds lag features: one hour ago, one day ago (24 hours), and one week ago (168 hours). These lag features capture past behavior, which is critical in time series forecasting. Without lags the model would not capture autocorrelation in the demand pattern.

---

## Running Forecasts

```python
def run_forecast(model, history_series, forecast_steps=48):
    dt_index = pd.date_range(start=history_series.index[-1] + timedelta(hours=1), periods=forecast_steps, freq='H')
    df_features = build_features(dt_index, list(history_series.values) + [np.nan]*forecast_steps)
    y_pred = model.predict(df_features.fillna(0))
    forecast_df = pd.DataFrame({"Datetime": dt_index, "Forecast": y_pred})
    return forecast_df
```

The `run_forecast` function generates future predictions. It first builds a new datetime index starting one hour after the last available point. It then calls the `build_features` helper to construct the inputs for the model. The model is applied to the features and predictions are returned in a DataFrame. I chose a default of 48 steps, which means two days ahead at an hourly level. The output DataFrame includes both the timestamps and the forecasted demand values.

---

## Loading Historical Data

```python
def load_history(file_path="COMED_hourly.csv"):
    df = pd.read_csv(file_path, parse_dates=['Datetime'])
    df = df.sort_values("Datetime")
    df = df.set_index("Datetime")
    return df
```

This helper loads the CSV data file. It parses the `Datetime` column into real timestamps, sorts the values, and sets the index to `Datetime`. By making the index a datetime type, plotting and resampling become straightforward. This step is critical because without a clean time index the forecasting pipeline would break.

---

## Streamlit App Layout

```python
def main():
    st.title("COMED Energy Demand Forecasting")
    st.write("This app forecasts hourly demand using historical data and a pre-trained model.")

    data = load_history()
    model = load_artifacts()

    st.subheader("Historical Data")
    st.line_chart(data["DEMAND"].tail(168))

    st.subheader("Forecast")
    forecast_df = run_forecast(model, data["DEMAND"], forecast_steps=48)
    st.line_chart(forecast_df.set_index("Datetime"))

if __name__ == "__main__":
    main()
```

The `main` function defines the user interface. It starts with a title and description. Then it loads the historical dataset and the trained model. It shows the last week of demand in a line chart for context. Next it runs the forecast for 48 hours and shows the results in another chart. Finally, the conditional `if __name__ == "__main__":` ensures that `main()` only runs when the file is executed directly, not when it is imported elsewhere. This layout keeps the app clean and intuitive.

---

## Supporting Files

### Dataset: COMED_hourly.csv

This CSV file contains the raw historical data. It includes hourly demand values for the COMED region. The app needs this file because the forecast depends on recent history. Without it, the lag features and model inputs would not work.

### Model: comed_forecaster.joblib

The pre-trained model is stored in the `models` folder. I used joblib to persist the model after training it offline. Streamlit loads this file using the `load_artifacts` function. By saving the model separately I avoid retraining each time the app runs. This also makes deployment lighter.

### Requirements.txt

The requirements file ensures consistent environments. Here are its contents:

```python
streamlit
pandas
numpy
matplotlib
joblib
holidays
```

This file lists dependencies that get installed in the hosting environment. By including it in GitHub, Streamlit Cloud or any other service can build the app reliably. It prevents version mismatch errors and makes the setup portable.

---

## Reflections

Working on this project showed me the value of clean feature engineering in time series tasks. Lag features turned out to be the most powerful inputs. Handling holidays and weekends also improved forecast stability. I realized how crucial it is to think about domain knowledge when preparing features. Electricity demand follows human behavior patterns, so encoding calendar information was key.

Deploying the project with Streamlit made it easy to share. I only needed to upload the code, data, and model into GitHub. Then the rest worked seamlessly. For anyone curious about forecasting energy demand, this project is a practical template to start with.

---

## Deeper Dive into Feature Engineering

The `build_features` function deserves a closer inspection because it lies at the core of making time series models work. In real-world scenarios, raw demand values are not enough for a model to understand future patterns. By expanding the datetime index into components like hour, day of week, and month, the model can capture seasonality effects. For example, evening peaks on weekdays happen at consistent hours, and weekend behavior often differs drastically.

Holidays introduce even more variation. Demand on national holidays like Thanksgiving or Christmas can be significantly lower compared to regular weekdays. That is why I attempted to integrate the `holidays` Python package. By checking if each date falls into the United States holiday calendar, I added another binary feature called `is_holiday`. If the package fails to load or the environment lacks it, I safely default to zero to avoid breaking the pipeline.

The lag features are another pillar. By including lag_1, lag_24, and lag_168, I let the model look at the immediate past, the same time yesterday, and the same time last week. These are common anchor points in electricity demand forecasting. They help the model align short-term momentum with medium- and long-term cycles. Without these lags, the model might only guess based on seasonal averages. With them, it becomes far more responsive to real fluctuations.

---

## Understanding the Forecast Horizon

When designing the `run_forecast` function, I had to decide how far into the future the app should look. I settled on 48 hours, which covers two full days. This choice balances usability and reliability. Forecasts too far into the future lose accuracy, while forecasts too short are less useful. Two days allow both operational planning and near-term adjustments. The app user can visualize what to expect tomorrow and the day after, with hourly resolution.

Another design decision was how to create the new datetime index. By starting exactly one hour after the last available timestamp, the forecast aligns seamlessly with the historical series. The index is generated with `pd.date_range`, which ensures equally spaced hourly intervals. This choice is vital because any gap or misalignment in time steps would create misleading plots and weaken the predictive power.

---

## Visualization Choices

The app currently uses `st.line_chart` to display both historical data and forecasts. While this is the simplest option, it is powerful because it updates dynamically as new data loads. I could have used matplotlib or Plotly charts inside Streamlit, but for this project, simplicity was my priority. The goal was to allow smooth interaction without overwhelming the viewer with options. By focusing on line charts, I highlight the temporal shape of demand, which is what really matters in this context.

---

## Why Streamlit Made Sense

One of the best parts of this project was how Streamlit simplified deployment. Normally, building a forecasting dashboard would require a backend service, a frontend interface, and a server for hosting. With Streamlit, I wrote everything in one Python file and pushed it to GitHub. The cloud platform could then run it directly. This let me focus on data preparation and forecasting logic rather than dealing with server administration.

Caching with `@st.cache_data` was another Streamlit feature that played a huge role. Without caching, loading the model file each time would slow down the app. By caching, Streamlit ensures that the model stays in memory once loaded. That performance gain is noticeable especially when users refresh the page or adjust parameters.

---

## Handling the Dataset

The `load_history` helper highlights the importance of data preprocessing. The raw CSV might contain unordered rows or misaligned timestamps. By explicitly sorting and indexing, I removed potential inconsistencies. This step often gets overlooked, but it prevents silent errors later. For example, if timestamps were out of order, the lag calculation would pull the wrong values. By enforcing order and indexing by datetime, I guaranteed chronological integrity.

I also deliberately avoided filling missing values here. For hourly demand, missing data can happen due to sensor outages or recording errors. In a production-grade system, I would consider interpolation or imputation. But for this project, I kept it minimal and left gaps as they were, since my model could handle them with default fills.

---

## Lessons Learned from Model Deployment

Packaging the trained model into a `.joblib` file was a practical choice. It ensured that the app did not depend on training scripts or heavy computation during startup. Instead, the pre-trained object could be reloaded instantly. This design mirrors what happens in many production ML systems, where training and serving pipelines are separated. By decoupling these steps, I made the app more stable and easier to run in different environments.

---

## Future Extensions

While this app already provides useful forecasts, I see multiple directions for future growth. I could extend the forecast horizon to a week with probabilistic intervals. I could add user inputs for selecting specific past ranges, which would adjust forecasts interactively. Another idea is to include temperature data, since energy demand is tightly linked with weather. Adding external regressors like this would enrich the feature set.

I could also work on improving visualization. Interactive charts with zooming, tooltips, and comparison lines would enhance exploration. Streamlit supports integration with Plotly, which could bring this level of interactivity. I intentionally started with simplicity, but the project has space to grow.

---

## Broader Reflections

This project reminded me that small-scale experiments can teach big lessons. I learned how to manage dependencies, structure a Streamlit app, and carefully prepare features. I also saw how caching, lag creation, and robust data loading all contribute to user experience. These skills transfer to many other projects, not just energy forecasting. Time series problems appear in finance, healthcare, retail, and even web traffic monitoring.

Most importantly, I realized how approachable forecasting becomes when framed as a step-by-step process. Breaking it into files, helpers, and charts turned a complex task into something manageable. By documenting it here, I also created a resource that I can revisit later when I take on more advanced forecasting challenges.

---


## Detailed Breakdown of Each Helper Function

### load_artifacts

The `load_artifacts` function does one simple but critical task. It retrieves the trained forecasting model from disk. By isolating this step into its own helper, I created a clean separation between model management and application logic. This means if I ever retrain the model with new data, I only need to replace the joblib file and the rest of the app remains unchanged. That modularity is what makes maintenance easier in the long term.

### build_features

This function deserves to be emphasized again because of its layered logic. It constructs a DataFrame starting from the datetime index. Then it builds temporal features, categorical flags, and lagged series. Each block of code adds a distinct layer of intelligence. The holiday block is wrapped in a try-except to ensure robustness in environments where the holidays package might not be installed. This design allows the app to degrade gracefully rather than fail completely.

### run_forecast

The `run_forecast` function is where the model actually interacts with future time steps. Its role is to take the history, generate new time stamps, build features, and produce predictions. It encapsulates the forecasting pipeline in a few concise steps. This separation also allows me to test it independently of the Streamlit interface. If I ever wanted to run batch forecasts offline, I could reuse this function in a script without modification.

### load_history

The `load_history` helper is simple but indispensable. It enforces chronological order and sets the time index, which are preconditions for any proper time series model. It is lightweight in code but heavy in impact. Without it, every downstream function would risk inconsistencies. By dedicating a function to this single task, I ensured clarity and reliability.

### main

The `main` function is the glue that brings everything together for the user. It manages loading, visualization, and interaction. Its role is not computation but orchestration. By isolating this as the entry point, I followed the common Python convention that improves readability and structure.

---

## Deployment and Hosting Details

For deployment, I uploaded this project to my GitHub repository. With the `requirements.txt` file in place, Streamlit Cloud was able to detect dependencies and install them automatically. The presence of `app.py` as the main entry script made the deployment process straightforward. Once deployed, the app became accessible through a public URL. I could share the link with others and they could see forecasts generated in real time.

This approach avoids the complexities of Docker images or cloud servers. It makes the project accessible to anyone familiar with GitHub. They only need to fork or clone the repo, push to Streamlit Cloud, and get the same result. This democratization of deployment is one of the reasons I enjoyed using Streamlit.

---

## Comparison with Other Approaches

If I had chosen Flask or Django, I would have needed to manage templates, routes, and a separate frontend. That would introduce more moving parts and slow me down. Similarly, if I had built a pure Jupyter Notebook solution, it would not have been as shareable for non-technical users. Streamlit strikes the right balance by being both simple and deployable.

In terms of modeling, I could have used advanced deep learning approaches like LSTMs or Transformers. But for this stage, a classical forecasting model with engineered features provided a better tradeoff between accuracy and interpretability. It also made the app lighter, since deep learning models would have increased both file size and computational requirements.

---

## Final Thoughts

Writing this blog was not only about sharing code but also about recording the thought process. Each decision, from caching to lagging, reflects practical tradeoffs. I balanced accuracy with performance, robustness with simplicity, and technical depth with accessibility. These tradeoffs are present in every real-world project. A perfect solution rarely exists, so engineering is about balancing constraints.

This project gave me a chance to experiment, learn, and build something that others can use. I know that in the future I will return to this foundation and expand it. For now, it stands as a complete and functional example of energy demand forecasting using historical data, engineered features, and a streamlined interface.

---

## Conclusion

This blog post explained every block of the code and all the files that I used. From the dataset and the model to the Streamlit app, each piece fits into the bigger picture of building a demand forecasting pipeline. I walked through helpers like `build_features`, the forecast runner, and the Streamlit layout. Every function had a specific role in turning raw historical data into actionable future predictions. With this foundation, I can extend the app further, like adding confidence intervals, custom forecast horizons, or even comparing models. For now, it stands as a working demonstration of how to combine data science, time series, and deployment into one coherent project.

---
