---
layout: default
title: "Creating my AI Electricity Demand Predictor"
date: 2023-06-12 12:45:31
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/electricity_demand.webp
demo_link: https://rahuls-ai-electricity-demand-forecasting.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_electricity_demand_forecasting
---

I have often faced situations where electricity usage patterns seemed unpredictable. During hot summers the air conditioners would run continuously, while in colder months heaters consumed most of the energy. These variations made me wonder how organizations can plan for electricity supply and demand with accuracy. A personal incident where a power outage disrupted my work made me think more deeply about the role of demand forecasting. That was the trigger for me to design a forecasting tool that could provide insights into how future electricity demand may behave. Dataset used [here](https://www.kaggle.com/datasets/rhtsingh/india-monthly-electricity-consumption-20192025).

The project presented here uses a machine learning model to forecast electricity demand in India. I built it as a web application so that anyone can interact with it directly. I decided to use the Prophet library because it is well suited for time series forecasting. The project is simple in its interface but powerful in the underlying logic. In the following sections, I will explain every file, every function, and each block of code that makes this project work.

---

## Files in the Repository

The repository contains three main components:

1. **app.py** - the main Streamlit application where the interface and forecasting logic are combined.
2. **requirements.txt** - a file listing the Python dependencies required to run the project.
3. **models/electricity_forecast.joblib** - the pre-trained forecasting model stored in serialized format.

Each of these files plays a crucial role. I will now explain them in depth and show how they connect to make the full application functional.

---

## The requirements.txt File

This file ensures that the correct versions of libraries are installed when running the project. The contents are shown below.

```python
streamlit
prophet
pandas
joblib
```

### Explanation

The **streamlit** package is required to build the interactive web interface. Without it the project would only run as a plain Python script. The **prophet** package provides the forecasting capabilities. It allows me to build models that capture trends and seasonality in the data. The **pandas** package is used for handling data frames, especially when preparing time series data and manipulating future dates. The **joblib** package is essential for loading the pre-trained model stored in the models folder. Together these packages form the environment in which the forecasting tool can run seamlessly.

---

## The app.py File

The app.py file is the heart of the application. It contains imports, the loading of the model, the user interface built with Streamlit, and the plotting of forecasts. The complete code is shown below.

```python
import streamlit as st
import joblib
import pandas as pd
from prophet import Prophet

st.title("India Electricity Demand Forecasting")

# Load model
model = joblib.load("models/electricity_forecast.joblib")

# User input
months = st.slider("Months to forecast", 1, 24, 12)

future = model.make_future_dataframe(periods=months, freq='M')
forecast = model.predict(future)

st.line_chart(forecast[['ds','yhat']].set_index('ds'))
```

---

### Import Statements

The script begins with four import statements. These lines bring in the necessary packages for the app. The **streamlit** import is labeled as **st** to keep the code concise. This convention allows me to call functions like st.title or st.slider without writing the full package name. The **joblib** import allows me to load the serialized model from disk. The **pandas** import is necessary because both Prophet and Streamlit require data frames as inputs. The **prophet** import brings in the forecasting library that provides the methods for generating future values. Each of these imports is critical for different parts of the project.

### Application Title

The line `st.title("India Electricity Demand Forecasting")` creates a heading at the top of the Streamlit app. It signals to the user what the purpose of the tool is. Streamlit automatically formats this as a visually prominent title. It makes the app user-friendly and professional.

### Model Loading

The statement `model = joblib.load("models/electricity_forecast.joblib")` loads the saved forecasting model. The model had already been trained and exported into the models directory. By loading it here I can reuse the training effort without having to rebuild the model every time the app runs. This is efficient and ensures consistent results.

### User Input Through Slider

The block `months = st.slider("Months to forecast", 1, 24, 12)` defines an interactive slider in the user interface. This slider lets the user choose how many months into the future they want the forecast to extend. The first number is the minimum value allowed, the second is the maximum, and the third is the default starting point. By default, it is set to 12 months. This interactive element makes the app flexible because different users can explore forecasts of different lengths.

### Creating Future Data Frame

The function `model.make_future_dataframe(periods=months, freq='M')` is used to generate a future set of dates. It creates rows for each month into the future as specified by the slider. The parameter **periods** equals the number chosen by the user. The parameter **freq='M'** indicates that the intervals are monthly. This helper is important because Prophet requires a properly formatted data frame of future dates before it can generate predictions.

### Forecast Generation

The call `model.predict(future)` uses the trained model to create forecasts for the future data frame. The output is a data frame with several columns. The most relevant ones here are **ds** for the date and **yhat** for the predicted demand. By calling this method I obtain predictions for every future month requested by the user.

### Displaying the Forecast

The last block `st.line_chart(forecast[['ds','yhat']].set_index('ds'))` displays a line chart. Streamlit takes care of the visualization, rendering an interactive chart in the browser. The chart plots dates on the x-axis and predicted electricity demand on the y-axis. This chart is the core output of the application because it translates numeric predictions into a visual story that users can understand at a glance.

---

## The Model File

The file `models/electricity_forecast.joblib` contains the serialized model. It was trained beforehand using Prophet on historical electricity demand data. Saving it in joblib format allows for quick loading. The model encodes all parameters, seasonal components, and learned patterns from the training stage. Although the training script is not included in this repository snapshot, the model itself carries all the intelligence required for future predictions. This approach is efficient because it keeps the deployed application lightweight.

---

## How the Components Work Together

1. The **requirements.txt** ensures that all dependencies are installed correctly.
2. The **app.py** file serves as the main script that ties user interaction with prediction logic.
3. The **joblib model file** provides the pre-trained intelligence needed to produce accurate forecasts.

When combined they allow the app to run smoothly on any environment that supports Streamlit.

---

## Technical Breakdown of Functions and Helpers

### joblib.load

The function `joblib.load` is a helper that reads a binary file from disk and reconstructs the original Python object. In this case it loads the Prophet model object with all its learned state. This is critical because it means training does not need to be repeated for each run.

### st.slider

The function `st.slider` is a Streamlit helper that builds an interactive slider component. It accepts parameters for label, minimum value, maximum value, and default value. In this app it helps users select how far into the future they want to look. This interactivity provides control to the end user without requiring code changes.

### model.make_future_dataframe

This function is part of the Prophet API. It generates a new data frame that extends into the future. It preserves the structure of the original training data while adding new dates. This helper is essential for forecasting tasks because predictions can only be made for dates that exist in the input frame.

### model.predict

The predict function uses the internal logic of the trained Prophet model. It applies seasonal patterns, trend analysis, and uncertainty intervals to produce future predictions. It is the key function that transforms raw future dates into meaningful forecasted values.

### st.line_chart

The line_chart function from Streamlit provides an easy way to plot time series data. It expects a data frame with an index and one or more columns. Streamlit automatically renders it as an interactive chart with hover information. This makes the forecast visually interpretable.

---


## Extended Breakdown of Code Blocks

### Detailed Explanation of Imports

The imports in app.py may look simple but each has an important role that must be considered. Streamlit is not only used for interface creation but also for layout handling and event-driven execution. Every time a user moves the slider or reloads the page the Streamlit runtime re-executes the script from top to bottom. This means imports must be lightweight and efficient.

Joblib is not just a serializer, it is optimized for large numpy arrays and models. It provides faster loading compared to pickle when dealing with machine learning models. Prophet itself is a powerful forecasting library built on top of Stan. It uses an additive model to capture non-linear trends with seasonality. The import from prophet gives access to both the Prophet class and its utilities. Together these imports form the backbone of the application.

### Expanded View on Title Definition

The st.title function is more than a cosmetic label. In practice, it defines the first impression of the app. A good title helps a user quickly understand the purpose. By keeping the title descriptive and short, I ensure that the message is clear. Streamlit automatically applies formatting and places it at the top of the page. This allows users to recognize the tool without confusion. It also helps when the app is embedded or shared because the title is often used in previews.

### Expanded Discussion on Model Loading

The line with joblib.load connects the app with the pre-trained forecasting brain. This design pattern is important because training time series models can be computationally expensive. For example, Prophet training may involve Bayesian parameter estimation which can take minutes to complete. By training once and saving the model I can skip this cost. Loading the model takes only a fraction of a second. Another advantage is reproducibility.

### Extended Explanation of User Slider

The slider in Streamlit is not only a way to choose a number. It represents a contract between the user and the app. By exposing the number of months as a slider I give control without overwhelming the user with multiple options. Sliders are intuitive because they mimic real-world controls. The parameters in st.slider ensure that values cannot go out of range. The minimum set at 1 ensures no empty forecast, the maximum of 24 ensures practical limits, and the default at 12 provides a balanced starting point.

### Deep Dive on Future Data Frame Creation

The make_future_dataframe function in Prophet deserves more explanation. When it generates future dates it aligns them with the same frequency as the training data. This consistency is crucial because mismatched frequency could break the forecasting process. For instance, if training data was monthly and future is generated daily, predictions would not match expectations. By explicitly setting freq='M' I enforce monthly granularity. The resulting data frame has one column ds which holds the dates.

### Extended Discussion on Forecast Generation

When model.predict is called it performs several internal operations. Prophet decomposes the time series into trend, yearly seasonality, weekly seasonality, and holiday effects if defined. It then applies these components to the future dates. The output includes columns like yhat_lower and yhat_upper which represent uncertainty intervals. In this app I focused on the yhat column because it provides the central prediction.

### Detailed Breakdown of Chart Display

The line chart in Streamlit is simple but effective. Behind the scenes Streamlit uses Vega-Lite to render charts in the browser. This means charts are interactive by default. Users can hover over points to see exact values. The call to set_index('ds') ensures that the date column becomes the index, aligning the data properly on the x-axis. Without setting the index the chart might display incorrectly.

---

## Extended Technical Commentary

### Why Prophet was Chosen

Prophet was selected because it balances simplicity with power. Traditional time series models like ARIMA require careful tuning of parameters and assumptions about stationarity. Prophet automates much of this by fitting additive models with defaults that work well in many business scenarios. It can handle missing data, outliers, and irregular trends. This flexibility makes it ideal for electricity demand which is influenced by many factors such as weather, holidays, and economic activity.

### Why Streamlit Was Used

Streamlit allows data scientists to turn scripts into apps with minimal effort. Instead of writing HTML, CSS, and JavaScript I can focus on Python logic. This reduces development time dramatically. The reactivity of Streamlit ensures that changes in input update the output immediately. For forecasting this is valuable because users want to explore multiple scenarios quickly. Streamlit also makes deployment simple, whether on Streamlit Cloud, Hugging Face Spaces, or custom servers.

### Handling Data with Pandas

Even though the app code looks small, Pandas is doing significant work in the background. It ensures that the forecast data is structured correctly for both Prophet and Streamlit. Pandas handles indexing, column selection, and reshaping. For example, when selecting forecast[['ds','yhat']] I rely on Pandas to filter columns. When calling set_index I rely on Pandas to restructure the data. These operations make the downstream visualization seamless.

### The Role of joblib

Joblib deserves a separate note because it ensures model persistence. In many projects, saving models incorrectly leads to compatibility issues. Joblib stores numpy arrays efficiently, compresses them if required, and loads them quickly. This makes it a preferred tool for serializing scikit-learn and Prophet models. Without joblib I would have to retrain models every time or manage fragile pickle files.

---

## Lessons Learned

Working on this project showed me that simplicity is powerful. A short app with only a few lines of code can still deliver strong impact when backed by the right libraries. Every helper in the code has a defined purpose. Streamlit makes the app accessible, Prophet provides the forecasting logic, Pandas structures the data, and Joblib connects the app with the pre-trained model. Together they create a smooth experience.

I also learned the importance of presenting forecasts visually. A numeric table might be accurate but it fails to engage. A line chart tells a story of rising and falling demand. This helps decision makers act faster. By exposing this app to others I realized how useful it is to create interactive tools that combine machine learning with user interface elements.

---



## Possible Extensions

This application in its current form provides monthly forecasts for up to two years. However, it can be extended in several directions. One extension is to allow users to upload their own historical data. For example, a company could upload daily electricity usage from their facilities. The app could then retrain a Prophet model on the fly and generate forecasts specific to their case. Another extension is to add multiple frequency options, such as weekly or daily forecasts.

Another valuable extension is adding external regressors. Electricity demand is often influenced by temperature, rainfall, or economic indicators. Prophet supports external regressors, so by including these factors the forecasts can become more accurate. In practice this could mean connecting the app to a weather API and pulling future temperature data. This would allow the model to adjust demand forecasts when higher temperatures are expected.

## Deployment Considerations

When deploying the app to production I must consider performance and reliability. The current model is small and loads quickly. But if more features are added, or if retraining is introduced, computation could increase. Streamlit provides caching mechanisms that can store results of expensive operations. For example, I could cache the loading of the model so it is not repeated every run. Another deployment factor is scaling.

Deployment also raises security concerns. If file upload is added, I need to sanitize inputs. If APIs are used, I need to manage API keys safely. These considerations are not visible in the current simple code, but they are important when planning real-world applications.

## Handling Edge Cases

Forecasting is never perfect. There are edge cases that must be considered. For example, if the model is trained on insufficient data the forecasts will be unreliable. If future dates are generated beyond the range where the model has learned seasonality, predictions may degrade. Streamlit sliders help constrain the input but cannot guarantee accuracy. Another edge case is missing dependencies. If a user tries to run the app without installing Prophet correctly, the app will fail.

## Scaling the Project

If I wanted to scale this project for nationwide deployment, several steps would be needed. First, the model must be retrained regularly with new data. Electricity demand patterns evolve over time, so models trained on old data may lose relevance. Second, the infrastructure must support thousands of users simultaneously. This could mean deploying the app behind a load balancer and running multiple instances. Third, logging and monitoring must be added to track performance and detect issues.

## Reflections

Building this project gave me confidence in combining machine learning with simple web interfaces. It reminded me that clarity is more important than complexity. The code is short but every line matters. From imports to visualization, each step plays a role. The final result is an accessible tool that forecasts electricity demand with just a few interactions. This kind of project demonstrates the power of combining libraries and frameworks thoughtfully.

---


## Closing Thoughts

This project taught me how powerful time series forecasting can be when combined with interactive tools. By preparing the environment correctly, training a model once, and exposing it through a Streamlit app, I was able to create a practical forecasting tool. The structure is simple but effective. Every function has a clear role and every file contributes to the overall workflow. I can extend this further by adding features like downloading the forecast data or comparing multiple regions. For now, the application is a working demonstration of how machine learning models can be deployed quickly for real world use.

---
