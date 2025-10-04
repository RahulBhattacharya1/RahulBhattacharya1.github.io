---
layout: default
title: "Creating my AI Asylum Application Trends"
date: 2024-01-27 09:23:41
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/asylum.webp
demo_link: https://rahuls-ai-asylum-trend-forecasting.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_asylum_trend_forecasting
featured: true
---

I once came across an official report that tried to predict how many asylum applications different countries in Europe would receive in the coming years. The report was full of technical terms, and while the graphs were helpful, I wanted something more interactive. I wanted to type a country code and immediately see a forecast line chart appear before me. That personal curiosity led me to imagine and then build this project. Dataset used [here](https://www.kaggle.com/datasets/gpreda/asylum-applicants-by-citizenship-in-europe).

I wanted to create a simple forecasting app where a user could enter a country code and view predicted asylum applications. Instead of relying only on static graphs in papers, I decided to use Streamlit to make a lightweight web application. By doing this, I could demonstrate forecasting concepts while also providing a working demo that anyone could explore from a browser. This blog post explains every piece of that project, file by file, block by block.


## requirements.txt

The first file I uploaded to GitHub was the `requirements.txt`. This file tells Streamlit Cloud or any other hosting platform which Python libraries must be installed for the application to run. Without this file, deployment would fail because the environment would not know what dependencies are required.

```python
streamlit
pandas
matplotlib
statsmodels
joblib

```

Each line here represents a library. `streamlit` provides the web framework for building the app. `pandas` is used for data manipulation and handling tabular structures. `matplotlib` helps with charting in some forecasting extensions, though Streamlit has a built-in chart function. `statsmodels` is often required when building forecasting models, so it is included for compatibility. `joblib` is needed because the forecasting model itself is stored as a serialized object in a `.pkl` file, and joblib is the package used to load it. By freezing these dependencies, I ensure reproducibility across environments.


## app.py

The core of the application is the `app.py` file. This script defines how the Streamlit app runs, how it loads the forecasting model, how it takes user input, and how it displays results. I will break it down into logical blocks and explain each.

```python
import streamlit as st
import pandas as pd
import joblib
```

The script begins with imports. `streamlit` is imported as `st` which allows me to call its functions to build the user interface. `pandas` is included here even though the minimal demo version does not use it heavily; in extended versions, pandas is used to structure and format forecasted values. `joblib` is imported so that I can load the saved forecasting model from disk.

```python
st.title("Asylum Applications Forecasting")
```

This line sets the title of the Streamlit web app. The title appears at the top of the page whenever the application is launched. It gives users immediate context on what the app is about.

```python
# Load model
model = joblib.load("models/forecast_model.pkl")
```

Here I load the forecasting model that I had trained separately in a notebook environment and then exported as a `.pkl` file. Joblib allows efficient serialization and deserialization of Python objects. By storing the model in the `models` folder, I separate the training artifact from the deployment script. This line is crucial because it connects the pre-trained forecasting logic with the Streamlit app.

```python
# User input
country = st.text_input("Enter country code (e.g., AT, BE, DE):", "AT")
```

This block defines a text input field where a user can type a two-letter country code such as AT for Austria, BE for Belgium, or DE for Germany. I set a default value of "AT" so that the application shows something immediately even if the user does not provide input. This improves usability and avoids rendering an empty state.

```python
# Demo: Just Austria for now
if country == "AT":
    st.write("Forecast for Austria next 5 years:")
    forecast = model.forecast(5)
    st.line_chart(forecast)
else:
    st.warning("Model currently supports only Austria. Extend in Colab for other countries.")
```

This conditional is the heart of the demo logic. It first checks if the entered country code matches "AT". If it does, the application writes a short text message on the page that clarifies the forecast is specifically for Austria. Then it calls `model.forecast(5)` to produce a five-step forecast. In practice, this returns predicted asylum applications for the next five years. The result is then displayed as a line chart using `st.line_chart`, which makes the forecast easier to understand visually.

The `else` branch shows a warning message if any other country code is entered. This prevents the application from breaking and also communicates to the user that model coverage is currently limited. In future versions, I could expand the model to handle multiple countries by training on broader datasets. This conditional block ensures graceful handling of unsupported inputs while still offering a functional demonstration for Austria.


## models/forecast_model.pkl

The forecasting model is stored inside the `models` folder as `forecast_model.pkl`. This file was generated outside this repository, using a training notebook where I experimented with time series forecasting methods. After I finalized the model, I saved it with joblib so that it could be re-used without retraining. The `.pkl` file is binary and cannot be read as plain text, but its purpose is simple: it holds the trained model object.

When the application starts, the app loads this file with `joblib.load`. That makes the model ready for generating forecasts directly. By separating the training step from the application, I ensure faster startup and also simplify deployment. Anyone cloning this repository does not need to repeat the training phase. They can directly use the saved model to run the forecast. This structure also reflects a common industry practice where model training and model serving are separated into distinct pipelines.


## Reflections and Extensions

By combining these three main files—`requirements.txt`, `app.py`, and the `forecast_model.pkl` inside the models folder—I was able to build and deploy a working forecasting demo. Each file had a clear role: dependencies setup, application logic, and trained model storage. GitHub provided the version control and hosting foundation, and Streamlit Cloud allowed deployment directly from the repository.

There are many directions this project can grow. I could expand the model to support multiple country codes by training on richer datasets. I could integrate pandas more deeply to clean and format forecast outputs. I could add interactive widgets that let the user change forecast horizons or toggle between different algorithms. I could also build a front-end visualization layer that uses matplotlib for customized charting. The structure is flexible enough to allow these additions.

What started as a personal curiosity about asylum application forecasts turned into a working demonstration of machine learning deployment. The value of such projects lies not only in the model’s predictions but also in how easily others can interact with it. This hands-on exploration reinforced for me the importance of building simple, reproducible demos that connect complex ideas with interactive experiences.


### Imports

The import section is small but powerful. I use `streamlit` to create UI components like titles, text inputs, and charts. Without this import, none of the interface elements would render. `pandas` may look unused in the short code but it provides the backbone for handling time series data. In extended builds, I would transform raw asylum application counts into dataframes, calculate rolling averages, and reshape structures before passing them to forecasting routines. `joblib` is essential here because my forecasting model is saved in a compressed and serialized form. Without importing joblib, the application could not read the model and forecasting would be impossible.
### Title Setup

Adding a title is not only about aesthetics. It sets context. A user opening the app knows immediately they are looking at a forecasting tool. This matters because Streamlit apps often combine different widgets and charts. Without a clear title, users might struggle to understand the theme. The title line also demonstrates Streamlit’s simplicity: one function call places a styled header at the top of the app.
### User Input

The text input widget defined by `st.text_input` is interactive. It provides a small box where users can type text and submit instantly. The default value `"AT"` ensures that the app works out of the box. If I left it blank, a new user might be confused about what to enter. Providing an example in the label `(e.g., AT, BE, DE)` clarifies expectations. That small addition makes the app more approachable. This block also shows how Streamlit simplifies interaction: I can capture user input with one line of code and store it directly in a variable for later logic.
### Conditional Logic

The conditional block plays two roles. First, it filters supported inputs. Second, it prevents errors when unsupported values are provided. Without this block, calling `model.forecast(5)` on an unknown country would either produce meaningless output or crash the app. By checking `if country == "AT":`, I build a safeguard. Inside the branch, I display a message that frames the forecast. Then I call the forecast method. In the `else` branch, I return a warning. This pattern demonstrates defensive programming: anticipating wrong inputs and handling them gracefully. It is a core skill in building reliable applications.
### The Model File

The `.pkl` file is an artifact of machine learning workflows. Training often happens in notebooks or scripts where raw asylum application data is fed into statistical models like ARIMA, SARIMA, or even Prophet. Once a model is fitted, we save it to disk. Loading that model later avoids repeating heavy computation. In deployment, speed matters. Streamlit apps should respond quickly to user inputs, so separating training and serving is smart. The pickle format works well because it can serialize Python objects, but joblib makes it even more efficient for large models. That is why the app relies on `joblib.load` instead of plain pickle.
### Reflections

This project taught me how small building blocks come together to form a deployable application. Even though the codebase is short, the thought process covers model training, serialization, dependency management, and user interaction. By packaging everything into a GitHub repository, I also gained practice in version control and public documentation. Future collaborators can fork the repo and extend it. Recruiters or hiring managers can run the app and see not only code but a working demo. These aspects turn a technical script into a portfolio piece.
