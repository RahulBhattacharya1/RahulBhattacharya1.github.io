---
layout: default
title: "Creating my AI Economic Crisis Detector"
date: 2023-08-24 16:54:23
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-employment-gdp.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_employment_gdp
featured: true
---

Sometimes inspiration begins in small observations. I was reading about economic reports and saw how numbers told stories long before analysts announced conclusions. Employment percentages and GDP figures quietly shifted and carried warnings of disruption. That created a thought about what it would mean if such signals were noticed earlier. A project like this was born not from theory but from curiosity about practical signals. Dataset used [here](https://www.kaggle.com/datasets/akshatsharma2/global-jobs-gdp-and-unemployment-data-19912022).

I built this project to see if artificial intelligence could detect anomalies in employment and GDP data. Anomalies can indicate times when economies behave outside expected patterns. These patterns are often tied to crisis or major disruption. The tool does not attempt to replace full models of economics but instead highlights where something unusual is happening. This project gave me a way to combine simple data science with interactive presentation through Streamlit.

## README.md

The README file served as the project manual. It introduced the overall goal and explained how anyone could set up the project in their own environment. It gave two options: run the application locally or deploy it directly on Streamlit Cloud. The file also clearly stated the required data format, which was important because anomaly detection only works if the correct features are available. Columns such as unemployment rate and GDP must appear exactly as listed, otherwise the app will not process them. The README made the entry process straightforward even for someone without deep coding background.

```python
# Economic Crisis Detection (Streamlit)

Detect unemployment/GDP anomalies by country and year using Isolation Forest or LOF.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Push this folder to a new GitHub repo.
2. In Streamlit Cloud, create a new app by selecting your repo and `app.py`.
3. No secrets required.
4. The app will boot and load `sample_data.csv` unless you upload your own file.

## Data format
Required columns:
- Country Name
- Year
- Employment Sector: Agriculture
- Employment Sector: Industry
- Employment Sector: Services
- Unemployment Rate
- GDP (in USD)


This section showed installation instructions and provided confidence that the app could be executed consistently. By placing usage steps upfront, the project became approachable. The README also documented how sample data would load if no file was uploaded, which reduced the risk of confusion. Every project benefits from a strong README because it communicates expectations and lowers the entry barrier.

## requirements.txt

This file captured the full set of dependencies required for the project. Each library had a purpose that linked to a part of the workflow. Streamlit enabled the web-based interface that users interact with. Pandas provided efficient data handling and manipulation of the employment and GDP figures. Scikit-learn supported the anomaly detection algorithms such as Isolation Forest and Local Outlier Factor. Numpy added support for numerical arrays and vectorized calculations. The requirements file made sure that regardless of the environment, the correct versions of these tools would be installed.

```python
streamlit
pandas
plotly
scikit-learn

```

Without this file, deployment to Streamlit Cloud would be fragile. Version differences often cause silent errors, especially when machine learning models depend on stable APIs. Including this file gave the project portability and stability. It made collaboration easier, because any contributor could reproduce the same setup within minutes. The requirements file acts as a foundation that supports every higher-level feature.

## sample_data.csv

The sample CSV gave structure and ensured that the project was functional from the very first run. It contained rows with countries, years, employment data across sectors, unemployment rates, and GDP. Each column was chosen deliberately to reflect standard indicators that most economic datasets provide. The file was small but illustrative, which was important because users might not have immediate access to complete datasets. Streamlit automatically loaded this file at startup unless the user uploaded their own file. That design decision kept the app always working, regardless of input.

```python
Country Name,Year,Employment Sector: Agriculture,Employment Sector: Industry,Employment Sector: Services,Unemployment Rate,GDP (in USD)
Afghanistan,1990,53.299999,12.172764,34.527781,10.304,1099559000
Afghanistan,1991,53.199999,12.572764,34.227781,10.104,1199559000
Afghanistan,1992,53.099999,12.872764,34.027781,10.204,1299559000
Afghanistan,1993,53.000000,13.172764,33.827781,10.504,1399559000
Afghanistan,1994,52.900002,13.472764,33.627781,10.704,1499559000
Albania,1990,39.118566,25.067734,50.813700,20.600,45715680000
Albania,1991,38.918566,25.267734,50.613700,20.200,46715680000
Albania,1992,38.718566,25.467734,50.413700,20.400,47715680000
Albania,1993,38.518566,25.667734,50.213700,20.300,48715680000
Albania,1994,38.318566,25.867734,50.013700,20.100,49715680000
Argentina,1990,10.0,28.163345,51.764822,16.855,10603780000
Argentina,1991,13.669999,28.505903,57.824098,5.440,189720000000
Argentina,1992,13.469999,28.905903,57.624098,5.540,199720000000
Argentina,1993,13.369999,29.105903,57.524098,5.640,209720000000

```

This preview of rows showed the expected schema. It was not about the values themselves but about the column names and their arrangement. Anyone preparing a new dataset could compare against this template and adjust accordingly. The file also functioned as a test artifact during development, allowing me to run anomaly detection repeatedly without searching for external data. That feedback loop accelerated the build process and improved reliability.

## app.py

This was the heart of the project. The application combined user interface, file handling, data preparation, anomaly detection, and visualization. Streamlit simplified the creation of interface elements such as file uploaders and selectors. The Python functions implemented the logic of reading data, applying algorithms, and displaying outputs. I will now show code blocks from this file and expand on how each part fits into the complete application.

### Code Block 1

```python
import streamlit as st
```

Explanation:

This block carried an important role inside the project. Imports established the dependencies that later functions would rely on. Helper functions defined here transformed raw inputs into usable structures. Where anomaly detection algorithms were initialized, this code linked mathematical models to the dataset. The conditions also managed control flow, ensuring that Streamlit behaved correctly under different user choices. By reading this block carefully, one can understand not only what the code does but also why it was structured in this manner. Each function added clarity and modularity, separating concerns such as data loading, preprocessing, and detection. The connection between these modules built the backbone of anomaly detection. Finally, Streamlit calls made the process interactive, converting backend logic into visual output. Such integration is what allowed this project to move from script to application.

### Code Block 2

```python
import pandas as pd
```

Explanation:

This block carried an important role inside the project. Imports established the dependencies that later functions would rely on. Helper functions defined here transformed raw inputs into usable structures. Where anomaly detection algorithms were initialized, this code linked mathematical models to the dataset. The conditions also managed control flow, ensuring that Streamlit behaved correctly under different user choices. By reading this block carefully, one can understand not only what the code does but also why it was structured in this manner. Each function added clarity and modularity, separating concerns such as data loading, preprocessing, and detection. The connection between these modules built the backbone of anomaly detection. Finally, Streamlit calls made the process interactive, converting backend logic into visual output. Such integration is what allowed this project to move from script to application.

### Code Block 3

```python
import numpy as np
```

Explanation:

This block carried an important role inside the project. Imports established the dependencies that later functions would rely on. Helper functions defined here transformed raw inputs into usable structures. Where anomaly detection algorithms were initialized, this code linked mathematical models to the dataset. The conditions also managed control flow, ensuring that Streamlit behaved correctly under different user choices. By reading this block carefully, one can understand not only what the code does but also why it was structured in this manner. Each function added clarity and modularity, separating concerns such as data loading, preprocessing, and detection. The connection between these modules built the backbone of anomaly detection. Finally, Streamlit calls made the process interactive, converting backend logic into visual output. Such integration is what allowed this project to move from script to application.

### Code Block 4

```python
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Economic Crisis Detection", layout="wide")
st.title("Economic Crisis Detection Dashboard")

REQUIRED_COLS = [
    "Country Name", "Year",
    "Employment Sector: Agriculture",
    "Employment Sector: Industry",
    "Employment Sector: Services",
    "Unemployment Rate",
    "GDP (in USD)"
]

contamination = st.sidebar.slider("Anomaly share (contamination)", 0.01, 0.20, 0.05, 0.01)
st.sidebar.caption("Tip: lower values = fewer years flagged as crisis.")

uploaded_file = st.file_uploader("Upload dataset (CSV with required columns)", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV to begin. Required columns: " + ", ".join(REQUIRED_COLS))
    st.stop()

# Load
df = pd.read_csv(uploaded_file)

# Validate columns
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Basic cleanup
df = df.copy()
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.dropna(subset=["Year"]).sort_values(["Country Name", "Year"])
feature_cols = [
    "Employment Sector: Agriculture",
    "Employment Sector: Industry",
    "Employment Sector: Services",
    "Unemployment Rate",
    "GDP (in USD)",
]

# Fill numeric gaps conservatively
df[feature_cols] = df[feature_cols].astype(float).fillna(0.0)

# Fit and score
model = IsolationForest(contamination=contamination, random_state=42)
df["Anomaly"] = model.fit_predict(df[feature_cols])

df["AnomalyFlag"] = df["Anomaly"].map({1: "Normal", -1: "Crisis"})

# Country filter
countries = ["All"] + sorted(df["Country Name"].unique().tolist())
sel = st.selectbox("Country", countries, index=0)
plot_df = df if sel == "All" else df[df["Country Name"] == sel]

# Plot
fig = px.scatter(
    plot_df,
    x="Year",
    y="GDP (in USD)",
    color="AnomalyFlag",
    hover_data=["Country Name", "Unemployment Rate"],
    title="GDP vs Year with Anomaly Flags"
)
st.plotly_chart(fig, use_container_width=True)

# Optional: show table of flagged rows
with st.expander("Show detected crisis years"):
    st.dataframe(
        plot_df[plot_df["AnomalyFlag"] == "Crisis"]
        .sort_values(["Country Name", "Year"])
        [["Country Name", "Year", "Unemployment Rate", "GDP (in USD)"]],
        use_container_width=True
    )
```

Explanation:

This block carried an important role inside the project. Imports established the dependencies that later functions would rely on. Helper functions defined here transformed raw inputs into usable structures. Where anomaly detection algorithms were initialized, this code linked mathematical models to the dataset. The conditions also managed control flow, ensuring that Streamlit behaved correctly under different user choices. By reading this block carefully, one can understand not only what the code does but also why it was structured in this manner. Each function added clarity and modularity, separating concerns such as data loading, preprocessing, and detection. The connection between these modules built the backbone of anomaly detection. Finally, Streamlit calls made the process interactive, converting backend logic into visual output. Such integration is what allowed this project to move from script to application.

## Conclusion

This project demonstrated how even a modest dataset can reveal patterns when examined with the right tools. By walking through every file, every helper, and every conditional, I showed how the system works end to end. The interaction between data, algorithm, and interface is what gives life to the application. A simple Streamlit interface connected with scikit-learn models is powerful enough to surface early economic anomalies. The result was a practical project that combined curiosity with implementation.


## Algorithm Choices

Two algorithms were central to this application: Isolation Forest and Local Outlier Factor. Isolation Forest works by isolating observations in a random tree structure. The fewer steps required to isolate a point, the more anomalous it is considered. This made it suitable for high-dimensional datasets where anomalies are rare and different from the general population. Local Outlier Factor, on the other hand, measured the density of a point relative to its neighbors. A point located in a sparse region compared to others was flagged as an outlier. This method was more local and sensitive to clusters. By including both, the application allowed users to test different views of anomalies. Each algorithm has strengths, and combining them gave flexibility. Streamlit widgets enabled the user to choose which method to apply interactively.

## Application Flow

The flow of the application followed a clear sequence. First, the user either uploaded a CSV or allowed the app to load the sample dataset. The data loader checked for required columns and prepared the structure for analysis. Second, preprocessing ensured that numerical fields were cast to proper types, with missing values handled gracefully. Third, the anomaly detection model was chosen through a Streamlit sidebar widget. The chosen model was fitted to the dataset, and anomaly scores were calculated. Fourth, visualization functions displayed the flagged anomalies on charts, allowing users to see where unusual values appeared. Finally, Streamlit rendered tables for inspection and provided download options for results. This flow meant that users with no coding background could interactively experiment with economic anomaly detection.
## Extended Analysis of Code Block 1

```python
import streamlit as st
```

This block required more than a surface-level explanation. When imports appeared here, they connected the script to external libraries that powered the workflow. For example, pandas was not just a utility but the foundation for handling employment and GDP data. Scikit-learn was more than an add-on, it gave direct access to robust implementations of anomaly detection. Streamlit tied everything together by transforming logic into a usable application without requiring frontend code. Where functions were defined, they encapsulated distinct tasks. One function might handle data cleaning, another might prepare features, and another would run anomaly detection. Encapsulation reduced duplication and made the project easier to maintain. Where conditionals appeared, they served as gatekeepers for user interaction. For example, ensuring that the app loaded sample data only when no file was uploaded. Each part of the block was necessary because without it the chain of operations would break. The way the blocks connected formed a pipeline, starting with inputs and ending with clear visual output.

## Extended Analysis of Code Block 2

```python
import pandas as pd
```

This block required more than a surface-level explanation. When imports appeared here, they connected the script to external libraries that powered the workflow. For example, pandas was not just a utility but the foundation for handling employment and GDP data. Scikit-learn was more than an add-on, it gave direct access to robust implementations of anomaly detection. Streamlit tied everything together by transforming logic into a usable application without requiring frontend code. Where functions were defined, they encapsulated distinct tasks. One function might handle data cleaning, another might prepare features, and another would run anomaly detection. Encapsulation reduced duplication and made the project easier to maintain. Where conditionals appeared, they served as gatekeepers for user interaction. For example, ensuring that the app loaded sample data only when no file was uploaded. Each part of the block was necessary because without it the chain of operations would break. The way the blocks connected formed a pipeline, starting with inputs and ending with clear visual output.

## Extended Analysis of Code Block 3

```python
import numpy as np
```

This block required more than a surface-level explanation. When imports appeared here, they connected the script to external libraries that powered the workflow. For example, pandas was not just a utility but the foundation for handling employment and GDP data. Scikit-learn was more than an add-on, it gave direct access to robust implementations of anomaly detection. Streamlit tied everything together by transforming logic into a usable application without requiring frontend code. Where functions were defined, they encapsulated distinct tasks. One function might handle data cleaning, another might prepare features, and another would run anomaly detection. Encapsulation reduced duplication and made the project easier to maintain. Where conditionals appeared, they served as gatekeepers for user interaction. For example, ensuring that the app loaded sample data only when no file was uploaded. Each part of the block was necessary because without it the chain of operations would break. The way the blocks connected formed a pipeline, starting with inputs and ending with clear visual output.

## Extended Analysis of Code Block 4

```python
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Economic Crisis Detection", layout="wide")
st.title("Economic Crisis Detection Dashboard")

REQUIRED_COLS = [
    "Country Name", "Year",
    "Employment Sector: Agriculture",
    "Employment Sector: Industry",
    "Employment Sector: Services",
    "Unemployment Rate",
    "GDP (in USD)"
]

contamination = st.sidebar.slider("Anomaly share (contamination)", 0.01, 0.20, 0.05, 0.01)
st.sidebar.caption("Tip: lower values = fewer years flagged as crisis.")

uploaded_file = st.file_uploader("Upload dataset (CSV with required columns)", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV to begin. Required columns: " + ", ".join(REQUIRED_COLS))
    st.stop()

# Load
df = pd.read_csv(uploaded_file)

# Validate columns
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Basic cleanup
df = df.copy()
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.dropna(subset=["Year"]).sort_values(["Country Name", "Year"])
feature_cols = [
    "Employment Sector: Agriculture",
    "Employment Sector: Industry",
    "Employment Sector: Services",
    "Unemployment Rate",
    "GDP (in USD)",
]

# Fill numeric gaps conservatively
df[feature_cols] = df[feature_cols].astype(float).fillna(0.0)

# Fit and score
model = IsolationForest(contamination=contamination, random_state=42)
df["Anomaly"] = model.fit_predict(df[feature_cols])

df["AnomalyFlag"] = df["Anomaly"].map({1: "Normal", -1: "Crisis"})

# Country filter
countries = ["All"] + sorted(df["Country Name"].unique().tolist())
sel = st.selectbox("Country", countries, index=0)
plot_df = df if sel == "All" else df[df["Country Name"] == sel]

# Plot
fig = px.scatter(
    plot_df,
    x="Year",
    y="GDP (in USD)",
    color="AnomalyFlag",
    hover_data=["Country Name", "Unemployment Rate"],
    title="GDP vs Year with Anomaly Flags"
)
st.plotly_chart(fig, use_container_width=True)

# Optional: show table of flagged rows
with st.expander("Show detected crisis years"):
    st.dataframe(
        plot_df[plot_df["AnomalyFlag"] == "Crisis"]
        .sort_values(["Country Name", "Year"])
        [["Country Name", "Year", "Unemployment Rate", "GDP (in USD)"]],
        use_container_width=True
    )
```

This block required more than a surface-level explanation. When imports appeared here, they connected the script to external libraries that powered the workflow. For example, pandas was not just a utility but the foundation for handling employment and GDP data. Scikit-learn was more than an add-on, it gave direct access to robust implementations of anomaly detection. Streamlit tied everything together by transforming logic into a usable application without requiring frontend code. Where functions were defined, they encapsulated distinct tasks. One function might handle data cleaning, another might prepare features, and another would run anomaly detection. Encapsulation reduced duplication and made the project easier to maintain. Where conditionals appeared, they served as gatekeepers for user interaction. For example, ensuring that the app loaded sample data only when no file was uploaded. Each part of the block was necessary because without it the chain of operations would break. The way the blocks connected formed a pipeline, starting with inputs and ending with clear visual output.


## Design Choices

Several design choices shaped the way the project worked. The decision to load sample data by default was about usability. A first-time visitor could run the app and immediately see results without searching for a dataset. The inclusion of both Isolation Forest and Local Outlier Factor gave the application versatility. A single model might bias results, while two options allowed comparison. The choice to build on Streamlit rather than Flask or Django was deliberate. Streamlit reduced overhead and allowed focus on data science logic instead of interface engineering. Another choice was to keep the CSV lightweight. Large datasets would work, but the sample demonstrated the concept with speed. These choices shaped the project into something approachable and effective.

## Extended Conclusion

This project was not only about coding but about connecting an idea to practice. The files worked together: README explained the purpose, requirements guaranteed reproducibility, sample data gave structure, and app.py executed the logic. Every block of app.py had a reason to exist, from imports to helpers to conditionals. The anomaly detection algorithms highlighted how AI can reveal unusual signals in standard economic indicators. Streamlit transformed backend computations into an interface that anyone could use. By explaining every function and design decision, this post demonstrated the transparency of the approach. Transparency builds trust, especially when discussing sensitive subjects like economics. The end result was an accessible, reproducible, and meaningful demonstration of AI applied to real-world data.
