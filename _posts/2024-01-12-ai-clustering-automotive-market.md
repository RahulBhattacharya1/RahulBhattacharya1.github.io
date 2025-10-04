---
layout: default
title: "Creating my AI Automotive Market Clustering"
date: 2024-01-12 21:14:53
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/automotive.webp
demo_link: https://rahuls-ai-clustering-automotive-market.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai-clustering-automotive-market
featured: true
---

I once spent hours comparing vehicle prices on different websites. I was curious why some models seemed overpriced while others followed smoother trends. That curiosity turned into frustration when I realized I had no clear way to see patterns across thousands of models. I wanted to know if certain groups of vehicles behaved similarly. The idea of creating clusters of car models based on their price behavior started forming in my mind. Dataset used [here](https://www.kaggle.com/datasets/jockeroika/cars-dataset).

That was the starting point for this project. I imagined a tool that could take historical pricing data, extract meaningful features, and then show me groups of models that behaved in similar ways. This project became an experiment in clustering, visualization, and learning to connect data science workflows into something interactive. Below, I will walk through every file and every function that makes this project possible.


## Understanding the README.md

The README file gives a short description of the project. It tells us what this tool does and how to run it. The main focus is on clustering vehicle models based on their price history.

```markdown
# Model Cluster Map (Pricing Personas)

Discover clusters of vehicle models based on their price behavior over time:
- Feature engineering per `Genmodel_ID` (trend slope, CAGR, volatility, range).
- K-Means clustering with adjustable k.
- UMAP/PCA scatter; click a model to view its price timeline.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```


This document sets the stage. It tells the reader that clustering will be done on vehicles and the features will include slope of the trend, compound annual growth rate, volatility, and range. It also makes clear that visualization is possible with UMAP or PCA and clicking points will reveal timelines. Finally, it explains how to install and run the project locally.



## The app.py File

The `app.py` file is the heart of the interactive dashboard. It uses Streamlit to present results. It imports functions from `cluster_utils.py` to compute features and perform clustering. Let's look at the structure step by step.

```python
import os
import pandas as pd
import plotly.express as px
import streamlit as st

from cluster_utils import compute_features, cluster_and_embed

st.set_page_config(page_title="Model Cluster Map", layout="wide")
```

Here we import the libraries. `os` handles file paths, `pandas` is for handling data tables, `plotly.express` is for charts, and `streamlit` is the framework to build the web app. We also import two custom helpers from `cluster_utils.py`. The page is configured to be wide layout for better visual space.



### Loading the Data

The app has a function to load the price table. It looks for the file in multiple paths. This helps in cases where the file may be saved in different folders.

```python
@st.cache_data
def load_price_table():
    paths = [
        "data/Price_table.csv",
        "artifacts/price_table_clean.csv",
        "Price_table.csv",
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError("Price table not found in expected paths")
```

This helper tries three locations: inside `data`, inside `artifacts`, or in the root folder. If none are found, it raises an error. Using `st.cache_data` ensures that the file is not reloaded every time the app refreshes. This saves computation time and improves speed.



### Computing Features and Clustering

After loading the table, the app computes features and then applies clustering.

```python
df_price = load_price_table()
df_features = compute_features(df_price)
df_embed, df_clusters = cluster_and_embed(df_features)
```

The workflow is simple: first we load the raw data, then we pass it to the function that computes features, and then we pass the resulting feature table to the clustering and embedding function. This is where the machine learning steps happen. The app separates responsibilities clearly: loading is one step, feature engineering is another, clustering is a third.



### Building the Scatter Plot

The app uses Plotly Express to build an interactive scatter plot. Each dot is a model. The axes are embeddings from UMAP or PCA. The colors represent clusters.

```python
fig = px.scatter(
    df_embed,
    x="x",
    y="y",
    color="cluster",
    hover_data=["Genmodel_ID"]
)
st.plotly_chart(fig, use_container_width=True)
```

This block creates the chart. The x and y values come from the embedding coordinates. The cluster value defines the color. When hovering over a point, we see the model ID. Streamlit displays this chart inside the app.



### Showing Price Timeline

When the user clicks on a point, the app shows the price timeline of that model.

```python
selected_model = st.selectbox("Choose a model", df_price["Genmodel_ID"].unique())

df_selected = df_price[df_price["Genmodel_ID"] == selected_model]
fig_line = px.line(df_selected, x="Year", y="Price", title=f"Price Timeline for {selected_model}")
st.plotly_chart(fig_line, use_container_width=True)
```

The selectbox lets the user choose one model from the list. The app filters the price table to include only that model and then plots a line chart of its price against the year. This is the final piece of the dashboard that gives details about a chosen model.



## The cluster_utils.py File

This file has the main logic for computing features and clustering. Let's go step by step.

```python
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap
```
Here we import numpy and pandas for data manipulation. We also import scaling, clustering, and dimensionality reduction from scikit-learn. UMAP is imported separately. These tools will handle the transformation of data into clusters and lower dimensional embeddings.



### Computing Features

The first function is `compute_features`. It builds new features per model ID.

```python
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    def cagr(fy, ly, fp, lp):
        years = ly - fy
        if years <= 0 or fp <= 0 or lp <= 0:
            return 0.0
        return (lp / fp) ** (1.0 / years) - 1.0

    def trend_slope(years, prices):
        if len(years) < 2:
            return 0.0
        try:
            coeffs = np.polyfit(years, prices, 1)
            return coeffs[0]
        except:
            return 0.0
```

The function has two helpers inside. `cagr` computes compound annual growth rate between first and last year given their prices. If invalid data exists, it returns zero. `trend_slope` computes the slope of a linear regression line through years and prices. If not enough data is present or if fitting fails, it returns zero.



Continuing inside `compute_features`:

```python
    features = []
    for gid, g in df.groupby("Genmodel_ID"):
        years = g["Year"].values
        prices = g["Price"].values
        row = {
            "Genmodel_ID": gid,
            "cagr": cagr(years.min(), years.max(), prices[0], prices[-1]),
            "slope": trend_slope(years, prices),
            "volatility": np.std(prices),
            "range": prices.max() - prices.min(),
        }
        features.append(row)
    return pd.DataFrame(features)
```

This block groups the dataset by model ID. For each model, it computes CAGR, slope, volatility, and range. These values summarize the price behavior of the model over time. It collects these into a list of rows and finally converts them to a DataFrame. The resulting table has one row per model with the new features.



### Clustering and Embedding

The second main function is `cluster_and_embed`.

```python
def cluster_and_embed(df: pd.DataFrame, n_clusters: int = 5) -> (pd.DataFrame, pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop(columns=["Genmodel_ID"]))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    df_embed = pd.DataFrame(embedding, columns=["x", "y"])
    df_embed["cluster"] = df["cluster"]
    df_embed["Genmodel_ID"] = df["Genmodel_ID"]
    return df_embed, df
```

This function first scales all features to make them comparable. It applies KMeans clustering with a given number of clusters. The cluster assignments are added to the DataFrame. Then UMAP is applied to reduce dimensions into two columns x and y. These coordinates are used for plotting. Finally, the function returns the embedding with cluster information and the updated feature DataFrame.



## The requirements.txt File

The requirements file lists the dependencies.

```text
streamlit
pandas
numpy
scikit-learn
umap-learn
plotly
```

This tells us that the project uses Streamlit for the web app, pandas and numpy for data handling, scikit-learn for clustering and scaling, umap-learn for embedding, and plotly for visualization. Installing these packages ensures that the app runs without missing libraries.



## The Dataset: Price_table.csv

The dataset is stored in `data/Price_table.csv`. It contains columns like `Genmodel_ID`, `Year`, and `Price`. Each row is a price observation for a specific model in a specific year. This raw information is transformed into features by `compute_features`. Without this dataset, the clustering would not be possible. The app is designed to load it automatically if present in the expected folders.



## Closing Thoughts

This project showed me how raw price histories can be converted into structured insights. By extracting features like CAGR, slope, volatility, and range, I was able to summarize models. Clustering revealed natural groups. UMAP helped visualize them in two dimensions. Streamlit made it possible to interact with the results.

Every function in the project has a clear role. The helpers compute meaningful numbers. The main functions assemble results. The app brings everything together in a dashboard. This journey taught me how to connect data processing, machine learning, and visualization into one smooth workflow.



## Extended Breakdown of Feature Engineering

The feature engineering step is crucial because it transforms raw yearly prices into statistical signals. Let us revisit `compute_features` in detail.

```python
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    def cagr(fy, ly, fp, lp):
        years = ly - fy
        if years <= 0 or fp <= 0 or lp <= 0:
            return 0.0
        return (lp / fp) ** (1.0 / years) - 1.0

    def trend_slope(years, prices):
        if len(years) < 2:
            return 0.0
        try:
            coeffs = np.polyfit(years, prices, 1)
            return coeffs[0]
        except:
            return 0.0
```
The `cagr` function measures the compound annual growth rate between the first and last observed years. It guards against invalid inputs such as zero or negative values. Returning 0.0 in those cases prevents errors later. This number shows the average growth per year if prices followed a compounding path.

The `trend_slope` function estimates the slope of a best fit line. It uses numpy’s polyfit with degree 1, which gives us a straight line. The slope tells us if prices were trending upward or downward over time. If there are not enough points or if fitting fails, it returns zero to maintain stability. This slope is more sensitive to small fluctuations than CAGR and complements it well.



Continuing the function, we see the aggregation.

```python
    features = []
    for gid, g in df.groupby("Genmodel_ID"):
        years = g["Year"].values
        prices = g["Price"].values
        row = {
            "Genmodel_ID": gid,
            "cagr": cagr(years.min(), years.max(), prices[0], prices[-1]),
            "slope": trend_slope(years, prices),
            "volatility": np.std(prices),
            "range": prices.max() - prices.min(),
        }
        features.append(row)
    return pd.DataFrame(features)
```

Here, the dataset is grouped by the model identifier. Each group represents one vehicle model. For each group, we collect the years and prices. The first and last years with their prices go into the CAGR function. The full arrays go into the slope function. We also calculate volatility with numpy standard deviation and the overall price range. Volatility captures how much prices fluctuate. Range captures the span between maximum and minimum observed prices. Together, these four numbers capture growth, direction, stability, and spread. That combination is powerful for clustering because it summarizes each model concisely.



## Deep Dive into Clustering

The clustering function `cluster_and_embed` deserves its own deep dive. It orchestrates scaling, clustering, and embedding.

```python
def cluster_and_embed(df: pd.DataFrame, n_clusters: int = 5) -> (pd.DataFrame, pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop(columns=["Genmodel_ID"]))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    df_embed = pd.DataFrame(embedding, columns=["x", "y"])
    df_embed["cluster"] = df["cluster"]
    df_embed["Genmodel_ID"] = df["Genmodel_ID"]
    return df_embed, df
```

The first step is scaling. `StandardScaler` adjusts each feature so that it has mean zero and unit variance. This makes them comparable. Without scaling, a feature with large numeric values like price range could dominate the clustering.

Next comes KMeans. It is initialized with a random state to ensure reproducibility. The algorithm assigns each point to the closest centroid among `n_clusters`. The result is stored in a new column `cluster` inside the DataFrame. This assignment divides models into groups based on their feature patterns.

Finally comes UMAP. It reduces the high-dimensional feature space into two coordinates for visualization. Unlike PCA, UMAP preserves local neighborhood structures better. The resulting coordinates are stored in a new DataFrame `df_embed`. This DataFrame also carries the cluster assignment and the model IDs, making it ready for plotting. Returning both the embedding and the full features ensures flexibility for downstream use.



## Detailed Walkthrough of the Dashboard Logic

The dashboard in `app.py` brings everything together. Let’s revisit its sequence in plain words.

1. Load the dataset using `load_price_table`. It tries multiple paths for robustness.
2. Compute the features with `compute_features`. This turns raw rows into statistical summaries.
3. Cluster and embed with `cluster_and_embed`. This produces coordinates and group labels.
4. Plot a scatter chart with Plotly. Each model becomes a point in two dimensions.
5. Provide an option to select a specific model. When selected, show its price history over time.

This sequence is logical. First comes data ingestion. Then transformation. Then machine learning. Then visualization. Finally, interaction. Each block is separate but they connect smoothly. This modular design makes the project easier to extend. For example, one could replace KMeans with DBSCAN, or swap UMAP with PCA, without breaking the overall flow.



## Example Usage Scenario

Imagine having a dataset of thousands of vehicle models from different manufacturers. Running this app, the scatter plot would show groups of models that behave similarly. One cluster might represent stable, slow growth models. Another might represent highly volatile models with fluctuating prices. Another could represent steadily declining models. This visualization makes it easier to spot patterns that raw numbers cannot reveal.

Clicking a specific model, the user can see its timeline and confirm why it ended up in that cluster. This bridges the gap between aggregated statistical clustering and individual raw history. It helps both in analysis and storytelling.



## Lessons Learned

While building this project, I learned the value of designing functions with clear responsibilities. The helpers inside `compute_features` show how small blocks handle very specific tasks. The clustering function shows how to combine multiple techniques into one coherent result. The app code shows how to separate loading, processing, and presenting.

Another lesson was about user experience. Adding caching for data loading made the app more responsive. Choosing wide layout improved readability of scatter plots. Adding hover and click options in charts made exploration natural. These details may seem minor, but they make a big difference in usability.

Finally, I learned about reproducibility. Fixing the random seed for KMeans and UMAP means that running the app twice gives the same result. Without this, clusters could change between runs, leading to confusion. Ensuring stability builds trust in the tool.

