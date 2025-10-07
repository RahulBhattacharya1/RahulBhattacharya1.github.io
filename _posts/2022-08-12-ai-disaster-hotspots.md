---
layout: default
title: "Building my AI Global Disaster Hotspot Predictor"
date: 2022-08-12 10:46:35
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/global_disaster.webp
thumbnail_mobile: /assets/images/disaster_hotspot_sq.webp
demo_link: https://rahuls-ai-disaster-hotspots.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_disaster_hotspots
---

The thought behind this project started during a time when I was reflecting on how disasters strike without warning. I realized that if there were a way to anticipate which regions were most prone to a certain type of disaster, then leaders could prepare better. That reflection inspired me to design a simple predictive application. I wanted to transform raw data into an accessible tool. The vision was not to create a perfect forecast but to highlight where attention should be focused. It was about making risks visible. Dataset used [here](https://www.kaggle.com/datasets/cyberevil545/global-natural-disasters-dataset-19602018).

Another reason I created this project was the challenge of merging machine learning with interactive visualization. I had worked with static models before, but I wanted something that anyone could use directly in a browser. I knew Streamlit could give me the interface, scikit-learn could provide the model, and Plotly could draw maps. The challenge was in stitching them together in a clean way. By combining these tools, I produced an application that predicts and maps likely disaster hotspots at the country level.

---

## Files I Uploaded to GitHub

### README.md

This file acted as the entry point for anyone who found my repository. It described the purpose of the project and gave step-by-step instructions for running the application. I mentioned that the dataset should be located in the `data` folder. I also provided an option for using an environment variable to point to an external CSV URL. The README made it clear that the app was lightweight, trained models quickly, and provided a map interface for exploration. Without this file, new users would not know where to start.

### requirements.txt

The requirements file contained explicit versions of the dependencies. Each entry had a purpose:

- **streamlit==1.39.0**: Responsible for the entire app interface. It allowed me to structure text, tables, and plots into a web layout without writing HTML or JavaScript.
- **pandas==2.2.2**: This handled the structured dataset. I relied on it to filter, transform, and select columns.
- **scikit-learn==1.5.2**: Provided machine learning utilities. It supported splitting data, encoding categories, building pipelines, training models, and evaluating results.
- **plotly==5.24.1**: Offered advanced plotting capabilities. It let me create a choropleth map that was interactive and visually engaging.
- **pycountry==24.6.1**: Helped resolve inconsistencies in country names by mapping them to ISO codes. This was essential for the map to work correctly.

### data/nasa_disaster_dataset.csv

This dataset was the raw material for the project. It contained information about disasters observed across countries. By feeding this dataset into the model, I could identify patterns and predict the most likely disaster type per country. The size of the file was large enough to provide variation but small enough for quick training in the browser.

### app.py

This was the central script. It defined the entire workflow: from loading data, preprocessing features, training a logistic regression model, evaluating it, and visualizing the results. Streamlit elements were spread across the file to ensure feedback at every step. The structure was linear, so users could scroll and follow the process naturally.

---

## Step-by-Step Code Breakdown

### Importing Libraries

```python
import os
import io
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import pycountry
```

This section collected every tool the app needed. I kept imports grouped. The standard libraries `os` and `io` allowed me to manage files and streams. The `requests` library was included in case I needed to fetch datasets over the web. Pandas structured tabular data. Streamlit rendered everything. Plotly made charts. The scikit-learn modules allowed encoding, splitting, training, and evaluating. Pycountry standardized country references. Without this group of imports, none of the later steps would be possible.

---

### Page Configuration

```python
st.set_page_config(page_title="Disaster Hotspot Prediction", layout="wide")
st.title("Disaster Hotspot Prediction")
st.caption("Colab + GitHub + Streamlit (no local setup)")
```

This section defined the look of the Streamlit app. I set the page title so the browser tab displayed a clear name. I used the wide layout so the map and tables had enough space. The title appeared at the top as a clear header. I added a caption to stress that the app was designed to work without local setup. This reassured users they could run it easily.

---

### Data Loading Helper

```python
@st.cache_data
def load_data(path_or_url):
    if path_or_url.startswith("http"):
        df = pd.read_csv(path_or_url)
    else:
        df = pd.read_csv(path_or_url)
    return df
```

This helper was central to performance. By caching results, Streamlit avoided reloading the dataset every time the page refreshed. The function took either a URL or a file path. It checked the prefix and then read the CSV. The return was a DataFrame. By isolating data loading in one helper, I ensured that any changes to how data was fetched would require modification in just one place.

---

### Using the Loader

```python
data_path = os.getenv("RAW_CSV_URL", "data/nasa_disaster_dataset.csv")
df = load_data(data_path)
st.write("Data shape:", df.shape)
```

Here I decided where to fetch the data from. The `os.getenv` call looked for an environment variable. If it was missing, the loader defaulted to the local CSV path. After loading, I wrote the shape to the Streamlit page. This gave users confirmation of how many rows and columns were in the dataset. Immediate feedback like this was important for transparency.

---

### Displaying Samples

```python
st.subheader("Sample of Data")
st.dataframe(df.head())
```

This code created a preview. By displaying the first few rows, users could inspect column names and data formats. This was often the first checkpoint for validation. If the dataset was not what was expected, the error would become visible immediately.

---

### Feature and Target Split

```python
X = df.drop("disaster_type", axis=1)
y = df["disaster_type"]
```

I separated predictors from the label. The predictors included country and other features. The label was the type of disaster. This step prepared the dataset for scikit-learn utilities, which required X and y to be distinct.

---

### Identifying Categorical Columns

```python
categorical_features = X.select_dtypes(include=["object"]).columns
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)
```

I programmatically identified which columns were categorical. By using pandas type checking, I avoided hardcoding column names. Then I built a column transformer. It applied a one-hot encoder to categorical features. By ignoring unknown categories, the encoder avoided errors during inference. This design made the pipeline robust.

---

### Building the Model Pipeline

```python
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
```

This pipeline chained preprocessing with logistic regression. First, the transformer encoded categories. Then the classifier trained on the processed data. The pipeline ensured the same transformations were applied during training and prediction. The maximum iterations parameter guaranteed convergence for logistic regression.

---

### Training and Splitting

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
clf.fit(X_train, y_train)
```

This block split the dataset into training and testing sets. I reserved 20 percent for testing. By stratifying on y, I kept class proportions balanced. The random seed ensured reproducibility. Finally, I trained the pipeline using the training set. This line marked the actual learning step of the project.

---

### Model Evaluation

```python
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write("Accuracy:", acc)
st.text(classification_report(y_test, y_pred))
```

After training, I predicted on the test set. I computed accuracy and displayed it in the app. I also produced a classification report with precision, recall, and f1-score. This allowed me to judge not just overall correctness but also performance across each class. By printing these metrics, users could see strengths and weaknesses of the model.

---

### Predictions for All Countries

```python
country_preds = df.copy()
country_preds["prediction"] = clf.predict(X)
country_preds["confidence"] = clf.predict_proba(X).max(axis=1)
```

Here I created a new DataFrame to hold predictions. The `prediction` column stored the predicted disaster type for each record. The `confidence` column kept the maximum probability of the prediction. By attaching these columns, I enriched the dataset for visualization. Confidence values were especially important for interpreting reliability.

---

### ISO Code Mapping

```python
def get_country_iso(country_name):
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except:
        return None

country_preds["iso_alpha"] = country_preds["country"].apply(get_country_iso)
```

This helper turned country names into ISO alpha-3 codes. The `pycountry` library looked up standardized names. I used try-except so errors did not crash the app. The output was stored in a new column. This mapping was necessary for Plotly, which required codes instead of raw names. Without this step, the map would not align data with geographic regions.

---

### Creating the Choropleth Map

```python
fig = px.choropleth(
    country_preds,
    locations="iso_alpha",
    color="prediction",
    hover_name="country",
    hover_data=["confidence"],
    color_discrete_sequence=px.colors.qualitative.Set1,
    title="Predicted Disaster Type by Country"
)
st.plotly_chart(fig, use_container_width=True)
```

This was the visualization block. I used `px.choropleth` to create a map where each country was shaded based on predicted disaster type. Hovering displayed the country name and confidence. The discrete color set made the map clear. The title gave context. I rendered the map using Streamlit with full width. This map was the centerpiece of the app because it transformed model output into a global picture.

---

## Conclusion

The entire project showed how to integrate machine learning with interactive visualization. Each block contributed a specific function. The data loader made it simple to fetch input. The preprocessing pipeline encoded categories. Logistic regression trained quickly and effectively. The evaluation step provided accuracy and detailed metrics. Mapping helpers transformed country names into ISO codes. Finally, Plotly visualized predictions in a way that anyone could interpret.

What I gained from this was not just a functioning app but a process of turning an idea into something shareable. The application was portable and ran entirely in the browser. It required no heavy computation yet delivered insight. By structuring the code carefully and explaining every block, I created a tool that was both educational and practical.

