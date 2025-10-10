---
layout: default
title: "Building my AI Avocado Price Predictor"
date: 2024-08-11 10:21:33
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/avocado.webp
thumbnail_mobile: /assets/images/avocado_price_sq.webp
demo_link: https://rahuls-ai-avocado-price-prediction.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_avocado_price_prediction
featured: true
---

I once stood in a grocery store aisle staring at a pile of avocados. Prices had changed from the week before, and I began to wonder what drove the fluctuations. It felt random at first glance, yet I knew deeper factors like seasonality, supply volumes, and regional differences influenced those numbers. That personal curiosity was enough to spark this project. I decided to build a predictive tool that could estimate avocado prices based on historical and structural features. Dataset used [here](https://www.kaggle.com/datasets/neuromusic/avocado-prices).

This blog is not only about showing a functional app but also about breaking it down piece by piece. I want to describe every file, every function, and every helper routine I used to make this possible. By walking carefully through each element, I can show how all the components fit together into a system that feels both understandable and reproducible. This is my complete journey from idea to deployment.

---

## Repository Structure

When I unzipped my project, the structure looked like this:

```
ai_avocado_price_prediction-main/
│
├── app.py
├── requirements.txt
└── models/
    ├── avocado_price_model.pkl
    └── metadata.json
```

Each of these files served a unique role. The `app.py` file contained all the application logic. The `requirements.txt` listed dependencies needed for the app to run consistently across environments. The `models/` folder stored the machine learning artifacts, including the trained model and a metadata dictionary that held important mapping rules and feature descriptions. I will now break down each of these components carefully.

---

## requirements.txt

The dependencies required to run this app were defined here.

```python
streamlit>=1.37
scikit-learn==1.6.1
joblib==1.5.2
pandas==2.2.2
numpy==2.0.2

```

I locked the key libraries with specific versions. Streamlit powered the interactive interface. Scikit-learn and joblib were used for model training and persistence. Pandas and NumPy handled the data manipulation routines. Having these versions pinned ensured that if someone else cloned the repository and installed requirements, they would replicate the exact environment I had when building and testing the app.

---

## metadata.json

This file contained metadata describing model inputs, dummy variable prefixes, and categories.

```python
{
  "feature_columns": [
    "Total Volume",
    "4046",
    "4225",
    "4770",
    "Total Bags",
    "Small Bags",
    "Large Bags",
    "XLarge Bags",
    "year_from_date",
    "month",
    "week",
    "type_conventional",
    "type_organic",
    "region_Albany",
    "region_Atlanta",
    "region_BaltimoreWashington",
    "region_Boise",
    "region_Boston",
    "region_BuffaloRochester",
    "region_California",
    "region_Charlotte",
    "region_Chicago",
    "region_CincinnatiDayton",
    "region_Columbus",
    "region_DallasFtWorth",
    "region_Denver",
    "region_Detroit",
    "region_GrandRapids",
    "region_GreatLakes",
    "region_HarrisburgScranton",
    "region_HartfordSpringfield",
    "region_Houston",
    "region_Indianapolis",
    "region_Jacksonville",
    "region_LasVegas",
    "region_LosAngeles",
    "region_Louisville",
    "region_MiamiFtLauderdale",
    "region_Midsouth",
    "region_Nashville",
    "region_NewOrleansMobile",
    "region_NewYork",
    "region_Northeast",
    "region_NorthernNewEngland",
    "region_Orlando",
    "region_Philadelphia",
    "region_PhoenixTucson",
    "region_Pittsburgh",
    "region_Plains",
    "region_Portland",
    "region_RaleighGreensboro",
    "region_RichmondNorfolk",
    "region_Roanoke",
    "region_Sacramento",
    "region_SanDiego",
    "region_SanFrancisco",
    "region_Seattle",
    "region_SouthCarolina",
    "region_SouthCentral",
    "region_Southeast",
    "region_Spokane",
    "region_StLouis",
    "region_Syracuse",
    "region_Tampa",
    "region_TotalUS",
    "region_West",
    "region_WestTexNewMexico"
  ],
  "regions": [
    "Albany",
    "Atlanta",
    "BaltimoreWashington",
    "Boise",
    "Boston",
    "BuffaloRochester",
    "California",
    "Charlotte",
    "Chicago",
    "CincinnatiDayton",
    "Columbus",
    "DallasFtWorth",
    "Denver",
    "Detroit",
    "GrandRapids",
    "GreatLakes",
    "HarrisburgScranton",
    "HartfordSpringfield",
    "Houston",
    "Indianapolis",
    "Jacksonville",
    "LasVegas",
    "LosAngeles",
    "Louisville",
    "MiamiFtLauderdale",
    "Midsouth",
    "Nashville",
    "NewOrleansMobile",
    "NewYork",
    "Northeast",
    "NorthernNewEngland",
    "Orlando",
    "Philadelphia",
    "PhoenixTucson",
    "Pittsburgh",
    "Plains",
    "Portland",
    "RaleighGreensboro",
    "RichmondNorfolk",
    "Roanoke",
    "Sacramento",
    "SanDiego",
    "SanFrancisco",
    "Seattle",
    "SouthCarolina",
    "SouthCentral",
    "Southeast",
    "Spokane",
    "StLouis",
    "Syracuse",
    "Tampa",
    "TotalUS",
    "West",
    "WestTexNewMexico"
  ],
  "types": [
    "conventional",
    "organic"
  ],
  "type_dummy_prefix": "type_",
  "region_dummy_prefix": "region_",
  "numeric_inputs": [
    "Total Volume",
    "4046",
    "4225",
    "4770",
    "Total Bags",
    "Small Bags",
    "Large Bags",
    "XLarge Bags",
    "year_from_date",
    "month",
    "week"
  ]
}
```

The `feature_columns` list specified the exact ordering of model features. The `regions` and `types` arrays outlined categorical variables expanded into dummy columns. The `numeric_inputs` section described which variables required numeric entry from the user. This file acted as the backbone of consistent input formatting. Without it, the app could not reliably transform raw form values into the correct feature vector expected by the model.

---

## models/avocado_price_model.pkl

This file was the trained machine learning model saved using joblib. It represented a regression estimator that learned relationships between avocado supply metrics, categorical encodings, and prices. I did not manually open this file in the app. Instead, I loaded it through joblib inside a dedicated helper. The presence of this file meant that training was already completed elsewhere, and this app only focused on inference.

---

## app.py

This was the heart of the application. Below is the full script and then a detailed breakdown section by section.

```python
import json
import joblib
import pandas as pd
import streamlit as st
from datetime import date

# -------------------------
# Load model & metadata
# -------------------------
MODEL_PATH = "models/avocado_price_model.pkl"
META_PATH = "models/metadata.json"

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_assets()

st.title("Avocado Price Predictor")

st.write("Enter inputs and click Predict. "
         "This model uses volumes, bag counts, seasonality, and region/type.")

# -------------------------
# UI controls
# -------------------------
# Numeric inputs
num_defaults = {
    "Total Volume": 1_000_00.0,   # feel free to change defaults
    "4046": 30_000.0,
    "4225": 30_000.0,
    "4770": 2_000.0,
    "Total Bags": 60_000.0,
    "Small Bags": 45_000.0,
    "Large Bags": 14_000.0,
    "XLarge Bags": 1_000.0,
}
col1, col2 = st.columns(2)

with col1:
    total_volume = st.number_input("Total Volume", min_value=0.0, value=float(num_defaults["Total Volume"]), step=1000.0)
    plu_4046 = st.number_input("4046", min_value=0.0, value=float(num_defaults["4046"]), step=100.0)
    plu_4225 = st.number_input("4225", min_value=0.0, value=float(num_defaults["4225"]), step=100.0)
    plu_4770 = st.number_input("4770", min_value=0.0, value=float(num_defaults["4770"]), step=50.0)

with col2:
    total_bags = st.number_input("Total Bags", min_value=0.0, value=float(num_defaults["Total Bags"]), step=1000.0)
    small_bags = st.number_input("Small Bags", min_value=0.0, value=float(num_defaults["Small Bags"]), step=500.0)
    large_bags = st.number_input("Large Bags", min_value=0.0, value=float(num_defaults["Large Bags"]), step=100.0)
    xlarge_bags = st.number_input("XLarge Bags", min_value=0.0, value=float(num_defaults["XLarge Bags"]), step=10.0)

# Date (for seasonality features)
d = st.date_input("Date (for seasonality)", value=date(2017, 6, 15))

# Type & Region
type_choice = st.selectbox("Type", options=meta["types"], index=0)
region_choice = st.selectbox("Region", options=meta["regions"], index=0)

# -------------------------
# Build feature row to match training columns
# -------------------------
def make_feature_row():
    # seasonality numeric features derived from the chosen date
    year_from_date = d.year
    month = d.month
    week = pd.Timestamp(d).isocalendar().week

    base = {
        "Total Volume": total_volume,
        "4046": plu_4046,
        "4225": plu_4225,
        "4770": plu_4770,
        "Total Bags": total_bags,
        "Small Bags": small_bags,
        "Large Bags": large_bags,
        "XLarge Bags": xlarge_bags,
        "year_from_date": year_from_date,
        "month": month,
        "week": int(week),
    }

    # Start a zero row for all expected columns
    row = pd.Series(0.0, index=pd.Index(meta["feature_columns"], name="feature"))

    # Fill numeric inputs
    for k, v in base.items():
        if k in row.index:
            row[k] = float(v)

    # Turn on the correct type & region dummy columns
    type_col_name = f"{meta['type_dummy_prefix']}{type_choice}"
    region_col_name = f"{meta['region_dummy_prefix']}{region_choice}"

    if type_col_name in row.index:
        row[type_col_name] = 1.0
    if region_col_name in row.index:
        row[region_col_name] = 1.0

    # Return as single-row DataFrame with columns in the exact order used in training
    return pd.DataFrame([row.values], columns=row.index)

if st.button("Predict"):
    X_new = make_feature_row()
    pred = float(model.predict(X_new)[0])
    st.success(f"Predicted AveragePrice: ${pred:.2f}")

```

---

## Deep Dive Into Feature Engineering

One of the most important design decisions was how to transform user inputs into the format that the model expected. The training data had been encoded with dummy variables for both type and region. That meant when a user chose "organic" or "LosAngeles", I had to make sure that the corresponding dummy columns were set to one while every other column remained zero. Without this alignment, the model would misinterpret the input.

This method may look simple, but it prevented subtle bugs. If I had only created keys for selected inputs, the DataFrame might have misaligned with the training order. Scikit-learn models are strict about feature ordering, so even a small mismatch would break predictions. The metadata file held the exact list of expected columns, which meant I could always trust that dictionary comprehension would initialize them in a predictable way. This was a safeguard against both human mistakes and environment differences.

---

## Streamlit Layout Considerations

The layout of the app might appear cosmetic, but it played a large role in usability. By using two columns for numeric inputs and then another two for temporal and categorical fields, I guided the user’s eyes through a logical sequence. The top half of the interface focused on quantities and volumes. The bottom half then shifted to contextual attributes like time and location. This natural grouping mirrored how analysts often think about data.

Streamlit’s column API made it easy to balance the screen. Without it, all inputs would have stacked vertically, creating a long and intimidating form. The choice of splitting the screen meant fewer scrolls and better engagement. Small design details like these can decide whether a casual user actually tries an application or leaves after a glance. I learned that technical accuracy must be paired with visual clarity for a tool to feel approachable.

---

## Reflection on Model Integration

The regression model was already trained outside of this script, but integrating it into a live app still required thought. The model itself was lightweight thanks to joblib serialization. Loading it directly every time would have been costly, so I relied on caching. The `@st.cache_resource` decorator turned out to be the right balance between simplicity and performance. With it, Streamlit remembered the object and only reloaded if files changed. This gave me near-instant responses during prediction.

The key lesson was that machine learning does not end at training. Deployment choices like caching, metadata alignment, and error handling often determine whether the system is usable. An accurate model without these details would remain locked away. By investing effort here, I converted a regression file into an interactive assistant for price prediction.

---

## Lessons Learned

Building this app taught me several lessons that extend beyond avocados. First, metadata is not just a side artifact but the core of reproducibility. Second, interface choices can make or break adoption. Third, even simple numeric inputs require validation and defaults if they are to be used reliably. Finally, prediction output must be clear and concise. Users do not want a vector of numbers. They want a simple answer they can interpret instantly.

Another lesson was the discipline of version pinning. Too often, projects break months later because of dependency drift. By fixing versions in `requirements.txt`, I made sure the repository was frozen in time. If I revisit it years later, the environment can be reconstructed with confidence. This habit is one of the most practical defenses against fragility in software projects.

---

## Broader Implications

What started as a curiosity about produce prices became an example of how machine learning can turn everyday questions into structured insights. The pipeline I used here—data preprocessing, model training, metadata storage, interface design, and deployment—could be applied to many other domains. It might be house prices, weather forecasts, or retail demand. The same pattern works across contexts. The details change, but the workflow remains robust.

This project reminded me that curiosity is often the first step in innovation. By following a simple question, I ended up with a deployable app that others could use. That experience continues to motivate me to transform observations into code and eventually into applications that can help decision making. Avocados were only the beginning.


---

## Error Handling and Validation

While the app looked simple, I also had to think about how to prevent bad input. For example, negative values in volume or bag counts would not make sense. By setting `min_value=0.0` in every numeric input widget, I eliminated the chance of entering invalid numbers. This validation was built into the interface itself, which is often more effective than downstream checks. The user could never progress past the input stage with a nonsensical entry.

I also added sensible step sizes for each widget. A field like "XLarge Bags" used a step of fifty, while "Total Volume" used a step of a thousand. These differences were not arbitrary. They reflected realistic scales of measurement. Users scrolling through values felt natural increments rather than confusing jumps. This alignment of interface controls with real-world scales built trust in the tool’s design.

---

## Testing the Application

Before publishing, I spent time testing the app with different scenarios. I selected dates from various months, toggled between conventional and organic, and switched across multiple regions. The goal was to see if the interface consistently built a complete feature vector. If any dummy column was missing or if the DataFrame misaligned, the model would raise errors. My testing confirmed that the metadata-driven initialization worked in all cases.

I also stress tested numeric ranges. Entering very high numbers did not crash the app, though predictions became unrealistic. That was acceptable because the model itself was only trained on realistic ranges. The key was that the application handled the input gracefully without raising exceptions. This gave me confidence to deploy publicly.

---

## Deployment Workflow

Once the app ran reliably on my machine, I turned to deployment. The repository structure was already streamlined. All I had to do was push the code to GitHub and then connect it with Streamlit Cloud. Because I had requirements pinned, the cloud service was able to reproduce the environment quickly. Within minutes, the application was live on a shareable URL.

Another detail was the presence of the `models/` folder. I uploaded both the pickle file and metadata to the repository. This made the repository self-contained. Anyone cloning it did not need to retrain or fetch missing pieces. Everything required for inference was already there. This principle of bundling artifacts is essential for portability. It meant my tool could travel with me to any environment without special setup.

---

## Extending the App

After deploying, I started thinking about possible extensions. One idea was to add charts that displayed historical price trends for the selected region and type. Another idea was to show confidence intervals alongside point predictions. Streamlit’s charting tools could make this straightforward. I also considered adding file upload functionality so that a user could predict multiple rows at once. These ideas remain future steps, but the foundation is strong enough to support them.

I also reflected on model retraining. The avocado market changes over time. A model frozen at one point may eventually become outdated. By keeping the training pipeline separate and saving new pickles, I could refresh the deployed app without rewriting the interface. This separation of concerns was intentional. The app consumed models but did not train them. That kept the runtime light and the interface responsive.

---

## Detailed Data Flow

The data flow from input to prediction followed a clear pipeline. First, numeric and categorical values were collected from widgets. Second, they were combined into a dictionary initialized with all zero values. Third, this dictionary was converted into a pandas DataFrame with strict column ordering. Fourth, the model received this DataFrame and produced a numeric prediction. Fifth, the prediction was formatted and displayed back on the screen.

The clarity of this pipeline was a major reason why debugging was straightforward. If something went wrong, I could isolate whether the issue was with input collection, dictionary population, DataFrame alignment, or model prediction. Each block had a defined responsibility. This modularity is a quality I now aim for in all projects.

---

## Closing Reflection

The avocado price predictor may appear narrow in scope, but it represents a larger philosophy. It showed me that structured curiosity, supported by careful design, can turn into a product others can use. I did not need a massive dataset or an advanced algorithm. What I needed was discipline in engineering details, respect for usability, and attention to reproducibility. These values transformed a passing thought in a grocery store into a running application.

Looking ahead, I see this project as a template. The same principles apply whether I am building a forecast for commodities, demand for retail items, or patterns in healthcare data. The context changes, but the structure endures. This lesson ensures that the time I invested here will continue to pay off in other domains for years to come.
