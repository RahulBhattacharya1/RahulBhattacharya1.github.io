---
layout: default
title: "Creating my AI Retail Sales Forecasting"
date: 2022-02-19 11:43:26
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-retail-sales-forecasting.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_retail_sales_forecasting
featured: true
---

I remember visiting a local supermarket one weekend and noticing how some shelves were empty while others were overstocked. That small observation made me think about how difficult it is for businesses to predict demand with accuracy. When items are out of stock, customers feel disappointed and may not return. When items are overstocked, the business loses money in storage and markdowns. This personal experience pushed me to imagine how machine learning could help avoid such situations. Dataset used [here](https://www.kaggle.com/datasets/manjeetsingh/retaildataset).

Later, while reflecting on that visit, I realized that many retailers struggle with balancing supply and demand. Forecasting sales is not only about numbers but about improving customer trust and operational efficiency. I wanted to build something practical that could turn data into insights. That thought gave birth to my project: a **Retail Sales Forecasting App** using Python, Streamlit, and machine learning. In this blog I will walk through every file, every function, and every helper that I had to design and upload into my GitHub repository to make the app run smoothly.


## requirements.txt

The project required a few core Python libraries, and I captured them in the `requirements.txt` file. Without this file, others who want to run the project would not know the exact dependencies. Listing them ensures reproducibility and makes deployment seamless. The file looked like this:

```python
streamlit>=1.37
scikit-learn==1.5.1
joblib==1.4.2
pandas==2.2.2
numpy==2.0.2

```

Each library had a specific purpose. Streamlit allowed me to build the interactive web app. Scikit-learn provided machine learning utilities, although I had saved a pre-trained model. Joblib helped me load the serialized model. Pandas and NumPy gave me data structures and mathematical functions to handle input data. I kept the versions fixed to prevent conflicts across environments.

## models/feature_schema.txt

This text file was simple but critical. It listed all the features that the trained model expected. When creating a forecasting pipeline, having a consistent schema prevents mismatches between training and inference. The content was:

```python
Store
Dept
IsHoliday
year
month
week
quarter
dayofyear
week_sin
week_cos
month_sin
month_cos
lag1
lag4_mean
```

The schema defined both categorical and engineered time-based features. For example, `Store` and `Dept` identified location and department. `IsHoliday` marked promotional or holiday weeks. Time features like `year`, `month`, `week`, and `quarter` allowed the model to learn seasonal cycles. The sine and cosine columns such as `week_sin` and `week_cos` encoded cyclic behavior. The lag features like `lag1` and `lag4_mean` gave the model short-term memory of past sales. By uploading this schema file to GitHub, I ensured clarity about what the model consumed.

## app.py

The `app.py` file served as the main entry point of the application. I structured it so that every important step was clear and reusable. I will now break it down into blocks, show the code, and explain its purpose.


### Code Block 1

```python
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date
```

This block handled an important part of the application. I will now explain its role in detail.

This section imported all the necessary libraries. I needed os for file paths, joblib for loading the model, numpy and pandas for data manipulation, and streamlit for the web interface. The datetime module helped with date inputs. By grouping imports at the top, the code became easy to maintain and anyone could quickly see the dependencies.


### Code Block 2

```python
st.set_page_config(page_title="Weekly Sales Forecaster", page_icon="📈", layout="centered")
```

This block handled an important part of the application. I will now explain its role in detail.

Here I defined the Streamlit page configuration. I specified the page title, the icon, and the centered layout. This step enhanced the user experience by giving a professional look and avoiding clutter.


### Code Block 3

```python
# ===== Load model
MODEL_PATH = os.path.join("models", "sales_forecast_hgb.pkl")
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found at models/sales_forecast_hgb.pkl. Please upload it to the models/ folder.")
    st.stop()
```

This block handled an important part of the application. I will now explain its role in detail.

This block loaded the trained model bundle. I first checked if the pickle file existed under the models folder. If it was missing, the application stopped and displayed an error. If present, I loaded it using joblib and extracted both the model and the expected features. This design avoided runtime failures and gave users clear guidance.


### Code Block 4

```python
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]
```

This block handled an important part of the application. I will now explain its role in detail.

This block loaded the trained model bundle. I first checked if the pickle file existed under the models folder. If it was missing, the application stopped and displayed an error. If present, I loaded it using joblib and extracted both the model and the expected features. This design avoided runtime failures and gave users clear guidance.


### Code Block 5

```python
# ===== Feature function (must match training)
def compute_features(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
```

This block handled an important part of the application. I will now explain its role in detail.

This function transformed raw input data into the features expected by the model. I converted Store and Dept to numeric values, parsed the Date field, and ensured IsHoliday became numeric. The function was central to maintaining parity with the training process. Without this helper, the model would misinterpret the user input and predictions would be unreliable.


### Code Block 6

```python
d["Store"] = pd.to_numeric(d["Store"], errors="coerce").astype("Int64")
    d["Dept"] = pd.to_numeric(d["Dept"], errors="coerce").astype("Int64")
```

This block handled an important part of the application. I will now explain its role in detail.

This block handled additional logic in the application. It connected different steps, ensured user input was validated, and prepared output. Each part worked together to maintain flow from data ingestion to prediction.


### Code Block 7

```python
# Date parsing
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce", dayfirst=True)
```

This block handled an important part of the application. I will now explain its role in detail.

This block handled additional logic in the application. It connected different steps, ensured user input was validated, and prepared output. Each part worked together to maintain flow from data ingestion to prediction.


### Code Block 8

```python
# IsHoliday → 0/1
    if d["IsHoliday"].dtype == bool:
        d["IsHoliday"] = d["IsHoliday"].astype(int)
    else:
        d["IsHoliday"] = (
            d["IsHoliday"].astype(str).str.strip().str.lower()
            .isin(["true", "1", "yes", "y", "t"]).astype(int)
        )
```

This block handled an important part of the application. I will now explain its role in detail.

This block handled additional logic in the application. It connected different steps, ensured user input was validated, and prepared output. Each part worked together to maintain flow from data ingestion to prediction.


### Code Block 9

```python
d = d.dropna(subset=["Store", "Dept", "Date"])
```

This block handled an important part of the application. I will now explain its role in detail.

This block handled additional logic in the application. It connected different steps, ensured user input was validated, and prepared output. Each part worked together to maintain flow from data ingestion to prediction.


### Code Block 10

```python
# Time features
    d["year"] = d["Date"].dt.year
    d["month"] = d["Date"].dt.month
    d["week"] = d["Date"].dt.isocalendar().week.astype(int)
    d["quarter"] = d["Date"].dt.quarter
    d["dayofyear"] = d["Date"].dt.dayofyear
```

This block handled an important part of the application. I will now explain its role in detail.

This block handled additional logic in the application. It connected different steps, ensured user input was validated, and prepared output. Each part worked together to maintain flow from data ingestion to prediction.


### Code Block 11

```python
d["week_sin"] = np.sin(2 * np.pi * d["week"] / 52.0)
    d["week_cos"] = np.cos(2 * np.pi * d["week"] / 52.0)
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12.0)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12.0)
```

This block handled an important part of the application. I will now explain its role in detail.

This block handled additional logic in the application. It connected different steps, ensured user input was validated, and prepared output. Each part worked together to maintain flow from data ingestion to prediction.


### Code Block 12

```python
# Lag features — user can supply manually; default = 0
    d["lag1"] = 0.0
    d["lag4_mean"] = 0.0
```

This block handled an important part of the application. I will now explain its role in detail.

This block handled additional logic in the application. It connected different steps, ensured user input was validated, and prepared output. Each part worked together to maintain flow from data ingestion to prediction.


### Code Block 13

```python
return d
```

This block handled an important part of the application. I will now explain its role in detail.

This block handled additional logic in the application. It connected different steps, ensured user input was validated, and prepared output. Each part worked together to maintain flow from data ingestion to prediction.


### Code Block 14

```python
# ===== Streamlit UI
st.title("📈 Weekly Sales Forecaster")
st.write("Predict weekly sales for a Store & Dept, considering holidays and seasonal patterns.")
```

This block handled an important part of the application. I will now explain its role in detail.

This block defined the user interface layout. I displayed titles and headers to guide users through the steps. Streamlit's functions allowed me to present input forms and results in a structured manner.


### Code Block 15

```python
col1, col2 = st.columns(2)
with col1:
    store = st.number_input("Store ID", min_value=1, step=1, value=1)
    dept = st.number_input("Dept ID", min_value=1, step=1, value=1)
with col2:
    the_date = st.date_input("Week Date", value=date(2012, 12, 14))
    is_holiday = st.checkbox("Is Holiday Week?", value=False)
```

This block handled an important part of the application. I will now explain its role in detail.

This block handled additional logic in the application. It connected different steps, ensured user input was validated, and prepared output. Each part worked together to maintain flow from data ingestion to prediction.


### Code Block 16

```python
with st.expander("Advanced: Provide sales context (optional)"):
    use_context = st.checkbox("Provide recent sales info")
    last_week = st.number_input("Last Week's Sales (lag1)", min_value=0.0, value=0.0, step=100.0)
    last_4wk_mean = st.number_input("Mean of Previous 4 Weeks (lag4_mean)", min_value=0.0, value=0.0, step=100.0)
```

This block handled an important part of the application. I will now explain its role in detail.

This block handled additional logic in the application. It connected different steps, ensured user input was validated, and prepared output. Each part worked together to maintain flow from data ingestion to prediction.


### Code Block 17

```python
if st.button("Predict Sales"):
    df_in = pd.DataFrame([{
        "Store": store,
        "Dept": dept,
        "Date": pd.Timestamp(the_date),
        "IsHoliday": int(is_holiday)
    }])
```

This block handled an important part of the application. I will now explain its role in detail.

This block handled additional logic in the application. It connected different steps, ensured user input was validated, and prepared output. Each part worked together to maintain flow from data ingestion to prediction.


### Code Block 18

```python
feats = compute_features(df_in)
```

This block handled an important part of the application. I will now explain its role in detail.

This block handled additional logic in the application. It connected different steps, ensured user input was validated, and prepared output. Each part worked together to maintain flow from data ingestion to prediction.


### Code Block 19

```python
if use_context:
        feats["lag1"] = float(last_week)
        feats["lag4_mean"] = float(last_4wk_mean)
```

This block handled an important part of the application. I will now explain its role in detail.

This block handled additional logic in the application. It connected different steps, ensured user input was validated, and prepared output. Each part worked together to maintain flow from data ingestion to prediction.


### Code Block 20

```python
X = feats.reindex(columns=FEATURES, fill_value=0.0).astype(float)
```

This block handled an important part of the application. I will now explain its role in detail.

This block handled additional logic in the application. It connected different steps, ensured user input was validated, and prepared output. Each part worked together to maintain flow from data ingestion to prediction.


### Code Block 21

```python
if X.isna().any().any():
        st.error("Inputs produced invalid values. Please adjust and try again.")
    else:
        yhat = model.predict(X)[0]
        st.success(f"Predicted Weekly Sales: ${yhat:,.2f}")
```

This block handled an important part of the application. I will now explain its role in detail.

This was the core prediction block. After computing the features using my helper, I called the model's predict method. I then created a new column for predicted sales and presented it in the app. This step completed the journey from raw input to forecast.


### Code Block 22

```python
st.markdown("---")
st.markdown("### Example inputs you can try:")
st.code(
    "Store=1, Dept=1, Date=2012-12-14, IsHoliday=True\n"
    "Store=5, Dept=12, Date=2011-11-25, IsHoliday=True\n"
    "Store=20, Dept=3, Date=2011-03-18, IsHoliday=False",
    language="text"
)
```

This block handled an important part of the application. I will now explain its role in detail.

This block handled additional logic in the application. It connected different steps, ensured user input was validated, and prepared output. Each part worked together to maintain flow from data ingestion to prediction.


## Conclusion

By combining Python, Streamlit, and a pre-trained model, I built a forecasting app that helps retail teams plan inventory more effectively. Every file I uploaded to GitHub had a purpose. The requirements file pinned dependencies. The schema file documented the model's expectations. The pickle file held the model itself. The app.py script tied everything together into an accessible tool. Building this taught me that good machine learning projects are not just about models but about thoughtful engineering, clear structure, and usability. Anyone reading this can now follow my steps, clone the repository, and deploy their own sales forecasting tool.

###Additional Reflection:

In this project I learned how important it is to document every function and helper. A well documented project does not confuse future contributors. The code blocks show that even small helpers like compute_features have a major impact. Without them predictions would not align with training. By writing this blog I also realized that sharing thought process is as valuable as sharing code. Readers benefit not only from running the app but also from understanding the reasoning behind each design choice. In this project I learned how important it is to document every function and helper. A well documented project does not confuse future contributors. The code blocks show that even small helpers like compute_features have a major impact. Without them predictions would not align with training. By writing this blog I also realized that sharing thought process is as valuable as sharing code. Readers benefit not only from running the app but also from understanding the reasoning behind each design choice.
