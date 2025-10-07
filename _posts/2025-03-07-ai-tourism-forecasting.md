---
layout: default
title: "Creating my AI Tourism Forecasting App"
date: 2025-03-07 13:46:39
categories: [ai]
tags: [python,streamlit,bert,self-trained]
thumbnail: /assets/images/tourism.webp
thumbnail_mobile: /assets/images/tourism_forecasting_sq.webp
demo_link: https://rahuls-ai-tourism-forecasting.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_tourism_forecasting
---

This post walks through a small forecasting project that I packaged as a simple Streamlit app. I
kept it self contained, so it does not reach out to any external APIs or big datasets. The goal is
to show a clean path from a tidy time series to a short horizon forecast. I will go through each code
block, every helper, and each file I committed to make the app run on my machine and in the cloud.

Project files I committed
- `app.py` — the Streamlit app with all logic and UI
- `requirements.txt` — pinned dependencies so runs are repeatable
- `data/tourism_monthly_sample.csv` — a tiny monthly series sample
- `data/tourism_quarterly_sample.csv` — a tiny quarterly series sample
- `data/tourism_yearly_sample.csv` — a tiny yearly series sample
- `README.md` — a short note about the repo purpose


## Complete `app.py`


I like to start by sharing the working script in full. The original file I committed had the exact structure below. I filled in the elided sections so the app runs end to end without any placeholders. The sections that follow break the script into focused blocks and explain what each piece does.

```python
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Tourism Forecasting (Self-contained)", layout="wide")
st.title("Tourism Forecasting — Self-contained Starter (No external APIs)")

# ----------------------
# Data registries
# ----------------------
DATA_MAP = {
    "Monthly":   "data/tourism_monthly_sample.csv",
    "Quarterly": "data/tourism_quarterly_sample.csv",
    "Yearly":    "data/tourism_yearly_sample.csv",
}
# Forecast horizon by frequency (how many steps ahead to predict)
H_MAP = {"Monthly": 12, "Quarterly": 4, "Yearly": 3}
# Seasonality period by frequency (m=12 for monthly, 4 for quarterly, 1 for yearly/no season)
M_MAP = {"Monthly": 12, "Quarterly": 4, "Yearly": 1}

# ----------------------
# UI: frequency + dataset
# ----------------------
freq = st.selectbox("Frequency", list(DATA_MAP.keys()), index=0)
csv_path = Path(DATA_MAP[freq])
H = int(H_MAP[freq])
m = int(M_MAP[freq])

# ----------------------
# Data loading
# ----------------------
@st.cache_data(show_spinner=True)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    # small hygiene: ensure proper dtypes
    df["item_id"] = df["item_id"].astype(str)
    df = df.sort_values(["item_id", "date"]).reset_index(drop=True)
    return df

df = load_data(csv_path)

# ----------------------
# UI: pick an item/series
# ----------------------
ids = sorted(df["item_id"].unique().tolist())
default_idx = 0 if len(ids) == 0 else 0
series_id = st.selectbox("Choose a series (item_id)", ids, index=default_idx if ids else 0)

# Filter to selected series and set index
series_df = df.loc[df["item_id"] == series_id, ["date", "value"]].copy()
series_df = series_df.set_index("date")["value"].asfreq(
    {"Monthly":"MS", "Quarterly":"Q", "Yearly":"A"}[freq], method=None
)

# Basic guardrails
if series_df.isna().any():
    series_df = series_df.fillna(method="ffill").fillna(method="bfill")

# ----------------------
# Train / test split
# ----------------------
if len(series_df) > H:
    y_train = series_df.iloc[:-H]
    y_test  = series_df.iloc[-H:]
else:
    y_train = series_df.copy()
    y_test  = pd.Series(dtype=float)

# ----------------------
# Model fit (SARIMAX)
# ----------------------
fit = None
err = None
try:
    fit = SARIMAX(
        y_train,
        order=(1,1,1),
        seasonal_order=(1,1,1,m) if m > 1 else (0,0,0,0),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
except Exception as e:
    err = str(e)

# ----------------------
# Forecast
# ----------------------
if fit is not None:
    steps = H if len(y_test) else max(1, H)
    fc = fit.forecast(steps=steps)
    # If yearly, ensure index increments by 1 year; for monthly/quarterly, pandas handles
    y_fc = fc
else:
    y_fc = pd.Series(dtype=float)

# ----------------------
# Plot
# ----------------------
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(series_df.index, series_df.values, label="history", linewidth=2)
if len(y_fc):
    ax.plot(y_fc.index, y_fc.values, label="forecast", linewidth=2)
if len(y_test):
    ax.plot(y_test.index, y_test.values, label="holdout", linewidth=2, linestyle="--")
ax.set_title(f"{series_id} — {freq} forecast")
ax.set_xlabel("date")
ax.set_ylabel("value")
ax.legend()
st.pyplot(fig)

# ----------------------
# Metrics
# ----------------------
def mape(a, f):
    a, f = np.array(a, float), np.array(f, float)
    return float(np.mean(np.abs((a - f) / np.maximum(1e-8, np.abs(a))))) * 100

if len(y_test) and len(y_fc) == len(y_test):
    mae = float(np.abs(y_test - y_fc).mean())
    rmse = float(np.sqrt(((y_test - y_fc)**2).mean()))
    st.write(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape(y_test, y_fc):.2f}%")
elif len(y_test):
    st.write("Forecast and holdout lengths do not match; skipping metrics.")

# ----------------------
# Download
# ----------------------
out = pd.DataFrame({"date": y_fc.index, "forecast": y_fc.values})
st.download_button("Download forecast CSV", out.to_csv(index=False).encode(), file_name="forecast.csv", mime="text/csv")
```


## Imports and page setup


This block declares the core libraries and configures the Streamlit page. I use `pandas` and `numpy` for data handling, `statsmodels` for SARIMAX, and `matplotlib` for plotting. `Path` gives me clean path joins across systems. `set_page_config` sets a wide layout and a clear page title so the app looks consistent.

```python
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Tourism Forecasting (Self-contained)", layout="wide")
st.title("Tourism Forecasting — Self-contained Starter (No external APIs)")
```


## Data registry and forecast settings


These small dictionaries act as a registry. `DATA_MAP` links a frequency label to a local CSV path. `H_MAP` sets a reasonable forecast horizon per cadence, while `M_MAP` holds the seasonal period used by SARIMAX. Keeping these values in one place makes the app easy to extend when I add more series.

```python
# Data registries
DATA_MAP = {
    "Monthly":   "data/tourism_monthly_sample.csv",
    "Quarterly": "data/tourism_quarterly_sample.csv",
    "Yearly":    "data/tourism_yearly_sample.csv",
}
H_MAP = {"Monthly": 12, "Quarterly": 4, "Yearly": 3}
M_MAP = {"Monthly": 12, "Quarterly": 4, "Yearly": 1}
```


## Frequency selection UI


I start at the top of the UI with a single dropdown. When I choose a frequency, the code looks up the CSV, the forecast horizon, and the seasonal period. This keeps downstream logic simple because every step depends on `freq` and pulls the correct constants. Streamlit updates instantly when I switch the control.

```python
freq = st.selectbox("Frequency", list(DATA_MAP.keys()), index=0)
csv_path = Path(DATA_MAP[freq])
H = int(H_MAP[freq])
m = int(M_MAP[freq])
```


## Cached data loader


I wrap file IO in a helper and decorate it with `@st.cache_data`. The cache keeps reloads snappy when I switch series or redraw plots. The function parses the date column up front and normalizes the types to avoid subtle bugs later. A stable sort makes the time axis deterministic across runs.

```python
@st.cache_data(show_spinner=True)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["item_id"] = df["item_id"].astype(str)
    df = df.sort_values(["item_id", "date"]).reset_index(drop=True)
    return df

df = load_data(csv_path)
```


## Pick a series and prepare the index


Many datasets carry multiple series under an `item_id`. I let the user choose one id and then I isolate that slice into a clean `Series`. I also set an explicit frequency on the datetime index, which lets the model and the plots behave well when I forecast. A short forward and backward fill handles small gaps without overcomplicating the example.

```python
ids = sorted(df["item_id"].unique().tolist())
series_id = st.selectbox("Choose a series (item_id)", ids, index=0)

series_df = df.loc[df["item_id"] == series_id, ["date", "value"]].copy()
series_df = series_df.set_index("date")["value"].asfreq(
    {"Monthly":"MS", "Quarterly":"Q", "Yearly":"A"}[freq], method=None
)

if series_df.isna().any():
    series_df = series_df.fillna(method="ffill").fillna(method="bfill")
```


## Train and holdout split


I prefer to keep evaluation honest with a time ordered split. If the series has enough history, the last horizon becomes holdout and the rest is training. Otherwise the app trains on everything and skips metrics gracefully. This keeps the demo predictable across the small sample files.

```python
if len(series_df) > H:
    y_train = series_df.iloc[:-H]
    y_test  = series_df.iloc[-H:]
else:
    y_train = series_df.copy()
    y_test  = pd.Series(dtype=float)
```


## SARIMAX training


For this starter I use a single SARIMAX specification that works across the three cadences. The seasonal order activates only when the seasonal period is greater than one. I also turn off strict stationarity and invertibility to avoid brittle failures on short samples. In production I would search the hyperparameters, but here I keep the focus on clarity.

```python
fit = None
err = None
try:
    fit = SARIMAX(
        y_train,
        order=(1,1,1),
        seasonal_order=(1,1,1,m) if m > 1 else (0,0,0,0),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
except Exception as e:
    err = str(e)
```


## Out-of-sample forecast


After the model fits, I request the number of steps defined by the horizon. When there is a holdout slice, the forecast length matches it for a fair comparison. The index carries forward with the same frequency, which makes charting simple. When the model fails for any reason, the app falls back to an empty series.

```python
if fit is not None:
    steps = H if len(y_test) else max(1, H)
    fc = fit.forecast(steps=steps)
    y_fc = fc
else:
    y_fc = pd.Series(dtype=float)
```


## Visualization


I keep the chart minimal because the point is the workflow, not the styling. The history series anchors the context, the forecast extends it, and the dashed holdout gives me a quick sanity check. Readable labels and a compact size help the plot sit well in a sidebar and a wide content layout. Streamlit handles the rendering with a single call.

```python
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(series_df.index, series_df.values, label="history", linewidth=2)
if len(y_fc):
    ax.plot(y_fc.index, y_fc.values, label="forecast", linewidth=2)
if len(y_test):
    ax.plot(y_test.index, y_test.values, label="holdout", linewidth=2, linestyle="--")
ax.set_title(f"{series_id} — {freq} forecast")
ax.set_xlabel("date")
ax.set_ylabel("value")
ax.legend()
st.pyplot(fig)
```


## Evaluation helpers


Metrics belong near the plot so I can judge the forecast in context. I calculate MAE and RMSE directly from the arrays and add a small `mape` helper that avoids division by zero. If lengths drift for any reason, the app explains why metrics are skipped instead of throwing an error. This keeps the teaching example calm under minor data issues.

```python
def mape(a, f):
    a, f = np.array(a, float), np.array(f, float)
    return float(np.mean(np.abs((a - f) / np.maximum(1e-8, np.abs(a))))) * 100

if len(y_test) and len(y_fc) == len(y_test):
    mae = float(np.abs(y_test - y_fc).mean())
    rmse = float(np.sqrt(((y_test - y_fc)**2).mean()))
    st.write(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape(y_test, y_fc):.2f}%")
elif len(y_test):
    st.write("Forecast and holdout lengths do not match; skipping metrics.")
```


## Export


A small export goes a long way in demos. I package the forecast as two columns so it is easy to join back on the original dataset. The generated CSV helps me validate downstream steps like plotting in another tool or aggregating by groups. Streamlit streams the bytes directly without writing a file to disk.

```python
out = pd.DataFrame({"date": y_fc.index, "forecast": y_fc.values})
st.download_button("Download forecast CSV", out.to_csv(index=False).encode(), file_name="forecast.csv", mime="text/csv")
```


## What I pushed to GitHub and why


- `app.py` holds the UI and logic in one place so the post can reference a single script. - `requirements.txt` pins specific versions for Streamlit, pandas, numpy, statsmodels, and matplotlib so deployments reproduce results. - `data/*.csv` carry small, tidy samples with three consistent columns: `item_id`, `date`, and `value`. - `README.md` documents the intent of the repository in plain language and sets expectations about size and scope.


## Data shape and assumptions


The CSVs follow a narrow, long format with `item_id`, `date`, and `value`. Dates are parseable ISO strings. Each file contains a single cadence so the frequency dropdown can activate the right settings. If I add more series later, I only need to append rows and keep the column names the same.


## Running it locally


```python
# create and activate a virtual environment if you prefer
pip install -r requirements.txt
streamlit run app.py
```

The app starts in a browser tab at the default Streamlit port. I can change the frequency, pick a series, and download the forecast without touching any credentials. Because the dependencies are pinned, a teammate sees the same behavior on their machine. This tight loop helps me test ideas without long setup steps.


## Deploying on Streamlit Cloud


I pointed Streamlit Cloud to the repository and set the entry point to `app.py`. Since the data lives under `data/` and the app never calls external endpoints, the container stays small and predictable. The default hardware is plenty for SARIMAX on these samples. Cold starts are quick because the import set is modest.


## Why I chose SARIMAX for the starter


SARIMAX is a mature classical baseline that handles trend and seasonality with very little ceremony. For a teaching example it keeps the math close to the series and avoids heavy dependencies. It also works without an internet connection, which matters in constrained environments. When I want to compare with machine learning models later, this script gives me a reliable yardstick.


## Easy extensions


The registries make it simple to add weekly or daily cadences with a few extra lines. I can also expose model knobs in the sidebar to tune orders and compare metrics across specs. If I need multiple series forecasts, a loop over `item_id` with a small progress bar can write a combined CSV. These changes fit naturally without refactoring the whole script.
