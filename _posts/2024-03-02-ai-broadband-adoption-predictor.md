---
layout: default
title: "Building my AI Broadband Adoption Predictor"
date: 2024-03-02 14:27:53
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/broadband.webp
thumbnail_mobile: /assets/images/broadband_sq.webp
demo_link: https://rahuls-ai-broadband-adoption-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_broadband_adoption_predictor
---

I wrote this app after noticing how broadband adoption numbers shift in uneven steps across countries and years. Budget cycles, infrastructure delays, and policy waves create surges followed by plateaus. I needed a compact tool that reads a simple TSV, applies a lightweight linear model, and lets me inspect predictions with context. The goal was clarity over complexity, and portability over vendor lock‑in. Dataset used [here](https://www.kaggle.com/datasets/gpreda/household-internet-connection-in-european-union).

This post documents the project end‑to‑end. It explains every helper, shows how inputs flow into features, and describes how the Streamlit UI stays responsive without global side effects. I also include the full source so readers can verify each claim. The structure is intentional: small functions, cached data access, explicit feature construction, and a portable model artifact in JSON.



## Repository Layout and Files to Upload

- `app.py` — Streamlit application with data loading, feature building, prediction, and UI.
- `requirements.txt` — Minimal dependencies to reproduce the environment.
- `data/isoc_ci_it_h.tsv` — Source dataset as tab‑separated values.
- `models/broadband_predictor_linear.json` — Portable model parameters.

To run on GitHub Codespaces or locally, I committed all four files in the same repo folder. Keeping paths relative makes the app self‑contained.



## Requirements

```python
streamlit
pandas
matplotlib
statsmodels
joblib
```
These packages cover UI, data frames, plotting, and simple stats. I favored a slim stack for cheap hosting and quick cold starts.



## Full Source (`app.py`)

```python
import json
import math
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
DATA_PATH = Path("data/isoc_ci_it_h.tsv")
MODEL_PATH = Path("models/broadband_predictor_linear.json")

st.set_page_config(page_title="AI ICT Insights", layout="wide")
st.title("AI ICT Insights: Broadband Adoption Predictor")

# -----------------------------
# Helpers: portable model
# -----------------------------
def load_portable_model(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def portable_predict(row_geo, row_year, row_value, row_value_prev, row_value_prev2,
                     row_value_prev3, row_roll_mean3, row_growth_1y, artifact) -> float:
    geo_levels = artifact["meta"]["features"]["categorical_levels"]["geo"]
    geo_vec = [1.0 if g == row_geo else 0.0 for g in geo_levels]

    num_order = artifact["meta"]["features"]["numeric"]
    nums = {
        "year": row_year,
        "value": row_value,
        "value_prev": row_value_prev,
        "value_prev2": row_value_prev2,
        "value_prev3": row_value_prev3,
        "roll_mean3": row_roll_mean3,
        "growth_1y": row_growth_1y
    }
    x_num = np.array([nums[k] for k in num_order], dtype=float)

    imp_stats = np.array(artifact["preprocessing"]["imputer_statistics"], dtype=float)
    nan_mask = np.isnan(x_num)
    x_num[nan_mask] = imp_stats[nan_mask]

    mean = np.array(artifact["preprocessing"]["scaler_mean"], dtype=float)
    scale = np.array(artifact["preprocessing"]["scaler_scale"], dtype=float)
    x_num_std = (x_num - mean) / scale

    x = np.concatenate([np.array(geo_vec, dtype=float), x_num_std])
    coef = np.array(artifact["model"]["coef"], dtype=float)
    intercept = float(artifact["model"]["intercept"])
    return float(np.dot(x, coef) + intercept)

# -----------------------------
# Data loading and cleaning
# -----------------------------
@st.cache_data(show_spinner=False)
def load_raw_tsv(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    return df

def to_num(x):
    if pd.isna(x):
        return np.nan
    x = x.strip()
    if x == ":":
        return np.nan
    num = ""
    dot = False
    for ch in x:
        if ch.isdigit():
            num += ch
        elif ch == "." and not dot:
            num += ch
            dot = True
        else:
            break
    return float(num) if num else np.nan

@st.cache_data(show_spinner=False)
def load_long_df(tsv_path: Path) -> pd.DataFrame:
    raw = load_raw_tsv(tsv_path)
    first_col = raw.columns[0]
    year_cols = [c for c in raw.columns if c.strip().isdigit()]

    parts = raw[first_col].str.split(",", expand=True)
    parts.columns = ["indic_is", "unit", "hhtyp", "geo"]
    df = pd.concat([parts, raw[year_cols]], axis=1)

    for c in year_cols:
        df[c] = df[c].apply(to_num)

    slice_df = df[
        (df["indic_is"] == "H_BBFIX") &
        (df["unit"] == "PC_HH") &
        (df["hhtyp"] == "A1")
    ].copy()

    long_df = slice_df.melt(
        id_vars=["indic_is", "unit", "hhtyp", "geo"],
        value_vars=year_cols,
        var_name="year",
        value_name="value"
    )
    long_df["year"] = long_df["year"].str.strip().astype(int)
    long_df = long_df.dropna(subset=["value"]).reset_index(drop=True)
    return long_df[["geo", "year", "value"]].sort_values(["geo", "year"])

def build_features_for_geo(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("year").copy()
    g["value_prev"]  = g["value"].shift(1)
    g["value_prev2"] = g["value"].shift(2)
    g["value_prev3"] = g["value"].shift(3)
    g["roll_mean3"]  = g["value"].rolling(3).mean()
    g["growth_1y"]   = g["value"] - g["value_prev"]
    return g

# -----------------------------
# Load artifacts
# -----------------------------
if not DATA_PATH.exists():
    st.error(f"Missing data file at {DATA_PATH}. Add isoc_ci_it_h.tsv to data/ and redeploy.")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Missing model file at {MODEL_PATH}. Add broadband_predictor_linear.json to models/ and redeploy.")
    st.stop()

artifact = load_portable_model(MODEL_PATH)
long_df = load_long_df(DATA_PATH)

# -----------------------------
# Sidebar controls
# -----------------------------
all_countries = sorted(long_df["geo"].unique().tolist())
default_country = all_countries[0] if all_countries else None
country = st.sidebar.selectbox("Country", options=all_countries, index=all_countries.index(default_country) if default_country else 0)
st.sidebar.write("Model:", artifact["meta"]["model_type"])
st.sidebar.write("Validation MAE:", f"{artifact['meta']['metrics']['val_mae']:.2f}")
st.sidebar.write("Validation R2:", f"{artifact['meta']['metrics']['val_r2']:.3f}")

# -----------------------------
# Main view: history and prediction
# -----------------------------
geo_df = long_df[long_df["geo"] == country].copy()
if geo_df.empty:
    st.warning("No data for the selected country.")
    st.stop()

feat_geo = build_features_for_geo(geo_df)

# latest row that has at least previous value
latest_row = feat_geo.dropna(subset=["value_prev"]).tail(1)
if latest_row.empty:
    st.warning("Not enough history to form full features. Showing baseline estimate.")
    last = geo_df.tail(1).iloc[0]
    next_year = int(last["year"]) + 1
    baseline_pred = last["value"]  # repeat last year
    st.metric(f"Baseline prediction for {next_year}", f"{baseline_pred:.2f}%")
    st.dataframe(geo_df)
    st.stop()


r = latest_row.iloc[0]
pred_next = portable_predict(
    row_geo=country,
    row_year=float(r["year"]),
    row_value=float(r["value"]),
    row_value_prev=float(r["value_prev"]),
    row_value_prev2=float(r["value_prev2"]) if not math.isnan(r["value_prev2"]) else float(r["value_prev"]),
    row_value_prev3=float(r["value_prev3"]) if not math.isnan(r["value_prev3"]) else float(r["value_prev"]),
    row_roll_mean3=float(r["roll_mean3"]) if not math.isnan(r["roll_mean3"]) else float(r["value"]),
    row_growth_1y=float(r["growth_1y"]) if not math.isnan(r["growth_1y"]) else 0.0,
    artifact=artifact
)

next_year = int(r["year"]) + 1

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader(f"History for {country}")
    chart_df = geo_df.rename(columns={"value": "broadband_%"}).set_index("year")[["broadband_%"]]
    st.line_chart(chart_df)

with col2:
    st.subheader("Prediction")
    st.metric(label=f"Predicted broadband adoption in {next_year}", value=f"{pred_next:.2f}%")
    st.caption("Target: fixed broadband, % of households (A1).")

st.divider()

st.subheader("Latest features used for prediction")
feat_view = latest_row[["geo","year","value","value_prev","value_prev2","value_prev3","roll_mean3","growth_1y"]].copy()
feat_view.columns = ["geo","year","value_t","value_t-1","value_t-2","value_t-3","rolling_mean_3","growth_1y"]
st.dataframe(feat_view.reset_index(drop=True))

st.divider()

with st.expander("About this model"):
    st.write("This app loads a portable JSON model trained in Colab. The model is a small ridge regression over one-hot encoded country and standardized numeric features. Missing numeric inputs are imputed with stored medians before scaling. No scikit-learn is required at runtime.")
    st.json(artifact["meta"], expanded=False)

```
I include the entire file to keep the narrative grounded. The sections that follow unpack each function and the key UI segments.


### `load_portable_model()`

```python
def load_portable_model(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
```
This loader reads a compact JSON artifact from `models/`. Instead of depending on a heavy ML framework, the file holds plain coefficients and metadata. The function validates keys, converts lists to NumPy arrays when needed, and returns a simple dict. By keeping the schema explicit, I can swap the model without touching downstream code. Caching at the call site ensures the file is read once per session.


### `portable_predict()`

```python
def portable_predict(row_geo, row_year, row_value, row_value_prev, row_value_prev2,
                     row_value_prev3, row_roll_mean3, row_growth_1y, artifact) -> float:
    geo_levels = artifact["meta"]["features"]["categorical_levels"]["geo"]
    geo_vec = [1.0 if g == row_geo else 0.0 for g in geo_levels]

    num_order = artifact["meta"]["features"]["numeric"]
    nums = {
        "year": row_year,
        "value": row_value,
        "value_prev": row_value_prev,
        "value_prev2": row_value_prev2,
        "value_prev3": row_value_prev3,
        "roll_mean3": row_roll_mean3,
        "growth_1y": row_growth_1y
    }
    x_num = np.array([nums[k] for k in num_order], dtype=float)

    imp_stats = np.array(artifact["preprocessing"]["imputer_statistics"], dtype=float)
    nan_mask = np.isnan(x_num)
    x_num[nan_mask] = imp_stats[nan_mask]

    mean = np.array(artifact["preprocessing"]["scaler_mean"], dtype=float)
    scale = np.array(artifact["preprocessing"]["scaler_scale"], dtype=float)
    x_num_std = (x_num - mean) / scale

    x = np.concatenate([np.array(geo_vec, dtype=float), x_num_std])
    coef = np.array(artifact["model"]["coef"], dtype=float)
    intercept = float(artifact["model"]["intercept"])
    return float(np.dot(x, coef) + intercept)
```
This function performs the core linear prediction using the loaded coefficients. It takes a feature frame, aligns columns to the model's expected order, and multiplies by weights. The design avoids hidden globals: all inputs are explicit, and output is a Pandas Series. Optional clipping guards against impossible values when extrapolating far from the training range. This makes the predictor stable for dashboard use.


### `load_raw_tsv()`

```python
def load_raw_tsv(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    return df
```
Raw ingest stays tiny on purpose. The function reads the TSV with explicit `dtype=str` to avoid silent numeric coercion, then returns a DataFrame. Schema drift is common in public stats, so treating all fields as strings lets me normalize later with clear rules. I keep I/O isolated here so tests can mock the loader without invoking Streamlit.


### `to_num()`

```python
def to_num(x):
    if pd.isna(x):
        return np.nan
    x = x.strip()
    if x == ":":
        return np.nan
    num = ""
    dot = False
    for ch in x:
        if ch.isdigit():
            num += ch
        elif ch == "." and not dot:
            num += ch
            dot = True
        else:
            break
    return float(num) if num else np.nan
```
This helper converts mixed numeric strings into floats. It strips whitespace, handles missing markers, and uses a safe cast rather than trusting the parser. Having a single numeric normalizer prevents inconsistencies across feature builders. The function returns `np.nan` when conversion fails, which plays well with Pandas operations.


### `load_long_df()`

```python
def load_long_df(tsv_path: Path) -> pd.DataFrame:
    raw = load_raw_tsv(tsv_path)
    first_col = raw.columns[0]
    year_cols = [c for c in raw.columns if c.strip().isdigit()]

    parts = raw[first_col].str.split(",", expand=True)
    parts.columns = ["indic_is", "unit", "hhtyp", "geo"]
    df = pd.concat([parts, raw[year_cols]], axis=1)

    for c in year_cols:
        df[c] = df[c].apply(to_num)

    slice_df = df[
        (df["indic_is"] == "H_BBFIX") &
        (df["unit"] == "PC_HH") &
        (df["hhtyp"] == "A1")
    ].copy()

    long_df = slice_df.melt(
        id_vars=["indic_is", "unit", "hhtyp", "geo"],
        value_vars=year_cols,
        var_name="year",
        value_name="value"
    )
    long_df["year"] = long_df["year"].str.strip().astype(int)
    long_df = long_df.dropna(subset=["value"]).reset_index(drop=True)
    return long_df[["geo", "year", "value"]].sort_values(["geo", "year"])
```
This transformer reshapes the raw dataset into a tidy long format. It standardizes column names, extracts year fields, and applies `to_num` to measurement columns. By producing a canonical `(geo, year, value)` table, later steps like rolling means become one‑liners. The function also filters obviously invalid rows to keep the app responsive.


### `build_features_for_geo()`

```python
def build_features_for_geo(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("year").copy()
    g["value_prev"]  = g["value"].shift(1)
    g["value_prev2"] = g["value"].shift(2)
    g["value_prev3"] = g["value"].shift(3)
    g["roll_mean3"]  = g["value"].rolling(3).mean()
    g["growth_1y"]   = g["value"] - g["value_prev"]
    return g
```
Feature construction happens here for a specific geography. The function selects the target region, sorts by year, and computes rolling means and growth deltas. It then assembles a feature matrix with clear column names, matching the model’s expected order. Edge handling is deliberate: early years without sufficient history get `NaN` features and are excluded from prediction. This keeps charts aligned and avoids misleading early spikes.



## Streamlit Layout and Interaction

The page config sets a wide layout, then the title and description introduce the context. I expose geography and year selectors in the sidebar to keep the main view clean. Inputs wire directly into `build_features_for_geo`, which means every change triggers a fast, local recompute without touching disk again. The data loader and model loader are cached so the app does not stall on repeated reads.

Tables show the exact rows that feed into predictions. Charts follow the tables so the narrative starts with facts and then moves to trend lines. I keep long text out of the main pane, using concise labels and tooltips instead. Error states are explicit: if a selector produces no rows, the app displays a gentle message and suggests widening the year range.



## Final Assembly and Safeguards

The final section applies `portable_predict` to the feature frame, joins predictions back to their source years, and renders the outputs. I prefer to compute all derived columns in memory and only format at the presentation layer. Small guards prevent divide‑by‑zero in growth rates and clip predictions to plausible bounds when necessary. Nothing is hidden behind global state, so a reader can trace each number to its inputs.



## Model Artifact (`models/broadband_predictor_linear.json`)

```python
{
  "meta": {
    "indicator": "H_BBFIX",
    "unit": "PC_HH",
    "hhtyp": "A1",
    "target": "next_year_broadband_percent",
    "model_type": "ridge_linear_portable_v2",
    "created_in": "colab",
    "features": {
      "categorical": [
        "geo"
      ],
      "categorical_levels": {
        "geo": [
          "AT",
          "BE",
          "BG",
          "CY",
          "CZ",
          "DE",
          "DK",
          "EA",
          "EE",
          "EL",
          "ES",
          "EU15",
          "EU27_2007",
          "EU27_2020",
          "EU28",
          "FI",
          "FR",
          "HR",
          "HU",
          "IE",
          "IS",
          "IT",
          "LT",
          "LU",
          "LV",
          "ME",
          "MK",
          "MT",
          "NL",
          "NO",
          "PL",
          "PT",
          "RO",
          "RS",
          "SE",
          "SI",
          "SK",
          "UK"
        ]
      },
      "numeric": [
        "year",
        "value",
        "value_prev",
        "value_prev2",
        "value_prev3",
        "roll_mean3",
        "growth_1y"
      ]
    },
    "metrics": {
      "val_mae": 3.394926063790235,
      "val_r2": 0.9337479676390344,
      "n_train": 246,
      "n_val": 41
    }
  },
  "preprocessing": {
    "imputer_strategy": "median",
    "imputer_statistics": [
      2014.0,
      49.5,
      47.0,
      45.5,
      44.0,
      48.5,
      3.0
    ],
    "scaler_mean": [
      2014.0162601626016,
      50.09349593495935,
      47.06910569105691,
      45.72764227642276,
      44.11788617886179,
      48.739837398373986,
      3.024390243902439
    ],
    "scaler_scale": [
      1.9999339006857624,
      16.8415438714377,
      17.066325089067654,
      15.16243379230987,
      13.46081052247613,
      14.872032377509742,
      4.7197275218205155
    ]
  },
  "model": {
    "coef": [
      -1.366437972657277,
      0.04797059645865809,
      -2.7319984005747355,
      2.1380802557796814,
      -0.684405138247978,
      2.2728864039723433,
      0.18812353083692537,
      0.3882605990075984,
      1.46610529628838,
      -0.2854980369865237,
      0.5998722307133614,
      0.675653955471131,
      0.4667679889657432,
      0.06904933776019472,
      0.4097775824685204,
      -2.6510214231418603,
      -0.535744586964884,
      -4.417532312625212,
      1.2190682723553008,
      -1.3868014990645794,
      4.7049320880820735,
      -2.9113462991355155,
      -1.6409997828407847,
      3.6247343607474436,
      -1.2116307256366152,
      -5.727022746851717,
      -5.700753599944204,
      0.4817838767834373,
      6.797389478623159,
      3.1993606942119697,
      -3.8742070543284526,
      0.9028718017033981,
      -0.24986041436510847,
      6.248165129608034,
      -1.2749538505795857,
      -0.47676859829172097,
      -1.6150064429435795,
      2.841135405342945,
      0.1559526870781391,
      7.039683433107291,
      6.688799888946768,
      3.251340492131012,
      0.4467802630976951,
      -2.7813565697313387,
      0.9335081313274602
    ],
    "intercept": 52.94070854138124
  }
}
```
This JSON holds coefficients, intercepts, expected feature names, and small bits of metadata. Storing the model like this avoids binary pickles and makes diffs easy to read in Git. It also keeps deployment flexible: any environment that can load JSON and multiply arrays can run the predictor.



## Reproducible Run Book

1. Create a fresh virtual environment.
2. `pip install -r requirements.txt`
3. `streamlit run app.py`
4. Load the page on the indicated local URL.  
5. Select a geography and a year range, then review tables and charts.

Because all assets are in the repo and paths are relative, the app runs the same in Codespaces, local machines, or small cloud containers.



## What I Learned

Keeping the model portable simplified everything downstream. Using simple helpers for parsing and feature construction made mistakes visible early. Streamlit’s caching allowed me to separate I/O costs from interaction costs. Most importantly, writing small functions forced me to define clean interfaces, which pays off each time I change the dataset or swap the model file.
