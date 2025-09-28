---
layout: default
title: "Creating my AI Election Forecasting App"
date: 2023-03-25 08:57:31
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-election-forecast.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_election_forecast
featured: true
---

I once tracked election polls daily and wanted a forecast that revealed every step rather than a black box. That curiosity turned into a project that reads public polling data, engineers time‑series features, and predicts weekly margins. I chose Streamlit because I could share a transparent, reproducible app that runs directly from a GitHub repository. The goal is simple: show how the national generic ballot margin moves week by week, and document the full pipeline that makes it possible. Dataset used [here](https://www.kaggle.com/datasets/iamtanmayshukla/2024-u-s-election-generic-ballot-polling-data).

While exploring datasets, I found that careful feature design matters as much as the model. Lagged margins capture momentum, rolling averages tame noise, and poll quality indicators help weigh stronger surveys. This blog explains every file I uploaded, every helper function I wrote, and how the parts fit together into a working forecast.

---
## Repository Layout and Dependencies
I uploaded four key elements to GitHub: the Streamlit app (`app.py`), a `data/` folder with the polling CSV, a `models/` folder with the trained estimator and metadata, and `requirements.txt` to pin versions.
```python
streamlit==1.36.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
joblib==1.4.2
matplotlib==3.8.4

```
These dependencies are the exact versions I tested. Streamlit renders the interface, pandas/numpy handle data, scikit‑learn powers the regressor, joblib loads the artifact, and matplotlib draws the time‑series charts.

## Data Snapshot
The app expects `data/generic_ballot_polls.csv`. Here is a small preview:
```python
   poll_id  pollster_id  pollster sponsor_ids   sponsors                           display_name  pollster_rating_id                   pollster_rating_name  numeric_grade  pollscore                   methodology  transparency_score  state start_date end_date  sponsor_candidate_id  sponsor_candidate  sponsor_candidate_party  question_id  sample_size population subpopulation population_full tracking     created_at notes                                                                                                                 url  source internal partisan  race_id  cycle office_type  seat_number seat_name election_date    stage  nationwide_batch   dem   rep  ind
0    87781         1102   Emerson         NaN        NaN                        Emerson College                  88                        Emerson College            2.9       -1.1  IVR/Online Panel/Text-to-Web                 7.0    NaN    8/12/24  8/14/24                   NaN                NaN                      NaN       206005       1000.0         lv           NaN              lv      NaN  8/15/24 09:30   NaN                                     https://emersoncollegepolling.com/august-2024-national-poll-harris-50-trump-46/     NaN      NaN      NaN     9549   2024  U.S. House          NaN   Generic       11/5/24  general             False  47.5  45.5  NaN
1    87760          568    YouGov         352  Economist                                 YouGov                 391                                 YouGov            2.9       -1.1                  Online Panel                 9.0    NaN    8/11/24  8/13/24                   NaN                NaN                      NaN       205807       1407.0         rv           NaN              rv      NaN  8/14/24 10:02   NaN                                            https://d3nkl3psvxxpe9.cloudfront.net/documents/econtoplines_W3lebBm.pdf     NaN      NaN      NaN     9549   2024  U.S. House          NaN   Generic       11/5/24  general             False  45.0  44.0  NaN
2    87774          320  Monmouth         NaN        NaN  Monmouth University Polling Institute                 215  Monmouth University Polling Institute            2.9       -0.9        Live Phone/Text-to-Web                 9.0    NaN     8/8/24  8/12/24                   NaN                NaN                      NaN       205881        801.0         rv           NaN              rv      NaN  8/14/24 11:15   NaN                                          https://www.monmouth.edu/polling-institute/reports/MonmouthPoll_US_081424/     NaN      NaN      NaN     9549   2024  U.S. House          NaN   Generic       11/5/24  general             False  48.0  46.0  NaN
3    87791         1347    Cygnal         NaN        NaN                                 Cygnal                  67                                 Cygnal            2.1       -1.3                           NaN                 4.0    NaN     8/6/24   8/8/24                   NaN                NaN                      NaN       206110       1500.0         lv           NaN              lv      NaN  8/15/24 20:42   NaN  https://www.cygn.al/national-poll-2024-election-has-re-entered-stasis-while-trump-republicans-maintains-advantage/     NaN      NaN      NaN     9549   2024  U.S. House          NaN   Generic       11/5/24  general             False  46.4  47.1  NaN
4    87696          568    YouGov         352  Economist                                 YouGov                 391                                 YouGov            2.9       -1.1                  Online Panel                 9.0    NaN     8/4/24   8/6/24                   NaN                NaN                      NaN       205394       1413.0         rv           NaN              rv      NaN   8/7/24 10:45   NaN                                            https://d3nkl3psvxxpe9.cloudfront.net/documents/econtoplines_OHhDhBP.pdf     NaN      NaN      NaN     9549   2024  U.S. House          NaN   Generic       11/5/24  general             False  45.0  44.0  NaN
```
The model focuses on Democratic and Republican support to compute a margin, while also using poll quality fields and sample sizes as predictors. Rows are later grouped by calendar week for stable comparison.

## Training Metadata
I store training details in `models/train_metadata.json` so inference knows what to compute:
```python
{
  "feature_cols": [
    "polls_per_week",
    "avg_sample",
    "avg_grade",
    "avg_transparency",
    "avg_pollscore",
    "margin_lag1",
    "margin_lag2",
    "margin_lag3",
    "margin_lag4",
    "margin_lag5",
    "margin_lag6",
    "margin_lag7",
    "margin_lag8",
    "margin_rollmean_3",
    "margin_rollmean_5",
    "margin_rollmean_8"
  ],
  "train_weeks": 36,
  "test_weeks": 10,
  "last_train_date": "2024-06-03",
  "mae": 1.1721256437772012,
  "r2": -0.6977368496462497
}
```
The `feature_cols` list documents the expected lags and rolling windows. The file also records the training/test split and error metrics so I can compare future retrains.

## Application Script (`app.py`) — Merged Blocks with Explanations
### imports
```python
import os
import json
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

# -----------------------------
# App config
# -----------------------------
```
I import core libraries used across the app. `os` and `json` handle paths and configuration files. `joblib` loads the trained scikit‑learn estimator from disk. `numpy` and `pandas` power numeric work and data frames. `matplotlib.pyplot` lets me draw figures that Streamlit can display. `streamlit` is the UI engine. `datetime` utilities help me align polls by calendar week for stable comparisons.

### page_config
```python
st.set_page_config(page_title="Generic Ballot: Weekly Margin Forecast", layout="wide")
```
I configure the Streamlit page early. The wide layout gives charts enough space, and the title sets context. This helps the dashboard feel like a single‑purpose tool rather than a generic demo.

### title_header
```python
st.title("Generic Ballot: Weekly Dem–Rep Margin Forecast")

# -----------------------------
# Helper functions (match training logic)
# -----------------------------
DATE_COLS_CANDIDATES = ["end_date", "start_date"]
```
I add a visible title at the top of the app so users understand that the focus is a national generic ballot forecast. Establishing this heading makes the following controls and charts easier to digest.

### def:_to_datetime_safe
```python
def _to_datetime_safe(series):
    return pd.to_datetime(series, errors="coerce")
```
This helper converts a `pandas` series to timezone‑naive datetimes and coerces bad strings to `NaT`. Polling datasets sometimes include inconsistent date formats or empty cells; coercion prevents hard crashes. By returning a clean datetime series, I can safely group rows by ISO calendar week later in the pipeline.

### def:load_training_metadata
```python
def load_training_metadata(meta_path="models/train_metadata.json"):
    if not os.path.exists(meta_path):
        st.error("Missing models/train_metadata.json. Upload it to your repo's models/ folder.")
        st.stop()
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return meta
```
This loader reads `models/train_metadata.json` to retrieve the exact feature columns, training split sizes, and evaluation metrics. If the file is missing, the function warns through Streamlit so users know the model artifacts were not uploaded. Returning a dictionary keeps downstream code explicit about which engineered columns must be built for inference.

### def:load_model
```python
def load_model(model_path="models/national_margin_forecaster.joblib"):
    if not os.path.exists(model_path):
        st.error("Missing models/national_margin_forecaster.joblib. Upload it to your repo's models/ folder.")
        st.stop()
    return joblib.load(model_path)
```
This function loads `models/national_margin_forecaster.joblib`. The artifact contains a scikit‑learn regressor fitted on weekly features. By separating storage from code, I avoid retraining at runtime and keep app startup fast. The function raises a friendly error if the model file is not present, guiding setup.

### def:load_poll_csv
```python
def load_poll_csv():
    """
    Tries to read data/generic_ballot_polls.csv from the repo.
    If not present, lets the user upload the CSV at runtime.
    """
    default_path = "data/generic_ballot_polls.csv"
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        st.info("Loaded bundled data from data/generic_ballot_polls.csv")
        return df

    uploaded = st.file_uploader("Upload generic_ballot_polls.csv", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success("CSV uploaded.")
        return df

    st.warning("Please upload generic_ballot_polls.csv, or add it to data/ in the repo.")
    st.stop()
```
This function reads `data/generic_ballot_polls.csv` with `pandas.read_csv`. It selects the columns I rely on most, including pollster ratings, sample size, cycle, and Democratic/Republican support. Centralizing the load step makes it easy to swap in an updated CSV while keeping the rest of the code stable.

### def:basic_clean_and_weekly
```python
def basic_clean_and_weekly(df):
    """
    Match the training preprocessing:
      - parse dates
      - margin = dem - rep
      - weekly national average across pollsters
    """
    # Parse date columns if present
    for col in ["start_date", "end_date", "election_date"]:
        if col in df.columns:
            df[col] = _to_datetime_safe(df[col])

    # Numeric coercions used in training (safe)
    for col in ["dem", "rep", "ind", "numeric_grade", "pollscore", "transparency_score", "sample_size"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Margin
    if not {"dem", "rep"}.issubset(df.columns):
        st.error("CSV must include 'dem' and 'rep' columns.")
        st.stop()
    df["margin"] = df["dem"] - df["rep"]

    # Pick date column preference like training
    date_col = None
    if "end_date" in df.columns:
        date_col = "end_date"
    elif "start_date" in df.columns:
        date_col = "start_date"
    else:
        st.error("CSV must have end_date or start_date column.")
        st.stop()

    df = df.dropna(subset=[date_col, "margin"]).copy()

    # Weekly alignment (week start for stability)
    df["date"] = df[date_col].dt.to_period("W").dt.start_time

    # Aggregate national weekly averages
    weekly = (
        df.groupby("date", as_index=False)
          .agg(
              margin=("margin", "mean"),
              polls_per_week=("margin", "size"),
              avg_sample=("sample_size", "mean") if "sample_size" in df.columns else ("margin", "size"),
              avg_grade=("numeric_grade", "mean") if "numeric_grade" in df.columns else ("margin", "size"),
              avg_transparency=("transparency_score", "mean") if "transparency_score" in df.columns else ("margin", "size"),
              avg_pollscore=("pollscore", "mean") if "pollscore" in df.columns else ("margin", "size"),
          )
          .sort_values("date")
          .reset_index(drop=True)
    )

    # If helper cols are entirely NaN, set to 0.0 (like training)
    for c in ["avg_sample", "avg_grade", "avg_transparency", "avg_pollscore"]:
        if c not in weekly.columns:
            weekly[c] = 0.0
        elif weekly[c].isna().all():
            weekly[c] = 0.0

    return weekly
```
Here I standardize the raw poll rows and aggregate them into weekly observations. The function computes the weekly Democratic minus Republican margin and derives per‑week quality summaries like average grade and pollscore. It also counts polls per week and stores average sample size, which later act as predictors and diagnostics.

### def:make_lags
```python
def make_lags(df_ts, target_col="margin", lags=8, roll_windows=(3, 5, 8)):
    out = df_ts.copy()
    for L in range(1, lags + 1):
        out[f"{target_col}_lag{L}"] = out[target_col].shift(L)
    for w in roll_windows:
        out[f"{target_col}_rollmean_{w}"] = out[target_col].rolling(w).mean().shift(1)
    return out
```
This routine adds autoregressive structure by creating lagged margins and rolling means (3, 5, and 8 weeks). Lags capture momentum, while rolling windows smooth noise from volatile weekly samples. These engineered time‑series features match the `feature_cols` listed in metadata and are crucial for stable predictions.

### def:infer_lags_and_windows_from_features
```python
def infer_lags_and_windows_from_features(feature_cols):
    """
    Reads back the lags and rolling windows that were used during training
    from metadata['feature_cols'].
    """
    lags = set()
    rolls = set()
    for c in feature_cols:
        if c.startswith("margin_lag"):
            try:
                lags.add(int(c.replace("margin_lag", "")))
            except:
                pass
        if c.startswith("margin_rollmean_"):
            try:
                rolls.add(int(c.replace("margin_rollmean_", "")))
            except:
                pass
    lags = sorted(list(lags)) if lags else list(range(1, 9))
    rolls = sorted(list(rolls)) if rolls else [3, 5, 8]
    return lags, rolls
```
Given the `feature_cols` from metadata, this helper infers which lags and rolling windows the model expects. It parses column names like `margin_lag3` or `margin_rollmean_5` and returns the numeric offsets. This allows me to compute exactly the required features even if I later tweak the model’s specification.

### def:recursive_forecast
```python
def recursive_forecast(last_history_df, model, horizon_weeks, feature_cols):
    """
    last_history_df: weekly df with columns: date, margin, polls_per_week, avg_* ...
    We build lag/rolling features each step using the growing history (actual + predicted).
    Returns a dataframe with future dates and predicted margins.
    """
    # Determine lags and rolling windows from training features
    lags, rolls = infer_lags_and_windows_from_features(feature_cols)
    max_lag = max(lags) if lags else 8
    rolls = tuple(rolls) if rolls else (3, 5, 8)

    # Work on a copy
    hist = last_history_df.copy().reset_index(drop=True)

    # Ensure we have enough history to create features
    if hist.shape[0] < max(max_lag, max(rolls)):
        raise ValueError(f"Not enough weekly history to create features. Need at least {max(max_lag, max(rolls))} weeks, have {hist.shape[0]}.")

    # Start forecasting one week at a time
    preds = []
    last_date = hist["date"].max()
    for step in range(1, horizon_weeks + 1):
        next_date = last_date + pd.Timedelta(days=7)

        # Build a temporary series with all rows so far
        tmp = pd.concat([hist], ignore_index=True)
        tmp = make_lags(tmp, target_col="margin", lags=max_lag, roll_windows=rolls)

        # Feature row is the last row (after adding placeholder for next week)
        # Create a placeholder row for next_date by copying aux features from latest row
        latest_aux = tmp.iloc[-1][["polls_per_week", "avg_sample", "avg_grade", "avg_transparency", "avg_pollscore"]].to_dict()
        next_row = {
            "date": next_date,
            "margin": np.nan,  # unknown yet
            "polls_per_week": latest_aux.get("polls_per_week", 0.0),
            "avg_sample": latest_aux.get("avg_sample", 0.0),
            "avg_grade": latest_aux.get("avg_grade", 0.0),
            "avg_transparency": latest_aux.get("avg_transparency", 0.0),
            "avg_pollscore": latest_aux.get("avg_pollscore", 0.0),
        }
        tmp = pd.concat([tmp, pd.DataFrame([next_row])], ignore_index=True)

        # Recompute lags/rolls including the appended row
        tmp = make_lags(tmp, target_col="margin", lags=max_lag, roll_windows=rolls)

        # Select features for the last row (new week)
        X_row = tmp.iloc[[-1]][feature_cols].copy()

        # If any lag/roll features are still NaN (very early horizon), fill with last known values
        X_row = X_row.fillna(method="ffill", axis=1).fillna(0.0)

        # Predict next margin
        yhat = float(model.predict(X_row)[0])

        preds.append({"date": next_date, "pred_margin": yhat})

        # Append prediction into history so next step can use it
        hist = pd.concat(
            [hist, pd.DataFrame([{
                "date": next_date,
                "margin": yhat,
                "polls_per_week": next_row["polls_per_week"],
                "avg_sample": next_row["avg_sample"],
                "avg_grade": next_row["avg_grade"],
                "avg_transparency": next_row["avg_transparency"],
                "avg_pollscore": next_row["avg_pollscore"],
            }])],
            ignore_index=True
        )
        last_date = next_date

    return pd.DataFrame(preds)

# -----------------------------
# Load artifacts
# -----------------------------
meta = load_training_metadata("models/train_metadata.json")
feature_cols = meta.get("feature_cols", [])
model = load_model("models/national_margin_forecaster.joblib")

# -----------------------------
# Data input
# -----------------------------
df_raw = load_poll_csv()
weekly = basic_clean_and_weekly(df_raw)

if weekly.empty:
    st.error("No weekly data after cleaning. Check your CSV.")
    st.stop()

# Show a quick peek
with st.expander("Preview weekly aggregated data"):
    st.dataframe(weekly.tail(10))

# -----------------------------
# Forecast horizon & run button
# -----------------------------
col1, col2, col3 = st.columns([1,1,2])
with col1:
    horizon = st.slider("Forecast horizon (weeks ahead)", min_value=1, max_value=24, value=8, step=1)

with col2:
    run_forecast = st.button("Run Forecast")

# -----------------------------
# Compute lags for plotting latest fit vs actual (optional)
# -----------------------------
# (Not used for prediction directly; model already trained. We just plot recent history.)
# This also validates that our feature columns exist in the engineered frame.
lags, rolls = infer_lags_and_windows_from_features(feature_cols)
max_lag_needed = max(lags) if lags else 8
weekly_lagged = make_lags(weekly, target_col="margin", lags=max_lag_needed, roll_windows=tuple(rolls))
engineered_cols_ok = all([(c in weekly_lagged.columns) for c in feature_cols])

if not engineered_cols_ok:
    st.warning("Some training features are missing in the current engineered data. Forecast may be limited. Proceeding with available history.")

# -----------------------------
# Forecast and visualize
# -----------------------------
if run_forecast:
    try:
        preds_df = recursive_forecast(weekly, model, horizon, feature_cols)

        # Merge recent actuals and forecast for plotting
        recent_actual = weekly[["date", "margin"]].tail(40).copy()
        recent_actual = recent_actual.rename(columns={"margin": "Actual margin"})

        plot_df = pd.merge(
            recent_actual,
            preds_df.rename(columns={"pred_margin": "Forecast margin"}),
            on="date",
            how="outer"
        ).sort_values("date")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(plot_df["date"], plot_df["Actual margin"], label="Actual margin")
        ax.plot(plot_df["date"], plot_df["Forecast margin"], label="Forecast margin")
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title("Weekly Dem–Rep Margin: Actual vs Forecast")
        ax.set_xlabel("Week")
        ax.set_ylabel("Margin (Dem - Rep)")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Show forecast table
        st.subheader("Forecast Table")
        show_df = preds_df.copy()
        show_df["date"] = show_df["date"].dt.date
        show_df["pred_margin"] = show_df["pred_margin"].round(3)
        st.dataframe(show_df)

        # Simple interpretation
        last_row = show_df.tail(1).iloc[0]
        sign = "Democratic lead" if last_row["pred_margin"] > 0 else ("Republican lead" if last_row["pred_margin"] < 0 else "Tie")
        st.info(f"On {last_row['date']}, model forecasts margin {last_row['pred_margin']} → {sign}.")

    except Exception as e:
        st.error(f"Forecast failed: {e}")

# -----------------------------
# Notes panel
# -----------------------------
with st.expander("How this app works (summary)"):
    st.markdown(
        """
- Uses the same weekly aggregation and lag/rolling features as your training notebook.
- The model is a tiny Ridge regressor (scaled), so the model file is very small and under 25 MB.
- Forecasts are computed recursively, each week using lags of recent actual/predicted margins.
- To keep it reproducible, keep your `generic_ballot_polls.csv` in `data/` or upload it.
        """
    )
```
This function produces a forward path of weekly predictions. It builds features from the most recent observed weeks, then steps forward recursively, feeding predictions back as lag inputs. The loop respects the lag/window structure so that each new week uses realistic historical context.

## Final Notes
To run locally, install the requirements and start Streamlit with `streamlit run app.py`. The repository structure keeps data, model artifacts, and code separate but synchronized. Because every feature and assumption is visible, the forecast is easy to audit and extend.
