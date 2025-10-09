---
layout: default
title: "Strengthening my AI Movie Rating Predictor"
date: 2025-01-16 07:22:31
categories: [ai]
tags: [python,streamlit,openai]
thumbnail: /assets/images/movie_rating.webp
thumbnail_mobile: /assets/images/movie_rating_sq.webp
demo_link: https://rahuls-ai-movie-rating-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_movie_rating_predictor
---

I created a project that predicts a movie rating using a clean dataset and a model. The goal was clarity over flash. Every file in the repository is plain and easy to follow. The final app trains in seconds and gives a quick sense of how features push a rating up or down.

---

## What lives in the repository

```
ai_movie_rating_predictor-main/
├─ app.py
├─ requirements.txt
├─ data/
│  └─ movies.csv
└─ README.md
```

- `app.py` holds the Streamlit app. It loads the CSV, builds a scikit-learn pipeline, trains a linear model, and exposes a small predictor UI.
- `requirements.txt` lists the exact Python packages needed by Streamlit Cloud or any runner.
- `data/movies.csv` ships a tiny table with a handful of movies to keep the demo self-contained.
- `README.md` gives a quick description for the repo front page.

---

## The files I uploaded to GitHub

You only need the four files above. No notebooks are required. No hidden environment files are needed. The CSV sits inside a `data/` folder so the code can read it with a simple relative path. Pushing these files to a public repository is enough for Streamlit Cloud to build and run the app without extra steps.

---

## The dataset shipped with the repo

I bundled a minimal CSV of popular films. It includes a few basic features and a single numeric label called `rating`. The sample below shows the top rows that load in the app.

```python
title,genre,runtime,budget,votes,rating
Inception,Sci-Fi,148,160000000,2100000,8.8
Titanic,Romance,195,200000000,1200000,7.9
Avengers: Endgame,Action,181,356000000,900000,8.4
The Godfather,Crime,175,6000000,1700000,9.2
The Dark Knight,Action,152,185000000,2500000,9.0
Frozen,Animation,102,150000000,600000,7.5
Parasite,Thriller,132,11400000,800000,8.6
Interstellar,Sci-Fi,169,165000000,1700000,8.6
Joker,Drama,122,55000000,1200000,8.4
```

The app keeps the dataset tiny on purpose. Training is instant even on free hardware. You can grow the table later once the baseline is in place.

---

## Requirements for the app

The zip already provided a short `requirements.txt`. Here is the exact content that I pushed.

```python
streamlit
pandas
scikit-learn
```

That is all the app needs. Streamlit handles the UI. Pandas reads the CSV. Scikit-learn builds and trains the model. No other package is required for this version.

---

## The complete `app.py` I run in production

The original `app.py` in the zip had a placeholder where training would go. The version below is the complete file I committed so the app runs end to end. It is one page, yet the flow is clear and modular. Each helper has a focused job and returns clean values back to the main script.

```python
import streamlit as st
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="Movie Rating Predictor", layout="wide")

st.title("Movie Rating Predictor")

st.write("This simple app predicts an IMDB-style rating from a few movie features. It uses a small sample CSV shipped with the repo, trains a linear model, and lets you try your own values.")

# -------------------------------
# Data loading
# -------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str = "data/movies.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic cleaning to keep the demo robust
    df = df.dropna(subset=["rating", "runtime", "budget", "votes", "genre"])
    # Clip extremes to avoid wild values breaking a tiny demo model
    df["runtime"] = df["runtime"].clip(60, 240)
    df["budget"] = df["budget"].clip(1_000_000, 400_000_000)
    df["votes"] = df["votes"].clip(1_000, 3_000_000)
    return df

df = load_data()

st.subheader("Sample Data")
st.dataframe(df.head(), use_container_width=True)

# -------------------------------
# Train / test split
# -------------------------------
def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[["genre", "runtime", "budget", "votes"]]
    y = df["rating"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = split_data(df)

# -------------------------------
# Pipeline builder
# -------------------------------
def make_pipeline() -> Pipeline:
    numeric = ["runtime", "budget", "votes"]
    categorical = ["genre"]
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    model = LinearRegression()
    pipe = Pipeline([("pre", pre), ("model", model)])
    return pipe

# -------------------------------
# Model training
# -------------------------------
def train_model(pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipe.fit(X_train, y_train)
    return pipe

pipe = make_pipeline()
pipe = train_model(pipe, X_train, y_train)

# -------------------------------
# Evaluation
# -------------------------------
def evaluate_model(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    preds = pipe.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return preds, mse, r2

preds, mse, r2 = evaluate_model(pipe, X_test, y_test)

st.subheader("Model Performance")
col1, col2 = st.columns(2)
with col1:
    st.metric("Mean Squared Error", f"{mse:.2f}")
with col2:
    st.metric("R² Score", f"{r2:.3f}")

# -------------------------------
# Predictor UI
# -------------------------------
st.subheader("Try Your Own Inputs")

genres = sorted(df["genre"].unique().tolist())
ui_genre = st.selectbox("Genre", options=genres, index=0)
ui_runtime = st.number_input("Runtime (minutes)", min_value=60, max_value=240, value=int(df["runtime"].median()))
ui_budget = st.number_input("Budget (USD)", min_value=1_000_000, max_value=400_000_000, value=int(df["budget"].median()), step=1_000_000, format="%d")
ui_votes = st.number_input("Votes", min_value=1_000, max_value=3_000_000, value=int(df["votes"].median()), step=10_000, format="%d")

def build_input_row(genre: str, runtime: int, budget: int, votes: int) -> pd.DataFrame:
    return pd.DataFrame([{"genre": genre, "runtime": runtime, "budget": budget, "votes": votes}])

input_row = build_input_row(ui_genre, ui_runtime, ui_budget, ui_votes)
pred = float(pipe.predict(input_row)[0])

st.success(f"Predicted rating: {pred:.2f}")
```

---

## Block-by-block breakdown

### Page setup

```python
st.set_page_config(page_title="Movie Rating Predictor", layout="wide")
st.title("Movie Rating Predictor")
st.write("This simple app predicts an IMDB-style rating from a few movie features. It uses a small sample CSV shipped with the repo, trains a linear model, and lets you try your own values.")
```

This block configures the Streamlit page and sets a short title. The layout is wide so the table and metrics have room. A brief description tells users what to expect. No magic here, just a clean entry into the app.

---

### Data loading with caching

```python
@st.cache_data(show_spinner=False)
def load_data(path: str = "data/movies.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["rating", "runtime", "budget", "votes", "genre"])
    df["runtime"] = df["runtime"].clip(60, 240)
    df["budget"] = df["budget"].clip(1_000_000, 400_000_000)
    df["votes"] = df["votes"].clip(1_000, 3_000_000)
    return df
```

The `load_data` function reads the CSV and returns a DataFrame. I drop rows that miss any target or feature to avoid silent errors. Simple clipping keeps extreme values in a safe range, which helps linear models on tiny data. The `@st.cache_data` decorator avoids re-reading the same file on each widget change.

---

### Preview the top rows

```python
df = load_data()
st.subheader("Sample Data")
st.dataframe(df.head(), use_container_width=True)
```

This part pulls data and shows the first few rows. A small preview helps users trust the inputs. The `use_container_width` flag lets the table stretch and remain readable. The preview also confirms that the CSV path is correct inside the deployed app.

---

### Splitting features and label

```python
from typing import Tuple
from sklearn.model_selection import train_test_split

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[["genre", "runtime", "budget", "votes"]]
    y = df["rating"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = split_data(df)
```

`split_data` separates features from the label and performs a fixed random split. The function makes the intent explicit and keeps the top-level script tidy. Using a small test fold is enough here because the dataset is tiny. A constant seed keeps results repeatable during demos.

---

### Building a preprocessing and model pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

def make_pipeline() -> Pipeline:
    numeric = ["runtime", "budget", "votes"]
    categorical = ["genre"]
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    model = LinearRegression()
    pipe = Pipeline([("pre", pre), ("model", model)])
    return pipe
```

`make_pipeline` wires up preprocessing and the regression in a single object. Numeric features get standardized for a stable fit. The genre column is one-hot encoded and does not break if a new category appears. Returning a ready pipeline reduces boilerplate and makes training a one-liner later.

---

### Training the model

```python
def train_model(pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipe.fit(X_train, y_train)
    return pipe

pipe = make_pipeline()
pipe = train_model(pipe, X_train, y_train)
```

The `train_model` helper does one thing well. It fits the pipeline on the training split and returns the trained object. The pattern keeps main code readable and makes it trivial to swap estimators. Training is fast because the dataset is small and the model is linear.

---

### Evaluating performance

```python
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    preds = pipe.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return preds, mse, r2

preds, mse, r2 = evaluate_model(pipe, X_test, y_test)
st.subheader("Model Performance")
col1, col2 = st.columns(2)
with col1:
    st.metric("Mean Squared Error", f"{mse:.2f}")
with col2:
    st.metric("R² Score", f"{r2:.3f}")
```

`evaluate_model` computes two common metrics that are easy to read in a demo. MSE shows average squared error in rating points. R² gives a quick sense of fit despite the tiny sample. Displaying them as metrics keeps the UI neat and avoids long tables.

---

### Building a single-row input frame

```python
def build_input_row(genre: str, runtime: int, budget: int, votes: int) -> pd.DataFrame:
    return pd.DataFrame([{"genre": genre, "runtime": runtime, "budget": budget, "votes": votes}])
```

This helper creates a DataFrame with the same columns as the training set. The pipeline expects a frame, not a list, so the shape must match. Wrapping the row makes the predictor call clear and reduces repeated code. One small function helps avoid subtle ordering mistakes.

---

### The interactive predictor UI

```python
genres = sorted(df["genre"].unique().tolist())
ui_genre = st.selectbox("Genre", options=genres, index=0)
ui_runtime = st.number_input("Runtime (minutes)", min_value=60, max_value=240, value=int(df["runtime"].median()))
ui_budget = st.number_input("Budget (USD)", min_value=1_000_000, max_value=400_000_000, value=int(df["budget"].median()), step=1_000_000, format="%d")
ui_votes = st.number_input("Votes", min_value=1_000, max_value=3_000_000, value=int(df["votes"].median()), step=10_000, format="%d")

input_row = build_input_row(ui_genre, ui_runtime, ui_budget, ui_votes)
pred = float(pipe.predict(input_row)[0])
st.success(f"Predicted rating: {pred:.2f}")
```

The app reflects the training schema in the widget design. It uses sensible limits and defaults taken from the dataset. A clean row builder feeds the pipeline and returns a float. The prediction appears as a single rounded value so the interface stays calm.

---

## Why these helpers exist

- **`load_data`** keeps I/O in one place and lets caching skip repeated reads. It also applies small guards like clipping for demo stability.
- **`split_data`** returns four clear values and avoids cross-talk between global variables.
- **`make_pipeline`** expresses the full data path from raw columns to model-ready vectors. One object captures both preprocessing and estimation.
- **`train_model`** and **`evaluate_model`** separate fit from scoring. This makes testing easier and keeps a short main script.
- **`build_input_row`** protects column order and dtypes when a user tries the predictor.

Each function is short, named well, and easy to test in isolation. Swapping the model, adding features, or changing encoders becomes a local change.

---

## How to run it locally (optional)

You can run the app with three commands. Create a virtual environment, install requirements, and start Streamlit.

```python
python -m venv .venv
source .venv/bin/activate  # on Windows use: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

This path is not required for Streamlit Cloud, but it is helpful during quick checks. The same repo works in both places.

---

## Deploying from GitHub to Streamlit Cloud

1. Push the repo to GitHub with the four files shown earlier.
2. Go to Streamlit Cloud and pick the repository.
3. Keep the default entry point `app.py` and Python version.
4. The build reads `requirements.txt` and installs three packages.
5. The app boots and reads `data/movies.csv` at start.

Streamlit caches the CSV and recompiles widgets as you move sliders. The model retrains on each start or code edit. That is fine for a tiny demo and keeps behavior simple.

---

## Ideas to extend this demo

- Replace `LinearRegression` with `Ridge` or `RandomForestRegressor` and compare metrics.
- Add `year` or `director_popularity` as features if you expand the CSV.
- Persist the trained pipeline to disk and reuse it across sessions.
- Add a residuals plot and a feature importance proxy using coefficients.

These are small, safe changes that keep the app readable. The current structure makes each change a local edit.

---

## Appendix — The original files from the zip

For context, here are the original `requirements.txt` and a preview of the CSV that shipped with the repo. The working app above uses the same packages and the same file layout.

```python
streamlit
pandas
scikit-learn
```

```python
title,genre,runtime,budget,votes,rating
Inception,Sci-Fi,148,160000000,2100000,8.8
Titanic,Romance,195,200000000,1200000,7.9
Avengers: Endgame,Action,181,356000000,900000,8.4
The Godfather,Crime,175,6000000,1700000,9.2
The Dark Knight,Action,152,185000000,2500000,9.0
Frozen,Animation,102,150000000,600000,7.5
Parasite,Thriller,132,11400000,800000,8.6
Interstellar,Sci-Fi,169,165000000,1700000,8.6
Joker,Drama,122,55000000,1200000,8.4
```

That is everything needed to run and deploy the project. The app stays in one file. The dataset is bundled. The environment is minimal. The post above walks through each block so we can adapt it quickly.
