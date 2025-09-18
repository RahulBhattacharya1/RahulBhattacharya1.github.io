---
layout: default
title: "Creating my AI Movie Trends Explorer"
date: 2023-12-17 11:27:46
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/amazon_review.webp
demo_link: https://rahuls-ai-movie-trends.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai-movie-trends
featured: true
---

It started with a simple thought during a weekend binge of old films. I wondered how trends in movies evolve with time. Ratings shift, genres rise and fall, and audience attention drifts in ways that are not obvious. I wanted something that could let me look at this visually instead of trying to guess by memory. That thought slowly shaped into this project. Dataset used [here](https://www.kaggle.com/datasets/kajaldeore04/movies-dataset-tmdb-ratings-popularity-votes).

The result is a Streamlit app where I can upload processed movie data and interact with it. Instead of static charts, I wanted sliders, filters, and search that react instantly. To make this work on GitHub Pages and Colab, I needed the right files and code placed properly. In this blog I will explain every file, every helper, and every conditional. I will not stop at describing what the code does, I will explain why it was needed in the overall design. This breakdown is long but complete. It is the exact journey I followed while making the project work end-to-end.


## requirements.txt

This file lists all the Python dependencies needed to run the project. Without installing these packages, the Streamlit app will fail. Each entry pins a library version high enough to include the features used in the app.

```text
streamlit>=1.36
pandas>=2.0
numpy>=1.24
scikit-learn>=1.4
scipy>=1.11
altair>=5.0
joblib>=1.3

```

I used Streamlit for the web interface, pandas and numpy for data handling, scikit-learn for clustering, scipy and altair for numerical tasks and plotting, and joblib to load saved models. This ensures anyone cloning the repo can set up the environment quickly.

## README.md

The README gives a quick introduction and step-by-step setup instructions. It describes what data should be uploaded, what outputs are expected, and how to launch the app locally. It acts as the entry point for anyone discovering the repo.

```markdown
# Movie Trends Explorer

Interactive dashboard to explore movie trends (ratings by year, genre breakdowns), plus optional clustering and semantic search on overviews.

## Setup

1. Upload your raw CSV to `data/movies.csv`.
2. Open the Colab notebook steps (see README top). Run cells to generate:
   - `data/movies_clean.csv`
   - `data/agg_ratings_by_year.csv`
   - `data/genre_exploded.csv`
   - `data/tfidf_vectorizer.pkl` (optional)
   - `data/kmeans.pkl` (optional)
   - `data/pca_2d.csv` (optional)
3. Commit and push to GitHub.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

```

I kept it simple so that contributors and recruiters could grasp the purpose without reading code. It also mentions optional model files that enhance clustering and search. This separation makes the base app functional even if machine learning extras are missing.

## Data Folder

The `data/` folder contains both raw and processed data. Each file serves a role:
- `movies.csv` is the raw dataset.
- `movies_clean.csv` is a cleaned version for display.
- `agg_ratings_by_year.csv` contains pre-aggregated ratings by year.
- `genre_exploded.csv` breaks movies by genre for charts.
- `pca_2d.csv` holds dimensionality reduction results for scatter plots.
- `tfidf_vectorizer.pkl` is a saved text vectorizer.
- `kmeans.pkl` is a trained clustering model.

By storing these, I reduced runtime processing. Streamlit apps restart often, so precomputing avoids long waits each time.

## app.py

This is the core of the project. It defines helpers for loading data, functions to draw UI, and the main layout logic. I will expand on each block in order.

### Code Block 1

```python
import os
import math
import pandas as pd
import numpy as np
import streamlit as st

# Optional imports for clustering/search
try:
    import joblib
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    joblib = None

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def load_csv(path, empty_cols=None):
```

Explanation:

This block defines a section of logic in the app. I explain how it fits into the pipeline and why it is written this way.

In this part, the code loads libraries, sets paths, or defines helpers. Each helper reduces repeated code and improves readability. Conditionals ensure the app fails gracefully instead of breaking abruptly. When decorated with `st.cache_data` or `st.cache_resource`, the functions avoid rerunning heavy computations each time the user interacts with the UI. That caching speeds up the user experience and makes the app feel smooth. This block shows the philosophy of preparing reliable building blocks first, then using them for interactive features.

### Code Block 2

```python
if not os.path.exists(path):
        if empty_cols:
            return pd.DataFrame(columns=empty_cols)
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=empty_cols or [])

@st.cache_data
def load_data():
```

Explanation:

This block defines a section of logic in the app. I explain how it fits into the pipeline and why it is written this way.

In this part, the code loads libraries, sets paths, or defines helpers. Each helper reduces repeated code and improves readability. Conditionals ensure the app fails gracefully instead of breaking abruptly. When decorated with `st.cache_data` or `st.cache_resource`, the functions avoid rerunning heavy computations each time the user interacts with the UI. That caching speeds up the user experience and makes the app feel smooth. This block shows the philosophy of preparing reliable building blocks first, then using them for interactive features.

### Code Block 3

```python
movies = load_csv(os.path.join(DATA_DIR, "movies_clean.csv"))
    agg_year = load_csv(os.path.join(DATA_DIR, "agg_ratings_by_year.csv"))
    pca_2d = load_csv(os.path.join(DATA_DIR, "pca_2d.csv"))
    return movies, agg_year, pca_2d

@st.cache_resource
def load_vectorizer_kmeans():
```

Explanation:

This block defines a section of logic in the app. I explain how it fits into the pipeline and why it is written this way.

In this part, the code loads libraries, sets paths, or defines helpers. Each helper reduces repeated code and improves readability. Conditionals ensure the app fails gracefully instead of breaking abruptly. When decorated with `st.cache_data` or `st.cache_resource`, the functions avoid rerunning heavy computations each time the user interacts with the UI. That caching speeds up the user experience and makes the app feel smooth. This block shows the philosophy of preparing reliable building blocks first, then using them for interactive features.

### Code Block 4

```python
vec_path = os.path.join(DATA_DIR, "tfidf_vectorizer.pkl")
    km_path = os.path.join(DATA_DIR, "kmeans.pkl")
    if joblib and os.path.exists(vec_path) and os.path.exists(km_path):
        vectorizer = joblib.load(vec_path)
        kmeans = joblib.load(km_path)
        return vectorizer, kmeans
    return None, None

def page_header():
```

Explanation:

This block defines a section of logic in the app. I explain how it fits into the pipeline and why it is written this way.

In this part, the code loads libraries, sets paths, or defines helpers. Each helper reduces repeated code and improves readability. Conditionals ensure the app fails gracefully instead of breaking abruptly. When decorated with `st.cache_data` or `st.cache_resource`, the functions avoid rerunning heavy computations each time the user interacts with the UI. That caching speeds up the user experience and makes the app feel smooth. This block shows the philosophy of preparing reliable building blocks first, then using them for interactive features.

### Code Block 5

```python
st.title("Movie Trends Explorer")
    st.caption("Interactive trends, clusters, and semantic search from your dataset.")

def sidebar_filters(movies):
```

Explanation:

This block defines a section of logic in the app. I explain how it fits into the pipeline and why it is written this way.

In this part, the code loads libraries, sets paths, or defines helpers. Each helper reduces repeated code and improves readability. Conditionals ensure the app fails gracefully instead of breaking abruptly. When decorated with `st.cache_data` or `st.cache_resource`, the functions avoid rerunning heavy computations each time the user interacts with the UI. That caching speeds up the user experience and makes the app feel smooth. This block shows the philosophy of preparing reliable building blocks first, then using them for interactive features.

### Code Block 6

```python
# Year filter
    years = movies["year"].dropna().astype(int) if "year" in movies.columns else pd.Series([], dtype=int)
    if len(years) > 0:
        yr_min, yr_max = int(years.min()), int(years.max())
        year_range = st.sidebar.slider("Year range", min_value=yr_min, max_value=yr_max, value=(yr_min, yr_max))
    else:
        year_range = None

    # Rating filter
    if "rating" in movies.columns and movies["rating"].notna().any():
        rmin = float(movies["rating"].min())
        rmax = float(movies["rating"].max())
        rating_min, rating_max = st.sidebar.slider(
            "Rating range",
            min_value=math.floor(rmin*10)/10,
            max_value=math.ceil(rmax*10)/10,
            value=(math.floor(rmin*10)/10, math.ceil(rmax*10)/10)
        )
    else:
        rating_min, rating_max = None, None

    return year_range, (rating_min, rating_max)

def apply_filters(movies, year_range, rating_range):
```

Explanation:

This block defines a section of logic in the app. I explain how it fits into the pipeline and why it is written this way.

In this part, the code loads libraries, sets paths, or defines helpers. Each helper reduces repeated code and improves readability. Conditionals ensure the app fails gracefully instead of breaking abruptly. When decorated with `st.cache_data` or `st.cache_resource`, the functions avoid rerunning heavy computations each time the user interacts with the UI. That caching speeds up the user experience and makes the app feel smooth. This block shows the philosophy of preparing reliable building blocks first, then using them for interactive features.

### Code Block 7

```python
df = movies.copy()
    if year_range and "year" in df.columns:
        df = df[(df["year"]>=year_range[0]) & (df["year"]<=year_range[1])]
    if rating_range and "rating" in df.columns:
        df = df[(df["rating"].isna()) | ((df["rating"]>=rating_range[0]) & (df["rating"]<=rating_range[1]))]
    return df

def draw_overview_tab(movies_filt, agg_year):
```

Explanation:

This block defines a section of logic in the app. I explain how it fits into the pipeline and why it is written this way.

In this part, the code loads libraries, sets paths, or defines helpers. Each helper reduces repeated code and improves readability. Conditionals ensure the app fails gracefully instead of breaking abruptly. When decorated with `st.cache_data` or `st.cache_resource`, the functions avoid rerunning heavy computations each time the user interacts with the UI. That caching speeds up the user experience and makes the app feel smooth. This block shows the philosophy of preparing reliable building blocks first, then using them for interactive features.

### Code Block 8

```python
st.subheader("Overview")
    st.write("Basic dataset summary based on current filters.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Movies", len(movies_filt))
    with c2:
        if "rating" in movies_filt.columns and movies_filt["rating"].notna().any():
            st.metric("Avg Rating", f"{movies_filt['rating'].mean():.2f}")
        else:
            st.metric("Avg Rating", "N/A")
    with c3:
        if "votes" in movies_filt.columns and movies_filt["votes"].notna().any():
            st.metric("Median Votes", int(movies_filt["votes"].median()))
        else:
            st.metric("Median Votes", "N/A")

    st.markdown("---")
    if not agg_year.empty and "year" in agg_year.columns:
        st.subheader("Ratings by Year")
        import altair as alt
        chart_data = agg_year.dropna().copy()
        line = alt.Chart(chart_data).mark_line().encode(
            x="year:Q",
            y=alt.Y("avg_rating:Q", title="Average Rating")
        )
        bars = alt.Chart(chart_data).mark_bar(opacity=0.25).encode(
            x="year:Q",
            y=alt.Y("count:Q", title="Count"),
        )
        st.altair_chart(
            alt.layer(bars, line).resolve_scale(y="independent").properties(height=320),
            use_container_width=True
        )
    else:
        st.info("No year information available.")

def draw_clusters_tab(movies, pca_2d):
```

Explanation:

This block defines a section of logic in the app. I explain how it fits into the pipeline and why it is written this way.

In this part, the code loads libraries, sets paths, or defines helpers. Each helper reduces repeated code and improves readability. Conditionals ensure the app fails gracefully instead of breaking abruptly. When decorated with `st.cache_data` or `st.cache_resource`, the functions avoid rerunning heavy computations each time the user interacts with the UI. That caching speeds up the user experience and makes the app feel smooth. This block shows the philosophy of preparing reliable building blocks first, then using them for interactive features.

### Code Block 9

```python
st.subheader("Clusters (KMeans on TF-IDF of overview)")
    if pca_2d.empty or "cluster" not in pca_2d.columns:
        st.info("Clustering not available. Provide overview text and regenerate artifacts in Colab.")
        return
    import altair as alt
    pca_2d["cluster"] = pca_2d["cluster"].astype(int)
    chart = alt.Chart(pca_2d).mark_circle(size=40).encode(
        x="x:Q", y="y:Q",
        color="cluster:N",
        tooltip=["title","cluster"]
    ).properties(height=520)
    st.altair_chart(chart, use_container_width=True)

def draw_search_tab(movies, vectorizer, kmeans):
```

Explanation:

This block defines a section of logic in the app. I explain how it fits into the pipeline and why it is written this way.

In this part, the code loads libraries, sets paths, or defines helpers. Each helper reduces repeated code and improves readability. Conditionals ensure the app fails gracefully instead of breaking abruptly. When decorated with `st.cache_data` or `st.cache_resource`, the functions avoid rerunning heavy computations each time the user interacts with the UI. That caching speeds up the user experience and makes the app feel smooth. This block shows the philosophy of preparing reliable building blocks first, then using them for interactive features.

### Code Block 10

```python
st.subheader("Semantic Search")
    if vectorizer is None:
        st.info("Vectorizer not available. Generate tfidf_vectorizer.pkl in Colab.")
        return
    query = st.text_input("Describe a movie you want to find (e.g., space adventure with strong female lead)")
    n = st.slider("Results", 5, 50, 10)
    if query:
        texts = movies["overview"].fillna("").astype(str).tolist()
        X = vectorizer.transform(texts)
        q = vectorizer.transform([query])
        sims = cosine_similarity(q, X).ravel()
        idx = np.argsort(-sims)[:n]
        cols = [c for c in ["title","year","rating","overview"] if c in movies.columns]
        res = movies.iloc[idx][cols].copy()
        res["similarity"] = sims[idx]
        st.dataframe(res)

def main():
```

Explanation:

This block defines a section of logic in the app. I explain how it fits into the pipeline and why it is written this way.

In this part, the code loads libraries, sets paths, or defines helpers. Each helper reduces repeated code and improves readability. Conditionals ensure the app fails gracefully instead of breaking abruptly. When decorated with `st.cache_data` or `st.cache_resource`, the functions avoid rerunning heavy computations each time the user interacts with the UI. That caching speeds up the user experience and makes the app feel smooth. This block shows the philosophy of preparing reliable building blocks first, then using them for interactive features.

### Code Block 11

```python
page_header()
    movies, agg_year, pca_2d = load_data()
    vectorizer, kmeans = load_vectorizer_kmeans()

    with st.sidebar:
        st.header("Filters")
        yr_rng, rating_rng = sidebar_filters(movies)

    movies_filt = apply_filters(movies, yr_rng, rating_rng)

    tabs = st.tabs(["Overview", "Clusters", "Search"])
    with tabs[0]:
        draw_overview_tab(movies_filt, agg_year)
    with tabs[1]:
        draw_clusters_tab(movies, pca_2d)
    with tabs[2]:
        draw_search_tab(movies, vectorizer, kmeans)

if __name__ == "__main__":
    main()
```

Explanation:

This block defines a section of logic in the app. I explain how it fits into the pipeline and why it is written this way.

In this part, the code loads libraries, sets paths, or defines helpers. Each helper reduces repeated code and improves readability. Conditionals ensure the app fails gracefully instead of breaking abruptly. When decorated with `st.cache_data` or `st.cache_resource`, the functions avoid rerunning heavy computations each time the user interacts with the UI. That caching speeds up the user experience and makes the app feel smooth. This block shows the philosophy of preparing reliable building blocks first, then using them for interactive features.

