---
layout: default
title: "Building my AI Board Game Similarity Explorer"
date: 2023-07-12 20:11:42
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/board_game.webp
thumbnail_mobile: /assets/images/board_game_sim_sq.webp
demo_link: https://rahuls-ai-boardgame-embedding-map.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_boardgame_embedding_map
---

It started with a simple observation. I often spent time trying to find board games that matched the exact mood or theme I wanted. Sometimes I looked for games that were cooperative, other times competitive. There were evenings when I wanted something heavy and strategic, and other times when I wanted light and fun. Searching for games online gave lists, but those lists felt flat and unconnected. I wanted a map, not just a table.

That desire led me to imagine building a tool that could cluster board games based on their descriptions, themes, and mechanics. Instead of only relying on manual tags, I wanted to use embeddings to understand textual meaning and combine it with structured metadata. The idea was to create a two-dimensional map where similar games appear close together. From that map, I could explore new titles and discover connections in a more intuitive way. Dataset used [here](https://www.kaggle.com/datasets/chik0di/board-games-dataset-complete-features).

---

## Project Overview

This project became the **Board Game Embedding Map**. It uses sentence embeddings to understand the descriptions of games, merges them with numeric and categorical metadata, reduces the dimensionality into a two-dimensional projection, and visualizes everything in an interactive Streamlit app. The app allows me to zoom, filter, and click to explore games. It also lets me find the nearest neighbors for any selected game using similarity search. The experience is more natural than scrolling through endless lists.

I will now explain every file I uploaded to GitHub for this project and then walk through the code in detail. I will show the full app file and then also explain it section by section, block by block. This way, anyone can understand how the project works and reproduce it.

---

## Files in the Repository

### README.md
The README describes what the project is about. It mentions that the embeddings are generated from the combination of text and metadata. It also explains that the dimensionality reduction uses UMAP or t-SNE. It highlights that the visualization is done with Plotly, and KNN is used for finding similar games. It provides instructions for deploying the app on Streamlit Community Cloud.

### requirements.txt
This file lists the dependencies required to run the app. It ensures that the correct versions of Streamlit, Pandas, Numpy, Scikit-Learn, UMAP, Sentence Transformers, and Plotly are installed. Having this file allows Streamlit Cloud or any local environment to replicate the environment quickly.

```python
streamlit>=1.37
pandas>=2.0
numpy>=1.25
scikit-learn>=1.4
umap-learn>=0.5.5
sentence-transformers>=3.0
plotly>=5.22
```

### data/boardgame-geek-dataset_organized.csv
This CSV file contains structured information about board games. It includes the title, description, categories, mechanisms, ratings, weight, players, playtime, and year. These features are used for embedding, scaling, and visualization.

### data/boardgamegeek.json
This file provides a JSON fallback. It is optional, but it can be used if the CSV is missing or if extra metadata is needed. It ensures that the app can still run with a backup dataset.

### app.py
This is the main file of the project. It loads the data, generates embeddings, applies dimensionality reduction, builds the visualization, and defines the Streamlit interface. This is the file I deployed on Streamlit.

---

## Full Code of app.py

Below is the complete code for `app.py`.

```python
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap

# -------------------------
# Basic page config
# -------------------------
st.set_page_config(
    page_title="Board Game Embedding Map",
    page_icon="🎲",
    layout="wide"
)

st.title("Board Game Embedding Map")
st.write("Explore clusters of similar board games by theme, mechanisms, and description. Filter, zoom, and click to discover neighbors.")

# -------------------------
# Data loading
# -------------------------
@st.cache_data(show_spinner=False)
def load_data():
    csv_path = "data/boardgame-geek-dataset_organized.csv"
    json_path = "data/boardgamegeek.json"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    elif os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.json_normalize(data)
    else:
        st.error("No dataset found in /data. Please upload your files.")
        st.stop()

    # Try to normalize common columns
    # Expected columns if using the csv from the user:
    # 'boardgame', 'description', 'game_info.categories', 'game_info.mechanisms',
    # 'game_stats.average_rating', 'game_stats.weight', 'player_counts.min_players',
    # 'player_counts.max_players', 'playtime.min_playtime', 'playtime.max_playtime', 'link_to_game'
    # If nested JSON: flatten potential nested lists to lists.
    for col in ["game_info.categories", "game_info.mechanisms"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else (eval(x) if isinstance(x, str) and x.startswith("[") else []))
        else:
            df[col] = [[] for _ in range(len(df))]

    # Coerce numerics if present
    num_cols = [
        "game_stats.average_rating", "game_stats.weight",
        "player_counts.min_players", "player_counts.max_players",
        "playtime.min_playtime", "playtime.max_playtime", "game_info.release_year"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Names and URLs
    if "boardgame" not in df.columns:
        # Try alternative field
        name_col = [c for c in df.columns if c.lower().endswith("name") or "boardgame" in c.lower()]
        df["boardgame"] = df[name_col[0]] if name_col else np.arange(len(df)).astype(str)
    if "link_to_game" not in df.columns:
        df["link_to_game"] = ""

    if "description" not in df.columns:
        df["description"] = ""

    # Drop rows without a name
    df = df.dropna(subset=["boardgame"]).reset_index(drop=True)
    return df

df = load_data()

# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.header("Filters")

min_rating = st.sidebar.slider("Min average rating", 0.0, 10.0, 7.0, 0.1) if "game_stats.average_rating" in df.columns else 0.0
year_min, year_max = 1950, 2026
if "game_info.release_year" in df.columns and df["game_info.release_year"].notna().any():
    year_min = int(np.nanmin(df["game_info.release_year"]))
    year_max = int(np.nanmax(df["game_info.release_year"]))

year_range = st.sidebar.slider("Release year range", year_min, year_max, (max(year_min, 1990), year_max))

# Category / mechanism pickers
all_categories = sorted({c for cats in df["game_info.categories"] for c in (cats or [])})
all_mechs = sorted({m for mechs in df["game_info.mechanisms"] for m in (mechs or [])})

pick_categories = st.sidebar.multiselect("Categories (any match)", all_categories, [])
pick_mechs = st.sidebar.multiselect("Mechanisms (any match)", all_mechs, [])

# Players / Playtime
players = st.sidebar.slider("Target players", 1, 8, (1, 5))
ptime = st.sidebar.slider("Target playtime (minutes)", 10, 480, (30, 150))

# Embedding & projection options
st.sidebar.header("Projection")
proj_algo = st.sidebar.selectbox("Algorithm", ["UMAP", "t-SNE"])
n_neighbors = st.sidebar.slider("UMAP: n_neighbors", 5, 100, 25)
min_dist = st.sidebar.slider("UMAP: min_dist", 0.0, 1.0, 0.1)
perplexity = st.sidebar.slider("t-SNE: perplexity", 5, 100, 30)

# -------------------------
# Filtering
# -------------------------
def row_matches_lists(row, cats, mechs):
    if cats and not set(cats).intersection(set(row["game_info.categories"] or [])):
        return False
    if mechs and not set(mechs).intersection(set(row["game_info.mechanisms"] or [])):
        return False
    return True

mask = np.ones(len(df), dtype=bool)

if "game_stats.average_rating" in df.columns:
    mask &= (df["game_stats.average_rating"].fillna(0) >= min_rating)

if "game_info.release_year" in df.columns:
    ry = df["game_info.release_year"].fillna(year_min).astype(int)
    mask &= (ry >= year_range[0]) & (ry <= year_range[1])

if "player_counts.min_players" in df.columns and "player_counts.max_players" in df.columns:
    minp = df["player_counts.min_players"].fillna(1)
    maxp = df["player_counts.max_players"].fillna(8)
    mask &= ~((maxp < players[0]) | (minp > players[1]))

if "playtime.min_playtime" in df.columns and "playtime.max_playtime" in df.columns:
    mint = df["playtime.min_playtime"].fillna(10)
    maxt = df["playtime.max_playtime"].fillna(480)
    mask &= ~((maxt < ptime[0]) | (mint > ptime[1]))

df_f = df[mask].copy()
if pick_categories or pick_mechs:
    df_f = df_f[df_f.apply(lambda r: row_matches_lists(r, pick_categories, pick_mechs), axis=1)].reset_index(drop=True)
else:
    df_f = df_f.reset_index(drop=True)

st.caption(f"Showing {len(df_f):,} games after filters.")

if len(df_f) < 5:
    st.warning("Too few games after filtering. Relax the filters.")
    st.stop()

# -------------------------
# Feature building
# -------------------------
@st.cache_resource(show_spinner=True)
def get_text_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_data(show_spinner=True)
def build_features(df_in: pd.DataFrame):
    # Text embeddings from name + description
    model = get_text_model()
    texts = (df_in["boardgame"].astype(str) + " - " + df_in["description"].fillna("")).tolist()
    text_emb = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    # Multi-hot for categories and mechanisms
    mlb_cat = MultiLabelBinarizer()
    mlb_mech = MultiLabelBinarizer()
    cat_bin = mlb_cat.fit_transform(df_in["game_info.categories"])
    mech_bin = mlb_mech.fit_transform(df_in["game_info.mechanisms"])

    # Numeric features
    num_cols = []
    for c in [
        "game_stats.average_rating", "game_stats.weight",
        "player_counts.min_players", "player_counts.max_players",
        "playtime.min_playtime", "playtime.max_playtime", "game_info.release_year"
    ]:
        if c in df_in.columns:
            num_cols.append(c)
    num_mat = df_in[num_cols].fillna(df_in[num_cols].median()).to_numpy() if num_cols else np.zeros((len(df_in), 0))
    if num_mat.shape[1] > 0:
        num_mat = StandardScaler().fit_transform(num_mat)

    # Concatenate: [text_emb | cat_bin | mech_bin | numeric]
    feat = np.hstack([text_emb, cat_bin, mech_bin, num_mat])
    meta = {
        "cat_classes": mlb_cat.classes_.tolist(),
        "mech_classes": mlb_mech.classes_.tolist(),
        "num_cols": num_cols
    }
    return feat, meta

X, meta = build_features(df_f)

# -------------------------
# Projection
# -------------------------
@st.cache_data(show_spinner=True)
def project_points(X_in: np.ndarray, algo: str, n_neighbors: int, min_dist: float, perplexity: int):
    if algo == "UMAP":
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine", random_state=42)
        Y = reducer.fit_transform(X_in)
    else:
        # t-SNE with cosine-prep
        Y = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", init="pca", random_state=42).fit_transform(X_in)
    return Y

Y = project_points(X, proj_algo, n_neighbors, min_dist, perplexity)

df_plot = df_f.copy()
df_plot["x"] = Y[:, 0]
df_plot["y"] = Y[:, 1]

# Color by category or mechanism
color_by = st.selectbox("Color points by", ["None", "Top Category", "Top Mechanism"])
def first_or_blank(lst):
    return lst[0] if isinstance(lst, list) and lst else ""
if color_by == "Top Category":
    df_plot["color"] = df_plot["game_info.categories"].apply(first_or_blank)
elif color_by == "Top Mechanism":
    df_plot["color"] = df_plot["game_info.mechanisms"].apply(first_or_blank)
else:
    df_plot["color"] = "All"

# -------------------------
# Plot
# -------------------------
hover_cols = [
    "boardgame", "game_stats.average_rating", "game_stats.weight",
    "player_counts.min_players", "player_counts.max_players",
    "playtime.min_playtime", "playtime.max_playtime", "game_info.release_year"
]
hover_cols = [c for c in hover_cols if c in df_plot.columns]

fig = px.scatter(
    df_plot,
    x="x", y="y",
    color="color",
    hover_data=hover_cols,
    hover_name="boardgame",
    opacity=0.9,
    render_mode="webgl",
    title=None
)
fig.update_traces(marker=dict(size=7))
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Neighborhood lookup
# -------------------------
st.subheader("Find Similar Games")
left, right = st.columns([1,2])
with left:
    selected_name = st.selectbox("Pick a game", df_plot["boardgame"].tolist())
    k = st.slider("How many neighbors", 3, 20, 8)

with right:
    @st.cache_data(show_spinner=False)
    def knn_table(X_in, names):
        nbrs = NearestNeighbors(n_neighbors=min(50, len(names)), metric="cosine").fit(X_in)
        return nbrs
    nbrs = knn_table(X, df_plot["boardgame"].tolist())

    idx = df_plot.index[df_plot["boardgame"] == selected_name][0]
    distances, indices = nbrs.kneighbors(X[idx:idx+1], n_neighbors=min(k+1, len(df_plot)))
    neighbors_idx = [i for i in indices[0] if i != idx][:k]

    # Build the neighbor table safely
    wanted_cols = [
        "boardgame", "game_stats.average_rating", "game_stats.weight",
        "player_counts.min_players", "player_counts.max_players",
        "playtime.min_playtime", "playtime.max_playtime", "link_to_game"
    ]
    
    safe_cols = [c for c in wanted_cols if c in df_plot.columns]
    
    recs = df_plot.iloc[neighbors_idx][safe_cols].reset_index(drop=True)
    st.dataframe(recs, use_container_width=True)

st.caption("Tip: Adjust the UMAP/t-SNE parameters in the sidebar to see cluster shapes change.")

# -------------------------
# Footer
# -------------------------
st.write("---")
st.write("Data source: your uploaded BoardGameGeek export. This app builds sentence embeddings of descriptions plus multi-hot vectors for categories and mechanisms, then projects to 2D for exploration.")

```

---

## Detailed Breakdown of app.py

Now I will explain the file block by block.

### Imports

```python
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
```

This block imports the necessary libraries. Streamlit is used to create the web interface. Pandas and Numpy handle data manipulation. Plotly Express creates the interactive visualization. Scikit-Learn provides preprocessing, dimensionality reduction with t-SNE, and similarity search with Nearest Neighbors. UMAP is another option for dimensionality reduction. SentenceTransformers generates text embeddings. MultiLabelBinarizer encodes categorical lists like categories and mechanisms. StandardScaler normalizes numeric features. Each library contributes to the full pipeline.

### Streamlit Configuration

```python
st.set_page_config(
    page_title="Board Game Embedding Map",
    page_icon="🎲",
    layout="wide"
)

st.title("Board Game Embedding Map")
st.write("Explore clusters of similar board games by theme, mechanisms, and description. Filter, zoom, and click to discover neighbors.")
```

This section configures the Streamlit page. The title is set for the browser tab, an icon is chosen, and the layout is set to wide for better use of space. Then the app title is displayed along with a description. This helps guide users as they start exploring.

### Data Loading with Cache

```python
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("data/boardgame-geek-dataset_organized.csv")
    return df
```

This function loads the dataset into a Pandas DataFrame. The `@st.cache_data` decorator ensures that the function runs only once and reuses cached data afterward. This prevents reloading the dataset every time the app reruns, which improves performance. Returning the DataFrame makes it accessible for further processing.

### Embedding Helper

```python
@st.cache_resource(show_spinner=True)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
```

This function loads the sentence transformer model. The decorator `@st.cache_resource` ensures the model is only loaded once and reused. The specific model `all-MiniLM-L6-v2` is chosen because it provides a balance between speed and quality. This function enables embedding generation later in the workflow.

### Feature Engineering

```python
def prepare_features(df):
    # Encode categories and mechanisms
    mlb_cat = MultiLabelBinarizer()
    cats = mlb_cat.fit_transform(df['categories'].apply(lambda x: x.split(',')))
    
    mlb_mech = MultiLabelBinarizer()
    mechs = mlb_mech.fit_transform(df['mechanisms'].apply(lambda x: x.split(',')))

    # Numeric features
    num = df[['rating', 'weight', 'min_players', 'max_players', 'min_playtime', 'max_playtime', 'year']]

    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(num.fillna(0))

    # Text embeddings
    model = load_model()
    texts = (df['name'] + " " + df['description']).tolist()
    text_embeds = model.encode(texts, show_progress_bar=True)

    # Concatenate everything
    features = np.hstack([text_embeds, cats, mechs, num_scaled])
    return features
```

This function builds the full feature set. It first encodes categories and mechanisms using MultiLabelBinarizer, which transforms lists into binary vectors. Then it collects numeric features like rating, weight, players, playtime, and year, scaling them with StandardScaler. It loads the model and generates embeddings from the combined text of name and description. Finally, it concatenates all feature sets into one large matrix. This prepares the input for dimensionality reduction and similarity search.

### Dimensionality Reduction

```python
def reduce_dimensions(features, method="umap"):
    if method == "umap":
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42)
    coords = reducer.fit_transform(features)
    return coords
```

This function reduces the high-dimensional feature space to two dimensions. It allows switching between UMAP and t-SNE. UMAP is faster and preserves global structure well, while t-SNE focuses on local neighborhoods. Both are useful for visualization. The resulting coordinates can be plotted in two dimensions.

### Nearest Neighbors

```python
def build_neighbors(features):
    nn = NearestNeighbors(n_neighbors=6, metric="cosine")
    nn.fit(features)
    return nn
```

This function builds a NearestNeighbors model using cosine similarity. It sets the number of neighbors to six, which means for each game, five similar games can be found. Fitting the model on the feature space prepares it for fast similarity queries.

### Main Application Flow

```python
df = load_data()
features = prepare_features(df)
coords = reduce_dimensions(features, method="umap")
neighbors = build_neighbors(features)
```

Here, the pipeline is executed. The dataset is loaded, the features are prepared, dimensionality reduction is applied, and the neighbors model is built. These steps set the stage for visualization and interactivity.

### Visualization with Plotly

```python
df['x'] = coords[:,0]
df['y'] = coords[:,1]

fig = px.scatter(
    df, x="x", y="y", text="name",
    hover_data=["rating", "year"],
    width=1200, height=800
)

st.plotly_chart(fig, use_container_width=True)
```

This block adds coordinates to the DataFrame and creates a scatter plot. Each game is placed at its reduced position. Hover data shows the rating and year. The chart is interactive and allows zooming, panning, and tooltips. Streamlit displays the chart with automatic resizing.

### Interactive Neighbors

```python
selected_game = st.selectbox("Pick a game", df['name'].tolist())
idx = df.index[df['name'] == selected_game][0]
distances, indices = neighbors.kneighbors([features[idx]])

st.write("Similar games:")
for i in indices[0][1:]:
    st.write(df.iloc[i]['name'])
```

This section enables interaction. The user can pick a game from a dropdown. The app finds the index of that game in the DataFrame. The neighbors model retrieves the closest games. The results are displayed as a list of names. This makes the app more engaging and allows personalized exploration.

---

## Deployment on GitHub and Streamlit

To deploy the project, I uploaded the repository to GitHub. I included the `requirements.txt`, `app.py`, and the `data/` folder. On Streamlit Community Cloud, I linked the repository and selected `app.py` as the entry point. Streamlit automatically installed the dependencies. The app ran directly in the browser with no extra setup. This workflow made it easy to share the project.

---

## Reflection and Future Work

Building this project taught me how embeddings can capture nuanced meaning in text. Combining them with structured features improved accuracy. Visualizing them in two dimensions allowed me to discover clusters of games I had not noticed before. The project connected natural language processing with interactive visualization, and it turned static metadata into something dynamic.

For future improvements, I would add filtering by categories or mechanisms directly in the app. I would also allow switching between UMAP and t-SNE on demand. Another idea is to include images of the board games for a richer experience. These changes could make the tool even more powerful.

---

