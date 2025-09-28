---
layout: default
title: "Building my AI University Semantic Search App"
date: 2025-09-27 10:16:34
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-university-semantic.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_university_semantic
---

The idea for this project came from a personal search problem. I often wanted to explore universities by describing them in plain language. I imagined scenarios where I would type phrases like *technical university in Germany, old* or *medical college in Japan established before 1900*. Regular search tools did not let me combine country, founding year, and theme in one flexible query. That limitation pushed me to think about semantic search. Dataset used [here](https://www.kaggle.com/datasets/ibrahimqasimi/wilierds).

I decided to create an application that would accept natural language queries and map them to real universities. Instead of fixed keyword matching, the system uses embeddings. This allowed me to capture meaning rather than surface tokens. The process was not only educational but also showed me how embeddings can make structured data more accessible. This is the story of how I built, packaged, and deployed the system.

---

## Repository Overview

The repository contains several important files:

- `app.py` — the main Streamlit application code.
- `requirements.txt` — the dependency specification for deployment.
- `README.md` — the guide that documents the pipeline.
- `data/universities_embedded.parquet` — metadata of universities including names, countries, and inception years.
- `data/universities_embedded.npz` — precomputed embeddings created in Colab.

Each of these files is critical. The application cannot function without the embeddings and metadata. The app file orchestrates the pipeline while the requirements lock dependencies to consistent versions.

---

## Requirements File

The requirements file ensures the runtime has all necessary packages. Below is the content:

```python
streamlit==1.37.1
pandas==2.2.2
pyarrow==16.1.0
numpy==1.26.4
scikit-learn==1.4.2
sentence-transformers==2.7.0
```

Each dependency plays a direct role.  
- **Streamlit** creates the user interface.  
- **Pandas** handles structured metadata stored in parquet.  
- **PyArrow** is required to read parquet efficiently.  
- **NumPy** manages the embeddings in arrays.  
- **Scikit‑learn** provides cosine similarity functions.  
- **Sentence‑transformers** loads the same model used for embedding text queries.

Without locking versions, the app may behave differently on Streamlit Cloud. Pinning exact versions removes that risk.

---

## README File

The README explains the flow. It shows that data is produced in Colab and uploaded here. It clarifies that two files, the parquet and npz, must exist in the `data/` folder. The document also gives deployment steps. For example, it instructs how to connect the repository to Streamlit Cloud and specify `app.py` as the entry file. Finally, it lists example queries that demonstrate the natural language search capability.

This README is short but vital. It provides instructions for others to reproduce results. Without it, the context of how embeddings were generated would be missing. That gap would stop new users from understanding why those artifacts exist.

---

## Application Code

The application is defined entirely inside `app.py`. It contains about 176 lines. I will go block by block and explain the purpose and design decisions.

### Import Block

```python
import os
import numpy as np
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
```

This block loads core libraries. `os` checks for file existence. `numpy` manages numeric arrays including embeddings. `pandas` reads parquet files and handles metadata. `streamlit` powers the interface. The external libraries are `SentenceTransformer` to load the embedding model and `cosine_similarity` from scikit‑learn to compute similarity between query vectors and university vectors. Every later step depends on these imports.

---

### Page Configuration

```python
st.set_page_config(page_title="University Semantic Search", layout="wide")
st.title("University Semantic Search")
st.caption(
    "Type natural language: e.g., 'technical university in Germany, old' or "
    "'medical college in Japan established before 1900'."
)
```

This section configures the page layout. The title provides immediate context for the user. The caption demonstrates usage by giving realistic queries. The `wide` layout ensures the results table has enough horizontal space. Without these settings the interface would look cramped. Streamlit provides these helpers to make apps professional.

---

### Resource Loading Function

```python
@st.cache_resource
def load_resources():
    meta_path = "data/universities_embedded.parquet"
    vec_path = "data/universities_embedded.npz"

    if not os.path.exists(meta_path) or not os.path.exists(vec_path):
        st.error("Missing data artifacts. Make sure both files exist: "
                 "`data/universities_embedded.parquet` and `data/universities_embedded.npz`.")
        st.stop()

    df_meta = pd.read_parquet(meta_path)
    npz = np.load(vec_path)
    emb = npz["emb"].astype("float32")

    if "row_id" not in df_meta.columns:
        df_meta = df_meta.copy()
        df_meta["row_id"] = np.arange(len(df_meta))

    if len(df_meta) != emb.shape[0]:
        st.error(f"Row count mismatch: parquet has {len(df_meta)} rows but embeddings have {emb.shape[0]} vectors.")
        st.stop()

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if "inception" in df_meta.columns:
        df_meta["inception_num"] = pd.to_numeric(df_meta["inception"], errors="coerce")
    else:
        df_meta["inception_num"] = np.nan

    valid_years = df_meta["inception_num"].dropna().astype(int)
    has_year = valid_years.size > 0
    if has_year:
        year_min = int(valid_years.min())
        year_max = int(valid_years.max())
    else:
        year_min, year_max = None, None

    return df_meta, emb, model, has_year, year_min, year_max
```

This is the most important helper in the app. It is decorated with `@st.cache_resource` so results are cached between runs. That avoids reloading data repeatedly. The function loads metadata and embeddings from the two files. It then validates that the number of rows matches the number of vectors. This prevents silent misalignment errors.

A `row_id` column is created if missing. This ensures that every university has a stable index. The model is then loaded using the same transformer used during embedding creation. This guarantees consistent vector spaces. Finally, the inception year column is cleaned into numeric form, and min and max years are calculated if present. This supports optional filtering later. Without this function, the entire search pipeline cannot run.

---

### Query Encoding Function

```python
def encode_query(model, text):
    vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return vec
```

This helper converts a string into an embedding vector. It takes the loaded model and applies it to the query. The embedding is normalized which simplifies cosine similarity later. This function encapsulates encoding so the main logic does not repeat code. It represents the bridge from natural language into numeric form.

---

### Search Function

```python
def search(df_meta, emb, model, query, topk=10, year_range=None):
    qvec = encode_query(model, query)
    sims = cosine_similarity(qvec, emb)[0]
    df_meta = df_meta.copy()
    df_meta["score"] = sims

    if year_range is not None:
        lo, hi = year_range
        df_meta = df_meta[
            (df_meta["inception_num"].isna()) |
            ((df_meta["inception_num"] >= lo) & (df_meta["inception_num"] <= hi))
        ]

    df_top = df_meta.sort_values("score", ascending=False).head(topk)
    return df_top
```

This function performs the actual semantic search. It encodes the query, computes similarity between query vector and all embeddings, and attaches the score to metadata. If a year range is provided, results are filtered to universities founded within that range. The final step sorts by similarity score and keeps the top results. This function ties together encoding, similarity, and filtering.

---

### Sidebar Inputs

```python
df_meta, emb, model, has_year, year_min, year_max = load_resources()

with st.sidebar:
    query = st.text_input("Search query")
    topk = st.slider("Top K results", min_value=5, max_value=50, value=10)
    year_range = None
    if has_year:
        year_range = st.slider("Year range", min_value=year_min, max_value=year_max,
                               value=(year_min, year_max))
```

Here the application builds the sidebar. It first loads resources. Then it creates a text input for the query. A slider controls how many results to show. If inception years exist, another slider lets the user restrict the range. This block collects all user input needed for search. Streamlit automatically re‑runs the script when inputs change.

---

### Main Execution

```python
if query:
    df_top = search(df_meta, emb, model, query, topk=topk, year_range=year_range)
    st.dataframe(df_top[["label", "country", "inception", "score"]])
else:
    st.info("Enter a query to start searching.")
```

This conditional checks if a query has been entered. If yes, it calls the search function and displays the resulting dataframe with relevant columns. The score column shows similarity strength. If no query is given, an info message guides the user. This conditional ensures the app does nothing until input is provided, which avoids wasted computation.

---

## Explanation of Data Artifacts

Two files must be generated externally.  
- `universities_embedded.parquet` holds metadata with names, countries, and inception years.  
- `universities_embedded.npz` stores the embedding array with vectors for each university.  

These are produced in Colab using the same transformer model. Once generated, they are uploaded into the `data/` folder. The Streamlit app only consumes them; it does not rebuild them. This separation keeps deployment lightweight.

---

## Deployment Flow

Deployment was done on Streamlit Cloud. After pushing the repository to GitHub, I connected it to Streamlit, pointed to `app.py`, and set dependencies from `requirements.txt`. Because the data files were under 100 MB, they could be stored in GitHub directly. The caching mechanism in `load_resources` ensured fast reloads during usage. After deployment, the app was ready to accept queries immediately.

---

## Lessons Learned

Building this taught me several lessons. Semantic search becomes very practical when combined with embeddings and structured data. Streamlit provides a straightforward way to expose machine learning models to end users. The biggest challenge was ensuring data alignment between parquet and npz. Adding validation checks saved time during debugging. Another lesson was that explicit version pinning in `requirements.txt` is essential for stability.

---

## Deep Dive into Each Function and Conditional

### Why Cache Resources?

Caching resources in Streamlit is not only about speed but also about user experience. Each time a widget changes, Streamlit reruns the script. If resources were reloaded on every rerun, the app would become sluggish. The decorator `@st.cache_resource` ensures that heavy data loading and model initialization happen only once. This makes the application responsive even when embeddings are large.

### Validating Input Files

The function `load_resources` checks if both the parquet and npz files exist. This conditional prevents cryptic runtime errors. If a file is missing, the user sees a clear error message. The program then stops execution using `st.stop()`. This pattern is safer than letting the code fail later with a stack trace. It improves reliability of the interface.

### Creating Row IDs

Another conditional checks if `row_id` exists. This is important because embeddings are linked by index. If the parquet metadata does not carry an explicit index, a new one is created with `np.arange`. This guarantees one‑to‑one alignment. The design choice here avoids data mismatches. Without this guard, similarity results might point to the wrong university.

### Matching Metadata and Embeddings

The function also verifies that the length of metadata matches the embedding array. If not, it raises an error and stops. This ensures that every row has a corresponding vector. Data alignment is one of the most common pitfalls in machine learning systems. Adding this explicit check makes the project more robust.

### Handling Inception Years

Not all records have inception years. The code converts the column into numeric values using `pd.to_numeric` with `errors="coerce"`. Any invalid entries become NaN. This design allows later filtering without crashing. The conditional then calculates min and max only if valid years exist. This extra step supports the year slider in the interface.

---

## Understanding the Search Pipeline

### Encoding Queries

The helper `encode_query` is minimal but powerful. It encapsulates embedding logic so other functions remain clean. By normalizing embeddings, cosine similarity becomes equivalent to dot product. This optimization is subtle but important. It avoids unnecessary magnitude differences.

### Cosine Similarity

The search function uses `cosine_similarity` to compare query vectors with all embeddings. The result is a score for each university. Attaching the score back to the dataframe integrates numeric results with metadata. Sorting by score produces the ranking.

### Year Range Filtering

The optional `year_range` argument is a key feature. It allows users to focus on universities within a historical window. The conditional uses logical operators to keep rows with inception years inside the range or NaN. This inclusive design ensures that institutions without years are not unfairly discarded.

---

## Streamlit Sidebar Explained

Streamlit automatically rebuilds the page when sidebar inputs change. The sidebar groups controls neatly. The `text_input` widget lets users type queries. The `slider` for top K results is intuitive for adjusting output length. Another `slider` for year range uses min and max calculated earlier. This dynamic binding makes the app interactive with minimal code.

---

## Displaying Results

The main conditional displays results only when a query exists. This prevents confusion when the app first loads. The table shows label, country, inception, and score. Presenting only these columns keeps the interface clean. The choice to include score is deliberate; it reveals ranking strength. This transparency helps users trust the output.

---

## Future Extensions

Several extensions are possible. The metadata could be enriched with student population, research output, or global ranking. The embeddings could be fine‑tuned for better performance on educational queries. The interface could allow faceted search by continent or type. These features can be layered on the current pipeline without major redesign.

---

## Reflections

Looking back, the most striking aspect is how small code can deliver powerful search. Less than 200 lines integrate deep learning, efficient similarity, and a user‑friendly interface. The project demonstrates how modern NLP tools reduce barriers. Anyone with basic Python skills can now create semantic systems that once required large teams.

---

## Thoughts

The University Semantic Search project began from a simple frustration but evolved into a showcase of applied machine learning. It emphasizes the importance of validation, caching, and modular functions. The design choices highlight practical engineering: explicit checks, clear messages, and reproducible data pipelines. This balance of theory and practice makes the project a valuable template for future work.


---

## Deployment Notes

Deploying on Streamlit Cloud is straightforward but a few points are worth noting. The platform automatically installs dependencies from `requirements.txt`. However, large models are not downloaded during runtime. That is why the embeddings were precomputed in Colab. This separation reduces cold start time. Streamlit Cloud has memory limits, so keeping the app lightweight matters. By uploading only essential artifacts, I avoided hitting those limits.

Another deployment detail is caching behavior. The first run after deployment can take a few seconds because it loads the parquet and npz. Later runs are instant due to caching. Users rarely notice delays, which improves perception of performance. Sharing the app link allows anyone to experiment without local setup. This accessibility was a primary motivation.

---

## Lessons

This project shows that semantic search can be applied beyond universities. The same pattern works for products, books, or research papers. The critical step is building embeddings for items and storing metadata. Once that is done, the rest of the pipeline is identical. Streamlit makes it easy to reuse the interface for new domains. Understanding this generality was a big lesson.

Another lesson was about balancing transparency and abstraction. I chose to show similarity scores to users, but I hid the raw vectors. The interface reveals enough for trust but avoids overwhelming users. This principle can guide other applications too. Always decide which internal details should surface and which should remain hidden.

---

## Remarks

The journey from idea to deployment taught me how modern tools shorten development cycles. Ten years ago, building such a system would require custom servers and complex front ends. Now, a single Python file and a few data artifacts are enough. This democratization of technology is powerful. It opens doors for students, researchers, and hobbyists to explore semantic methods. I built this for a personal reason, but the lessons apply widely. That is the real success of the project.

---

## Conclusion

The University Semantic Search app demonstrates how semantic embeddings can power intuitive queries. By breaking down the pipeline into data preparation, embedding, caching, searching, and displaying, the system becomes reliable and reproducible. The modular design makes it easy to extend, such as adding new filters or expanding metadata. This project began as a personal need but grew into a demonstration of how modern NLP can transform structured search.
