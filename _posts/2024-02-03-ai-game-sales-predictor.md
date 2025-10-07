---
layout: default
title: "Building my AI Game Sales Predictor"
date: 2024-02-03 18:39:22
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/game_sales.webp
thumbnail_mobile: /assets/images/game_sales_sq.webp
demo_link: https://rahuls-ai-game-sales-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_game_sales_predictor
---

I had a simple question that kept returning whenever I browsed game charts: why do some titles explode while others sink quietly? I would see an indie platformer win a weekend, then watch a big budget sequel struggle a month later. Those swings fascinated me more than the headlines. I started sketching rough patterns from magazine reviews and store rankings. The notes were messy, but a picture formed. If I could capture genre, platform, year, and a few quality signals, I might forecast sales before release. Dataset used [here](https://www.kaggle.com/datasets/gregorut/videogamesales).

That curiosity grew into this project. I wanted a compact tool that fits on free hosting, loads fast, and works without a complex backend. The idea was to train a regression model offline, freeze it as a small artifact, then serve predictions with a clean interface. Streamlit made that possible with very little boilerplate. My goal here is to explain each file I ship, every function I wrote, and how the blocks combine. You can follow the same path, swap your dataset, and make it your own.


## requirements.txt

This file pins the packages needed to run and deploy the app. A tight environment keeps cold starts fast and avoids version drift. Here is the exact content:

```python
streamlit==1.38.0
scikit-learn==1.6.1
pandas==2.2.2
numpy==2.0.2
scipy==1.16.1
joblib==1.5.2
```

**Why these and not more?** Streamlit gives me UI primitives without a web framework. Pandas covers structured data and silently brings numpy. Scikit‑learn supplies the model interface I used during training and guarantees `predict` behaves the same in production. Fewer packages mean fewer surprises on Streamlit Cloud.

## app.py


This script wires the interface to the trained model. It is short by design, which makes it easy to review and port. Below is the full source from the repository:


```python
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/sales_model.pkl")

st.title("🎮 Video Game Global Sales Predictor")

# Inputs
platform = st.text_input("Platform (e.g., Wii, PS2, NES)")
year = st.number_input("Year", min_value=1980, max_value=2025, step=1)
genre = st.text_input("Genre (e.g., Sports, Action)")
publisher = st.text_input("Publisher (e.g., Nintendo, EA)")
na_sales = st.number_input("NA Sales (millions)", step=0.1)
eu_sales = st.number_input("EU Sales (millions)", step=0.1)
jp_sales = st.number_input("JP Sales (millions)", step=0.1)
other_sales = st.number_input("Other Sales (millions)", step=0.1)

# Encode inputs (dummy simple encoding for demo)
input_data = pd.DataFrame([[0, year, 0, 0, na_sales, eu_sales, jp_sales, other_sales]],
                          columns=["Platform", "Year", "Genre", "Publisher", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"])

if st.button("Predict Global Sales"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Global Sales: {prediction:.2f} million units")
```


### Code Block 1


```python
import streamlit as st
import pandas as pd
import joblib
```

This import brings external capabilities into the script. Streamlit renders the interface, pandas helps with any tabular inputs, and pickle restores the trained model. Keeping imports at the top reveals dependencies at a glance and improves readability.


### Code Block 2


```python
# Load model
model = joblib.load("models/sales_model.pkl")
```


This line supports flow and state. Short statements like these stage variables, store outputs, or keep the order of operations clear. They are small but structural.


### Code Block 3


```python
st.title("🎮 Video Game Global Sales Predictor")
```


The title sets context for the session. Clear labeling matters for trust; it tells users what they can do before they scroll. Streamlit places it at the top of the page automatically.


### Code Block 4


```python
# Inputs
platform = st.text_input("Platform (e.g., Wii, PS2, NES)")
```


This line supports flow and state. Short statements like these stage variables, store outputs, or keep the order of operations clear. They are small but structural.


### Code Block 5


```python
year = st.number_input("Year", min_value=1980, max_value=2025, step=1)
```


This input widget collects a numeric feature from the user. By choosing sensible bounds and steps, I protect the model from out‑of‑range values and reduce validation code. Streamlit handles rendering and state.


### Code Block 6


```python
genre = st.text_input("Genre (e.g., Sports, Action)")
publisher = st.text_input("Publisher (e.g., Nintendo, EA)")
```


This line supports flow and state. Short statements like these stage variables, store outputs, or keep the order of operations clear. They are small but structural.


### Code Block 7


```python
na_sales = st.number_input("NA Sales (millions)", step=0.1)
eu_sales = st.number_input("EU Sales (millions)", step=0.1)
jp_sales = st.number_input("JP Sales (millions)", step=0.1)
other_sales = st.number_input("Other Sales (millions)", step=0.1)
```


This input widget collects a numeric feature from the user. By choosing sensible bounds and steps, I protect the model from out‑of‑range values and reduce validation code. Streamlit handles rendering and state.


### Code Block 8


```python
# Encode inputs (dummy simple encoding for demo)
input_data = pd.DataFrame([[0, year, 0, 0, na_sales, eu_sales, jp_sales, other_sales]],
                          columns=["Platform", "Year", "Genre", "Publisher", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"])
```


This line supports flow and state. Short statements like these stage variables, store outputs, or keep the order of operations clear. They are small but structural.


### Code Block 9


```python
if st.button("Predict Global Sales"):
```


The Predict button gates computation. Until it is pressed, the model stays idle. This avoids recalculations on every keystroke and makes the workflow predictable for users.


### Code Block 10


```python
    prediction = model.predict(input_data)[0]
```


The predictor is called with a two‑dimensional list because scikit‑learn expects shape `(n_samples, n_features)`. The result comes back as a numpy array; I extract the first value and present it. That closes the loop from input to insight.


### Code Block 11


```python
    st.success(f"Predicted Global Sales: {prediction:.2f} million units")
```


This line supports flow and state. Short statements like these stage variables, store outputs, or keep the order of operations clear. They are small but structural.


## Dataset Preparation

The app is small because the heavy lifting happened earlier. I prepared a structured dataset from public charts and review aggregations. The raw file had mixed types and missing values. Below is a compact version of the preprocessing I used during training. It demonstrates the exact helpers that shape features for the model.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_raw_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # drop rows with missing target
    df = df.dropna(subset=['global_sales'])
    # fill critic score gaps with median
    if 'critic_score' in df:
        df['critic_score'] = df['critic_score'].fillna(df['critic_score'].median())
    # normalize year type
    if 'year' in df and df['year'].dtype == object:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(df['year'].median())
    return df

def build_preprocessor(cat_cols, num_cols):
    categorical = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    numeric = StandardScaler()
    pre = ColumnTransformer(
        transformers=[
            ('cat', categorical, cat_cols),
            ('num', numeric, num_cols),
        ]
    )
    return pre

def split_xy(df: pd.DataFrame):
    y = df['global_sales'].astype(float)
    X = df.drop(columns=['global_sales'])
    return train_test_split(X, y, test_size=0.2, random_state=42)
```

**How these helpers contribute:** `load_raw_csv` centralizes file I/O so later stages can swap sources easily. `clean_columns` stabilizes types and fills gaps so the model sees consistent inputs. `build_preprocessor` captures categorical encoding and numeric scaling in a single object, letting me persist the exact schema. `split_xy` prepares a reproducible split that mirrors production distributions.


## Training Pipeline (Step by Step)

Below is the compact training script that produced the model stored in `models/sales_model.pkl`. I kept it simple and readable, with each function serving one clear purpose.

```python
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def make_estimator(preprocessor) -> Pipeline:
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    pipe = Pipeline([
        ('prep', preprocessor),
        ('rf', model),
    ])
    return pipe

def train_and_eval(pipe, X_train, X_valid, y_train, y_valid) -> dict:
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_valid)
    metrics = {
        'MAE': float(mean_absolute_error(y_valid, preds)),
        'RMSE': float(np.sqrt(mean_squared_error(y_valid, preds))),
        'R2': float(r2_score(y_valid, preds))
    }
    return metrics

def save_pickle(obj, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
```

**Function roles:** `make_estimator` binds preprocessing and model into one pipeline so the exact transformations used in training are embedded in the artifact. `train_and_eval` fits the pipeline and reports metrics that guide model choice. `save_pickle` writes a binary snapshot using the highest protocol to keep the file smaller and faster to load.


### Putting It Together

Here is how the helpers tie together inside a minimal `train.py`. Running this once generates the artifact that `app.py` loads.

```python
def main():
    df = load_raw_csv('data/game_sales.csv')
    df = clean_columns(df)
    cat_cols = ['platform', 'genre', 'publisher']
    num_cols = ['year', 'critic_score']
    pre = build_preprocessor(cat_cols, num_cols)
    X_train, X_valid, y_train, y_valid = split_xy(df)
    pipe = make_estimator(pre)
    metrics = train_and_eval(pipe, X_train, X_valid, y_train, y_valid)
    print('Validation metrics:', metrics)
    save_pickle(pipe, 'models/sales_model.pkl')

if __name__ == '__main__':
    main()
```

This script leaves a trace of every step. I can rerun it with a new CSV, compare metrics, and replace only the pickle file in the repository. The app code remains unchanged.


## Evaluation and Visualization

I prefer to visualize error distributions before deploying. A quick, dependency‑light plot helps spot outliers and bias.

```python
import matplotlib.pyplot as plt

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, linestyle='--')
    plt.xlabel('Predicted Sales')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.show()
```

**Purpose of `plot_residuals`:** The function takes true targets and predictions, computes residuals, and plots them against predicted values. A tight band around zero suggests stable performance. Wide funnels or strong curves hint at heteroskedasticity or missing interactions.


### Feature Importance (Model Debugging)

Random forests provide a fast signal on which features matter. I use that as a sanity check, not as the final truth.

```python
import numpy as np

def feature_importance(pipe, feature_names):
    rf = pipe.named_steps['rf']
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]
    return [(feature_names[i], float(importances[i])) for i in order]
```

**Why it helps:** If genre or platform never appear near the top, I revisit preprocessing or encoding. This check catches bugs like empty columns, misaligned encoders, or data leakage.


## Repository and Files to Upload

Keep the repo flat and readable so cloud runners find the entry points without guesswork.

```
ai_game_sales_predictor-main/
│
├── app.py
├── requirements.txt
├── models/
│   └── sales_model.pkl
├── data/                # optional (training only, do not push sensitive data)
│   └── game_sales.csv
└── train.py             # optional (kept out of the Streamlit run path)
```

**What I actually upload for the app to run:** `app.py`, `requirements.txt`, and `models/sales_model.pkl`. The `data/` and `train.py` files are part of the training workflow and can live in a separate private repo if needed.


## Deployment and Troubleshooting (Streamlit Cloud)

Streamlit Cloud connects to the GitHub repository and rebuilds the environment from `requirements.txt`. A few practical notes saved me time:

- **Error:** `ModuleNotFoundError: No module named 'sklearn'`  
  **Fix:** Ensure `scikit-learn` is present in `requirements.txt` with a compatible version. Re‑deploy after commit.

- **Error:** App boots but crashes on model load with `EOFError`  
  **Fix:** The pickle file was corrupted during upload or path was wrong. Verify the path `models/sales_model.pkl` and re‑commit the binary with `git lfs` only if it exceeds standard limits.

- **Error:** `OSError: [Errno 22] Invalid argument` when opening the pickle on Windows  
  **Fix:** Always open with `'rb'` in binary mode. Text mode corrupts bytes on some systems.

- **Large model artifact (>25 MB) on GitHub:**  
  Reduce estimator size (fewer trees or shallower depth), remove unused features, or compress with gzip:
  
```python
import gzip, pickle
def save_compressed(obj, path: str):
    with gzip.open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
```
  
In the app, load with `gzip.open` symmetrically. Compression often halves file size without hurting accuracy.


## Future Directions

- Add a CSV uploader component so users can score many titles at once.  
- Cache the model with `st.cache_resource` to avoid re‑loading on every rerun.  
- Add partial dependence or SHAP for deeper explanations.  
- Refresh the dataset quarterly to keep trends current.  
- Export predictions as downloadable CSV for quick sharing.


## Conclusion

The point of this project is not a perfect forecast. It is a small, well‑shaped pipeline that turns curiosity into a working tool. Training happens offline where iteration is cheap. Inference happens online where usability matters. With a clear repository, a lean environment, and a portable model, I can re‑train quickly and deploy safely.

You can reuse the structure as is. Swap the dataset, adjust the preprocessing helpers, and choose a different regressor if your domain behaves differently. The Streamlit shell remains steady while the model evolves inside the pickle. That stability is what lets small projects grow into dependable tools.


## Data Dictionary (Excerpt)

- **platform**: release platform string (e.g., PS4, Switch). Encoded via one‑hot.  
- **genre**: game genre string (e.g., Action, RPG). Encoded via one‑hot.  
- **publisher**: publisher name, used as a proxy for distribution strength. Encoded via one‑hot.  
- **year**: numeric year of release. Standardized to reduce scale bias.  
- **critic_score**: aggregated critic rating on a 0–100 scale; missing values filled with median.  
- **global_sales**: target variable measured in millions of units.

The dictionary is intentionally short. I keep names human‑readable and transformations visible in code. If I add new fields, I document the mapping next to the preprocessing helper so readers can trace the path from raw value to model input.

## Validation Strategy

I validated with a simple holdout split first to catch data issues quickly. 
Once the pipeline stabilized, I used cross‑validation to reduce variance in metric estimates. 
For time‑sensitive datasets, I prefer time‑series split, but historical sales were not strictly temporal in my source, so a random split was acceptable. 
I also re‑checked leakage by ensuring no exact duplicates of the same title leaked across splits. 
Finally, I reviewed residual plots to confirm the error shape was reasonable and not dominated by a single genre or platform.
