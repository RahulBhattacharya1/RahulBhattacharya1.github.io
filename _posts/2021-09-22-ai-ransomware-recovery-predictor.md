---
layout: default
title: "Building my AI Ransomware Recovery Time Predictor"
date: 2021-09-22 10:27:34
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/ransomware.webp
thumbnail_mobile: /assets/images/ransomware_sq.webp
demo_link: https://rahuls-ai-ransomware-recovery-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_ransomware_recovery_predictor
---

A winter outage story pushed this build into motion. A regional healthcare network suffered a ransomware event, and the updates from the incident response team read like weather reports that never got better. Recovery windows moved from days to weeks, and every new dependency uncovered another delay. I kept wondering if a simple model could at least frame the timeline in a grounded way for leaders who need to plan. The question was not about certainty, but about getting a baseline that could be refined with more data. That line of thinking grew into a small tool I could ship fast and improve over time. Dataset used [here](https://www.kaggle.com/datasets/rivalytics/healthcare-ransomware-dataset).

I did not want a heavy dashboard or an intricate backend, so I used Streamlit to make a clear interface in one file. The model behind it is a light regression trained on structured factors that influence recovery effort. Inputs capture organization type, size, infection rate, and whether backups or data were compromised. The point is not to replace an analyst, but to turn a vague debate into a concrete starting estimate. From there the conversation becomes specific and the tradeoffs become visible. This post explains every file I committed and every block of code that turns the model into a working page.

## Repository layout and the files I committed
I kept the repository small and obvious because that reduces confusion during deployment. The root contains `app.py` as the Streamlit entry point, `requirements.txt` to lock the runtime, and a `models/` directory that stores the serialized estimator. A compact layout helps reviewers read the code in one sitting and helps me avoid burying logic in nested folders. The model file lives under `models/` to keep large artifacts out of the root and to leave space for logs or reports later. The structure below is exactly what I pushed to GitHub so anyone can clone and run the same tree.

```
ai_ransomware_recovery_predictor/
    app.py
    requirements.txt
    models/
        recovery_time_model.pkl
```

## requirements.txt explained in plain terms
Dependencies are precise because small version drifts cause breakage at prediction time. Streamlit renders the UI, Pandas shapes the row that flows to the model, NumPy underpins efficient numeric work, and scikit‑learn provides the regression implementation. The `skops` line is included because I sometimes share scikit‑learn models across environments; it does not change app behavior but helps with portability. Joblib is not listed directly because scikit‑learn includes it as a dependency, which simplifies the file. This section explains what each line gives me and why I fixed the versions instead of letting them float.

```python
streamlit>=1.36.0
pandas>=2.0,<2.3
numpy==1.26.4
scikit-learn==1.4.2
skops==0.9.0
```

## app.py at a glance
The application code is short by design so that a reader can hold the full logic in working memory. It loads a model, draws inputs with Streamlit widgets, builds a single‑row DataFrame, one‑hot encodes categorical columns, and asks the model for a numeric estimate. I kept variable names descriptive and close to their on‑screen labels so the mapping from UI to features is unambiguous. The model expects a fixed set of columns, and the code uses `reindex` to align the runtime row to that set. The snippet below is the exact file I committed, with no edits or placeholders.

```python
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/recovery_time_model.pkl")

st.title("AI Ransomware Recovery Time Predictor")

# User inputs
org_type = st.selectbox("Organization Type", ["Hospital","Clinic","Pharma","Insurance","Research Lab"])
org_size = st.selectbox("Organization Size", ["Small","Medium","Large"])
facilities = st.number_input("Facilities Affected", 1, 100)
infection_rate = st.slider("Ransomware Infection Rate (%)", 0, 100, 50)
backup_comp = st.selectbox("Backup Compromised?", [True, False])
data_encrypted = st.selectbox("Data Encrypted?", [True, False])
entry_method = st.selectbox("Entry Method", ["Phishing Email","Exploited Vulnerability","Compromised Credentials"])

# Prepare input
X_input = pd.DataFrame([[org_type, org_size, facilities, infection_rate, backup_comp, data_encrypted, entry_method]],
                       columns=['org_type','org_size','facilities_affected',
                                'ransomware_infection_rate_(%)','backup_compromised',
                                'data_encrypted','entry_method'])
X_input = pd.get_dummies(X_input).reindex(columns=model.feature_names_in_, fill_value=0)

# Prediction
pred = model.predict(X_input)[0]
st.success(f"Estimated Recovery Time: {pred:.1f} days")
```

## Imports block
The import trio is minimal because the app does not need anything beyond UI, data shaping, and model loading. Streamlit drives interaction and layout, Pandas gives me a reliable DataFrame API, and Joblib loads the estimator quickly without boilerplate. I did not add exotic utilities or custom helpers here because each extra import increases cognitive load. A small import block is also easier for code scanners and makes security review faster. The tradeoff is that some validations remain inline instead of being extracted into separate modules.

## Model loading block
The estimator is stored at `models/recovery_time_model.pkl` and loaded once when the app starts. This keeps the first page render a little heavier, but every interaction after that stays fast because the object remains in memory. Loading early also means the app fails fast if the file is missing or corrupted, which is good for debugging. Using a relative path keeps the repository portable across machines and free tiers. The model is a plain scikit‑learn regressor, and Joblib deserializes it in a single call.

## Title and purpose
The call to `st.title` sets the page intention before any inputs appear. Clear context keeps people from guessing what the sliders and dropdowns mean. Titles also anchor screenshots that end up in tickets or message threads because the product name is captured as text. I favor a short title here since the estimate is the star of the page. The rest of the narrative lives below in labeled controls and the result banner.

## Input widgets: why these fields exist
The dropdowns capture structure that meaningfully shifts recovery effort without drowning the user in jargon. The size of the organization moves the estimate because recovery coordination and testing scale with headcount and system count. The slider for infection rate gives a fast way to model breadth of impact without exposing raw device counts. Two boolean flags represent whether backups were compromised and whether data was encrypted, both of which tend to add days. The entry method captures operational friction, since a credential compromise often implies a wider privilege review.

## DataFrame creation block
The code takes the widget values and places them into a single row with stable column names. I do this in one call because it keeps the code compact and avoids accidental mis‑ordering of features. Naming is accurate and mirrors the labels so future contributors can map the row back to the UI without guessing. Collecting a single row keeps the mental model simple for a new reader and matches how Streamlit passes a single scenario at a time. If I later add batch predictions, the same pattern will scale to a table of rows.

## One‑hot encoding and reindexing
`pd.get_dummies` converts categories into indicator columns so the numeric model can read them. The `reindex` step is the guardrail that aligns whatever the user picked with the exact set of columns the regressor saw during training. Missing columns are filled with zeros so inference does not crash when a category is absent. Extra columns are dropped because the model does not expect them, which keeps the vector shape correct. This is the simple, explicit approach that mirrors how I trained the estimator.

## Prediction and display
The `predict` call returns a one‑element array, and I format the first value with one decimal to make the number easier to scan. Displaying with `st.success` adds a visual cue that the operation completed, but it does not imply the situation is trivial. If an unexpected value appears, the small surface area makes it easy to add a `st.write(X_input)` line for inspection during support calls. Keeping the message short respects how people read on dashboards where attention spans are small. This is the end of the flow from inputs to estimate.

## The model artifact under models/
The file `recovery_time_model.pkl` is a binary artifact produced by scikit‑learn. It contains the coefficients and trained state needed to turn a shaped row into a number. I keep it in version control only because the file is small; if it grows beyond the GitHub limit, I would store it in object storage and download it at app start. The code relies on the estimator exposing `feature_names_in_`, so the training step must fit the model with a Pandas DataFrame so the attribute is populated. For longer term work, I would record the training hash and dataset version next to the file for traceability.

## Reproducing the model: a compact training script
This script shows how I built a small synthetic dataset and trained a ridge regressor that exposes `feature_names_in_`. It uses plain `pd.get_dummies` for consistency with the app and saves the model under the same path the app expects. The synthetic generator is intentionally simple so the relationships are obvious and easy to audit. A real program would replace the generator with curated incident data that has well defined labels. I include this code to make the repository self‑contained for readers who want to rebuild the artifact.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import joblib

# --- Synthetic training data builder ---
def make_training_frame(n: int = 800, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    org_type = rng.choice(["Hospital","Clinic","Pharma","Insurance","Research Lab"], size=n)
    org_size = rng.choice(["Small","Medium","Large"], size=n, p=[0.35, 0.45, 0.20])
    facilities = rng.integers(1, 101, size=n)
    infection_rate = rng.integers(0, 101, size=n)
    backup_comp = rng.choice([True, False], size=n, p=[0.3, 0.7])
    data_encrypted = rng.choice([True, False], size=n, p=[0.55, 0.45])
    entry_method = rng.choice(["Phishing Email","Exploited Vulnerability","Compromised Credentials"], size=n)

    base = 3 + 0.08 * facilities + 0.06 * infection_rate
    base += np.where(org_size == "Large", 6, np.where(org_size == "Medium", 3, 0))
    base += np.where(org_type == "Hospital", 4, np.where(org_type == "Research Lab", 2, 0))
    base += np.where(backup_comp, 10, 0)
    base += np.where(data_encrypted, 7, 0)
    base += np.where(entry_method == "Compromised Credentials", 2.5, 0)
    noise = rng.normal(0, 2.5, size=n)
    y = np.clip(base + noise, 1, None)  # recovery time in days

    df = pd.DataFrame({
        "org_type": org_type,
        "org_size": org_size,
        "facilities_affected": facilities,
        "ransomware_infection_rate_(%)": infection_rate,
        "backup_compromised": backup_comp,
        "data_encrypted": data_encrypted,
        "entry_method": entry_method,
        "recovery_days": y,
    })
    return df

# --- Model training ---
def train_and_save(path: str = "models/recovery_time_model.pkl") -> None:
    df = make_training_frame()
    X = df.drop(columns=["recovery_days"])
    X = pd.get_dummies(X)
    y = df["recovery_days"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Ridge(alpha=1.2, random_state=42)
    model.fit(X_train, y_train)

    # Ensure feature_names_in_ exists on the estimator
    assert hasattr(model, "feature_names_in_")
    joblib.dump(model, path)

if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    train_and_save()
```

## Training code explained function by function
**make_training_frame** creates a repeatable synthetic table with columns that match the UI. It samples categories with controlled probabilities, adds counts that resemble plausible operations, and computes recovery days from a clear linear recipe plus noise. The target is clipped at one day because negative recovery makes no sense. The point is to create a dataset that exercises the model without pretending to be real-world truth. **train_and_save** transforms the frame with `get_dummies`, splits a test fold, and trains a ridge regressor with a small regularization factor. Fitting with a DataFrame means the estimator captures column names inside `feature_names_in_`, which the app uses to align encoding. The function saves the model with Joblib and creates the `models` folder if needed. This keeps the training step one command away from a fresh clone.

## Optional refactor with helpers (kept separate from the original)
I left the committed `app.py` short, but for maintainers I also wrote a small refactor that extracts helpers and explains each one. Streamlit’s `@st.cache_resource` memoizes the model so the app only deserializes once per session. A `build_input_dataframe` helper turns the widget dictionary into a DataFrame with fixed column names, and `encode_to_model_space` performs one‑hot encoding plus safe reindexing. The `predict_days` helper returns a float for easy formatting. The `main` function glues the pieces together and keeps the top level tidy.

```python
import streamlit as st
import pandas as pd
import joblib
from typing import Dict, Any

@st.cache_resource
def load_model() -> Any:
    return joblib.load("models/recovery_time_model.pkl")

def build_input_dataframe(form: Dict[str, Any]) -> pd.DataFrame:
    row = [[
        form["org_type"], form["org_size"], form["facilities_affected"], form["ransomware_infection_rate_(%)"],
        form["backup_compromised"], form["data_encrypted"], form["entry_method"]
    ]]
    cols = ["org_type","org_size","facilities_affected","ransomware_infection_rate_(%)","backup_compromised","data_encrypted","entry_method"]
    return pd.DataFrame(row, columns=cols)

def encode_to_model_space(df: pd.DataFrame, model) -> pd.DataFrame:
    X = pd.get_dummies(df)
    return X.reindex(columns=model.feature_names_in_, fill_value=0)

def predict_days(model, X: pd.DataFrame) -> float:
    return float(model.predict(X)[0])

def main() -> None:
    st.title("AI Ransomware Recovery Time Predictor")

    with st.form("inputs"):
        org_type = st.selectbox("Organization Type", ["Hospital","Clinic","Pharma","Insurance","Research Lab"])
        org_size = st.selectbox("Organization Size", ["Small","Medium","Large"])
        facilities = st.number_input("Facilities Affected", 1, 100)
        infection_rate = st.slider("Ransomware Infection Rate (%)", 0, 100, 50)
        backup_comp = st.selectbox("Backup Compromised?", [True, False])
        data_encrypted = st.selectbox("Data Encrypted?", [True, False])
        entry_method = st.selectbox("Entry Method", ["Phishing Email","Exploited Vulnerability","Compromised Credentials"])
        submitted = st.form_submit_button("Estimate")

    if submitted:
        form = {
            "org_type": org_type, "org_size": org_size, "facilities_affected": facilities,
            "ransomware_infection_rate_(%)": infection_rate, "backup_compromised": backup_comp,
            "data_encrypted": data_encrypted, "entry_method": entry_method
        }
        model = load_model()
        X_row = build_input_dataframe(form)
        X_enc = encode_to_model_space(X_row, model)
        days = predict_days(model, X_enc)
        st.success(f"Estimated Recovery Time: {days:.1f} days")

if __name__ == "__main__":
    main()
```

## Helper functions explained
**load_model** loads and caches the estimator so repeated submissions do not re-open the file. Caching here reduces file I/O and keeps the UI responsive on free tiers. **build_input_dataframe** isolates the mapping from UI values to the DataFrame schema, which makes later changes less risky and easier to test. By holding a single responsibility it becomes a natural unit to cover in tests. **encode_to_model_space** handles one-hot encoding and the strict alignment to the training columns, which is the most error-prone step during inference. The reindex call is explicit, so any drift shows up as missing columns rather than silent mistakes. **predict_days** wraps the estimator call and coerce the result into a float, which keeps formatting clean and moved out of the main routine. **main** hosts layout, form submission, and the call chain from dictionary to estimate, which keeps the top-level code clean and readable.

## Local run steps
Clone the repository, create a fresh virtual environment, and install dependencies. Running `streamlit run app.py` opens a local server that renders the widgets in your browser. I prefer to copy the `models/` folder as-is rather than rebuilding the model on the first run, because it shortens the feedback loop. When the page loads, try the extremes in the slider and dropdowns to check that the number prints and that no errors leak to the UI. This quick smoke test catches path issues and dependency mismatches immediately.

```python
# quickstart
# python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
# pip install -r requirements.txt
# streamlit run app.py
```

## Streamlit Cloud deployment notes
Create a public repository, connect Streamlit Cloud to your GitHub account, and point a new app at the repo with the working branch. The platform reads `requirements.txt`, installs the pinned versions, and launches the app with the default command. Place `recovery_time_model.pkl` under `models/` in the repository so the relative path resolves in the cloud container. Keep the artifact smaller than the 25 MB GitHub limit or store it in a bucket and download it at startup. After the first deploy, edits to the repository will trigger a rebuild and redeploy.

## Input contract and edge cases
Numeric inputs are bounded in the widget to avoid invalid counts or percentages. The encoding step fills missing categories with zeros, so the prediction call remains stable even if a certain category was never seen in the training frame. If you later add a new category to the UI, retrain and commit a new model so `feature_names_in_` includes the fresh column name. Out-of-range slider values are impossible by design because Streamlit enforces the bounds. These choices prevent the most common user errors without adding heavy validation code.

## Testing approach that fits a single-file app
I rely on a small set of tests that exercise the helpers in the refactor module and a couple of golden inputs for the original script. One test builds a dictionary, runs it through `build_input_dataframe` and `encode_to_model_space`, and asserts on the final column set shape. Another test feeds an obviously heavy scenario and checks that the prediction is above a threshold. A final test loads the app in headless mode and ensures the top level imports without throwing. This gives me confidence without introducing a full web test harness.

## Troubleshooting checklist
If the page loads but the estimate does not appear, first check the terminal logs for a path error on the model file. Next ensure that the model exposes `feature_names_in_`; if not, rebuild the artifact using a DataFrame during `fit`. If the app fails to import `joblib`, upgrade scikit‑learn to the pinned version from `requirements.txt`. When the page is blank on Streamlit Cloud, review the build logs for a dependency pin conflict or a Python version mismatch. Most issues resolve by aligning the training and inference encoding steps and by verifying the model path.

## Security and data handling notes
The app collects no credentials and writes no user inputs to disk. It performs a single in‑memory prediction and clears state on reload. If loaded with real incident data for batch scoring, I would add a redaction step that strips organization identifiers and reduces stored fields to the minimum set. The open nature of Streamlit calls for careful review before adding uploads or persistent logs. For now the design is intentionally minimal to avoid surprises.

## Performance considerations
The estimator is small and predictions are instantaneous even on free CPUs. Encoding dominates the runtime, which is fine because the row is a handful of features. Caching the model removes repeated disk I/O, and using a short dependency list keeps container build times low. If the app later adds batch scoring of thousands of rows, I would switch to vectorized prediction and remove per‑row loops. Those changes are orthogonal to the single‑scenario interface shown here.

## Roadmap ideas
A future iteration could surface percentile bands by training an error model on residuals and sampling a simple distribution around the mean estimate. Another idea is to move from a single number to a small confidence band to better communicate uncertainty. I could also add a short narrative explaining why a scenario looks slow and which factors drive the estimate. The UI can remain small while giving more decision support. Each addition would arrive with the same discipline applied here: small code, pinned dependencies, and testable helpers.

## Closing reflection
This project started from a practical need to replace ambiguity with a steady estimate. A short Streamlit page and a clear model do not solve an incident, yet they create a shared frame for decisions about staffing and risk. The design is intentionally boring because boring software is easier to trust in a stressful moment. By documenting each file and each block, I make the work easy to audit and easy to evolve. That is the kind of tool I want on a shelf when the call comes in.
