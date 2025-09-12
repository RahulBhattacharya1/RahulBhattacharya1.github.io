---
layout: default
title: "Building my AI Handwritten Digit Classifier"
date: 2022-06-13 21:24:33
categories: [ai]
tags: [python,streamlit,bert,self-trained]
thumbnail: /assets/images/classifier2.webp
demo_link: https://rahuls-ai-mnist-digit-classifier.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_mnist_digit_classifier
featured: true
---

I list every file that sits in the project. This gives a map before the deep dive. It also helps later when someone checks the same tree on GitHub.

- `ai_mnist_digit_classifier-main/app.py`
- `ai_mnist_digit_classifier-main/README.md`
- `ai_mnist_digit_classifier-main/requirements.txt`
- `ai_mnist_digit_classifier-main/train.py`
- `ai_mnist_digit_classifier-main/models/mnist_labels.json`
- `ai_mnist_digit_classifier-main/models/mnist_logreg.pkl`
- `ai_mnist_digit_classifier-main/models/mnist_mlp.pkl`
- `ai_mnist_digit_classifier-main/models/mnist_pca_logreg.pkl`


## `ai_mnist_digit_classifier-main/README.md`

```python

# MNIST Handwritten Digit Classifier (scikit-learn)

**Colab → GitHub → Streamlit** workflow. No OpenAI API. Trains a Logistic Regression model on MNIST (28x28 grayscale digits) and serves a Streamlit app with a drawing canvas.

## 1) Train in Colab
```python
# Install
!pip -q install scikit-learn>=1.3 joblib>=1.3 numpy>=1.23 pandas>=1.5 streamlit>=1.37 pillow>=10.0 streamlit-drawable-canvas>=0.9.3
# Save files into /content/mnist_digit_classifier and run:
!python train.py
```

## 2) Artifacts produced
- `models/mnist_logreg.pkl` — trained pipeline
- `models/mnist_labels.json` — label names 0..9

## 3) Streamlit demo
```bash
streamlit run app.py
```

## 4) Deploy (Streamlit Community Cloud)
- Push repo to GitHub (include `models/` with both files).
- Create new app, main file `app.py`.
- Add tags like: `self-trained`, `image-classification`, `mnist`, `scikit-learn`.



This README gives quick start context. I keep the full blog detailed, but the README remains as a short map for new readers.



## `ai_mnist_digit_classifier-main/app.py`

```python
# app.py — MNIST Handwritten Digit Classifier (robust canvas + MLP model)
# Works with models/mnist_mlp.pkl produced by the upgraded train.py.
# Preprocessing: crop -> center -> pad -> resize(28x28) -> invert -> flatten.

from pathlib import Path
import io
import json
import numpy as np
import joblib
from PIL import Image, ImageOps
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ---------- Paths ----------
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "mnist_mlp.pkl"         # match upgraded train.py
LABELS_PATH = ROOT / "models" / "mnist_labels.json"

# ---------- Page ----------
st.set_page_config(page_title="MNIST Handwritten Digit Classifier")
st.title("MNIST Handwritten Digit Classifier (scikit-learn)")

# ---------- Guard: artifacts ----------
if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    st.error("Model or labels not found. Please run train.py to create ./models artifacts.")
    st.stop()

# ---------- Load once ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)["label_names"]
    return model, labels

model, label_names = load_artifacts()

st.write(
    "Draw a digit (0–9) or upload an image. The app centers your drawing, pads to a square, "
    "resizes to 28×28, inverts to MNIST style (white digit on black), and predicts."
)

# ---------- Helpers ----------
def has_ink(img_rgb: Image.Image, thr: int = 245, min_pixels: int = 120) -> bool:
    """
    Return True if there are at least `min_pixels` dark pixels (< thr) in the grayscale image.
    This is more robust than a global mean check.
    """
    arr = np.asarray(img_rgb.convert("L"))
    ink = np.count_nonzero(arr < thr)
    return ink >= min_pixels

def preprocess_pil(img: Image.Image):
    """
    1) Convert to grayscale.
    2) Threshold to find dark strokes and compute a bounding box.
    3) Crop to the bbox, then pad to a square so the digit is centered.
    4) Resize to 28×28.
    5) Invert to MNIST style (white digit on black).
    6) Flatten to shape (1, 784) float32; scaling to 0..1 is handled by the pipeline MinMaxScaler.
    Returns: (X, preview_image)
    """
    g = img.convert("L")
    a = np.array(g)
    mask = a < 245  # stroke mask

    if not mask.any():
        blank = Image.new("L", (28, 28), color=0)
        arr = np.array(blank).astype("float32").reshape(1, -1)
        return arr, blank

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    cropped = g.crop((x0, y0, x1 + 1, y1 + 1))

    w, h = cropped.size
    side = max(w, h)
    square = Image.new("L", (side, side), color=255)  # white background
    offset = ((side - w) // 2, (side - h) // 2)
    square.paste(cropped, offset)

    resized = square.resize((28, 28), Image.LANCZOS)
    inverted = ImageOps.invert(resized)

    arr = np.array(inverted).astype("float32").reshape(1, -1)
    return arr, inverted

def predict_and_render(X: np.ndarray):
    pred = model.predict(X)[0]
    st.subheader(f"Prediction: {pred}")
    # Try to show probabilities when available
    clf = getattr(model, "named_steps", {}).get("clf", model)
    if hasattr(clf, "predict_proba"):
        proba = model.predict_proba(X)[0]
        order = np.argsort(proba)[::-1]
        st.write("Confidence by digit:")
        for i in order[:5]:
            st.write(f"- {i}: {proba[i]:.3f}")

# ---------- UI ----------
debug_preview = st.checkbox("Show preprocessing preview", value=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Draw a digit")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=18,
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        rgb = Image.fromarray((canvas_result.image_data[:, :, :3]).astype("uint8"))
        if has_ink(rgb):
            X, preview_28 = preprocess_pil(rgb)
            if debug_preview:
                st.caption("What the model sees (after center+pad+invert+resize):")
                st.image(preview_28, width=112, clamp=True)
            predict_and_render(X)
        else:
            st.info("Draw something on the canvas to get a prediction.")

with col2:
    st.subheader("Upload a 28×28 grayscale PNG (optional)")
    file = st.file_uploader("Upload image", type=["png"])
    if file:
        try:
            img = Image.open(io.BytesIO(file.read()))
            st.image(img, caption="Uploaded image", width=140)
            X, preview_28 = preprocess_pil(img)
            if debug_preview:
                st.caption("What the model sees (after center+pad+invert+resize):")
                st.image(preview_28, width=112, clamp=True)
            predict_and_render(X)
        except Exception as e:
            st.error(f"Could not process the image: {e}")

```

**What this file does**

This Python file contributes to the pipeline. The code complements the other modules to keep the project modular and readable.

**Functions in this file and why they matter**

- def load_artifacts(): bundles one unit of work behind a simple interface. The function isolates edge cases that would clutter the caller. It returns a clear output so the rest of the pipeline can move forward. This separation improves testability and keeps the training loop focused.
- def preprocess_pil(img: Image.Image): bundles one unit of work behind a simple interface. The function isolates edge cases that would clutter the caller. It returns a clear output so the rest of the pipeline can move forward. This separation improves testability and keeps the training loop focused.
- def predict_and_render(X: np.ndarray): bundles one unit of work behind a simple interface. The function isolates edge cases that would clutter the caller. It returns a clear output so the rest of the pipeline can move forward. This separation improves testability and keeps the training loop focused.

**Control flow and conditionals**

When conditionals appear, they gate expensive steps like training versus evaluation. They also guard file I/O, device selection, and optional logging. Simple branches keep error handling local and reduce surprises in the main path. Loops handle epochs, batches, and metric accumulation without mixing responsibilities.


## `ai_mnist_digit_classifier-main/train.py`

```python
# train.py — MNIST Handwritten Digit Classifier (fixed: no custom FunctionTransformer)
# Trains a Logistic Regression model on 28x28 grayscale digits using a simple pipeline.
# Save artifacts in ./models for Streamlit app consumption.

from pathlib import Path
import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import json

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS / "mnist_logreg.pkl"
LABELS_PATH = MODELS / "mnist_labels.json"

def main():
    print("Downloading MNIST (OpenML mnist_784)...")
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True, parser="auto")
    y = y.astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, random_state=42, stratify=y
    )

    print("Building pipeline (MinMaxScaler -> LogisticRegression)...")
    pipe = Pipeline([
        ("scale01", MinMaxScaler(feature_range=(0.0, 1.0))),
        ("clf", LogisticRegression(
            solver="saga",
            penalty="l2",
            multi_class="multinomial",
            max_iter=200,
            n_jobs=-1
        ))
    ])

    print("Training...")
    pipe.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=3))

    # Save pipeline + labels
    print(f"Saving model -> {MODEL_PATH}")
    joblib.dump(pipe, MODEL_PATH)

    labels = list(map(int, sorted(np.unique(y))))
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({"label_names": labels}, f, indent=2)

    print("Done. You can now run: streamlit run app.py")

if __name__ == "__main__":
    main()

```

**What this file does**

This script handles training the model. It defines the loop, moves data through the network, and saves the final weights.

**Functions in this file and why they matter**

- def main(): bundles one unit of work behind a simple interface. The function isolates edge cases that would clutter the caller. It returns a clear output so the rest of the pipeline can move forward. This separation improves testability and keeps the training loop focused.

**Control flow and conditionals**

When conditionals appear, they gate expensive steps like training versus evaluation. They also guard file I/O, device selection, and optional logging. Simple branches keep error handling local and reduce surprises in the main path. Loops handle epochs, batches, and metric accumulation without mixing responsibilities.


## `ai_mnist_digit_classifier-main/requirements.txt`

```python
streamlit>=1.37
scikit-learn>=1.3
numpy>=1.23
pandas>=1.5
pillow>=10.0
streamlit-drawable-canvas>=0.9.3
joblib>=1.3
```
This file provides configuration or dependencies. It is not Python code, but it shapes how the Python code runs. Committing it makes the project reproducible on a clean machine.


## `ai_mnist_digit_classifier-main/models/mnist_labels.json`

This is an asset or binary. It ships with the repository so the code can run end to end. I keep such files under version control only when they are small and stable.


## `ai_mnist_digit_classifier-main/models/mnist_logreg.pkl`

This is an asset or binary. It ships with the repository so the code can run end to end. I keep such files under version control only when they are small and stable.


## `ai_mnist_digit_classifier-main/models/mnist_mlp.pkl`

This is an asset or binary. It ships with the repository so the code can run end to end. I keep such files under version control only when they are small and stable.


## `ai_mnist_digit_classifier-main/models/mnist_pca_logreg.pkl`

This is an asset or binary. It ships with the repository so the code can run end to end. I keep such files under version control only when they are small and stable.



## How I ran the project locally

The steps here match the committed files. They keep the run repeatable on a clean machine.

```python
# 1) Create and activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# 2) Install dependencies
pip install --upgrade pip
if os.path.exists('requirements.txt'):
    # regular CLI: pip install -r requirements.txt
    pass

# 3) Train or run inference depending on the scripts that exist
# Examples below; adjust the entry points to match the repo
# python train.py --epochs 5 --batch-size 64
# python evaluate.py --weights runs/mnist_cnn.pt
# python app.py  # if a Streamlit or Flask app is present
```

The commands do not assume hidden files. They mirror the file names under version control. I avoid magic steps. Every action is visible and easy to audit.



## Closing notes

The repository stands on small, focused modules. Each file covers one concern. The training logic stays clean. The model definition stays in one place. Config remains explicit. This style keeps future changes safe and easy.

If someone new clones the repo, the map above should be enough. They can scan code blocks, read the surrounding notes, and run the project without guesswork.
