---
layout: default
title: "Creating my AI Semiconductor Wafer Defect Classification App"
date: 2025-09-14 15:43:29
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/wafer.webp
thumbnail_mobile: /assets/images/wafer_sq.webp
demo_link: https://rahuls-ai-semiconductor-defect-classification.streamlit.app
github_link: https://github.com/RahulBhattacharya1/semiconductor_defect_classification
featured: true
---

I once visited a manufacturing floor where large silicon wafers were stacked in clean trays. Each wafer carried a story of precision and fragility. A minor scratch or a faint ring pattern could mean the difference between thousands of functional chips and wasted effort. Watching that process made me think about how engineers and analysts could benefit from fast tools that reveal defect patterns. That curiosity led me to build a small application that simulates wafer maps, predicts defect types, and displays dashboards to analyze performance.

The project became a learning experience about connecting machine learning with visualization. I imagined a scenario where quick insights about wafer quality could help in training models, teaching students, or prototyping research ideas. Streamlit provided a way to build interactive pages. Python libraries helped generate synthetic data, extract features, train models, and visualize outcomes. In this blog, I will describe every part of the project, file by file and block by block. The goal is to make the explanation clear enough that someone else could recreate it with the same files in their own GitHub repository.

## Main Application (app/app.py)

```python
# app/app.py
# Wafer Map Defect Classifier – Streamlit demo
# - Fixes st.pyplot misuse (uses use_container_width=True)
# - Adds robust error handling and a light heuristic fallback “model”
# - Works in two modes: Generate (synthetic) and Upload (.npy or image)

from __future__ import annotations

import io
import os
import sys
import json
import math
from typing import Tuple, Optional

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------
# App / page config
# -------------------------------
st.set_page_config(
    page_title="Wafer Map Defect Classifier",
    page_icon="🧪",
    layout="wide",
)

TITLE = "Wafer Map Defect Classifier"
CLASSES = ["center", "edge_ring", "scratch", "donut", "random"]

# -------------------------------
# Utilities
# -------------------------------

def show_versions_banner(expected: dict[str, str] | None = None) -> None:
    """Optionally warn if critical packages differ from what you pinned."""
    try:
        import numpy as _np
        import matplotlib as _mpl
        vers = {"numpy": _np.__version__, "matplotlib": _mpl.__version__, "python": ".".join(map(str, sys.version_info[:3]))}
        if expected:
            mismatch = {k: (vers.get(k, "?"), v) for k, v in expected.items() if vers.get(k) != v}
            if mismatch:
                st.info(f"Environment versions: {vers}. Expected: {expected}.")
        else:
            st.caption(f"Env: numpy {vers['numpy']} • matplotlib {vers['matplotlib']} • Python {vers['python']}")
    except Exception:
        pass


def as_uint8(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return (x * 255).astype(np.uint8)


def draw_wafer(ax: plt.Axes, arr: np.ndarray, title: str = "Input") -> None:
    ax.imshow(arr, cmap="viridis", interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


def gaussian_2d(h: int, w: int, cx: float, cy: float, sx: float, sy: float) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    return np.exp(-(((x - cx) ** 2) / (2 * sx ** 2) + ((y - cy) ** 2) / (2 * sy ** 2)))


def synthesize_wafer(label: str, size: int = 64, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 0.15, (size, size)).astype(np.float32)

    if label == "center":
        base += 1.8 * gaussian_2d(size, size, size / 2, size / 2, size / 6, size / 6)

    elif label == "edge_ring":
        g_outer = gaussian_2d(size, size, size / 2, size / 2, size / 1.9, size / 1.9)
        g_inner = gaussian_2d(size, size, size / 2, size / 2, size / 3.2, size / 3.2)
        base += 1.4 * (g_outer - g_inner)

    elif label == "scratch":
        # thin diagonal line
        for i in range(size):
            j = int(0.2 * size + 0.6 * i) % size
            base[i, max(0, j - 1):min(size, j + 2)] += 1.8
        base = (base + rng.normal(0, 0.05, base.shape)).astype(np.float32)

    elif label == "donut":
        g_outer = gaussian_2d(size, size, size / 2, size / 2, size / 5, size / 5)
        g_center = gaussian_2d(size, size, size / 2, size / 2, size / 10, size / 10)
        base += 2.0 * (g_outer - g_center)

    elif label == "random":
        base += rng.normal(0.0, 0.6, base.shape).astype(np.float32)

    # clip + scale
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)
    return base


# -------------------------------
# Lightweight heuristic classifier
# -------------------------------

def radial_profile(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y, x = np.mgrid[0:h, 0:w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rbin = np.clip(r.astype(int), 0, max(h, w))
    prof = np.bincount(rbin.ravel(), weights=img.ravel(), minlength=rbin.max() + 1)
    cnts = np.bincount(rbin.ravel(), minlength=rbin.max() + 1) + 1e-8
    return (prof / cnts)[: (min(h, w) // 2 + 1)]


def simple_features(img: np.ndarray) -> dict:
    rp = radial_profile(img)
    center_mean = img[img.shape[0]//2 - 4:img.shape[0]//2 + 4, img.shape[1]//2 - 4:img.shape[1]//2 + 4].mean()
    edge_mean = np.mean([rp[-3:], rp[-6:-3]])
    ringness = (rp[int(len(rp)*0.75)] - rp[int(len(rp)*0.45)])
    donut_gap = rp[int(len(rp)*0.35)] - rp[int(len(rp)*0.15)]
    var = img.var()
    # scratch detector via Sobel-like gradient energy
    gy, gx = np.gradient(img)
    diag_energy = np.mean((gx + gy) ** 2)
    return {
        "center_mean": float(center_mean),
        "edge_mean": float(edge_mean),
        "ringness": float(ringness),
        "donut_gap": float(donut_gap),
        "var": float(var),
        "diag_energy": float(diag_energy),
    }


def heuristic_predict(img: np.ndarray) -> Tuple[str, dict]:
    f = simple_features(img)
    scores = {k: 0.0 for k in CLASSES}
    # center: high center_mean relative to edge
    scores["center"] = f["center_mean"] * 3.0 - f["edge_mean"]
    # edge_ring: high edge relative to mid + some variance
    scores["edge_ring"] = f["ringness"] * 4.0 + 0.3 * f["var"]
    # scratch: strong diagonal gradient energy
    scores["scratch"] = f["diag_energy"] * 5.0 + 0.2 * f["var"]
    # donut: strong outer vs inner contrast with inner dip
    scores["donut"] = (f["ringness"] * 3.0 + max(0.0, -f["donut_gap"]) * 2.0)
    # random: high variance but weak structure → baseline on var
    scores["random"] = 0.8 * f["var"]

    # softmax-ish
    arr = np.array(list(scores.values()), dtype=np.float32)
    exp = np.exp(arr - arr.max())
    probs = exp / (exp.sum() + 1e-8)
    pred_idx = int(np.argmax(probs))
    pred = CLASSES[pred_idx]
    out = {c: float(p) for c, p in zip(CLASSES, probs)}
    return pred, out


# -------------------------------
# Optional: load a real model if present
# -------------------------------

def try_load_model(path: str = "model.pkl"):
    """Return (predict_fn, label) where predict_fn(img)->(pred, probs)."""
    if not os.path.isfile(path):
        return None, "fallback-heuristic (no model found)"
    try:
        import joblib
        model = joblib.load(path)

        def _predict(img: np.ndarray) -> Tuple[str, dict]:
            x = img.astype(np.float32).reshape(1, -1)
            prob = model.predict_proba(x)[0]
            idx = int(np.argmax(prob))
            return CLASSES[idx], {c: float(p) for c, p in zip(CLASSES, prob)}
        return _predict, "joblib-model"
    except Exception as e:
        st.warning(f"Model load failed ({e}). Falling back to heuristic.")
        return None, "fallback-heuristic (load error)"


# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.header("Input")

mode = st.sidebar.radio("Input Mode", ["Generate", "Upload"], index=0)
default_class = st.sidebar.selectbox("Default class", CLASSES, index=0)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)

# -------------------------------
# Main title
# -------------------------------
st.title(TITLE)
st.caption("Synthetic demo for yield engineering: center, edge_ring, scratch, donut, random")

show_versions_banner()

# -------------------------------
# Prepare input wafer map
# -------------------------------
img: Optional[np.ndarray] = None

col_input, col_pred = st.columns([1, 1])

with col_input:
    st.subheader("Wafer Map")

    if mode == "Generate":
        img = synthesize_wafer(default_class, size=64, seed=int(seed))
    else:
        up = st.file_uploader("Upload wafer (.npy grayscale 2D) or image (PNG/JPG)", type=["npy", "png", "jpg", "jpeg"])
        if up is not None:
            try:
                if up.name.lower().endswith(".npy"):
                    arr = np.load(io.BytesIO(up.read()))
                    if arr.ndim == 3:
                        arr = arr.mean(axis=2)  # to grayscale
                    img = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                else:
                    from PIL import Image
                    im = Image.open(up).convert("L").resize((64, 64))
                    img = np.asarray(im, dtype=np.float32) / 255.0
            except Exception as e:
                st.error(f"Failed to read upload: {e}")

    # Always draw something (placeholder if none)
    fig, ax = plt.subplots(layout="constrained", figsize=(4, 4))
    if img is None:
        placeholder = np.zeros((64, 64), dtype=np.float32)
        draw_wafer(ax, placeholder, title="Waiting for input…")
    else:
        draw_wafer(ax, img, title="Input")

    # ✅ FIX: use_container_width instead of width="content"
    st.pyplot(fig, use_container_width=True)

# -------------------------------
# Prediction
# -------------------------------
with col_pred:
    st.subheader("Prediction")

    if img is None:
        st.info("Provide an input to see predictions.")
    else:
        predict_fn, model_label = try_load_model()
        if predict_fn is None:
            pred, probs = heuristic_predict(img)
        else:
            pred, probs = predict_fn(img)

        st.markdown(f"**Predicted class:** `{pred}`  \n_Model:_ {model_label}")

        # bar chart of probabilities
        fig2, ax2 = plt.subplots(layout="constrained", figsize=(5, 3))
        ax2.bar(list(probs.keys()), list(probs.values()))
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("probability")
        ax2.set_title("Class probabilities")
        for i, (k, v) in enumerate(probs.items()):
            ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        st.pyplot(fig2, use_container_width=True)

        # show raw features to help debug (optional)
        with st.expander("Debug: features (heuristic)"):
            st.json(simple_features(img))
```



This file defines the entry point of the Streamlit app. It handles input, either by generating a synthetic wafer image or by uploading a file. The file contains functions to normalize images, draw wafers using matplotlib, and predict classes using either a loaded model or a heuristic fallback. Each function is written to ensure the app does not break if the model file is missing. The app also uses `st.pyplot` with `use_container_width=True` so figures fit correctly. The sidebar allows mode selection and seed configuration. The main area shows the wafer plot and prediction results side by side. The prediction output includes both the predicted class and the probability distribution across all classes.



## Dashboard Page (app/pages/1_📈_Dashboard.py)



```python
from __future__ import annotations

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard • Wafer Map Defect Classifier", layout="wide")

st.title("Dashboard")
st.caption("Model-free demo dashboard with synthetic metrics for the wafer classes.")

CLASSES = ["center", "edge_ring", "scratch", "donut", "random"]

# Controls
colA, colB, colC = st.columns(3)
with colA:
    n_samples = st.slider("Samples", 100, 5000, 1000, step=100)
with colB:
    rng_seed = st.number_input("Seed", 0, 99999, 42, step=1)
with colC:
    show_conf = st.checkbox("Show confusion matrix", True)

rng = np.random.default_rng(int(rng_seed))

# Fake per-class support and accuracy just to render charts without a trained model.
support = rng.integers(low=max(10, n_samples // 40), high=max(20, n_samples // 10), size=len(CLASSES))
support = (support / support.sum() * n_samples).astype(int)
per_class_acc = np.clip(rng.normal(loc=0.82, scale=0.08, size=len(CLASSES)), 0.4, 0.98)

# Summary metrics
overall_acc = float(np.average(per_class_acc, weights=support))
st.metric("Overall accuracy (demo)", f"{overall_acc*100:.1f}%")

# Bar: per-class accuracy
fig1, ax1 = plt.subplots(layout="constrained", figsize=(6, 3))
ax1.bar(CLASSES, per_class_acc)
ax1.set_ylim(0, 1)
ax1.set_ylabel("accuracy")
ax1.set_title("Per-class accuracy (demo)")
for i, v in enumerate(per_class_acc):
    ax1.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
st.pyplot(fig1, use_container_width=True)  # ✅ correct usage

# Support chart
fig2, ax2 = plt.subplots(layout="constrained", figsize=(6, 3))
ax2.bar(CLASSES, support)
ax2.set_ylabel("samples")
ax2.set_title("Per-class support (demo)")
st.pyplot(fig2, use_container_width=True)  # ✅ correct usage

# Confusion matrix (demo)
if show_conf:
    # Construct a plausible confusion matrix consistent with the supports and accuracies
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=float)
    for i, s in enumerate(support):
        tp = int(round(s * per_class_acc[i]))
        fp = s - tp
        cm[i, i] = tp
        if fp > 0:
            off = rng.dirichlet(np.ones(len(CLASSES) - 1)) * fp
            cm[i, np.arange(len(CLASSES)) != i] = off
    # Normalize row-wise to show rates
    row_sum = cm.sum(axis=1, keepdims=True) + 1e-9
    cm_norm = cm / row_sum

    fig3, ax3 = plt.subplots(layout="constrained", figsize=(6, 6))
    im = ax3.imshow(cm_norm, interpolation="nearest")
    ax3.set_title("Confusion matrix (row-normalized, demo)")
    ax3.set_xticks(range(len(CLASSES))); ax3.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax3.set_yticks(range(len(CLASSES))); ax3.set_yticklabels(CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax3.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=9)
    st.pyplot(fig3, use_container_width=True)  # ✅ correct usage
```



The dashboard page creates an overview of model performance. In this project, synthetic metrics are generated for demonstration. The page includes a slider for sample size, a number input for random seed, and a checkbox to toggle confusion matrix display. The script generates random per-class accuracy and support counts, then plots bar charts for both. It also creates a row-normalized confusion matrix for visualization. Streamlit metrics show the overall accuracy. Each chart is plotted with matplotlib and displayed with `st.pyplot`. This page shows how to quickly summarize classification performance even when the numbers are simulated.



## Model Analysis Page (app/pages/2_🔎_Model_Analysis.py)



```python
# app/pages/2_🔎_Model_Analysis.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Model Analysis • Wafer Map", layout="wide")
st.title("🔎 Model Analysis")
st.caption("Synthetic, self-contained analysis to keep the page working even without a trained model.")

CLASSES = ["center", "edge_ring", "scratch", "donut", "random"]

# -------------------------------
# Minimal synth + heuristic (match main app behavior)
# -------------------------------
def gaussian_2d(h, w, cx, cy, sx, sy):
    y, x = np.mgrid[0:h, 0:w]
    return np.exp(-(((x - cx) ** 2) / (2 * sx ** 2) + ((y - cy) ** 2) / (2 * sy ** 2)))

def synthesize_wafer(label: str, size: int = 64, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    base = rng.normal(0.0, 0.15, (size, size)).astype(np.float32)

    if label == "center":
        base += 1.8 * gaussian_2d(size, size, size / 2, size / 2, size / 6, size / 6)
    elif label == "edge_ring":
        g_outer = gaussian_2d(size, size, size / 2, size / 2, size / 1.9, size / 1.9)
        g_inner = gaussian_2d(size, size, size / 2, size / 2, size / 3.2, size / 3.2)
        base += 1.4 * (g_outer - g_inner)
    elif label == "scratch":
        for i in range(size):
            j = int(0.2 * size + 0.6 * i) % size
            base[i, max(0, j - 1):min(size, j + 2)] += 1.8
        base = (base + rng.normal(0, 0.05, base.shape)).astype(np.float32)
    elif label == "donut":
        g_outer = gaussian_2d(size, size, size / 2, size / 2, size / 5, size / 5)
        g_center = gaussian_2d(size, size, size / 2, size / 2, size / 10, size / 10)
        base += 2.0 * (g_outer - g_center)
    elif label == "random":
        base += rng.normal(0.0, 0.6, base.shape).astype(np.float32)

    base = (base - base.min()) / (base.max() - base.min() + 1e-8)
    return base

def radial_profile(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y, x = np.mgrid[0:h, 0:w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rbin = np.clip(r.astype(int), 0, max(h, w))
    prof = np.bincount(rbin.ravel(), weights=img.ravel(), minlength=rbin.max() + 1)
    cnts = np.bincount(rbin.ravel(), minlength=rbin.max() + 1) + 1e-8
    return (prof / cnts)[: (min(h, w) // 2 + 1)]

def simple_features(img: np.ndarray) -> dict:
    rp = radial_profile(img)
    center_mean = img[img.shape[0]//2 - 4:img.shape[0]//2 + 4, img.shape[1]//2 - 4:img.shape[1]//2 + 4].mean()
    edge_mean = np.mean([rp[-3:], rp[-6:-3]])
    ringness = (rp[int(len(rp)*0.75)] - rp[int(len(rp)*0.45)])
    donut_gap = rp[int(len(rp)*0.35)] - rp[int(len(rp)*0.15)]
    var = img.var()
    gy, gx = np.gradient(img)
    diag_energy = np.mean((gx + gy) ** 2)
    return {
        "center_mean": float(center_mean),
        "edge_mean": float(edge_mean),
        "ringness": float(ringness),
        "donut_gap": float(donut_gap),
        "var": float(var),
        "diag_energy": float(diag_energy),
    }

def heuristic_predict(img: np.ndarray) -> str:
    f = simple_features(img)
    scores = {k: 0.0 for k in CLASSES}
    scores["center"] = f["center_mean"] * 3.0 - f["edge_mean"]
    scores["edge_ring"] = f["ringness"] * 4.0 + 0.3 * f["var"]
    scores["scratch"] = f["diag_energy"] * 5.0 + 0.2 * f["var"]
    scores["donut"] = (f["ringness"] * 3.0 + max(0.0, -f["donut_gap"]) * 2.0)
    scores["random"] = 0.8 * f["var"]
    arr = np.array(list(scores.values()), dtype=np.float32)
    return CLASSES[int(np.argmax(arr))]

# -------------------------------
# Controls
# -------------------------------
left, right = st.columns([1, 1])
with left:
    per_class = st.slider("Samples per class", 20, 500, 120, step=20)
with right:
    seed = st.number_input("Seed", 0, 99999, 7, step=1)

rng = np.random.default_rng(int(seed))

# -------------------------------
# Run synthetic evaluation
# -------------------------------
y_true, y_pred = [], []
for c in CLASSES:
    for _ in range(per_class):
        img = synthesize_wafer(c, size=64, rng=rng)
        y_true.append(c)
        y_pred.append(heuristic_predict(img))

# Confusion matrix
cm = pd.crosstab(pd.Series(y_true, name="true"),
                 pd.Series(y_pred, name="pred"),
                 dropna=False).reindex(index=CLASSES, columns=CLASSES, fill_value=0)

# Normalize rows to rates for visualization
cm_rates = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)

# -------------------------------
# Plots / Tables
# -------------------------------
c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("Confusion Matrix (counts)")
    fig1, ax1 = plt.subplots(layout="constrained", figsize=(6, 6))
    im = ax1.imshow(cm.values, interpolation="nearest", cmap="Blues")
    ax1.set_xticks(range(len(CLASSES))); ax1.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax1.set_yticks(range(len(CLASSES))); ax1.set_yticklabels(CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax1.text(j, i, str(cm.values[i, j]), ha="center", va="center", fontsize=9)
    fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    st.pyplot(fig1, use_container_width=True)  # ✅ correct

with c2:
    st.subheader("Confusion Matrix (row-normalized)")
    fig2, ax2 = plt.subplots(layout="constrained", figsize=(6, 6))
    im2 = ax2.imshow(cm_rates.values, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax2.set_xticks(range(len(CLASSES))); ax2.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax2.set_yticks(range(len(CLASSES))); ax2.set_yticklabels(CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax2.text(j, i, f"{cm_rates.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    st.pyplot(fig2, use_container_width=True)  # ✅ correct

st.subheader("Confusion Matrix (table)")
# ✅ FIX: use_container_width instead of width="stretch"
st.dataframe(cm.style.background_gradient(cmap="Blues"), use_container_width=True)
```



The model analysis page runs a small experiment. It generates synthetic wafer images for each class, runs heuristic predictions, and builds a confusion matrix. The page displays the confusion matrix in three forms: raw counts, normalized heatmap, and styled dataframe. It explains model weaknesses by showing where misclassifications occur. This structure is helpful for teaching or for debugging new models. The code uses pandas crosstab to compute the confusion matrix and matplotlib to visualize it. The dataframe is styled with a gradient background. All figures use `use_container_width=True` to avoid layout errors.



## Data Generation (src/generate_data.py)



```python
import numpy as np

CLASSES = ["center", "edge_ring", "scratch", "donut", "random"]

def _disk(h, w, cx, cy, r):
    Y, X = np.ogrid[:h, :w]
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2

def _ring(h, w, cx, cy, r1, r2):
    Y, X = np.ogrid[:h, :w]
    rr = (X - cx) ** 2 + (Y - cy) ** 2
    return (rr >= r1 ** 2) & (rr <= r2 ** 2)

def _line(h, w, angle_deg=0, thickness=2):
    img = np.zeros((h, w), dtype=bool)
    angle = np.deg2rad(angle_deg)
    cx, cy = w // 2, h // 2
    for t in range(-w, w):
        x = int(cx + t * np.cos(angle))
        y = int(cy + t * np.sin(angle))
        for k in range(-thickness, thickness + 1):
            xx = x - int(k * np.sin(angle))
            yy = y + int(k * np.cos(angle))
            if 0 <= xx < w and 0 <= yy < h:
                img[yy, xx] = True
    return img

def synth_wafer(h=28, w=28, kind="center", seed=None):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), dtype=float)
    img += rng.normal(0, 0.03, size=(h, w))

    cx, cy = w // 2, h // 2
    if kind == "center":
        mask = _disk(h, w, cx, cy, r=rng.integers(h//8, h//4))
        img[mask] += rng.uniform(0.55, 0.9)
    elif kind == "edge_ring":
        r1 = rng.integers(h//3, h//2 - 3)
        r2 = r1 + rng.integers(1, 3)
        mask = _ring(h, w, cx, cy, r1, r2)
        img[mask] += rng.uniform(0.55, 0.9)
    elif kind == "scratch":
        ang = rng.uniform(0, 180)
        mask = _line(h, w, angle_deg=ang, thickness=rng.integers(1, 3))
        img[mask] += rng.uniform(0.55, 0.9)
    elif kind == "donut":
        r1 = rng.integers(h//6, h//4)
        r2 = r1 + rng.integers(2, 4)
        mask = _ring(h, w, cx, cy, r1, r2)
        img[mask] += rng.uniform(0.55, 0.9)
        hole = _disk(h, w, cx, cy, r=r1 - 1)
        img[hole] -= rng.uniform(0.2, 0.4)
    elif kind == "random":
        for _ in range(rng.integers(2, 5)):
            rr = rng.integers(h//10, h//4)
            mx = _disk(h, w, rng.integers(rr, w-rr), rng.integers(rr, h-rr), rr)
            img[mx] += rng.uniform(0.45, 0.85)
    else:
        raise ValueError(f"Unknown kind: {kind}")

    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    return img

def make_dataset(n=500, seed=0, classes=CLASSES):
    rng = np.random.default_rng(seed)
    imgs, ys = [], []
    for i in range(n):
        cls = classes[i % len(classes)]
        img = synth_wafer(kind=cls, seed=int(rng.integers(0, 1e9)))
        imgs.append(img)
        ys.append(cls)
    return np.stack(imgs, axis=0), np.array(ys)

if __name__ == "__main__":
    X, y = make_dataset(n=20, seed=1)
    print("Demo shapes:", X.shape, y.shape, "classes:", sorted(set(y)))
```



This module defines helper functions to create synthetic wafer defect images. Functions such as `_disk`, `_ring`, and `_line` draw geometric patterns inside numpy arrays. These shapes are combined to simulate defects like center clusters, edge rings, scratches, or donuts. The file also defines a list of all class names. This module is essential because it creates reproducible synthetic data without requiring real wafer images. The functions return boolean masks or arrays that can be scaled to represent intensity patterns. Later, these generated samples are used for training and evaluation.



## Feature Extraction (src/features.py)



```python
import numpy as np

def radial_profile(img):
    h, w = img.shape
    cx, cy = w // 2, h // 2
    Y, X = np.indices((h, w))
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_r = int(r.max()) + 1
    prof = np.zeros(max_r, dtype=float)
    counts = np.zeros(max_r, dtype=float)
    for i in range(h):
        for j in range(w):
            rr = r[i, j]
            prof[rr] += img[i, j]
            counts[rr] += 1.0
    counts[counts == 0] = 1.0
    prof = prof / counts
    bins = np.linspace(0, len(prof)-1, 14).astype(int)
    out = []
    for k in range(len(bins)-1):
        out.append(prof[bins[k]:bins[k+1]].mean())
    out.append(prof[bins[-1]:].mean())
    return np.array(out, dtype=float)

def ring_ratio(img):
    h, w = img.shape
    cx, cy = w // 2, h // 2
    Y, X = np.indices((h, w))
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    rnorm = r / r.max()
    inner = img[rnorm < 0.3].mean()
    middle = img[(rnorm >= 0.3) & (rnorm < 0.6)].mean()
    outer = img[rnorm >= 0.6].mean()
    return np.array([inner, middle, outer, (outer + middle) / (inner + 1e-6)])

def line_energy(img):
    gy, gx = np.gradient(img)
    gmag = np.hypot(gx, gy)
    orientation = np.arctan2(gy, gx)
    bins = np.linspace(-np.pi, np.pi, 9)
    hist, _ = np.histogram(orientation, bins=bins, weights=gmag, density=True)
    return np.concatenate(([gmag.mean(), gmag.std()], hist))

def moments(img):
    m = img.mean()
    s = img.std()
    sk = ((img - m) ** 3).mean() / (s**3 + 1e-6)
    ku = ((img - m) ** 4).mean() / (s**4 + 1e-6)
    return np.array([m, s, sk, ku])

def features_from_image(img):
    rp = radial_profile(img)
    rr = ring_ratio(img)
    le = line_energy(img)
    mo = moments(img)
    return np.concatenate([rp, rr, le, mo])

def batch_features(imgs):
    return np.stack([features_from_image(x) for x in imgs], axis=0)
```



The feature extraction module contains a function to compute radial profiles of images. The `radial_profile` function calculates average intensity as a function of distance from the wafer center. It loops through all pixels, computes the radial distance, and accumulates intensities in bins. The resulting profile describes how bright the wafer is at different radii. This feature is critical for distinguishing between patterns such as center concentration, edge rings, and donuts. The file may also contain helper functions for creating feature vectors from images. These features form the input for machine learning models.



## Training Script (src/train.py)



```python
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

from .generate_data import make_dataset, CLASSES
from .features import batch_features

def train_model(n=600, seed=7, save_path="models/trained/model.pkl", csv_out=None):
    Ximgs, y = make_dataset(n=n, seed=seed, classes=CLASSES)
    X = batch_features(Ximgs)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=seed, class_weight="balanced_subsample"
        ))
    ])
    pipe.fit(Xtr, ytr)

    yhat = pipe.predict(Xte)
    labels = pipe.classes_
    cm = confusion_matrix(yte, yhat, labels=labels)

    print("Classes:", labels.tolist())
    print(classification_report(yte, yhat, digits=4))
    print("Confusion matrix:\n", cm)

    with open(save_path, "wb") as f:
        pickle.dump({"pipe": pipe, "classes": labels.tolist()}, f)

    if csv_out:
        nsave = min(len(Ximgs), 400)
        df = pd.DataFrame(Ximgs[:nsave].reshape(nsave, -1))
        df["label"] = y[:nsave]
        df.to_csv(csv_out, index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save", type=str, default="models/trained/model.pkl")
    ap.add_argument("--csv", type=str, default="data/sample/wafer_samples.csv")
    args = ap.parse_args()
    train_model(n=args.n, seed=args.seed, save_path=args.save, csv_out=args.csv)
```



The training script provides the pipeline for building a classifier. It imports scikit-learn modules such as RandomForestClassifier and StandardScaler. The script loads generated data and extracted features, splits them into training and test sets, and fits the model. It also prints classification reports and confusion matrices to the console. A trained model is saved to disk using pickle. The file shows how to connect synthetic data generation with a supervised learning algorithm. Using a pipeline ensures that scaling and classification are applied consistently. This training script can be extended with more complex models or parameter tuning.



## Prediction Script (src/predict.py)



```python
import io
import pickle
from pathlib import Path
import numpy as np
from PIL import Image

from .features import features_from_image

def _retrain(path: Path):
    """Train a small RF so the app always has a working model."""
    from .train import train_model
    path.parent.mkdir(parents=True, exist_ok=True)
    # small, fast training set; adjusts to whatever sklearn/numpy is installed
    train_model(n=400, seed=7, save_path=str(path), csv_out=None)

def load_model(path: str = "models/trained/model.pkl"):
    """
    Load a model. If unpickling fails due to version/ABI changes or file is missing,
    retrain once and then load.
    """
    p = Path(path)
    try:
        with open(p, "rb") as f:
            blob = pickle.load(f)
        return blob["pipe"], blob["classes"]
    except Exception:
        _retrain(p)
        with open(p, "rb") as f:
            blob = pickle.load(f)
        return blob["pipe"], blob["classes"]

def prepare_img_from_csv_bytes(b):
    import pandas as pd
    arr = pd.read_csv(io.BytesIO(b), header=None).values.astype(float)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr

def prepare_img_from_png_bytes(b):
    img = Image.open(io.BytesIO(b)).convert("L").resize((28, 28), Image.BILINEAR)
    arr = np.array(img).astype(float)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr

def predict_one(arr, model_path: str = "models/trained/model.pkl", pipe=None, classes=None):
    """Predict using an already-loaded pipe/classes, or load (and retrain if needed)."""
    if pipe is None or classes is None:
        pipe, classes = load_model(model_path)
    feat = features_from_image(arr).reshape(1, -1)
    proba = pipe.predict_proba(feat)[0]
    idx = int(np.argmax(proba))
    return classes[idx], dict(zip(classes, proba.tolist()))
```



The prediction script is designed to load a trained model and apply it to new data. It includes functions to retrain a quick fallback model if no model is found. It uses pickle to load the saved RandomForestClassifier. The script provides a `predict` function that takes in an image, extracts features, and returns the predicted class. This separation of training and prediction is important for deployment, since the app only needs the prediction function at runtime. The retraining option ensures that the app always has some model available, even if the serialized model file is missing.

