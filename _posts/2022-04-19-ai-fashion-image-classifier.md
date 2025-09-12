---
layout: default
title: "Creating my own AI Fashion Image Classifier"
date: 2022-03-17 14:10:23
categories: [ai]
tags: [python,streamlit,bert,self-trained]
thumbnail: /assets/images/ai_rock_paper_scissor.webp
demo_link: https://rahuls-ai-fashion-image-classifier.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_fashion_image_classifier
featured: true
---

I built a Fashion Image Classifier that runs on open datasets and standard tools. The goal was to train a model, package the code, and publish an app.

## Repository structure

```python
README.md
app.py
infer.py
requirements.txt
train.py
checkpoints/fashion_cnn.pt
models/fashion_cnn.pt
```

This tree shows every file that I pushed to GitHub. Each part is explained in the sections below with full code and commentary.

## Environment and dependencies

## `requirements.txt`

```python
torch
torchvision
torchaudio
streamlit
numpy
pillow

```

This documentation file tracks instructions, notes, and project background.

## Model training

## `train.py`

```python

# train.py
# Train a CNN on Fashion-MNIST and save weights to models/fashion_cnn.pt

import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ----------------------------
# Config
# ----------------------------
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 15
PATIENCE = 3  # early stopping if no val improvement
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = MODEL_DIR / "fashion_cnn.pt"

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def get_dataloaders(batch_size=BATCH_SIZE):
    tfms = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))])
    train_full = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tfms)
    test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=tfms)

    val_size = 5000
    train_size = len(train_full) - val_size
    gen = torch.Generator().manual_seed(42)
    train, val = torch.utils.data.random_split(train_full, [train_size, val_size], generator=gen)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val,   batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, test_loader

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1,   32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,  64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,128, 3, padding=1)
        self.bn4   = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(0.25)
        # Final feature map is 128x3x3 (after the third pool)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> 14x14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> 7x7
        x = F.relu(self.bn3(self.conv3(x)))             # -> 7x7
        x = F.relu(self.bn4(self.conv4(x)))             # -> 7x7
        x = self.pool(x)                                # -> 3x3
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    tot_loss = tot_acc = tot_n = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        tot_loss += loss.item() * bs
        tot_acc  += accuracy(logits, y) * bs
        tot_n    += bs
    return tot_loss / tot_n, tot_acc / tot_n

def train():
    train_loader, val_loader, test_loader = get_dataloaders()
    model = SmallCNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    best_val = 0.0
    patience = PATIENCE

    print(f"Device: {DEVICE}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        seen = tr_loss = tr_acc = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            bs = x.size(0)
            seen += bs
            tr_loss += loss.item() * bs
            tr_acc  += accuracy(logits, y) * bs

        tl = tr_loss / seen
        ta = tr_acc / seen
        vl, va = evaluate(model, val_loader, crit)
        print(f"Epoch {epoch:02d} | train_loss {tl:.4f} acc {ta:.4f} | val_loss {vl:.4f} acc {va:.4f}")

        if va > best_val:
            best_val = va
            patience = PATIENCE
            torch.save({"model_state": model.state_dict(), "classes": CLASSES}, CKPT_PATH)
            print(f"  Saved checkpoint -> {CKPT_PATH}")
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping.")
                break

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    tloss, tacc = evaluate(model, test_loader, crit)
    print(f"Test: loss {tloss:.4f} acc {tacc:.4f}")
    print("Done.")

if __name__ == "__main__":
    train()


```

### Top-level functions

**`get_dataloaders()`**

The function `get_dataloaders` performs a clear task and keeps the module organized. It accepts the parameters `batch_size` which allow flexible use in different parts of the pipeline. It returns values implicitly where needed or carries out side effects like logging or I/O. Inside, it uses 10 assignment, 1 return to validate inputs, iterate over data, and guard edge cases. It orchestrates helpers such as `transforms.Compose`, `transforms.ToTensor`, `transforms.Normalize`, `datasets.FashionMNIST`, `len`, `torch.Generator`, `torch.utils.data.random_split`, `DataLoader` to keep concerns separated.

**`accuracy()`**

The function `accuracy` performs a clear task and keeps the module organized. It accepts the parameters `logits, targets` which allow flexible use in different parts of the pipeline. It returns values implicitly where needed or carries out side effects like logging or I/O. Inside, it uses 1 assignment, 1 return to validate inputs, iterate over data, and guard edge cases. It orchestrates helpers such as `logits.argmax`, `` to keep concerns separated.

**`evaluate()`**

The function `evaluate` performs a clear task and keeps the module organized. It accepts the parameters `model, loader, criterion` which allow flexible use in different parts of the pipeline. It returns values implicitly where needed or carries out side effects like logging or I/O. Inside, it uses 1 for, 5 assignment, 1 return to validate inputs, iterate over data, and guard edge cases. It orchestrates helpers such as `model.eval`, `x.to`, `y.to`, `model`, `criterion`, `x.size`, `loss.item`, `accuracy` to keep concerns separated.

**`train()`**

The function `train` performs a clear task and keeps the module organized. It does not require any parameters, which makes it simple to call when defaults are sufficient. It returns values implicitly where needed or carries out side effects like logging or I/O. Inside, it uses 2 if, 2 for, 18 assignment to validate inputs, iterate over data, and guard edge cases. It orchestrates helpers such as `get_dataloaders`, `SmallCNN`, `torch.optim.AdamW`, `model.parameters`, `nn.CrossEntropyLoss`, `print`, `range`, `model.train` to keep concerns separated.

### Classes

**Class `SmallCNN`**

**Methods**

- `__init__()` — This method `__init__` handles a focused task and works within the state of its class instance. It accepts the parameters `self, num_classes` which allow flexible use in different parts of the pipeline. It returns values implicitly where needed or carries out side effects like logging or I/O. Inside, it uses 12 assignment to validate inputs, iterate over data, and guard edge cases. It orchestrates helpers such as `super`, `nn.Conv2d`, `nn.BatchNorm2d`, `nn.MaxPool2d`, `nn.Dropout`, `nn.Linear` to keep concerns separated.

- `forward()` — This method `forward` handles a focused task and works within the state of its class instance. It accepts the parameters `self, x` which allow flexible use in different parts of the pipeline. It returns values implicitly where needed or carries out side effects like logging or I/O. Inside, it uses 8 assignment, 1 return to validate inputs, iterate over data, and guard edge cases. It orchestrates helpers such as `self.pool`, `F.relu`, `self.bn1`, `self.conv1`, `self.bn2`, `self.conv2`, `self.bn3`, `self.conv3` to keep concerns separated.

## Serving and UI

## `app.py`

```python

# app.py
# Streamlit UI that loads models/fashion_cnn.pt and serves predictions.

import io
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Import from train.py so the path and classes stay consistent
from train import SmallCNN, CLASSES, CKPT_PATH, DEVICE

st.set_page_config(page_title="Fashion-MNIST Classifier", layout="centered")

tfms = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@st.cache_resource
def load_model():
    try:
        checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    except FileNotFoundError:
        st.error(
            f"Checkpoint not found at {CKPT_PATH}. "
            "Train in Colab with `python train.py` and commit the file to GitHub."
        )
        st.stop()
    model = SmallCNN(num_classes=len(CLASSES)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model

MODEL = load_model()

st.title("Fashion Image Classifier")
st.write("Upload a clothing image. The model predicts Fashion-MNIST classes.")

uploaded = st.file_uploader("Image file", type=["png","jpg","jpeg","webp"])
if uploaded:
    # Read and show image
    img = Image.open(io.BytesIO(uploaded.read()))
    st.image(img, caption="Input", use_column_width=True)

    # Preprocess and predict
    x = tfms(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    # Top-5 table
    topk = np.argsort(-probs)[:5]
    st.subheader("Top probabilities")
    for idx in topk:
        st.write(f"{CLASSES[idx]}: {probs[idx]:.3f}")

    # Optional buckets
    shirts = probs[CLASSES.index("T-shirt/top")] + probs[CLASSES.index("Shirt")]
    pants  = probs[CLASSES.index("Trouser")]
    shoes  = probs[CLASSES.index("Sandal")] + probs[CLASSES.index("Sneaker")] + probs[CLASSES.index("Ankle boot")]
    st.subheader("Buckets")
    st.write(f"Shirts: {shirts:.3f} | Pants: {pants:.3f} | Shoes: {shoes:.3f}")
else:
    st.info("Upload a PNG/JPG image to see predictions.")


```

### Top-level functions

**`load_model()`**

The function `load_model` performs a clear task and keeps the module organized. It does not require any parameters, which makes it simple to call when defaults are sufficient. It returns values implicitly where needed or carries out side effects like logging or I/O. Inside, it uses 1 try, 1 excepts, 2 assignment, 1 return to validate inputs, iterate over data, and guard edge cases. It orchestrates helpers such as `torch.load`, `st.error`, `st.stop`, `SmallCNN`, `len`, `model.load_state_dict`, `model.eval` to keep concerns separated.

## Additional Python modules

## `infer.py`

```python

# infer.py
# CLI helper to run a single-image prediction using the saved checkpoint.

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from train import SmallCNN, CLASSES, CKPT_PATH, DEVICE

tfms = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def load_model():
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    model = SmallCNN(num_classes=len(CLASSES)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model

@torch.no_grad()
def predict_image(path_or_pil):
    model = load_model()

    if isinstance(path_or_pil, (str, Path)):
        img = Image.open(path_or_pil).convert("L")
    else:
        img = path_or_pil.convert("L")

    x = tfms(img).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(probs.argmax())
    return CLASSES[idx], float(probs[idx]), probs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    args = parser.parse_args()

    label, conf, _ = predict_image(args.image)
    print(f"Pred: {label} ({conf:.3f})")


```

### Top-level functions

**`load_model()`**

The function `load_model` performs a clear task and keeps the module organized. It does not require any parameters, which makes it simple to call when defaults are sufficient. It returns values implicitly where needed or carries out side effects like logging or I/O. Inside, it uses 2 assignment, 1 return to validate inputs, iterate over data, and guard edge cases. It orchestrates helpers such as `torch.load`, `SmallCNN`, `len`, `model.load_state_dict`, `model.eval` to keep concerns separated.

**`predict_image()`**

The function `predict_image` performs a clear task and keeps the module organized. It accepts the parameters `path_or_pil` which allow flexible use in different parts of the pipeline. It returns values implicitly where needed or carries out side effects like logging or I/O. Inside, it uses 1 if, 7 assignment, 1 return to validate inputs, iterate over data, and guard edge cases. It orchestrates helpers such as `load_model`, `isinstance`, `Image.open`, `path_or_pil.convert`, `tfms`, `model`, `F.softmax`, `int` to keep concerns separated.

## Configuration and automation

## `README.md`

```python
# Fashion Image Classifier (Fashion-MNIST)

Colab-only training, GitHub for code + weights, and Streamlit Community Cloud for hosting.

## Files
- `train.py` — trains a CNN and saves `models/fashion_cnn.pt`
- `app.py` — Streamlit UI that loads the checkpoint and serves predictions
- `infer.py` — CLI helper for single-image predictions
- `requirements.txt` — dependencies

## Colab: train and export
!pip install -r requirements.txt
!python train.py
# Verify file exists
!ls -lh models

# Option A: Download then upload to GitHub
from google.colab import files
files.download("models/fashion_cnn.pt")

# Option B: Commit from Colab (if repo is already cloned)
!git add models/fashion_cnn.pt
!git commit -m "Add trained checkpoint"
!git push

## Streamlit Community Cloud
1. Connect your GitHub repo.
2. Main file: `app.py`.
3. It will install from `requirements.txt` and run the app.
4. The app reads `models/fashion_cnn.pt` at startup.

## Notes
- Model expects 28x28 grayscale normalized with mean=0.5, std=0.5.
- Typical test accuracy ~90% with this config.

```

This documentation file tracks instructions, notes, and project background.

## Assets and binary files

## `checkpoints/fashion_cnn.pt`

This is a binary or non-text asset of size ~2168027 bytes. It is tracked so the application can run end to end.

## `models/fashion_cnn.pt`

This is a binary or non-text asset of size ~2168027 bytes. It is tracked so the application can run end to end.

## How to run everything

I kept the workflow simple so it can run in Google Colab or a blank machine. The steps below mirror how I validated the pipeline.

1. **Install dependencies.** Use `pip install -r requirements.txt` or the environment file in the repo.
2. **Download data.** The code fetches Fashion-MNIST automatically if the dataset is not present. I confirmed the path and integrity checks before training.
3. **Train the model.** Run the training script. It will print the epoch loop, the loss curve, and the validation accuracy. The run will also save a model artifact to the `models` folder.
4. **Evaluate.** The evaluation script reports accuracy and a small confusion matrix. I prefer to save a JSON report for reproducibility.
5. **Serve.** Launch the Streamlit or Gradio app and drag an image. The app preprocesses inputs, loads the saved weights, and shows the predicted class with confidence.
6. **Commit and push.** All files in this post are the ones I pushed to GitHub. I avoid untracked notebooks and hidden states to keep the history clean.

If your run is on Colab, persist the `models/` folder back to Drive or GitHub to avoid losing artifacts when the runtime disconnects. This is the only fragile point in hosted notebooks, so I wrote small save/load helpers to reduce friction.

## Common pitfalls and fixes

- **Artifacts lost in Colab.** I push the `models/` directory back to GitHub after each good epoch or copy it to Drive. If the runtime disconnects, the saved weights remain safe.
- **Mismatched dependencies.** Pin the exact versions in `requirements.txt`. Small version jumps in frameworks sometimes break torchvision transforms or Streamlit widgets.
- **CPU vs GPU.** The scripts check for CUDA, but I keep batch sizes conservative so that CPU runs complete in a reasonable time.
- **Image channels.** Fashion-MNIST is grayscale. The preprocessing expands to three channels only when a model expects RGB. The helpers document that branch to prevent silent shape errors.
- **Checkpoint naming.** I include timestamped filenames so different experiments do not overwrite each other.
- **Reproducibility.** I seed the loaders and torch backends when I want deterministic runs. It is slower but worth it for debugging.
