---
layout: default
title: "Creating my AI Sentiment Analysis App"
date: 2022-01-01 08:22:12
categories: [ai]
tags: [python,streamlit,openai]
thumbnail: /assets/images/ai_rock_paper_scissor.webp
demo_link: https://huggingface.co/spaces/RahulBhattacharya/Rahuls_Text_Classification_Sentiment_Analysis
github_link: https://github.com/RahulBhattacharya1/ai_sentiment_analyzer
featured: true
---

This post walks through my IMDB sentiment analysis app. I train a movie review classifier on IMDB. It uses the Transformers library for model and training. It wraps everything in a Gradio interface for the web. It also pushes the trained model to the Hugging Face Hub.

## Requirements I Used
Below is the exact requirements file I used. These versions work well together. They support training and the web UI. They also support pushing to the Hub.

```python
pip>=24.0
setuptools>=68
wheel
transformers>=4.44.0
datasets>=2.19.0
accelerate>=0.32.0
evaluate>=0.4.2
huggingface_hub>=0.24.0
scikit-learn>=1.1.0
torch
gradio>=4.37.2
```

I group libraries by their roles to explain intent. Transformers, datasets, and evaluate power the NLP pipeline. Accelerate, torch, and scikit-learn support fast training and metrics. Gradio powers the browser interface and event wiring. Hugging Face Hub libraries manage model upload and auth.

## Space Configuration
I run this project in a Hugging Face Space. The Space config pins the runtime and entry file. I mirror that config here as Python comments to keep formatting simple. I also keep the values short and readable.

```python
# ---
# title: Rahul's Text Classification Sentiment Analysis
# emoji: 🌖
# colorFrom: gray
# colorTo: gray
# sdk: gradio
# sdk_version: 5.44.1
# app_file: app.py
# pinned: false
# ---
# 
# Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
```

The title matches my project name and keeps things clear. The sdk field tells Spaces to run Gradio. The app_file points to the script that launches the UI. I keep pinned false so I can upgrade when needed.

## Application Code Walkthrough
I split the code into meaningful blocks. I show the block, then I explain it. I avoid line by line detail, and focus on the purpose.

### Imports, Constants, and Hub Login
This block pulls in libraries and sets a target repo. It reads an access token and tries an authenticated login. It sets a human readable status string for display.

```python
# app.py — Gradio UI that launches IMDB training and pushes to the Hub
import os, time, numpy as np, threading, queue
import gradio as gr
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer
)
from huggingface_hub import login, HfApi

TARGET_REPO = "RahulBhattacharya/Rahuls_Text_Classification_Sentiment_Analysis"

# ===== Hub auth (from Space secret) =====
HF_TOKEN = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or "").strip().strip('"').strip("'")
if not HF_TOKEN or not HF_TOKEN.startswith("hf_"):
    WARN = "[WARN] No valid HF_TOKEN secret found. Add one in Settings → Variables and secrets."
else:
    try:
        login(HF_TOKEN)
        me = HfApi().whoami(token=HF_TOKEN).get("name", "unknown")
        WARN = f"[OK] Logged in as: {me}"
    except Exception as e:
        WARN = f"[ERR] Token login failed: {e}"
```

I import numpy for metrics and arrays in evaluation. I import threading and queue to handle background training. I import Transformers components for tokenization, modeling, and training. I import Hugging Face Hub login helpers to push the model.

I set TARGET_REPO to the destination in my account. I look up the token from environment variables. I attempt a login and fetch my username to confirm. I set a WARN string so the UI can show current auth state.

### Logging Queue and Locks
This block defines a thread safe queue for logs. It also defines a mutex to keep only one training at a time. This is simple, but it prevents overlap. It keeps the UI predictable.

```python
# ===== Training worker =====
log_q: "queue.Queue[str]" = queue.Queue()
trainer_thread = None
_training_lock = threading.Lock()
```

The queue collects messages from the background thread. The lock serializes training sessions behind a single gate. This avoids resource contention on CPU or GPU. It also keeps the Hub push flow clean.

### Helper: _log(msg)
This small helper pushes a string to the queue. It hides queue usage behind one call. It keeps the worker readable and consistent. It also makes testing easier.

```python
def _log(msg: str):
    log_q.put(msg)
```

I keep the body one line for minimal overhead. The queue stores messages until the UI reads them. This prevents blocking writes during heavy work. It also preserves message ordering from the worker.

### Training Worker: train_worker
This function handles all training activities. It loads data, tokenizes it, and builds a model. It initializes the trainer, runs training, and evaluates results. It then pushes the model to the Hub.

```python
def train_worker(train_size: int, eval_size: int, epochs: int, lr: float, batch: int):
    try:
        _log(f"[WAIT] Loading dataset imdb …")
        ds = load_dataset("imdb")

        base = "distilbert-base-uncased"
        tok  = AutoTokenizer.from_pretrained(base)
```

I start by loading the IMDB dataset with a single call. I create a tokenizer from a base DistilBERT model. I map tokenization over the dataset and drop raw text to save memory. I subset train and eval splits to keep runs fast on CPU.

I define label maps for readability in outputs. I load a sequence classification head with two labels. I set id2label and label2id for clean downstream usage. I set the problem type for clarity and correctness.

The metric function computes simple accuracy over predictions. It uses argmax over logits and a vectorized compare. It returns a float accuracy for the trainer to log. It keeps the loop light and focused.

TrainingArguments control batch sizes, learning rate, and epochs. I disable external reporting to reduce noise. I enable push_to_hub with my repo and token. This allows direct upload after training finishes.

I pass all components to the Trainer in one place. I include a data collator to handle padding cleanly. I call train and then evaluate to measure performance. I finally push the model to the Hub with one method call.

I wrap the body in a try and catch to handle errors. I log nice messages for user feedback in the UI. I close with a finally block to post a final status. This ensures the UI state updates even on failure.

### Start Handler: start_training
This function runs when the Start button is clicked. It checks if another training is running and validates the token. It then starts a new background thread. It returns UI updates for status and logs.

```python
def start_training(train_size, eval_size, epochs, lr, batch):
    global trainer_thread
    if _training_lock.locked():
        return gr.update(), "[WARN] Training already running. Please wait."
    if not HF_TOKEN or not HF_TOKEN.startswith("hf_"):
        return gr.update(), "[ERR] Missing/invalid HF_TOKEN secret. Add it in Settings."

    # spawn worker
```

I check the lock to ensure only one run at a time. I check the token again to keep errors clear and early. I define a runner that enters the lock and calls the worker. I start a daemon thread so the server stays responsive.

### Log Stream: stream_logs
This generator yields new messages from the queue. It blocks briefly to avoid busy waiting. It sleeps on empty queue to reduce CPU usage. It is simple and robust.

```python
def stream_logs():
    # generator for live log textbox
    while True:
        try:
            line = log_q.get(timeout=0.5)
            yield line
        except queue.Empty:
            time.sleep(0.2)

# ===== Gradio UI + Prediction Widget =====
from transformers import pipeline

# load your model once (reused for every click)
clf = pipeline(
    "text-classification",
    model="RahulBhattacharya/Rahuls_Text_Classification_Sentiment_Analysis"
)
```

I use get with a timeout to avoid hanging forever. I catch queue.Empty to know when to sleep briefly. I loop forever because the UI controls the stop. This pattern works well for streaming text areas.

### Inference Pipeline and Prediction Helper
This block loads the model into a pipeline. It exposes a clean function to classify text. It also guards against empty input. It returns a simple label to score map.

```python
from transformers import pipeline

# load your model once (reused for every click)
clf = pipeline(
    "text-classification",
    model="RahulBhattacharya/Rahuls_Text_Classification_Sentiment_Analysis"
)

def predict_sentiment(text: str):
    text = (text or "").strip()
    if not text:
        return {"N/A": 0.0}
    res = clf(text)[0]  # {'label': 'POSITIVE', 'score': 0.99}
    return {res["label"]: res["score"]}
```

I use the text-classification pipeline for simplicity and speed. I pass the model repo name to load from the Hub. I strip and validate input so the call is safe. I return a dict that the label widget can display.

### Gradio Interface and Event Wiring
This block creates the full UI in one context. It adds controls to set sizes and hyperparameters. It wires the Start button to the training handler. It also adds a live log viewer and the prediction demo.

```python
with gr.Blocks(title="Rahul's Text Classification Sentiment Analysis") as demo:
    gr.Markdown(f"### Rahul's Text Classification Sentiment Analysis\n{WARN}\n\n**Target repo:** `{TARGET_REPO}`")

    # ---- Training controls ----
    with gr.Row():
        train_size = gr.Number(value=2000, label="Train subset size")
        eval_size  = gr.Number(value=500,  label="Eval subset size")
        epochs     = gr.Number(value=1,    label="Epochs")
        lr         = gr.Number(value=2e-5, label="Learning rate")
        batch      = gr.Number(value=16,   label="Batch size")

    start_btn = gr.Button("Start training")
    status = gr.Markdown("")
    logs = gr.Textbox(label="Logs (live)", lines=18, interactive=False, value="")

    # Start training (spawns background thread)
    start_btn.click(start_training, [train_size, eval_size, epochs, lr, batch], [logs, status])

    # --- Live log updater using a Timer ---
    def read_logs(prev_text):
        new_lines = []
        try:
            while True:
                new_lines.append(log_q.get_nowait())
        except queue.Empty:
            pass
        if not new_lines:
            return prev_text
        updated = (prev_text + ("\n" if prev_text else "") + "\n".join(new_lines)).splitlines()[-200:]
        return "\n".join(updated)

    timer = gr.Timer(1.0, active=True)
    timer.tick(read_logs, inputs=logs, outputs=logs)

    # ---- Prediction demo ----
    gr.Markdown("##  Try Out Sentiment Prediction")
    input_text = gr.Textbox(label="Enter a movie review")
    output = gr.Label(label="Prediction")
    analyze_btn = gr.Button("Analyze")
    analyze_btn.click(predict_sentiment, inputs=input_text, outputs=output)
```

The Blocks layout gives me rows and columns quickly. The Number inputs let me tune experiments without code changes. The Start button triggers the handler and returns status. The logs box streams server messages in near real time.

I define an inner function that drains the log queue. I cap the history so the browser stays fast. The Timer calls the reader each second to fetch fresh lines. This keeps the UI reactive without manual refreshes.

I add a simple prediction section at the end. It has a text input and a label output. The Analyze button calls the predict function above. This lets a recruiter try the model with one click.

### Launching the App
This small guard creates a queue for requests. It launches the Gradio server when the file runs directly. It keeps reuse clean when importing the module elsewhere. It is a common Python pattern.

```python
if __name__ == "__main__":
    demo.queue().launch()
```

I call queue() to enable request queuing in Gradio. I then launch the app with sane defaults on Spaces. It prints share links locally and binds on the right port on Spaces. It keeps setup very light.

## Design Notes
I choose DistilBERT for small size and good speed. It trains fast even on CPU with small subsets. It performs well for two label sentiment tasks. It is a good default for this kind of demo.

I use the Trainer API to reduce boilerplate code. It wraps the loop and evaluation in a clean interface. It saves checkpoints and logs at set steps. It keeps the code short and maintainable.

I use a background thread for training to keep the UI responsive. It decouples the long task from the click handler. It allows live logging without blocking the main loop. It feels smooth to the end user.

## Practical Tuning Tips
Start with a small training subset to verify the pipeline. Increase the subset once you are happy with the loop. Tune learning rate and batch size with small steps. Watch accuracy and adjust epochs with restraint.

Use a GPU if you need faster training on the full set. Use the CPU for quick correctness checks with small data. Cache the dataset to speed repeated runs during tuning. Track model cards on the Hub for your updates.

## Deployment Notes
Spaces make it easy to host this demo for free. Gradio handles the UI and web server for interactions. The Hub stores your model snapshots and version history. The login helper pushes artifacts after training ends.

## Troubleshooting
If the token string is missing the code refuses to run. Add the token as a secret in the Space settings. If training does not start check that another run is not active. Watch the logs for any errors in the worker.

If push fails verify the repo name and your permission. If evaluation fails check the subset sizes and label mapping. If the UI freezes reduce the log size in the reader. Keep the timer cadence to one second for balance.

## Closing Thoughts
This app is small but complete and easy to follow. It shows a clear path from data to a hosted model. It stays readable because each block does one job well. It is a practical template for many NLP demos.
