---
layout: default
title: "Enhancing my AI Summarizer"
date: 2022-01-01 08:22:12
categories: [ai]
tags: [python,streamlit,openai]
thumbnail: /assets/images/ai_rock_paper_scissor.webp
demo_link: https://huggingface.co/spaces/RahulBhattacharya/RahulsTextSummarizer
github_link: https://github.com/RahulBhattacharya1/ai_text_summarizer
featured: true
---

This project is a web app that summarizes long text. It uses a pretrained model from Hugging Face. It wraps the model with a friendly Gradio interface. I run it locally during development and also on Hugging Face Spaces. I keep the code short and focused so I can debug it fast.

---

# The Core Application: `app.py`

Below are the exact blocks from my `app.py` with a simple breakdown for each. I explain what the block does, why I wrote it that way, and how it fits the whole.

## 1) Imports and a Single Model Choice

```python
import gradio as gr
from transformers import pipeline

# Pretrained CNN/DailyMail summarizer
MODEL_ID = "sshleifer/distilbart-cnn-12-6"
```
This block brings in the two pieces I need. Gradio builds a small web UI without extra HTML. The `pipeline` helper loads a ready summarization model, so I do not write tokenization code. I put the chosen model id in a constant, so I can swap it later in one place. I picked `distilbart-cnn-12-6` because it is fast and gives readable summaries.

## 2) Building the Summarization Pipeline

```python
summarizer = pipeline("summarization", model=MODEL_ID, device=-1)
```
This single line hides a lot of work. The `pipeline` downloads weights on first run and caches them. It builds a tokenizer and a model for summarization and keeps them in memory. I pass `device=-1` so it runs on CPU by default, which is stable on most machines. If I have a GPU available, I can switch to `device=0` to speed it up.

## 3) The Only Function: `summarize`

```python
def summarize(text):
    if not text.strip():
        return "Please enter some text."
    result = summarizer(
        text,
        max_length=150,
        min_length=30,
        truncation=True
    )
    return result[0]["summary_text"]
```
This function is my thin wrapper. The first conditional guards against empty input and gives a clear message. The call to `summarizer` sets sane bounds with `max_length` and `min_length` so the model does not ramble or cut too short. I also turn on `truncation` to keep inputs under model limits. I return the string field from the first item because the pipeline returns a list of results.

## 4) A Minimal Gradio Interface

```python
demo = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(lines=12, label="Paste article text"),
    outputs=gr.Textbox(lines=8, label="Summary"),
    title="Rahul's Text Summarizer",
    description="Summarizer for long texts."
)
```
This block defines the UI with clear defaults. I wire the `summarize` function to the interface so the button calls my code. I use a large input box so I can paste long pieces comfortably. I use a smaller output box so the summary stays focused and readable. The title and description tell users what to expect before they try it.

## 5) Launching the App

```python
if __name__ == "__main__":
    demo.launch()
```
This guard runs the app when I execute the file directly. It avoids side effects when I import `summarize` in tests. The `launch` call spins up a local server and opens a browser tab. It gives me a fast loop to test changes without extra setup.

---

# Why Each Choice Works for Me

## Model: `sshleifer/distilbart-cnn-12-6`

I need clean summaries in a few seconds. This distil version of BART is a good trade off. It is smaller than full BART and still trained on CNN/DailyMail style news. It keeps the tone balanced and drops filler well. I can switch models later by changing one string.

## CPU by Default

I set `device=-1` to avoid CUDA errors on machines without GPUs. It keeps the app portable during demos. It is also fine for short articles and notes. If I need more speed, I run with `device=0` on a server that has a GPU.

## Guarding Empty Input

I do not want stack traces for empty text. The `strip` check returns a clear message and saves a model call. It is cheap and prevents confusion for new users. It also lets me test the error path quickly.

## Summary Length Bounds

I keep `min_length=30` to avoid one sentence stubs. I keep `max_length=150` to fit on a small screen. These values are not magic. They are pragmatic and match my content. I adjust them if my use case shifts.

---

# End‑to‑End Flow in Words

1. I paste an article into the input box.  
2. I press the button, and Gradio calls my `summarize` function.  
3. The function checks if the text is empty and returns early if so.  
4. Otherwise the pipeline tokenizes, runs the model, and decodes the summary.  
5. The interface shows the result in the output box.

This loop is simple to follow and easy to test. It uses only one function I wrote. It leans on stable libraries to do the hard work.

---

# Practical Setup and Versions

I pin the key libraries to stable versions so setup is smooth.

```python
# requirements.txt (expressed here as python comment lines for clarity)
# transformers>=4.41.0
# torch
# gradio>=4.0.0
```
I install these with pip in a clean environment. I prefer a virtualenv or conda so my global site-packages stay clean. After install, I run a quick version check to avoid mismatch issues.

```python
import transformers, torch, gradio
print("Transformers:", transformers.__version__)
print("Torch:", torch.__version__)
print("Gradio:", gradio.__version__)
```
This small check has saved me many times. If anything is off, I know before I debug app logic. It also helps when I compare behavior across machines.

---

# Quick Local Test Without the UI

I like to test the core function directly. It is faster than clicking a button.

```python
# save as quick_test.py next to app.py for a fast check
from app import summarize

text = """
Neural sequence-to-sequence models have improved summarization quality.
They still struggle with factual consistency on long inputs.
Smaller distilled models run faster while preserving key content.
"""

print(summarize(text))
```
This check proves the function works without the Gradio layer. It also lets me write future unit tests. I keep the input small so it runs in a few seconds. It helps me debug changes to the function parameters in isolation.

---

# Behavior on Long Inputs

Very long inputs can exceed model limits. The `truncation=True` flag keeps things safe but may drop tail content. For my use, this is acceptable because I paste articles or notes, not books. If I need full coverage, I can add simple chunking later. I keep the current app lean until that need is real.

```python
# example sketch if I ever need naive chunking (not used in app.py)
def naive_chunks(text, max_chars=1500):
    buf = []
    chunk = []
    count = 0
    for line in text.splitlines(True):
        if count + len(line) > max_chars and chunk:
            buf.append(''.join(chunk))
            chunk, count = [], 0
        chunk.append(line)
        count += len(line)
    if chunk:
        buf.append(''.join(chunk))
    return buf
```
I am not shipping this helper today. I keep it here as a note to self. Chunking trades context for coverage and needs careful stitching. I will add it only when my inputs truly demand it.

---

# Interface Details That Matter

I keep the inputs and outputs as text boxes. I do not add extra controls until I need them. The defaults make the app easy to grasp. A single screen with a title, a big input, and a button is enough for now.

```python
# a tiny variation I might use later to add an example
demo = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(lines=12, label="Paste article text", placeholder="Paste text here..."),
    outputs=gr.Textbox(lines=8, label="Summary"),
    examples=[["This is a small note that needs a compact summary."]],
    title="Rahul's Text Summarizer",
    description="Summarizer for long texts."
)
```
Examples help first time users understand the flow. I keep this optional because I want a clean screen. I add it when I do public demos.

---

# Running Locally

I run the app with a simple command:

```python
# run.py for a launcher, optional
import os, subprocess, sys
subprocess.run([sys.executable, "app.py"], check=True)
```
I usually just run `python app.py`. The `run.py` script is a convenience when I add more startup checks. It gives me one entry point when I grow the project.

---

# Deploying on Hugging Face Spaces

I keep a small configuration file that tells Spaces how to start my app. It sets the SDK to Gradio and points to my `app.py`. When I push the repo, the Space builds and runs the app automatically. This keeps deployment simple and repeatable.

```python
# expressed as a Python string for clarity; in the repo this lives in README.md style metadata
spaces_cfg = """
title: Rahul's Text Summarizer
colorFrom: pink
colorTo: pink
sdk: gradio
sdk_version: 5.44.1
app_file: app.py
pinned: false
"""
print(spaces_cfg)
```
I avoid UI emojis in the app screen and keep the content plain. Spaces gives me a public URL that I can share. I test it the same way I test locally.

---

# Common Troubles I Solved

**Cold start downloads feel slow.** The first run downloads model weights. I wait once. Later runs use the cache.  
**Memory spikes on big inputs.** I paste smaller chunks or lower `max_length`. It keeps the experience smooth.  
**Output is too short or too long.** I tune `min_length` and `max_length` in small steps and try again.

---

# How I Would Extend This Later

I would add chunking for very long texts and stitch summaries. I might switch to a model fine-tuned on my domain content. I could add a dropdown to pick a model at runtime. I could also add a textbox to control the length bounds. I will add each feature only when I truly need it.

```python
# not in the shipped app; sketch for a length control
import gradio as gr

length_slider = gr.Slider(60, 250, value=150, step=10, label="Max length")

def summarize_with_length(text, max_len):
    if not text.strip():
        return "Please enter some text."
    out = summarizer(text, max_length=int(max_len), min_length=30, truncation=True)
    return out[0]["summary_text"]

demo2 = gr.Interface(
    fn=summarize_with_length,
    inputs=[gr.Textbox(lines=12, label="Paste article text"), length_slider],
    outputs=gr.Textbox(lines=8, label="Summary"),
    title="Rahul's Summarizer (Adjustable)",
    description="Control maximum summary length."
)
```
These ideas stay on a branch or in a notebook until they earn a place in the main app. I like to merge only features that improve day to day value.

---

# Final Notes

I keep this project small so it stays reliable. The code fits in one file and is easy to read. The interface is direct and does what it says. When I need more power, I will grow it with care and tests. Until then, I enjoy the speed and simplicity of this build.

