---
layout: default
title: "My own AI Model for Sentiment Analysis"
date: 2025-05-09 12:23:49
categories: [ai]
tags: [python,streamlit,bert,self-trained]
thumbnail: /assets/images/sentiment.webp
demo_link: https://rahuls-ai-sentiment-analysis.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_sentiment_analysis
featured: true
---

# Building My IMDb Sentiment Classifier

In this post, I want to walk through how I created an AI-powered IMDb Sentiment Classifier. The goal of this project was simple: take a piece of text (a movie review) and classify it as **Positive** or **Negative**. I trained my own model, saved it, and deployed it as a live demo using Streamlit.

---

## Loading the IMDb Dataset

I started with the **IMDb dataset** available through Hugging Face Datasets. This dataset contains 50,000 movie reviews labeled with sentiment.

```python
from datasets import load_dataset
import pandas as pd

ds = load_dataset("imdb")

train_df = pd.DataFrame({"text": ds["train"]["text"], "label": ds["train"]["label"]})
test_df  = pd.DataFrame({"text": ds["test"]["text"],  "label": ds["test"]["label"]})
```

Here I converted the dataset into Pandas DataFrames for easier handling. Each row has a text review and a label (`0 = Negative`, `1 = Positive`).

---

## Text Preprocessing

Movie reviews often have HTML tags or strange punctuation. To make the model more effective, I normalized the text.

```python
import re

def normalize(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

train_df["text"] = train_df["text"].apply(normalize)
test_df["text"]  = test_df["text"].apply(normalize)
```

This step ensures that the reviews are lowercased, stripped of extra characters, and ready for vectorization.

---

## Vectorization with TF-IDF

Next, I converted the raw text into numeric features using **TF-IDF**. This technique captures which words are important in a review relative to all other reviews.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
X_train = vectorizer.fit_transform(train_df["text"])
X_test  = vectorizer.transform(test_df["text"])
```

Here I kept up to 50,000 features and considered both unigrams (single words) and bigrams (two-word phrases).

---

## Training the Classifier

I chose **Logistic Regression**, a simple yet powerful classifier for text problems.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

clf = LogisticRegression(solver="liblinear", max_iter=1000)
clf.fit(X_train, train_df["label"])

pred = clf.predict(X_test)
acc = accuracy_score(test_df["label"], pred)
print("Accuracy:", acc)
print(classification_report(test_df["label"], pred, digits=3))
```

This gave me a strong baseline sentiment classifier with accuracy well above random guessing.

---

## Saving the Model

After training, I combined the vectorizer and classifier into a pipeline and saved it as a `.pkl` file. This makes it easy to reuse later without retraining.

```python
from sklearn.pipeline import Pipeline
import joblib
import os

pipeline = Pipeline([
    ("tfidf", vectorizer),
    ("clf", clf)
])

os.makedirs("artifacts", exist_ok=True)
joblib.dump(pipeline, "artifacts/imdb_tfidf_logreg.pkl")
```

Now I had a portable model file that I could use in my web app.

---

## Building the Streamlit App

I wanted recruiters to try the model themselves, so I built a small web app using **Streamlit**.

```python
import joblib
import re
import streamlit as st
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "imdb_tfidf_logreg.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def normalize(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

st.set_page_config(page_title="IMDb Sentiment Classifier")
st.title("IMDb Sentiment Classifier")

model = load_model()

text = st.text_area("Enter a movie review:")
if st.button("Analyze") and text.strip():
    x = normalize(text)
    pred = model.predict([x])[0]
    label = "Positive" if pred == 1 else "Negative"
    st.subheader(f"Prediction: {label}")
```

With this app, a user can type or paste a review, click **Analyze**, and instantly see if the model thinks the review is Positive or Negative.

---

## Packaging Requirements

To make deployment work on Streamlit Cloud, I created a `requirements.txt` file:

```
streamlit>=1.37
scikit-learn>=1.3
joblib>=1.3
numpy>=1.23
```

This ensures that all necessary Python packages are installed automatically.

---

## Deployment

I pushed my code and model file to GitHub, then connected the repository to **Streamlit Community Cloud**. Streamlit automatically built the environment and launched the app at a free public URL.

Finally, I embedded the app into my GitHub Pages portfolio so recruiters could interact with it directly.

---

## What This Project Demonstrates

By completing this project, I was able to demonstrate:
- How to train a machine learning model for sentiment analysis using IMDb data.  
- How to clean and preprocess text for natural language processing.  
- How to package and deploy a model as a live demo using Streamlit.  
- How to integrate an AI project into a professional portfolio.

This project shows end-to-end ownership: from dataset to model to deployment.
