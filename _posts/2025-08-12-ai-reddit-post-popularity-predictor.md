---
layout: default
title: "Building my AI Reddit Post Popularity Predictor"
date: 2025-08-12 10:45:16
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/reddit.webp
thumbnail_mobile: /assets/images/reddit_post_sq.webp
demo_link: https://rahuls-ai-reddit-post-popularity-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_reddit_post_popularity_predictor
---

I remember browsing through online forums and noticing how some posts gained massive attention while others faded away quickly. I wanted to know if there was a way to anticipate this behavior in advance. That thought slowly grew into a project idea where I tried to predict the popularity of Reddit posts. The result was a simple but effective web application powered by machine learning. Dataset used [here](https://www.kaggle.com/datasets/gpreda/reddit-vaccine-myths).

This blog post is my attempt to document everything about the project. I will go through each file, explain every piece of code, and show why each helper or conditional was written the way it was. My goal is to write this in a natural style so anyone reading can follow without needing a deep technical background.

---

## Project Structure

When I unzipped the project into my GitHub repository, the following files were included:

- `app.py`  
- `requirements.txt`  
- `models/reddit_popularity_model.pkl`  

Each of these files serves a very specific purpose. The `app.py` file is the main driver of the application. It contains the code to run the Streamlit app. The `requirements.txt` file lists all the dependencies required so that GitHub Pages or Streamlit Cloud can recreate the environment. The model file is the pre-trained machine learning model stored in serialized form, ready to be used.

---

## The `requirements.txt` File

The requirements file is very small, only five lines. But each line is critical. Here is how it looked:

```python
streamlit
scikit-learn
pandas
numpy
joblib
```

This file allowed me to control the Python environment. Without it, deploying the app on Streamlit Cloud would have been impossible. Streamlit is the framework I used to build the web interface. Scikit-learn is the machine learning library that provided training and prediction functions. Pandas was needed for handling structured input data. Numpy underpinned a lot of the mathematical operations in the background. Joblib was the utility I used to load the trained model from disk. Each of these libraries came together to form the foundation of the project.

---

## The `app.py` File

This file is short in terms of line count but very powerful. It contains the actual logic of the web application. Let us look at the entire file before breaking it into blocks:

```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("models/reddit_popularity_model.pkl")

st.title("Reddit Post Popularity Predictor")

title = st.text_input("Enter the Reddit post title:")
upvotes = st.number_input("Enter number of upvotes:", min_value=0)
comments = st.number_input("Enter number of comments:", min_value=0)
subscribers = st.number_input("Enter number of subreddit subscribers:", min_value=0)

if st.button("Predict Popularity"):
    features = pd.DataFrame([[upvotes, comments, subscribers]],
                            columns=["upvotes", "comments", "subscribers"])
    prediction = model.predict(features)
    st.write("Predicted Popularity:", prediction[0])
```

---

### Import Section

```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
```

This first block imports all the modules needed for the app. Streamlit is given the alias `st` for convenience. Pandas is imported as `pd` so that I can quickly build data frames from user input. Numpy is imported as `np`, though in this version I only relied lightly on it. Joblib is brought in to handle loading of the model file.

These imports make the script self-contained. The application would fail at the very first line if any of these libraries were missing. That is why listing them in `requirements.txt` was mandatory.

---

### Loading the Model

```python
model = joblib.load("models/reddit_popularity_model.pkl")
```

This line loads the serialized machine learning model from disk. I trained the model separately and stored it using joblib. By doing this, I avoided training at runtime which would be slow and costly. Instead, the app instantly has access to a ready-to-use model.

This is the backbone of the application. Without a model file the app could show an interface, but it would not produce any prediction. This design decision made the app very responsive to users.

---

### Setting the Title

```python
st.title("Reddit Post Popularity Predictor")
```

This line controls the header displayed at the top of the web application. Streamlit makes this process simple by providing a single function. The title gives users immediate context about what the app does. It was important for me to be explicit, so that visitors knew what to expect right away.

---

### Input Fields

```python
title = st.text_input("Enter the Reddit post title:")
upvotes = st.number_input("Enter number of upvotes:", min_value=0)
comments = st.number_input("Enter number of comments:", min_value=0)
subscribers = st.number_input("Enter number of subreddit subscribers:", min_value=0)
```

This block defines the input elements visible on the page. The first is a text input for the Reddit post title. Even though I did not feed the title text into the model, I added this field to make the app feel realistic. The next three are numeric inputs: upvotes, comments, and subscribers. I used `min_value=0` for each so the user could not enter negative values. This is a small validation step that kept the input clean.

These inputs are central to the model. They collect all the values that are then shaped into a data frame to be passed into the predictor.

---

### Predict Button

```python
if st.button("Predict Popularity"):
    features = pd.DataFrame([[upvotes, comments, subscribers]],
                            columns=["upvotes", "comments", "subscribers"])
    prediction = model.predict(features)
    st.write("Predicted Popularity:", prediction[0])
```

This conditional is the decision point. It only executes when the user presses the button. Inside the block, I first build a Pandas DataFrame with a single row. This row contains the values entered by the user for upvotes, comments, and subscribers. Naming the columns was important so that the model could map input correctly.

Once the DataFrame was ready, I passed it into the model's `predict` function. The model responded with an array containing the prediction. Since I only needed the first element, I accessed `prediction[0]`. Finally, I displayed the result back on the page using `st.write`. This cycle of input, prediction, and output is the essence of the app.

---

## The Model File

The third major component is `models/reddit_popularity_model.pkl`. This file was not written by hand. Instead, I created it by training a scikit-learn model offline. I then serialized it using joblib. The file extension `.pkl` indicates that it is a pickle-style object. Streamlit itself does not care about the format, but joblib makes sure it can reload the exact trained model.

The model holds all the learning that happened during the training phase. It encodes relationships between upvotes, comments, subscribers, and the final measure of popularity. The app depends entirely on this file for its predictive ability.

---

## Thoughts on Design

Writing this project taught me several lessons. Keeping the code short was beneficial. It allowed me to quickly deploy on free services without facing memory or runtime errors. At the same time, the model file carried the heavy lifting. This separation of logic kept the interface light and accessible.

Another key insight was about user experience. By keeping inputs clear and limited, I made the app easier to use. Too many fields might have scared away users. Instead, I balanced realism with simplicity.

---

## Possible Extensions

There are many ways I could extend this project. I could include natural language processing on the post title itself, allowing the text content to influence the prediction. I could also include features like posting time or sentiment extracted from comments. Another extension would be storing user queries in a database so I can study usage patterns.

---

## Use Cases

One use case is for researchers studying online communities. They could simulate post popularity based on hypothetical input. Another use case is for marketers planning content campaigns. They can gauge if a topic is likely to succeed in a subreddit. It also has educational use. Students of machine learning can explore this small project to understand end-to-end deployment.

---

## Deep Dive Reflection

This section is written to reflect further on the design choices made during the project. I wanted to emphasize how critical each part of the workflow is when working in machine learning. The interface design through Streamlit gives an impression of simplicity but hides the complexity of the model behind it. I realized that exposing only the necessary fields leads to higher adoption. Many users do not want to see complicated forms. They want immediate results.

The serialized model file is another reflection point. By pre-training the model and storing it with joblib, I separated training from serving. This decision mirrored professional machine learning systems, where offline training and online serving are decoupled. It was a good practice for me to follow, and it taught me how industrial pipelines are often structured.

The choice of dependencies was minimalistic on purpose. Fewer dependencies mean fewer errors in deployment. It also lowers the risk of version conflicts. I kept only what was essential: streamlit, scikit-learn, pandas, numpy, and joblib. Each line in the requirements file earned its place. This discipline improved stability and maintainability.

Another reflection is about community value. This project might look small, but it represents the kind of analysis that happens on social media platforms every day. Predicting engagement is at the heart of content recommendation systems. Even though my model is basic, it demonstrates the mechanics that scale into recommendation engines seen on large platforms.

Finally, reflecting on my personal growth, this project gave me confidence to take on bigger ideas. Writing about it in such detail forced me to evaluate why I made each decision. That habit of reflection is what will guide me in future projects. I do not just write code now; I think about the why and the how.

---

## Closing Thoughts

When I first thought about this idea, it seemed small. But after finishing the deployment, I realized that even small projects carry valuable lessons. I learned how to package a model, how to connect it to a Streamlit interface, and how to host it online for others to use. That process gave me confidence for larger projects.

This post became long because I wanted to explain each block carefully. Every helper, conditional, and import played a role. In the end, this application is proof that simple design choices, when executed carefully, can produce a functional and valuable tool.

---
