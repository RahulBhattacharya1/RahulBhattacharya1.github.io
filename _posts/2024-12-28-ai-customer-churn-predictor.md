---
layout: default
title: "Building my AI Customer Response Predictor"
date: 2024-12-28 10:16:47
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/customer.webp
thumbnail_mobile: /assets/images/customer_response_sq.webp
demo_link: https://rahuls-ai-customer-churn-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_customer_churn_predictor
---

It started with a moment of curiosity when I was analyzing campaign reports. Numbers showed that many people never reacted to offers despite similar spending patterns. I began to wonder if there was a way to anticipate which customers would likely respond before sending the next promotion. That thought stayed with me as I considered the cost of each campaign and the missed opportunity when the wrong people were targeted. Predicting customer response became not just a technical project but a way to align marketing actions with data-driven insight. Dataset used [here](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis).

I imagined an application where input fields captured details about a customer’s spending habits and demographic information. Behind the scenes a model could use those signals to predict the likelihood of a response. The challenge was not just building a classifier but packaging it into a tool that marketing teams could actually use. Streamlit offered an ideal balance of interactivity and speed, so I chose it to bring this idea to life. What followed was a series of steps, each tied to a file that I uploaded into GitHub to complete the project.

---

## requirements.txt

The first file I prepared was the dependency list. It ensured that anyone running the project would have the same versions of the core libraries. Reproducibility is crucial, especially when a model is involved.

```python
streamlit>=1.37
scikit-learn==1.6.1
joblib==1.5.2
pandas==2.2.2
numpy==2.0.2

```

Each line pins a version. Streamlit provided the interface, scikit-learn trained the model, joblib handled saving and loading, pandas helped with data manipulation, and numpy supported the numerical work. Without this file, the project would easily break across machines.

---

## app.py

The central file of this project is `app.py`. It defined the interface and linked the user inputs with the machine learning model. Below is the entire script.

```python
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/response_model.pkl")

st.title("Customer Response Predictor")
st.write("Predict if a customer will respond to a marketing campaign")

# User inputs
age = st.number_input("Age", 18, 100, 40)
income = st.number_input("Income", 1000, 200000, 50000)
kidhome = st.number_input("Kids at home", 0, 5, 0)
teenhome = st.number_input("Teens at home", 0, 5, 0)
recency = st.slider("Recency (days since last purchase)", 0, 100, 30)
mnt_wines = st.number_input("Wine Spend", 0, 2000, 100)
mnt_meat = st.number_input("Meat Spend", 0, 2000, 100)
mnt_fish = st.number_input("Fish Spend", 0, 2000, 50)
mnt_sweets = st.number_input("Sweet Spend", 0, 2000, 20)
mnt_gold = st.number_input("Gold Spend", 0, 2000, 20)
deals = st.number_input("Deals Purchases", 0, 20, 1)
web = st.number_input("Web Purchases", 0, 20, 2)
catalog = st.number_input("Catalog Purchases", 0, 20, 1)
store = st.number_input("Store Purchases", 0, 20, 3)

# Predict
if st.button("Predict"):
    data = [[age,income,kidhome,teenhome,recency,
             mnt_wines,mnt_meat,mnt_fish,mnt_sweets,mnt_gold,
             deals,web,catalog,store]]
    pred = model.predict(data)[0]
    st.success("✅ Will Respond" if pred==1 else "❌ Will Not Respond")

```

### Imports and Model Loading

The script begins with imports of streamlit, pandas, and joblib. Streamlit drives the interface. Pandas is present even though no heavy data manipulation happens inside this script; it remains useful when integrating with more complex workflows. Joblib loads the trained model from the `models/response_model.pkl` file. This step connects the deployed app with the offline training process. Without the model being loaded, the interface would not have predictive power.

### Title and Description

The `st.title` function places a header at the top of the app. It signals clearly to users what the application does. Right after, `st.write` adds a smaller description to set context. These two commands give the app a professional look and reassure the user that the interface has purpose.

### User Input Widgets

The next section of code declares a series of input fields. They collect age, income, number of kids, number of teens, recency, and spending amounts across different categories. I used `st.number_input` for most of them. Each call sets a label, a minimum, a maximum, and a default value. This prevents invalid input and provides guidance to the user. For example, age is constrained between 18 and 100, which reflects a realistic customer range. Income is capped at 200000 to cover typical household values within the training dataset.

Recency is handled with `st.slider`. This choice gives a better sense of scale since days since last purchase is naturally visual. The spending amounts like wine, meat, fish, sweets, and gold are captured with similar number inputs. Each spending type reflects categories that were present in the dataset used to train the model. By aligning inputs with training data the app ensures consistency between prediction time and training time.

### Channel Purchases

I also included number of deals purchased, web purchases, catalog purchases, and store purchases. These represent interaction channels and behavior differences. Someone who buys through web might respond differently than someone who prefers physical stores. The model captures these subtle signals.

### Prediction Logic

At the bottom of the script lies the predictive trigger. The `st.button("Predict")` creates a button. When clicked, it executes the block of code beneath. Inside, I gather all inputs into a nested list. This matches the structure expected by scikit-learn models, where each row is an observation and each column is a feature.

The line `pred = model.predict(data)[0]` calls the model. It outputs a prediction of either 1 or 0. That value determines whether the customer is expected to respond. Finally, the app displays feedback using `st.success`. I chose clear messages: "Will Respond" or "Will Not Respond". These map binary values into human language. The conditional ensures that the correct message appears based on the model output.

---

## models/response_model.pkl

This file is the trained model itself. I used scikit-learn to build it offline and saved it with joblib. The file size is about 500 KB, which is small enough for version control but large enough to contain the coefficients and structure of the classifier. Uploading this file into the `models` directory made it possible for `app.py` to load it directly. Without this artifact, the interface would only show input fields without any predictive ability.

---

## Detailed Walkthrough of Each Functionality

### Streamlit Number Input Helper

The `st.number_input` calls act like helpers. They not only collect values but also enforce validation. Each call ensures the input falls into expected ranges. This prevents the model from receiving unrealistic data. For example, without constraints one might enter negative income or more than ten children. That would not align with the dataset distribution. The helper thus maintains integrity.

### Slider for Recency

The slider serves a dual purpose. It visualizes the distance from the last purchase while also constraining the number to a realistic range. Recency is a strong predictor in many customer churn and response studies. By making it easy to set, the app encourages exploration.

### Conditional for Prediction

The conditional tied to the predict button is simple yet powerful. It isolates model execution to a deliberate user action. This prevents unnecessary computation and gives the user control. Only when the button is clicked does the model run. This design aligns with interactive analysis where actions should be explicit.

### Model Prediction

The prediction itself is encapsulated in a single line but carries weight. It converts structured customer data into a binary outcome. This transformation from many features to a clear signal is what makes machine learning valuable in practice. The surrounding code frames this transformation within a user-friendly flow.

---

## Reflections

When I built this project I learned how small details in interface design matter as much as the underlying model. Setting reasonable input ranges, labeling fields clearly, and ensuring quick feedback improved usability. Even though the model could have been wrapped in a bare script, packaging it in Streamlit made it accessible. It turned raw prediction into a conversation between user and data.

---

## Closing Thoughts

This project combined three essential files: a dependency list, a Python script, and a model artifact. Together they formed a deployable application. Every block of code contributed a role. Helpers like input functions guarded validity. Conditionals structured flow. The model bridged past training with present interaction. Writing about it now, I realize how much thought goes into what may look like a short script. Behind every line stands a decision about usability, accuracy, and communication.

---


## Reflections

### Feature Importance in Context
Each input feature I selected carries meaning beyond just a number. Age links to maturity of spending habits. Income sets the boundary of what kind of promotions resonate. Recency captures memory of the last interaction and signals if the customer is drifting away. Spending categories reveal lifestyle choices. Deals and channel preferences show responsiveness to marketing channels. Together they paint a portrait of behavior that a model can transform into probabilities.

### Why Streamlit Was Ideal
I considered alternatives like Flask or Django, but they require more boilerplate and templates. Streamlit made it possible to build a prototype in a few lines. The interactive widgets lowered the barrier for non-technical users to experiment with inputs. This immediacy turned the model into a tool, not just code. That tradeoff between flexibility and simplicity worked in favor of speed, which mattered for this use case.

### Handling Model Artifacts
Saving a model is not just about persistence. It is about ensuring that weights, hyperparameters, and encoders stay intact across environments. Joblib offered a straightforward solution. Uploading the model into the repository created a bridge between offline training and online usage. This discipline avoids the common pitfall of models that work on one machine but fail elsewhere.

### On Validation and Constraints
I could have left inputs open-ended, but then predictions would lose credibility. Constraining age, income, and purchase counts grounded the app in reality. Validation is not restrictive—it guides. Each constraint encoded assumptions from the training dataset into the interface. This way users stay within the data distribution, which is critical for reliable predictions.

### Limitations and Improvements
The model behind the predictor is static. It reflects the patterns of the training dataset but cannot evolve without retraining. In practice, customer behavior shifts with time and context. Seasonal changes, new products, or shifts in economy all alter response rates. A future improvement would be automating retraining with pipelines so the app stays current. Another idea is adding probability scores instead of binary labels to capture uncertainty.

### Ethical Dimensions
Predicting human behavior always raises questions of fairness and ethics. Income as a feature might disadvantage certain groups. Spending categories can reveal sensitive information. Deploying such predictors responsibly means auditing for bias and ensuring transparency. In my project the goal was technical exploration, but I remain mindful of how similar systems operate in real businesses.

### Deployment Lessons
Deploying with Streamlit was smooth, but scaling to heavy usage would require thought. Streamlit apps are ideal for internal teams or demos. For production at scale, containerization and orchestration with Docker and Kubernetes would be more robust. Still, as a personal learning project, the simplicity of pushing to GitHub and running in Streamlit Cloud was more than enough.

### Personal Growth
Working on this app reminded me that even short scripts carry complexity when unpacked. Explaining every block forced me to explain why each line exists. This practice deepened my appreciation for documentation as part of development, not an afterthought. It also made me more confident in communicating technical details in plain language.
