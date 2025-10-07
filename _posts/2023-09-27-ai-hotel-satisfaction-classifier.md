---
layout: default
title: "Creating my AI Hotel Satisfaction Predictor"
date: 2023-09-27 10:56:29
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/hotel.webp
thumbnail_mobile: /assets/images/hotel_satisfaction_sq.webp
demo_link: https://rahuls-ai-hotel-satisfaction-classifier.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_hotel_satisfaction_classifier
---

A few years ago, while traveling through Europe, I stayed at a hotel that looked great on paper. The online reviews praised certain aspects like the wifi and comfort but criticized others like the food and cleanliness. When I finally stayed there, I noticed that my own impression was a mix of these reviews. The room was comfortable, but the restaurant quality was indeed low. That trip made me realize how subjective satisfaction is, and how much it depends on specific features that matter differently to each scenario. Dataset used [here](https://www.kaggle.com/datasets/ishansingh88/europe-hotel-satisfaction-score).

Later, as I learned more about machine learning, I wanted to connect real-life scenarios with predictive systems. I asked myself: what if there was a way to predict whether someone like me would be satisfied with a hotel stay, based on a set of features? The idea turned into this project. It became a tool that accepts structured information like travel purpose, booking type, and service ratings, and then produces a satisfaction prediction. Streamlit made it easy to present the idea as an interactive web app.

---

## Project Structure

The unzipped folder contained three major components:

- `app.py`: The Streamlit application script that defines the layout, input widgets, model loading, and output display. This is the heart of the project.
- `requirements.txt`: The dependency file that ensures the same environment can be recreated by anyone running the app. Each line locks a specific version of a package.
- `models/hotel_satisfaction_pipeline.joblib`: The trained pipeline saved with joblib. This binary object contains both preprocessing and the classifier that interprets inputs.

Each file had a specific role. The script controlled the flow, the requirements provided consistency, and the model file held the intelligence.

---

## requirements.txt Explained in Context

```python
streamlit
pandas==2.2.2
scikit-learn==1.6.1
numpy==2.0.2
scipy==1.16.1
joblib==1.5.2
```

### Streamlit
Streamlit enabled me to focus on user experience without writing raw HTML or JavaScript. It provided high-level widgets like `selectbox` and `slider` which I used to capture categorical and numerical values. It also offered functions like `st.success` and `st.dataframe` to render results. Without Streamlit, I would have spent significant time building a frontend.

### Pandas
The model pipeline expected data in a tabular format. Pandas was perfect for that because I could take all user input, wrap it into a dictionary, and then convert it into a single-row DataFrame. By explicitly setting columns equal to `feature_cols`, I ensured alignment with training. Pandas also made it easy to structure probability outputs into a sortable DataFrame for display.

### Scikit-learn
This library powered the training stage. Although not visible in app.py, the model artifact was trained using scikit-learn. The joblib file contained an entire pipeline object, which could include encoders, scalers, and a classifier. When I loaded the file, I instantly got access to `predict` and `predict_proba`. This minimized boilerplate code inside the app.

### NumPy
Even though I did not import NumPy directly in app.py, it remained a silent backbone. Every pandas DataFrame relies on NumPy arrays internally. Every scikit-learn prediction turns into NumPy operations. In this context, NumPy ensured that input arrays and model computations were efficient.

### SciPy
Scikit-learn internally uses SciPy for optimization and probability distributions. For example, if the classifier was logistic regression, SciPy handled optimization routines. Including SciPy in the requirements was essential for the pipeline to function smoothly, even if I never called it directly in the script.

### Joblib
Joblib connected training with deployment. The pipeline was serialized using joblib during training. In app.py, I used `joblib.load` to restore it. This provided a direct bridge between offline model development and real-time predictions in the web interface.

---

## app.py: Full Breakdown

### Import Section

```python
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
```

This section defined the foundation. Each library had a clear role. Streamlit gave the interface, pandas structured inputs, joblib restored the model, and Path handled filesystem checks in a safe way. By importing only what was needed, the script remained minimal yet functional.

### Page Setup

```python
st.set_page_config(page_title="Europe Hotel Booking Satisfaction Predictor", layout="centered")
st.title("Europe Hotel Booking Satisfaction Predictor")
```

I configured the page to have a centered layout and set a meaningful page title. A descriptive title reduces confusion for users. These two lines focused on presentation, making the app look professional rather than like a plain script.

### Model Loading

```python
MODEL_PATH = Path("models/hotel_satisfaction_pipeline.joblib")
if not MODEL_PATH.exists():
    st.error("Model file not found at models/hotel_satisfaction_pipeline.joblib. Please upload it.")
    st.stop()

pipe = joblib.load(MODEL_PATH)
```

This logic served as a safeguard. Instead of letting the script crash if the file was missing, it displayed an informative error. The conditional `if not MODEL_PATH.exists()` checked presence. `st.error` communicated the problem clearly. `st.stop` halted execution gracefully. Finally, if the file existed, joblib loaded the pipeline. This section demonstrated defensive programming.

### Feature Column Definition

```python
feature_cols = [
    "Gender",
    "Age",
    "purpose_of_travel",
    "Type of Travel",
    "Type Of Booking",
    "Hotel wifi service",
    "Hotel location",
    "Food and drink",
    "Stay comfort",
    "Cleanliness",
]
```

This list defined the schema expected by the pipeline. The names matched exactly with those used during training. A mismatch here would lead to runtime errors. By explicitly setting the order, I ensured that the DataFrame columns aligned with the model’s expectations. This section acted as a bridge between human inputs and machine requirements.

### Input Form Layout

```python
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 90, 30)
        purpose = st.selectbox("Purpose of Travel", ["Business", "Leisure"])
        travel_type = st.selectbox("Type of Travel", ["Solo", "Group"])
        booking = st.selectbox("Type Of Booking", ["Online", "Offline"])

    with col2:
        wifi = st.slider("Hotel wifi service", 1, 5, 3)
        location = st.slider("Hotel location", 1, 5, 3)
        food = st.slider("Food and drink", 1, 5, 3)
        comfort = st.slider("Stay comfort", 1, 5, 3)
        cleanliness = st.slider("Cleanliness", 1, 5, 3)

    submitted = st.form_submit_button("Predict Satisfaction")
```

This was the most interactive part. By wrapping widgets inside a form, I ensured that values were only processed when the user clicked the submit button. Dividing into two columns improved readability. The left column handled demographic and booking details. The right column focused on service ratings. The use of sliders with defined ranges minimized invalid inputs. The form structure reduced errors by grouping all inputs logically.

### Handling Submission

```python
if submitted:
    row = {
        "Gender": gender,
        "Age": age,
        "purpose_of_travel": purpose,
        "Type of Travel": travel_type,
        "Type Of Booking": booking,
        "Hotel wifi service": wifi,
        "Hotel location": location,
        "Food and drink": food,
        "Stay comfort": comfort,
        "Cleanliness": cleanliness,
    }
    X = pd.DataFrame([row], columns=feature_cols)
```

Once submitted, the script created a dictionary with all user inputs. This dictionary was then wrapped into a pandas DataFrame. The explicit `columns=feature_cols` enforced ordering. This step was critical because scikit-learn pipelines map features by position as well as name. Without strict alignment, predictions could have been nonsensical.

### Prediction Logic

```python
    pred = pipe.predict(X)[0]
    proba = getattr(pipe, "predict_proba", None)
    if proba is not None:
        probs = pipe.predict_proba(X)[0]
        if isinstance(pred, str):
            label = pred
        else:
            label = str(pred)
        st.success(f"Prediction: {label}")
        st.write("Class probabilities (sorted):")
        prob_df = pd.DataFrame({"class": pipe.classes_, "probability": probs})
        prob_df = prob_df.sort_values("probability", ascending=False).reset_index(drop=True)
        st.dataframe(prob_df, use_container_width=True)
    else:
        st.success(f"Prediction: {pred}")
```

This block executed the pipeline prediction. `pipe.predict` gave the predicted label. The use of `getattr` checked if `predict_proba` existed. If probabilities were available, they were retrieved and displayed as a sorted table. This transparency allowed users to see not just the top prediction but also how confident the model was in each class. The `st.success` call provided immediate positive feedback. If probabilities were not available, the app still showed the raw prediction.

---

## Deployment Considerations

I deployed this app using Streamlit Cloud. The presence of `requirements.txt` ensured that the environment matched exactly. The model file was placed in the `models` folder to be loaded at runtime. I tested the app with different inputs to confirm stability. Errors such as missing model files were handled gracefully. This made the app reliable even in shared environments.

---

## Lessons Learned

- Defensive programming is critical. By checking for file existence, I avoided silent crashes.
- Explicit feature column ordering prevents subtle bugs in predictions.
- User experience matters. By splitting widgets into columns, the form became more approachable.
- Transparency builds trust. Showing probabilities helped users understand the model’s confidence.
- Each library contributed a specific layer of value. Streamlit handled UI, pandas structured data, scikit-learn modeled, NumPy and SciPy supported computation, and joblib bridged training with deployment.

---

## Conclusion

This project started from a travel experience and evolved into a machine learning application. It showed me how personal insights can inspire data-driven tools. By carefully combining logic, libraries, and interface design, I built an app that predicts hotel satisfaction. More than a technical exercise, it reminded me that user trust and clarity matter as much as prediction accuracy. Every decision, from widget design to error handling, contributed to a smooth user journey.
