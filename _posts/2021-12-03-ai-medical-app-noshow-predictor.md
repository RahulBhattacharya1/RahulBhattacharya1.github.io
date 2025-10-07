---
layout: default
title: "Building my AI Medical Appointment No-Show Predictor"
date: 2021-12-03 11:43:26
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
thumbnail_mobile: /assets/images/medical_appointment.webp
demo_link: https://rahuls-ai-medical-appointment-noshow-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_medical_appointment_noshow_predictor
---

The inspiration for this project came from a real situation that left me reflecting on how clinics manage patient schedules. I had once booked an appointment, carefully arranged my commitments around it, and arrived early. The clinic, however, was backed up because several patients simply did not come. I realized that every no-show not only wasted the physician’s time but also created cascading delays for those who did arrive. This inefficiency was not just a matter of personal inconvenience. Dataset used [here](https://www.kaggle.com/datasets/joniarroba/noshowappointments).

Later, I thought about how technology might help reduce such waste. If we could predict the probability that a patient might skip an appointment, clinics could act. They could schedule reminders more aggressively for higher-risk patients or even slightly overbook to balance the schedule. That thought became the seed for this project. The goal was to create a practical yet educational app that demonstrates how machine learning models can solve a real-world issue.

---

## Repository Overview

The repository is structured to be simple yet functional. The top-level folder includes a README, a requirements file, and subfolders for the application, data, and models. Each element was carefully chosen to make deployment seamless.

- **README.md** explains the purpose of the project in plain text.  
- **requirements.txt** ensures dependencies can be replicated consistently.  
- **app/app.py** contains the Streamlit code that serves as the user-facing interface.  
- **data/README.md** explains the data folder, which is empty here due to privacy concerns.  
- **models/appointment_noshow_pipeline.joblib** is the pre-trained machine learning model serialized with joblib.

This organization follows best practices for small-scale machine learning demos. It separates code, model, and data clearly. It also ensures that if someone forks or deploys the repository, they know where everything belongs.

---

## requirements.txt in Detail

```python
streamlit
pandas
numpy
scikit-learn
joblib
```

This small file defines the environment. Every project benefits from an explicit requirements file. Streamlit enables the web interface, Pandas handles structured data, Numpy supports array math, Scikit-learn powers the machine learning, and Joblib ensures the pipeline can be loaded. These five libraries are enough to run the application.

I decided to keep dependencies minimal. Additional libraries could provide more features, but they would bloat the environment and make deployment harder. By limiting to essentials, the application remains lightweight. Anyone can deploy it on Streamlit Cloud without exceeding resource limits.

---

## README.md Purpose

The README is short but meaningful. It outlines what the project is, how it helps, and what one needs to run it. While not containing technical detail, it sets context. Without a README, a visitor might not know why the project exists. For this reason, I never consider a repository complete until it has a README.

---

## app/app.py Breakdown

The Streamlit application is the core of this project. It contains everything from interface design to model invocation.

### Imports and Setup

```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, date

st.set_page_config(page_title="Medical No-Show Predictor", page_icon="📅", layout="centered")
```

This block imports necessary libraries. I also configure the Streamlit app’s appearance. The centered layout helps keep forms aligned. Without such configuration, the app might look cramped on smaller devices. The inclusion of the calendar icon makes the application visually aligned with its theme. It is these small details that make a project feel polished.

### Title and Introductory Text

```python
st.title("Medical Appointment No-Show Predictor")
st.write("Predict the probability that a patient will miss an appointment.")
```

This establishes branding inside the app. Clear messaging is important. I use a concise one-line subtitle so that any user instantly understands purpose. Too much text at this stage would overwhelm. This balance of clarity and brevity is something I consider essential in user interface design.

### Cached Pipeline Loader

```python
@st.cache_resource
def load_pipeline():
    path = "models/appointment_noshow_pipeline.joblib"
    return joblib.load(path)
```

Here I introduce caching. Streamlit allows functions to be cached so that repeated calls do not reload heavy resources. In my case, the joblib pipeline is loaded once. Subsequent calls reuse the same object. This dramatically improves responsiveness. Without caching, every click would reload the model and slow the app. The decision to use caching therefore impacts both user experience and server resource usage.

### Days Wait Helper Function

```python
def compute_days_wait(scheduled_date, appointment_date):
    if scheduled_date and appointment_date:
        delta = (appointment_date - scheduled_date).days
        return max(delta, 0)
    return None
```

This function looks small but carries weight. It extracts the waiting period, a known factor influencing no-shows. If dates are inverted or invalid, it avoids negative numbers by returning zero. I added this safeguard because raw date differences can produce negative values if users input them incorrectly. The helper embodies one of my design philosophies: protect the pipeline from bad input.

### Model Loading with Error Handling

```python
pipe = None
load_error = None
try:
    pipe = load_pipeline()
except Exception as e:
    load_error = str(e)
```

This section shows resilience. Instead of assuming the model will load, I handle exceptions. If the joblib file is missing or corrupted, the app does not collapse. Instead, it stores the error message for later display. Such defensive programming prevents failures in real-world deployments. It also helps debugging because the error message is visible instead of hidden.

### Input Form Creation

```python
st.subheader("Input")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["F", "M"])
    age = st.number_input("Age", min_value=0, max_value=120, value=40, step=1)
    scholarship = st.selectbox("Scholarship (0 = No, 1 = Yes)", [0, 1], index=0)
    sms_received = st.selectbox("SMS Received (0 = No, 1 = Yes)", [0, 1], index=0)

with col2:
    hipertension = st.selectbox("Hipertension (0 = No, 1 = Yes)", [0, 1], index=0)
    diabetes = st.selectbox("Diabetes (0 = No, 1 = Yes)", [0, 1], index=0)
    alcoholism = st.selectbox("Alcoholism (0 = No, 1 = Yes)", [0, 1], index=0)
    handcap = st.number_input("Handcap (0-4)", min_value=0, max_value=4, value=0, step=1)
```

This block demonstrates how user-friendly interfaces can gather structured data. I split fields into two columns for clarity. Column one gathers demographic and communication fields, while column two handles medical background. Each field matches a feature the model expects. I intentionally use dropdowns and number inputs to restrict invalid entries. This decision reduces data entry errors. It also mirrors clinical intake forms where options are predefined.

### Appointment Timing Options

```python
st.markdown("---")
st.subheader("Appointment Timing")

use_dates = st.checkbox("Use Scheduled and Appointment Dates to compute DaysWait", value=True)
if use_dates:
    c1, c2 = st.columns(2)
    with c1:
        scheduled = st.date_input("Scheduled Date", value=date(2016, 5, 1))
    with c2:
        appointment = st.date_input("Appointment Date", value=date(2016, 5, 3))
    days_wait = compute_days_wait(scheduled, appointment)
    st.write(f"Computed DaysWait: {days_wait} day(s)")
else:
    days_wait = st.number_input("DaysWait", min_value=0, max_value=365, value=0, step=1)
```

I wanted users to have flexibility. Some might not want to pick exact dates and prefer to type days directly. The checkbox provides this choice. If selected, the helper function calculates days automatically. If not, users can manually specify. Transparency is enhanced because I display the computed days back to the user. That feedback loop builds trust in the system.

### Constructing Data and Making Predictions

```python
if pipe and not load_error:
    input_data = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Scholarship": scholarship,
        "SMS_received": sms_received,
        "Hipertension": hipertension,
        "Diabetes": diabetes,
        "Alcoholism": alcoholism,
        "Handcap": handcap,
        "DaysWait": days_wait
    }])

    if st.button("Predict"):
        try:
            prob = pipe.predict_proba(input_data)[0][1]
            st.success(f"Predicted probability of no-show: {prob:.2f}")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
else:
    st.error(f"Model load error: {load_error}")
```

This final block integrates all parts. I build a Pandas DataFrame from user input. This ensures that data aligns with the structure the pipeline expects. Then I add a button for prediction. When clicked, the pipeline returns probabilities, and I extract the no-show likelihood. The result is presented with formatting. Errors are again captured to prevent app crashes. This complete cycle—input, processing, prediction, output—embodies the application’s purpose.

---

## Data Folder

The `data` folder only contains a README in this project. The absence of raw medical data is intentional. Privacy concerns prevent sharing real datasets. However, I keep the folder as a placeholder. It signals to anyone extending the project where data should be placed. Documentation matters even for empty folders. It communicates intention.

---

## Models Folder

The model file inside `models` is the heart of the predictive system. It contains the trained scikit-learn pipeline saved as a joblib file. This pipeline includes preprocessing steps and the final classifier. By separating training and deployment, the app stays lean. The joblib file is simply loaded and used. This modularity makes it possible to retrain with new data without changing the app code.

---

## Reflections on Design Choices

### Why Streamlit?

I selected Streamlit because it bridges the gap between machine learning and accessible interfaces. A data scientist can quickly build a prototype without heavy front-end development. This decision mattered because I wanted to focus on the logic, not JavaScript or CSS. Streamlit also integrates well with cloud hosting, allowing a single Python script to become a web app in minutes. I find that empowering, especially for portfolio projects.

### Importance of Feature Engineering

The `DaysWait` feature might seem small, but it captures critical behavior. When patients wait too long, forgetfulness or competing priorities increase the chance of absence. Translating this social observation into a numeric feature is a key example of feature engineering. It shows how domain understanding feeds into modeling. Without such features, the model would rely only on static demographics, which limits predictive power.

### Handling User Inputs

I intentionally restricted fields to dropdowns and bounded number inputs. This prevents unrealistic values like negative age or a handcap of 10. In production systems, robust validation is necessary because user inputs are unpredictable. My implementation is simple, but it demonstrates the idea. A clean form also makes the app easier to use.

---

## Deployment Considerations

Hosting this project on GitHub and Streamlit Cloud demonstrates portability. A user only needs to clone the repository and deploy. Because the `requirements.txt` file is lightweight, the environment builds quickly. The model file is small enough to fit under free hosting limits. This shows that even with resource constraints, meaningful projects can be shared.

I also avoided including sensitive data. Many beginner projects mistakenly commit raw datasets into repositories. That creates privacy risks. By using only a model artifact and a placeholder README in the data folder, I highlight responsible sharing practices.

---

## Lessons Learned

This project taught me several lessons:

1. **Simplicity Works**: A small app can make a large point. It does not need hundreds of files.  
2. **Documentation Matters**: Writing this blog took more effort than the code itself. Yet it is necessary for others to understand the design.  
3. **Error Handling is Essential**: Without try-except blocks, the app could fail silently. That would hurt trust.  
4. **UI Design Impacts Adoption**: Splitting forms into two columns and using dropdowns improved usability.  
5. **Caching Improves Performance**: The cached pipeline ensures fast response even when many users interact with the app.

These lessons extend beyond this single project. They represent practices I will apply to future work.

---

## Future Enhancements

I see several directions to extend this project:

- **Integration with SMS Gateways**: The predictor could automatically send reminders to high-risk patients.  
- **Improved Feature Set**: Additional features like distance from clinic or weather conditions could improve accuracy.  
- **Dashboard for Clinics**: A clinic might want to view aggregate predictions across all scheduled patients for a day.  
- **Model Retraining**: Automating retraining with new appointment data would keep predictions current.  
- **Explainability**: Adding SHAP or LIME explanations could help clinics trust the predictions.

Each enhancement would turn this demo into a more production-ready tool. For now, the focus is on simplicity and clear demonstration.

---

## Broader Perspective

This project belongs to a wider set of applied machine learning problems. Healthcare scheduling is just one area where predictive modeling helps. Similar approaches can be used in education (student attendance), retail (customer visits), or logistics (delivery completion). What matters is the principle: using past behavior and contextual data to predict future outcomes.

The model is not perfect, and it is not intended to replace human judgment. Instead, it serves as an assistant, offering probabilities that inform decision-making. By documenting the reasoning behind each file and code block, I show how technical solutions emerge from real frustrations.

---

## Closing Thoughts

When I reflect on this project, I see more than code. I see the path from observation to solution. A small annoyance in a waiting room sparked a chain of learning, design, and sharing. This is why I value building portfolio projects. They turn abstract concepts into tangible demonstrations. They also show employers and collaborators that I can take an idea from inception to deployment, with documentation to match.

This blog post is deliberately long and detailed because I wanted to simulate a full walkthrough. Anyone who reads it should be able to recreate the project, understand the reasoning, and adapt it for their own ideas. That is the purpose of technical writing: not only to record but to empower others.

---


---

## Hypothetical Training Process

Although the repository does not contain raw training code, I want to describe what the pipeline likely involved. The dataset often used in no-show prediction tasks contains demographic and appointment details. The pipeline would include steps like:

1. **Data Cleaning**: Handling missing values, correcting inconsistent entries, and converting strings to categories.  
2. **Feature Engineering**: Creating `DaysWait` as described, encoding gender, and grouping ages.  
3. **Encoding**: Transforming categorical variables into numeric form, often with one-hot encoding.  
4. **Model Choice**: Logistic regression or random forest are typical choices due to interpretability.  
5. **Validation**: Splitting into training and testing sets, evaluating with accuracy and AUC.  
6. **Serialization**: Saving the pipeline as a joblib file to ensure reproducibility.

This process emphasizes the division of concerns. Training is heavy and not part of the Streamlit app. Prediction is lightweight and belongs inside the interface.

---

## Alternative Implementations

If I wanted to build the same predictor differently, I could have used Flask or Django as the web framework. That would give more flexibility at the cost of complexity. Another alternative would be to build a desktop tool using Tkinter or PyQt. Yet I chose Streamlit because my aim was demonstration, not enterprise deployment.

For the model, deep learning could be applied. A neural network might capture nonlinear relationships better. However, for tabular data with limited features, scikit-learn models are usually more practical. They are faster, interpretable, and require fewer resources. Again, the decision reflects balancing accuracy with usability.

---

## Detailed Code Block Expansions

### Error Handling Philosophy

The try-except block around pipeline loading is small but symbolic. It represents the philosophy that user-facing applications must not fail silently. Every piece of code that deals with external files or user input should have protection. Otherwise, one missing file could break the experience. In my projects, I always prioritize graceful failure. Showing a clear error message to the user is far better than leaving them with a blank screen.

### Prediction Trigger

The prediction occurs only when the button is clicked. This is intentional. Continuous prediction with every form change would overload the pipeline and confuse the user with changing outputs. By requiring a deliberate click, I give the user control. It also simulates how real systems often require a submit action before generating results.

### Success and Error Messages

The `st.success` and `st.error` calls are about communication. Green for success, red for error. These visual cues immediately tell the user what happened. Even if the text message were ignored, the color signals outcome. That matters in building trust with the interface.

---

## Reflections on Documentation

As I wrote this blog, I realized that documenting each function made me more aware of design choices. Often, developers code without reflecting. Writing forces clarity. For example, explaining the helper function reminded me that small safeguards can prevent major issues. Similarly, describing the input form highlighted the balance between flexibility and control.

This experience reinforces why I value blogging alongside coding. It is not just about showing finished products but about exposing the thought process. That transparency builds credibility.

---

## Broader Applications

Predicting no-shows is not only about healthcare. Airlines overbook flights based on no-show probabilities. Restaurants use similar logic for reservations. Even software services rely on churn prediction, which is a conceptual cousin. By working on this project, I connected a local observation to global practices. It reminded me that data-driven prediction is a universal tool.

---

## Conclusion

This project demonstrates how personal observation can lead to a working technical solution. What started as a thought in a waiting room became a functioning predictor hosted online. By explaining every file and every block, I hope I have made clear how a project should be documented. Each design decision, from input validation to error handling, reflects lessons learned. The blog itself is long because I believe context matters.

The final reflection is that projects like this bridge imagination and execution. A small idea—what if we predicted clinic no-shows—turns into a working demonstration. By walking through each line and file, I turned a simple script into a teachable artifact. Anyone reading this blog can now follow the logic, learn from the design, and perhaps build their own version.

I am glad I invested the time to make this explanation detailed. It shows not only technical ability but also communication. For me, that combination defines a strong engineer: someone who can build and also explain. That is the message I wanted my portfolio to send.
