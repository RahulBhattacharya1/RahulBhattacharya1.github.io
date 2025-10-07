---
layout: default
title: "Building my AI Water Quality Predictor"
date: 2024-04-22 09:16:41
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/water_quality.webp
thumbnail_mobile: /assets/images/water_quality_sq.webp
demo_link: https://rahuls-ai-indian-water.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_indian_water
---

It started with a simple thought while pouring a glass of water one late evening. The clarity of the water looked fine, but I wondered what invisible details could be hidden in that liquid. Access to clean water is something I never take lightly, and the idea of using machine learning to check water quality seemed both practical and exciting. That is how I began this small project, where I built an application that predicts water potability using trained models.

This project was not born in a laboratory. It came from daily life, where questions about safety and quality occur naturally. I wanted to see if I could take publicly available datasets, train a predictive model, and then wrap it all in an application that could be easily shared. The result was this project, built with Streamlit, Python, and a machine learning model saved in joblib format. What follows is a complete breakdown of every file, every block of code, and how each piece helps form the working system. Dataset used [here](https://www.kaggle.com/datasets/rishabchitloor/indian-water-quality-data-2021-2023).

---

# Step 1: Understanding the Project Structure

The project folder `ai_indian_water-main` contains four key components:

- **`app.py`** – the Streamlit application script that ties everything together.  
- **`requirements.txt`** – a list of dependencies needed to run the application.  
- **`models/schema.json`** – the JSON file that defines which inputs the model requires.  
- **`models/water_model.joblib`** – the trained machine learning model itself.  

Each of these files plays a different but equally important role. To make the system robust, I needed to ensure that they all worked in harmony. Let us now dive into them one by one, expanding each code block and reasoning behind it.

---

# Step 2: requirements.txt – Ensuring Reproducibility

A project is only useful if it can be replicated by others. That is why the first step was defining dependencies.

```python
streamlit
scikit-learn
pandas
joblib
```

### Why this matters
- **Streamlit** provides the user interface. Without it, users would have to run scripts from the terminal.  
- **scikit-learn** is the library where the water quality model was trained. It provides algorithms like logistic regression, random forest, and support vector machines.  
- **pandas** manages tabular data, allowing me to structure input values correctly before sending them to the model.  
- **joblib** saves and loads models efficiently. A model trained once can be reused without retraining.  

This file may look small, but it prevents countless hours of troubleshooting. Anyone can run `pip install -r requirements.txt` and get the exact same setup as I had. That reproducibility is key in machine learning projects.

---

# Step 3: models/schema.json – Defining the Input Contract

The schema describes exactly what input fields the model expects. If a user enters data in the wrong order or leaves out a field, predictions may not make sense. This file prevents that problem.

```python
{
  "columns": [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
  ]
}
```

### Why this matters
The schema lists nine columns, each representing a water quality parameter. Let us expand what each means:

- **ph** – Acidity or alkalinity of water.  
- **Hardness** – Concentration of calcium and magnesium.  
- **Solids** – Amount of dissolved solids in water.  
- **Chloramines** – Disinfectant compound level.  
- **Sulfate** – Level of sulfate ions.  
- **Conductivity** – How well water conducts electricity, linked to ion concentration.  
- **Organic_carbon** – Amount of organic matter.  
- **Trihalomethanes** – By-products of chlorination.  
- **Turbidity** – Clarity of water, measured by light scattering.  

By listing these features in JSON, the app knows the exact structure of expected inputs. This avoids mismatches between the training phase and the prediction phase. In short, schema.json acts as a contract.

---

# Step 4: models/water_model.joblib – The Learned Knowledge

The `.joblib` file contains the trained model. During training, scikit-learn used the dataset to learn relationships between water parameters and whether water was potable. The learned weights, thresholds, or tree structures are stored here.

This file is binary, so we do not read it manually. Instead, joblib restores it into memory. Once restored, the model can make predictions instantly. Without this file, the app would need to retrain the model every time, which is inefficient.

Think of this file as the brain of the project. It has already been taught from experience. The app’s role is to ask the brain questions in the right format, and then display the answers.

---

# Step 5: app.py – Building the Application

This is the most important file. It transforms the raw model into a usable tool by creating a user interface and handling input and output. Let us break it down carefully.

## Importing Required Libraries

```python
import streamlit as st
import pandas as pd
import joblib
import json
```

Here I import the core libraries. Each one brings a specific ability:

- **streamlit** (`st`) – Handles all user interface elements, such as buttons, text, and inputs.  
- **pandas** (`pd`) – Converts user inputs into tabular form, which is how the model expects data.  
- **joblib** – Restores the pre-trained model from its saved state.  
- **json** – Reads the schema file so the app knows which fields to request.  

This block lays the foundation. Without these imports, none of the following sections would work.

---

## Loading the Schema and Model

```python
with open("models/schema.json", "r") as f:
    schema = json.load(f)

model = joblib.load("models/water_model.joblib")
```

Two tasks happen here. First, I open `schema.json` and load it into a Python dictionary. This ensures the app knows what features to expect. Second, I load the pre-trained model from the `.joblib` file.

Why at the beginning? Because these resources should be ready before any user interaction. By loading them upfront, I avoid delays when the user presses the predict button. It also ensures any errors are caught immediately.

---

## Creating the Title and Description

```python
st.title("Water Potability Prediction")
st.write("Enter the water quality parameters below to check if it is safe to drink.")
```

This is the presentation layer. The title grabs attention, and the description gives users context. Without it, users might see empty input fields and wonder what the app does. These small text blocks guide the experience.

---

## Generating Input Fields Dynamically

```python
inputs = {}
for col in schema["columns"]:
    inputs[col] = st.number_input(f"Enter {col}", value=0.0)
```

This block is elegant. Instead of hardcoding nine input boxes, I loop over the schema. For every column listed in `schema.json`, I create a number input field in the interface. The result is flexible:

- If I add a new feature to schema.json, the app updates automatically.  
- If I remove one, the app also adjusts without errors.  

This dynamic link between schema and interface avoids duplication. It ensures the user always enters exactly what the model expects.

---

## Handling Predictions

```python
if st.button("Predict Potability"):
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("The water is safe to drink.")
    else:
        st.error("The water is not safe to drink.")
```

This is the action block. When the user clicks the button:

1. A DataFrame is created from the dictionary of inputs. This ensures the format matches what the model expects.  
2. The model’s `predict` function is called, producing either `0` (not potable) or `1` (potable).  
3. A conditional checks the output. If `1`, a green success message is shown. Otherwise, a red error message warns the user.  

Each step here is important. Without converting to DataFrame, the model would reject the input. Without the conditional, the user would only see a number instead of clear language. Streamlit’s `success` and `error` calls make the output human-friendly.

---

# Step 6: Reflections on Each Component

### Why requirements.txt was crucial
It solved the problem of environment inconsistency. If someone else cloned the repository, they could replicate the setup without confusion.

### Why schema.json was important
It acted as a blueprint. Instead of relying on memory or documentation, the app itself enforces correct input structure.

### Why the joblib model mattered
It separated training from usage. I trained once, saved the results, and reused them endlessly.

### Why app.py tied everything together
It turned raw machine learning into an interactive experience. Without it, the project would only exist as scripts. With it, anyone can test predictions with ease.

---

# Step 7: Extended Reflections

This project taught me how structure and clarity are as important as accuracy. A model may be accurate, but if users cannot interact with it, its value is limited. Streamlit provided an accessible path to share my work.

Another lesson was the importance of contracts between components. The schema ensured consistency. The joblib model ensured separation. These design choices made the project modular and future-proof.

I also realized that technical clarity must meet human usability. The app does not expose raw probabilities. It simply says safe or not safe. That clarity makes the system practical, even for people unfamiliar with machine learning.

---

# Step 8: Possible Extensions

While the current version is functional, several improvements could make it stronger:

- Adding probability scores instead of binary labels.  
- Supporting batch uploads via CSV files.  
- Expanding the model with deeper algorithms like gradient boosting.  
- Deploying the app online so others can access it without local setup.  
- Adding visualizations for each feature to give users context about ranges.  

---

# Conclusion

This project was more than writing Python scripts. It was about turning everyday curiosity into a working machine learning system. By carefully structuring files and documenting every part, I built something reproducible and shareable. Clean water remains a vital resource, and while this model is not a substitute for lab testing, it demonstrates how machine learning can create accessible tools.

The experience showed me that technical design and usability go hand in hand. A well-trained model is powerful, but a usable interface makes it impactful. That balance is the real lesson from this project.

