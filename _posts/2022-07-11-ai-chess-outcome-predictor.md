---
layout: default
title: "Building my AI Chess Game Outcome Predictor"
date: 2022-07-11 08:33:21
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/chess.webp
thumbnail_mobile: /assets/images/chess_game_sq.webp
demo_link: https://rahuls-ai-chess-outcome-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_chess_outcome_predictor
---

This project grew from a personal observation about how chess games unfold. I noticed patterns in player ratings and number of turns that often signaled the likely outcome. That curiosity led me to design a predictor application. I wanted it to be interactive, accessible, and built on reliable tools. Streamlit made sense as the framework, while a scikit-learn model handled the prediction. In this post I will go through every single file, code block, and helper in detail. Dataset used [here](https://www.kaggle.com/datasets/datasnaek/chess).

## Repository Structure
The repository is organized in a simple way:

```
ai_chess_outcome_predictor-main/
├── app.py
├── requirements.txt
└── models/
    └── game_outcome_model.pkl
```

Each part has a defined role. The application file controls logic and interface. The requirements file ensures the correct environment. The models folder stores the trained predictor. I will now explain each file carefully.


## requirements.txt

This file lists all the Python packages required for the project. Its content is:

```python
streamlit>=1.37
scikit-learn==1.6.1
joblib==1.5.2
pandas==2.2.2
numpy==2.0.2

```

This small block may look simple but it carries importance. Without it, deployment would fail. Streamlit Cloud and other services look for this file to know which dependencies to install. I specified explicit versions to avoid mismatches. Let me explain line by line.

- `streamlit>=1.37` makes sure the app uses a recent Streamlit version. Streamlit powers the interface so it must be present.
- `scikit-learn==1.6.1` is pinned because the model was trained with this exact version. Different versions may break compatibility with serialized models.
- `joblib==1.5.2` loads the model file. This tool is lighter than pickle and often recommended for scikit-learn.
- `pandas==2.2.2` is included even though not directly in app.py. It may have been used during training. Keeping it ensures consistency.
- `numpy==2.0.2` is required for numerical operations. Arrays passed to the model must be numpy arrays.

This file guarantees that when someone deploys the app, the right libraries load automatically. It makes the project portable.


## app.py

This is the main program. Below is the complete code:

```python
import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("models/game_outcome_model.pkl")

st.title("Chess Game Outcome Predictor")

# User inputs
white_rating = st.number_input("White Rating", min_value=800, max_value=2800, value=1500)
black_rating = st.number_input("Black Rating", min_value=800, max_value=2800, value=1500)
turns = st.number_input("Estimated Turns", min_value=1, max_value=200, value=40)

if st.button("Predict Outcome"):
    input_data = np.array([[white_rating, black_rating, turns]])
    pred = model.predict(input_data)[0]
    outcome = {{0:"White wins", 1:"Black wins", 2:"Draw"}}
    st.success(f"Predicted Outcome: {{outcome[pred]}}")

```

I will now break it into sections and explain each part thoroughly.


### Imports

```python
import streamlit as st
import joblib
import numpy as np
```

This block brings in the required libraries. Streamlit (`st`) builds the interface. Joblib loads the model. Numpy handles arrays. Each import is essential. Without Streamlit the interface cannot render. Without joblib the model stays inaccessible. Without numpy the model cannot receive inputs in the expected format. The imports form the foundation of the program.


### Loading the Model

```python
model = joblib.load("models/game_outcome_model.pkl")
```

This block restores the trained model from disk. The file path points to the models folder. Joblib ensures that the object structure and parameters come back as they were during training. If this line fails the entire app fails, because no prediction can occur without the model. Keeping the model separate from code is smart because training and deployment stay decoupled.


### Title of the App

```python
st.title("Chess Game Outcome Predictor")
```

This block sets the title in the Streamlit interface. It is not functional logic but presentation logic. A clear title helps users understand the purpose right away. It improves usability and makes the tool professional. Streamlit renders the title at the top of the page in large font.


### Input Fields

```python
white_rating = st.number_input("White Rating", min_value=800, max_value=2800, value=1500)
black_rating = st.number_input("Black Rating", min_value=800, max_value=2800, value=1500)
turns = st.number_input("Estimated Turns", min_value=1, max_value=200, value=40)
```

This block defines three numeric input fields. The first two are for player ratings. They allow values from 800 to 2800, which matches typical chess rating ranges. Default values are set at 1500 so the inputs are not empty. The third field captures estimated number of turns. It allows values between 1 and 200. A default of 40 keeps it practical. Streamlit ensures that only values in the allowed ranges can be entered. These controls reduce the chance of invalid data going into the model.


### Prediction Button and Logic

```python
if st.button("Predict Outcome"):
    input_data = np.array([[white_rating, black_rating, turns]])
    pred = model.predict(input_data)[0]
    outcome = {0:"White wins", 1:"Black wins", 2:"Draw"}
    st.success(f"Predicted Outcome: {outcome[pred]}")
```

This is the main logic block. It begins with a conditional tied to a button. Streamlit buttons return True only when clicked. That means the prediction code runs only when the user requests it. Inside the block, the inputs are wrapped into a two-dimensional numpy array. The shape `[ [white, black, turns] ]` ensures the model receives one sample with three features. The model's `predict` method returns an array. By indexing with `[0]`, only the single prediction is extracted. Then a dictionary maps numeric codes to text outcomes. Finally the result is displayed in a green success box. Each step is necessary. Without proper array shaping the model would throw errors. Without mapping the result would be unreadable. Without st.success the user would not see a clear response.


### Outcome Mapping

```python
outcome = {0:"White wins", 1:"Black wins", 2:"Draw"}
```

This dictionary makes model predictions human-friendly. Models usually return numeric codes. Numbers are efficient for computation but not for interpretation. By mapping them to text the app bridges machine output and human understanding. This design improves usability with minimal code.


## Model File Explanation

The file `models/game_outcome_model.pkl` contains the trained model. It was produced during a separate training process not shown here. The file holds all weights, coefficients, and learned patterns. Using joblib to store it ensures it can be loaded quickly. By separating training from deployment, I avoided retraining at every launch. This file is large but it is critical. Without it, the app has no intelligence.


## Deployment

Deployment was straightforward. The steps included:

1. Push the repository to GitHub.
2. Connect the repository to Streamlit Cloud.
3. Set `app.py` as the entry file.
4. Let the platform install dependencies from `requirements.txt`.
5. The application went live with a shareable URL.

GitHub Pages cannot run Python, but Streamlit Cloud can. That is why the actual app lives there, while GitHub hosts the code. This separation makes sense because static hosting is not enough for interactive machine learning apps. Cloud deployment makes it accessible to anyone without local setup.


## Reflections

While building this I learned that small details matter. Setting ranges for inputs prevented invalid data. Pinning library versions avoided incompatibility. Separating training and deployment improved speed. Writing this breakdown made me realize how much thought goes into each line, even in a small program. Every block has a reason. Skipping explanation would hide important lessons.

## Possibilities
When I reviewed this block of code I realized how even simple commands represent design decisions. For example, choosing to use joblib instead of pickle was not random. Joblib is optimized for large numpy arrays which often appear in scikit-learn models. It handles compression and efficiency better. That means the model loads faster and uses less memory. If I had used pickle, it might still work, but it would not be as reliable for arrays.

I also thought about the Streamlit number inputs. Setting minimum and maximum values may look cosmetic, but it actually serves as input validation. Many applications need separate validation functions, but Streamlit integrates this directly into the widget. That means fewer errors and cleaner code. It saves time and reduces bugs. It also demonstrates how picking the right framework can simplify many common problems.

Another lesson was about numpy arrays. Scikit-learn always expects a two dimensional input even for one sample. This requirement may confuse beginners but makes sense for consistency. By shaping the array as [[x, y, z]] I respected this rule. Ignoring it would throw errors. Understanding the expectations of libraries helps design correct pipelines. This project reminded me of that many times.

Finally I considered deployment. Locking versions in requirements.txt was not optional. Machine learning models are sensitive to version mismatches. Even a small change in scikit-learn can alter serialization format. That is why reproducibility depends on exact versioning. By pinning the version I ensured the model can be loaded anywhere. This discipline avoids painful debugging later. It may seem strict but it guarantees reliability.

## Conclusion

This project shows how a simple idea can turn into a working tool. By combining a trained model with Streamlit, I created an interactive predictor. The structure is small but effective. Each file serves a role. Each block of code adds value. I walked through everything step by step. Anyone following this can reproduce the project. More importantly, they can learn how to design clean and transparent applications.
