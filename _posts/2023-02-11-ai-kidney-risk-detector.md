---
layout: default
title: "Building my AI Kidney Disease Risk Detector"
date: 2023-02-11 22:35:17
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-kidney-disease-risk-classifier.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_kidney_disease_risk_classifier
featured: true
---

It started with a moment that stayed in my thoughts for days. I came across a news article that explained how many people discover kidney disease too late because the symptoms are subtle and often ignored. That made me think of the role technology could play if it could predict risks much earlier. The thought did not come from a conversation or a group setting, it came from a quiet realization while reflecting on the importance of preventive care. Dataset used [here](https://www.kaggle.com/datasets/miadul/kidney-disease-risk-dataset).

I wanted to see if I could create an application that would help in assessing the risk of kidney disease based on a set of medical features. The idea was not to replace medical professionals but to provide a supportive tool that can highlight risk early. With this motivation, I designed and deployed a small project that classifies whether a person might be at risk. It combined Python code, machine learning training, and a Streamlit application to make the interface simple for anyone to use.

---

## Files I Uploaded

To make the project work properly on GitHub Pages with Streamlit deployment, I uploaded several files into my repository:

- **app.py**: This is the main application file. It defines the Streamlit interface, loads the trained model, handles user input, and produces predictions. It is the center of the project because without this script, nothing would appear on screen and the user would not be able to interact with the model.  
- **requirements.txt**: This lists all the Python libraries needed for the application. Without this file, the deployment would fail because the environment would not know which packages to install. Listing dependencies in one file also makes the project easier to share and reproduce across machines.  
- **models/model.pkl**: This is the serialized trained model. It stores the machine learning classifier that was trained on kidney disease dataset. The `app.py` file loads this model to make predictions in real time. Saving the model ensures that training does not need to be repeated during deployment.  

Each of these files had a unique role in ensuring the app runs end to end. Their separation allows each piece to handle a clear function. That makes debugging easier and improves readability when the project grows.

---

## The Code in app.py

Now I will go block by block through the code in `app.py`. I will explain every function, every helper, and each conditional. The goal is to show why the code was written this way and how the pieces work together to make the application function.

### Importing Libraries

```python
import streamlit as st
import pickle
import numpy as np
import pandas as pd
```

This block imports the required Python libraries. `streamlit` is used for building the web interface. It allows me to turn a Python script into an interactive web application with minimal effort. `pickle` is used to load the trained machine learning model. It is Python’s built-in tool for serializing and deserializing objects. `numpy` is helpful for handling arrays and numerical data. It provides fast operations that would be hard to achieve otherwise.

Without this block, none of the later steps could function because the dependencies would be missing. This part sets the foundation of the script. Every subsequent block depends on these imports. The order is not strict but having them at the top keeps the script clean and readable.

### Loading the Model

```python
model = pickle.load(open('models/model.pkl', 'rb'))
```

This line opens the `model.pkl` file from the `models` folder and loads the trained machine learning model. Using `pickle.load` ensures that the model is deserialized back into Python memory. Having the model already trained means the web app does not need to train from scratch, which saves both time and compute resources. Training can take minutes or hours depending on dataset size, but loading takes seconds. It allows the app to focus purely on prediction.

If this line fails because the path is incorrect, the whole app cannot work. That is why the folder structure matters. The `models` directory must exist, and the file must be inside it. I deliberately placed the model in a folder rather than the root directory so that the repository stays organized. Grouping model files under one directory helps when multiple versions of models are trained.

### Defining the Prediction Function

```python
def predict(input_data):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return prediction
```

This helper function takes input data from the user, wraps it into a pandas DataFrame, and feeds it to the trained model for prediction. The reason for using `DataFrame([input_data])` rather than a plain list is that the model expects column names. A DataFrame preserves both the values and the column names. It then calls `model.predict(df)` to produce a prediction. Finally, `[0]` selects the first element because the output is a list-like structure even if we pass one sample.

Creating this function keeps the code modular and reusable. Instead of repeating the same DataFrame construction and prediction call multiple times, this function centralizes the process. It makes the logic easier to manage and reduces the chance of mistakes. If I decide to adjust preprocessing in the future, I only need to update this function, not every place where prediction happens.

### Building the Streamlit Interface

```python
def main():
    st.title("Kidney Disease Risk Classifier")
    st.write("This app predicts the risk of kidney disease based on user inputs.")
```

The `main` function starts by defining the app title and an explanatory text. `st.title` creates a large heading at the top of the app. `st.write` provides supporting text in smaller font. This sets the tone for the application and gives the user context on what the app is designed to do.

Starting with these lines in `main` helps the user feel oriented. Without them, the interface would appear blank, which can confuse or discourage someone using the app for the first time. The text provides reassurance that the system has a purpose.

### Collecting User Input

```python
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    bp = st.number_input("Blood Pressure", min_value=50, max_value=200, value=80)
    sg = st.number_input("Specific Gravity", min_value=1.0, max_value=1.05, value=1.02, format="%.2f")
    al = st.number_input("Albumin", min_value=0, max_value=5, value=0)
    su = st.number_input("Sugar", min_value=0, max_value=5, value=0)
```

This block adds input fields for several medical features. The user can enter their age, blood pressure, specific gravity, albumin, and sugar values. Each input has defined minimum and maximum values to prevent invalid entries. Default values are also provided. Collecting this data is critical because it feeds into the classifier and determines the prediction output.

The reason for using `st.number_input` is that it restricts inputs to numeric types. That prevents the user from entering text that could break the model. Streamlit also provides a consistent interface across browsers, which avoids errors that could occur if manual parsing were required. The defaults, such as age 30 or blood pressure 80, are chosen to be reasonable values so that the user sees a working example immediately.

### Organizing Inputs into a Dictionary

```python
    input_data = {
        'age': age,
        'bp': bp,
        'sg': sg,
        'al': al,
        'su': su
    }
```

Here the user inputs are grouped into a dictionary. This makes it easy to pass all the features at once to the `predict` function. The dictionary keys match the expected feature names that the model was trained on. This alignment is crucial; otherwise the model would not understand the data and the prediction would fail.

The choice of dictionary is intentional. A dictionary maps keys to values, which aligns with how DataFrame columns are named. That makes the transition smooth when converting to pandas DataFrame. If I had used a list, I would need to remember the correct ordering, which is error-prone.

### Running Prediction on Button Click

```python
    if st.button("Predict"):
        result = predict(input_data)
        if result == 1:
            st.error("High risk of kidney disease detected.")
        else:
            st.success("Low risk of kidney disease detected.")
```

This conditional listens for the click of the "Predict" button. If the button is pressed, the input data is passed to the `predict` function. The returned result is then checked. If the result is `1`, the app displays an error message in red, highlighting that high risk is detected. If the result is `0`, it displays a success message in green, indicating low risk. This simple flow keeps the user informed with clear and immediate feedback.

The structure of this block is essential. First, it checks for the button. Without that check, the prediction would run every time a number is adjusted, which could annoy the user. Second, it uses clear if-else branching. That ensures the app covers both possible outcomes. Finally, it uses `st.error` and `st.success`, which provide built-in styling so that users can interpret results at a glance.

### Standard Python Entry Point

```python
if __name__ == '__main__':
    main()
```

This line ensures that the `main` function runs when the script is executed directly. It is a standard Python convention to use this guard. Without it, the `main` function might not run in some execution contexts. This block ties the program together by starting the Streamlit application.

The use of this guard is common across many Python projects. It signals to other developers that the file can be imported without executing its contents immediately. That improves flexibility if I later decide to extend the project with modules.

---

## Explanation of requirements.txt

The `requirements.txt` file ensures that the deployment platform knows which packages to install. A minimal version of this file looks like this:

```python
streamlit
pandas
numpy
scikit-learn
```

Each line lists one dependency. `streamlit` powers the web interface. `pandas` and `numpy` handle data processing. `scikit-learn` is required because the model was trained using it. Having this file makes the environment reproducible so that the same setup can be created anywhere.

In practice, deployment platforms like Streamlit Cloud automatically read this file. If it is missing, the platform may default to an empty environment, which causes errors. Including `requirements.txt` is not optional, it is mandatory for successful deployment. I deliberately kept the list short because smaller environments install faster and reduce the risk of version conflicts.

---

## Explanation of model.pkl

The file `model.pkl` is not human-readable because it is a binary serialized object. It contains the trained classifier. During training, a machine learning algorithm was fit to kidney disease data and the resulting model was saved using Python's `pickle` library. This file is loaded by the application so that predictions can be made instantly. Without this file, the app would have no intelligence because the logic of the classifier resides here.

The benefit of saving the model this way is speed and consistency. I only need to train the model once, and after that, I can share it easily. Anyone with the pickle file and the same library versions can load and use the model. It makes collaboration easier. It also makes web deployment possible, because training a model on the fly during app load would not be efficient.

---

## Step-by-Step User Walkthrough

When a user opens the deployed app, they are greeted by a clear title. They read the description and know that the app predicts kidney disease risk. They then enter values in number fields. For example, they might enter age 45, blood pressure 120, specific gravity 1.02, albumin 1, and sugar 0. Once they are satisfied with their inputs, they press the Predict button.

Behind the scenes, the app gathers these inputs into a dictionary, converts it into a DataFrame, and sends it to the model. The model checks the data against the learned patterns and returns either 0 or 1. That result is then displayed in a styled message. The user immediately understands whether the risk is low or high. This flow demonstrates how machine learning can be integrated into a simple, interactive web app with minimal friction.

---

## Reflections

Working through this project reinforced how small technical steps can translate a simple idea into a working system. Every function in the code was written with a purpose. The prediction function organized the workflow. The conditionals provided feedback logic. The helpers allowed reuse. The external files supported portability. As I tested and deployed, the application came alive on Streamlit Cloud.

The entire exercise also taught me the importance of clear modular design. Each block of code was easier to explain when written cleanly. Each file uploaded to GitHub served one clear role. This separation of concerns allowed me to maintain the project with less confusion. It became not just a coding exercise but a practical application that merged personal motivation with technical execution.

Another reflection was on the responsibility of building health-related applications. I realized that predictions should be framed carefully. That is why the app displays results as risk indicators and not definitive diagnoses. It is a supportive tool, not a replacement for medical advice. Keeping that balance required careful wording in the user interface and thoughtful design of outputs.

---

## Conclusion

The Kidney Disease Risk Classifier stands as one example of how personal reflection can evolve into a machine learning solution. It combined data science, web deployment, and structured explanation into one flow. Building it required coding, model training, and thoughtful deployment practices. Sharing the code through GitHub and the app through Streamlit ensures that others can also see and use the tool.



---

## Deeper Dive into Function Design

The `predict` function is the core logic that connects user input to the trained model. It is small but powerful. To understand why it is written this way, consider three design choices:

1. **Why wrap input in a list**: The model expects multiple samples in a DataFrame. Even when predicting a single case, the input must be two-dimensional. Wrapping input_data inside a list creates that structure. Without this wrapper, pandas would misinterpret the data.

2. **Why use DataFrame**: Many scikit-learn models store feature names when trained. Passing a dictionary into a DataFrame ensures the model sees the same column labels it was trained with. That reduces mismatch errors. Using numpy array alone would lose that clarity.

3. **Why return prediction directly**: The function returns `prediction` as a simple integer. That keeps the interface clear. The function does not decide what message to show. That decision is left to the Streamlit UI code. This separation of concerns keeps logic layered and easy to extend later.

---

## Streamlit Widgets and Their Importance

Every `st.number_input` widget carries parameters that matter. The minimum and maximum values prevent unrealistic data entry. The default value makes the widget usable immediately. The format argument in specific gravity input ensures that the number displays with two decimal places. That makes the input precise and user friendly.

If I wanted, I could extend the interface with dropdowns or sliders. For example, albumin and sugar could be sliders from 0 to 5. Sliders make the UI feel different, while number inputs emphasize precision. Streamlit makes it easy to swap widgets. That flexibility is part of why I chose it for this project.

---

## Error Handling Considerations

The current code assumes the model file is present and that inputs are valid. In real-world deployments, errors can happen. For example, if the model file is missing, `pickle.load` would raise an exception. To handle that, I could add try-except blocks. That would allow the app to show a friendly error message rather than crashing.

Similarly, inputs could be checked for unusual ranges. For example, an age of 200 is unlikely, but the widget already prevents it. Streamlit widgets provide built-in validation that reduces errors before they happen. Still, adding extra checks in the code can provide additional safety.

---

## Alternative Architectures

While this project uses pickle and Streamlit, other choices exist. I could have exported the model using `joblib` which is often recommended for scikit-learn models. Joblib is optimized for large numpy arrays. In this case, pickle was sufficient because the model is small.

For the interface, I could have built a Flask or Django web app. That would give more control over routing and design. However, those frameworks require more code to handle forms, templates, and responses. Streamlit was faster to prototype and deploy. For a demonstration project, simplicity matters more than full control.

---

## Extended Reflections on Deployment

Deploying the app taught me the importance of environment consistency. Locally, the model might work because the versions of scikit-learn and pandas are the same as when it was trained. On a cloud platform, version differences can break pickle loading. That is why specifying exact versions in `requirements.txt` is sometimes needed. For example:

```python
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.25.0
```

By pinning versions, I ensure that the environment matches training. This avoids cryptic errors and makes the app stable.

---

## Long Term Possibilities

The current app uses five features only. The dataset contained more, like hemoglobin levels and red blood cell counts. I limited the app to a few inputs for clarity. In future, the interface can be expanded to include more fields. Each new feature could improve model accuracy but might also confuse users if too many fields are shown. Balancing usability with completeness is a design challenge.

I could also add charts to visualize risk. For example, plotting age against risk probability. Streamlit supports charts using matplotlib or plotly. That would make the app more engaging and provide richer feedback. Users often understand visuals better than plain text.

---

## Lessons Learned

Breaking down the project step by step helped me see the value of structure. Keeping files separate allowed each one to play a role without overlap. Explaining each function revealed why clarity in code helps not just developers but also readers. Thinking about deployment forced me to consider reproducibility. Reflecting on design choices highlighted that simplicity often wins for educational projects.

This journey shows that even a short script can hold many lessons when analyzed deeply. Every conditional, every helper, and every file tells part of the story.

---

