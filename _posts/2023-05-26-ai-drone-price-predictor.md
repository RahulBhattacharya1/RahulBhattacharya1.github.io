---
layout: default
title: "Creating my AI Drone Price Predictor"
date: 2023-05-26 08:42:31
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-drone-price-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_drone_price_predictor
---

When I was browsing drones in a local electronics market, I noticed a curious detail. The price tags did not seem random but followed a certain pattern. Models with longer ranges and sturdier weights were consistently priced higher. That experience sparked a thought in my mind. What if I could build a predictive tool that accepted specifications and returned an estimated price? The spark became the foundation of this project. Dataset used [here](https://www.kaggle.com/datasets/akshatsharma2/flipkart-e-commerce-drone-dataset).

Later as I explored online drone catalogs, the trend repeated. Every time I filtered drones by weight or control range, the prices shifted accordingly. The relationship between specifications and price was clear. I wanted to capture this relationship in a model and wrap it into a simple web app. That app became the AI Drone Price Predictor. It is a minimal yet complete project built with Streamlit, scikit-learn, pandas, and joblib, deployed through my GitHub workflow. This blog explains every file, every block of code, and why each piece matters.

## Requirements File: `requirements.txt`

The first file I created was `requirements.txt`. This file may look small but it plays a critical role. Without it, anyone trying to run the project would be forced to guess which packages to install. That often leads to version conflicts and wasted effort. By locking the requirements, I ensured smooth setup.

Here is the file content:

```
streamlit
pandas
scikit-learn
joblib
```

Each entry in this list serves a purpose.

1. **streamlit** – This package is the backbone of the user interface. Instead of spending time coding HTML, CSS, or JavaScript, I can declare components with Python functions. The benefit is speed of development and focus on logic rather than design boilerplate.

2. **pandas** – This package manages structured data. In my project, I used it to create a DataFrame from user inputs. A DataFrame is the structure my model expects, so pandas became the translator between the interface and the prediction function.

3. **scikit-learn** – The model itself was trained using scikit-learn. It provides regression algorithms, model evaluation tools, and preprocessing utilities. Even though training happened before deployment, scikit-learn was required to load the serialized model correctly because the object references its classes.

4. **joblib** – Saving and loading machine learning models can be memory-intensive. Joblib offers efficient serialization, especially for scikit-learn models. In this project, the `.pkl` model file was created and read using joblib. Without it, the trained model could not be shipped with the app.

By combining these four packages in the requirements file, I ensured that a single command `pip install -r requirements.txt` would recreate the exact environment. That is why this file is small in size but huge in importance.

## Application File: `app.py`

The application logic resides in `app.py`. This script glues together the interface, the model, and the prediction output. Below is the full code:

```python
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/drone_price_model.pkl")

st.title("AI Drone Price Predictor")

control_range = st.number_input("Control Range (m)", min_value=10, max_value=1000, value=100)
weight = st.number_input("Weight (grams)", min_value=50, max_value=5000, value=500)

if st.button("Predict Price"):
    data = pd.DataFrame([[control_range, weight]], columns=["Control Range", "Weight"])
    pred = model.predict(data)[0]
    st.success(f"Predicted Price: ${pred:.2f}")
```

### Import Section

The script begins with three imports: `streamlit`, `pandas`, and `joblib`. Each import has a clear purpose. Streamlit provides functions like `st.title`, `st.number_input`, and `st.success` that create interactive components. Pandas is used to wrap user inputs into a DataFrame, ensuring correct structure for the model. Joblib loads the trained model from disk so that predictions can be made in real time.

### Model Loading

The line `model = joblib.load("models/drone_price_model.pkl")` is critical. Without loading the model, the app would have no intelligence. This line pulls a pre-trained regression model into memory. The benefit is instant predictions without retraining every time. The `.pkl` file format ensures compact storage and quick retrieval.

### Title Definition

The line `st.title("AI Drone Price Predictor")` may look cosmetic but it sets the stage. A clear title guides the user and establishes context. Instead of staring at blank inputs, the user knows exactly what the tool does.

### Input Widgets

Two input fields were created with `st.number_input`. The first takes control range, the second takes weight. Both inputs have minimum and maximum bounds. These bounds prevent unrealistic values. For example, a control range below 10 meters does not make sense for drones, so the lower limit prevents misuse. Default values provide convenience, so the form is never empty. The benefit of these inputs is that they enforce valid, constrained, and usable values.

### Action Button

The line `if st.button("Predict Price"):` creates an action trigger. Until the user clicks this button, no prediction happens. This design saves resources because the model is not queried on every change. It also mirrors user intent: a prediction is only meaningful when the user explicitly requests it.

### Data Preparation

Inside the button block, the first step is wrapping the inputs in a DataFrame. The code `pd.DataFrame([[control_range, weight]], columns=["Control Range", "Weight"])` ensures that the structure matches what the model expects. A scikit-learn model cannot work directly on raw numbers unless wrapped in the same structure as training. Pandas guarantees that the column names align with training features. This step is essential for compatibility.

### Prediction

The line `pred = model.predict(data)[0]` passes the DataFrame into the model. The model returns a prediction array, so I extracted the first element with `[0]`. The benefit here is that the model output becomes a plain number, ready for display. This keeps the flow simple: input values go in, a single price comes out.

### Display Result

Finally, `st.success(f"Predicted Price: ${pred:.2f}")` prints the result in a green box. The formatting rounds the prediction to two decimals and appends a dollar sign. The success message visually stands out and signals completion. The benefit is clarity: the user sees a clean, human-readable output instead of raw numbers or logs.

Every block in `app.py` serves a purpose. Importing packages prepares the tools. Loading the model activates intelligence. Defining inputs structures the interface. The conditional button ensures intentional predictions. Wrapping data ensures compatibility. Prediction logic converts input into value. Display logic communicates results in a friendly way. Each step is necessary, none is redundant.

## Model File: `drone_price_model.pkl`

The project includes a `models` folder containing `drone_price_model.pkl`. This file represents the intelligence of the app. It was created by training a regression model using scikit-learn. The training dataset contained drone specifications and corresponding prices. Two features, control range and weight, were selected as predictors. After fitting the model, I saved it using joblib.

The benefit of saving the model is separation of concerns. Training can be heavy and time-consuming. By training once and saving the result, I decoupled training from deployment. The app simply loads the trained file and runs instantly. This is efficient because the prediction step is lightweight while training is resource-intensive. Users get fast results without waiting for model fitting.

The `.pkl` format is also portable. It can be shared, versioned, or replaced with an improved model in the future. If later I add more features like battery life or camera resolution, I can retrain and save a new file with the same name. The app will continue to work without code changes. That is the long-term benefit of modular design.

## Deep Dive into Imports

The import section looks short but it is the foundation of the app. Each package brings a benefit that justifies its presence.

- `import streamlit as st` – This line unlocks the full Streamlit API under the alias `st`. Without this alias, every call would be longer and less readable. The alias makes the code concise and consistent.

- `import pandas as pd` – Using `pd` as an alias is a convention. It makes code involving DataFrames shorter and easier to read. Since the model expects a structured dataset, pandas is unavoidable.

- `import joblib` – This import allows loading of serialized scikit-learn objects. Even though it is a single word, it hides a complex mechanism. Joblib optimizes memory usage during serialization, especially for NumPy arrays that are often part of scikit-learn models.

The benefit of keeping imports minimal is clarity. I did not import unnecessary libraries. Each import had a direct role in the app. This simplicity reduces cognitive load for anyone reading the code and speeds up execution since no extra modules are loaded.

## Detailed Look at Inputs

The input widgets are defined as:

```python
control_range = st.number_input("Control Range (m)", min_value=10, max_value=1000, value=100)
weight = st.number_input("Weight (grams)", min_value=50, max_value=5000, value=500)
```

These two lines might look repetitive but each parameter was chosen deliberately.

- **Label** – The label tells the user exactly what value to enter. Instead of vague wording, I chose descriptive labels with units. This reduces confusion and ensures consistent input.

- **Minimum value** – The minimum prevents nonsense. A control range less than 10 meters is not realistic. A weight less than 50 grams would not support standard drone components. These checks enforce domain validity.

- **Maximum value** – The maximum ensures practicality. While technically possible to imagine a 10,000-meter drone range, it would not belong to consumer categories. By capping the input, the app stays relevant.

- **Default value** – Defaults provide usability. A blank field often confuses users. By setting defaults, the interface becomes friendly and usable without immediate edits.

The benefit of designing inputs this way is balance. The user gets freedom to test different values but not so much that the system allows invalid inputs. This keeps predictions meaningful and prevents accidental errors.

## Conditional Trigger Explained

The line `if st.button("Predict Price"):` is more than a trigger. It is a guard clause. Without it, the prediction code would run automatically on every change, creating unnecessary load and a confusing user experience.

By wrapping the prediction code under this conditional, I ensured three benefits:

1. **Performance** – The model is only queried when needed. This reduces computational overhead.

2. **Clarity** – The user knows exactly when a prediction is happening. There is no ambiguity.

3. **Control** – The design respects user intention. A result is shown only when requested.

This simple conditional creates a clear boundary between setup and action. That separation makes the app predictable and user-friendly.

## Why DataFrame Wrapping Matters

The code `data = pd.DataFrame([[control_range, weight]], columns=["Control Range", "Weight"])` is critical. It may look like extra work since the values are already captured, but the model requires structured input.

Benefits of this step:

- **Column Names** – Models trained in scikit-learn associate input arrays with feature order. By explicitly naming columns, I guarantee alignment between training and prediction.

- **Scalability** – If more features are added later, I only need to add them to the DataFrame structure. This keeps the code flexible for growth.

- **Consistency** – DataFrames enforce shape consistency. Even if inputs come in different formats, the DataFrame standardizes them.

Without this wrapper, predictions could misalign, causing inaccurate results. The benefit of using pandas here is future-proofing and reliability.

## Prediction Step in Detail

The line `pred = model.predict(data)[0]` involves several layers.

1. **Model Call** – The method `predict` belongs to the scikit-learn estimator. It accepts structured input and returns an array of predictions.

2. **Array Output** – Even though I am predicting one instance, scikit-learn always returns an array. That ensures consistency whether predicting one or many.

3. **Indexing** – Extracting `[0]` converts the array into a scalar. This makes it ready for display. Otherwise, the result would look like `[245.76]` instead of `245.76`.

The benefit of this design is clarity. The output becomes a clean number, formatted in the next step for readability. Every detail matters because clarity improves trust in the result.

## Display Mechanism

The last step `st.success(f"Predicted Price: ${pred:.2f}")` finalizes the experience. Streamlit provides several display methods like `st.write`, `st.info`, `st.warning`, but I chose `st.success`. The green box immediately signals that the operation worked.

The format string rounds the prediction to two decimals. Prices are rarely shown with long decimal tails, so this formatting makes the output realistic. The dollar sign provides context. Without it, the number might be misinterpreted as grams or meters.

The benefit is communication. A machine learning model may do the hard math, but if the output is not clear, it loses value. This final step ensures the result is not just correct but also understandable.

## Real World Benefits of Each Package

It is important to not only know what each package does but also why it matters in real-world scenarios.

- **Streamlit** – In practice, building web apps can take weeks with traditional frameworks. Streamlit cuts this down to hours. The benefit is not just speed but also accessibility. Data scientists can create usable tools without needing web developers. That democratizes deployment of machine learning models.

- **Pandas** – Outside this project, pandas is the backbone of nearly every data analysis pipeline. Its ability to reshape, clean, and aggregate data makes it invaluable. In my case, it ensures that even if inputs expand, the structure remains consistent. The real benefit is adaptability. Models can only work if input data is formatted correctly, and pandas guarantees that.

- **Scikit-learn** – This package is a teaching tool and a production library at once. It allows fast prototyping of models while offering robust implementations. The benefit in this project was reliability. I trained a regression model without having to code the algorithm myself. Scikit-learn provided tested, optimized implementations.

- **Joblib** – The real-world benefit of joblib is efficiency. Large NumPy arrays can consume significant memory when serialized with pickle. Joblib handles them more gracefully. In projects with multiple models, this efficiency compounds. In this app, it meant the model loaded instantly instead of lagging.

Each package thus adds value not just technically but strategically. Streamlit brings speed, pandas ensures correctness, scikit-learn offers reliability, and joblib delivers efficiency. Together, they make the project both lightweight and practical.

## Extended View of the Model File

The model file, though invisible in code, is the heart of the project. Understanding its role gives insight into how prediction works.

The model was trained using historical drone data. Control range and weight were numeric predictors. Price was the target variable. During training, the algorithm learned coefficients that best fit the relationship. Once trained, those coefficients were frozen and saved.

The benefit of saving coefficients is repeatability. Predictions are not guesses; they are consistent mathematical applications of learned patterns. If I input the same values tomorrow, the result will match. That stability is crucial for trust.

Another benefit is separation of workflows. Training required a dataset, feature engineering, and evaluation. Prediction only requires inputs. By saving the model, I decoupled the heavy step from the light one. This reduces friction and makes deployment feasible even on limited machines.

Finally, the model file is swappable. If tomorrow I build a random forest instead of a regression, I can save it under the same file name. The app code remains unchanged. This plug-and-play design is a hidden strength of the approach.

## Future Improvements

This project is intentionally minimal, but there are many ways it could evolve.

1. **More Features** – Adding battery life, camera quality, and maximum altitude as predictors could increase accuracy. These features often influence drone pricing in real markets.

2. **Model Upgrades** – Switching from linear regression to ensemble models could improve precision. Methods like random forests or gradient boosting handle nonlinearities better.

3. **Data Expansion** – Training on larger datasets from online drone catalogs would make the model generalize better. More rows mean more patterns captured.

4. **Interface Styling** – Streamlit allows customization. Adding charts, sliders, or file uploads could enhance usability. A slider for control range, for instance, would give a more interactive feel.

5. **Deployment Scaling** – The app runs on a personal environment now, but it could be deployed on cloud platforms. This would allow handling more users at once.

Each of these improvements builds on the foundation already laid. The core remains simple: structured inputs, a trained model, and a clear interface. That simplicity ensures the app remains extendable without breaking.

## Final Reflection

Looking back, the biggest strength of this project was clarity. Every file had a purpose. Every block of code contributed directly to the functionality. There was no clutter, no filler.

The requirements file provided reproducibility. The app file tied everything together. The model file carried the intelligence. Imports gave me tools. Inputs structured user interaction. The conditional respected intent. The DataFrame aligned with training. The prediction converted numbers into meaning. The display communicated results.

The benefit of designing in such layers is maintainability. If tomorrow I replace the regression model with a neural network, only the model file changes. If I add new features like battery life, only the input section expands. The rest of the system remains intact. That is the hidden power of modular design: each part does its job, and together they form a whole.

This project may have been inspired by drone shopping, but the lesson applies everywhere. Observing patterns, structuring logic, packaging dependencies, and building clear interfaces turns an idea into a usable tool. That is what I achieved with the AI Drone Price Predictor.
