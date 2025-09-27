---
layout: default
title: "Building my AI Station Type Classifier"
date: 2023-08-11 13:27:51
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-station-type-classifier.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_station_type_classifier
featured: true
---

The origin of this project goes back to a recurring thought during my regular train commutes. I often moved between different types of stations: small suburban ones, medium-sized regional stops, and larger urban terminals. Each of them had unique characteristics like the number of tracks, size of platforms, presence of parking, and the flow of passengers. I imagined that if there was a simple tool that could take these attributes and instantly classify the type of a station, it would save time for planner. Dataset used [here](https://www.kaggle.com/datasets/headsortails/train-stations-in-europe).

This realization was not triggered by a technical necessity but by curiosity. I had been working with machine learning models in other domains, but I had never tried to apply them to something as practical and physical as transport stations. The thought stayed with me until I decided to implement it. That was when I built the AI Station Type Classifier. In this blog post, I will walk through every piece of code, every helper, every file I uploaded into GitHub, and explain them in detail.

---

## Project Structure

The repository has a very simple yet effective structure. It contains the following files and folders:

- `app.py` : The Streamlit application that forms the user interface, handles inputs, and displays the prediction results.
- `requirements.txt` : The dependency list that ensures the environment is set up properly with all necessary libraries.
- `models/station_classifier.joblib` : The serialized machine learning model that was pre-trained and exported for reuse in the app.

By keeping the project this lean, I ensured clarity and ease of deployment. Anyone cloning the repository can immediately understand how the system is organized.

---

## requirements.txt

This is the file that establishes the foundation of the project environment.

```python
streamlit
scikit-learn
joblib
pandas
```

### Explanation of Each Package

- **Streamlit**: This library provides the web interface. In this project, I used it to create input fields, buttons, and display prediction results. It transformed my Python code into a user-friendly app accessible through a browser without needing to know web development.

- **Scikit-learn**: This package contains the algorithms and tools I used to build and train the classifier. It provides functionality for data preprocessing, model training, and evaluation. The saved model that I imported into this app was originally trained using scikit-learn.

- **Joblib**: This is the package that allowed me to serialize the trained scikit-learn model and save it as a file. Later, in the app, I used joblib to load the model quickly. Without it, I would need to retrain or reimplement the model, which is inefficient.

- **Pandas**: This library allowed me to work with tabular data. In the app, I structured the user inputs into a pandas DataFrame. This step was essential because the model expects inputs in a format that matches the training schema. Pandas also makes it simple to manage column names and numerical conversions.

By combining these packages, I ensured that the app was complete, efficient, and deployable.

---

## app.py

This is the heart of the project. It contains every line that runs the web application. I will now explain it block by block.

```python
import streamlit as st
import pandas as pd
import joblib
```

### Explanation

This block imports all required packages into the app. I imported streamlit with an alias `st` so that I could call its functions in a concise way. I imported pandas to structure user input data into a DataFrame. This was important because the model was trained on a DataFrame with specific column names. I imported joblib to load the machine learning model that I had saved earlier. Each import directly corresponds to a part of the project. Without streamlit, the app would have no interface.

---

### Loading the Model

```python
model = joblib.load("models/station_classifier.joblib")
```

This line loads the machine learning model from the models folder. The model file is binary and contains the serialized classifier. Loading it at the start ensures the model is ready for predictions throughout the app. If I had loaded it inside a button function, it would reload every time, slowing down performance. Keeping it global improved efficiency. It also ensured that the model was always in memory during the lifetime of the session.

---

### Building the Interface

```python
st.title("AI Station Type Classifier")
st.write("This app predicts the type of a train station based on input features.")
```

### Explanation

These two lines created the header of the app. The title appears prominently at the top of the page. The write statement acts as a description and sets the context for the user. A visitor arriving on the page should know exactly what the app does without reading external documentation. Streamlit made this process very straightforward. The interface would not feel complete without such descriptions, as users need orientation before interacting with forms.

---

### Creating Input Fields

```python
platform_length = st.number_input("Platform length (in meters)", min_value=50, max_value=600, value=200)
num_tracks = st.number_input("Number of tracks", min_value=1, max_value=20, value=2)
daily_passengers = st.number_input("Average daily passengers", min_value=100, max_value=500000, value=5000)
has_parking = st.selectbox("Parking available?", ["Yes", "No"])
```

### Explanation

This block defined the interactive input fields. The first three are numeric fields for platform length, number of tracks, and daily passengers. Each has a minimum, maximum, and default value. This ensures that users cannot insert unrealistic values. For example, a platform length below fifty meters is not typical for a station. Similarly, more than twenty tracks is rare, so I capped it there. By adding validation, I made sure that the input was practical.

---

### Preparing the Input Data

```python
data = pd.DataFrame({
    "platform_length": [platform_length],
    "num_tracks": [num_tracks],
    "daily_passengers": [daily_passengers],
    "has_parking": [1 if has_parking == "Yes" else 0]
})
```

### Explanation

This step structured the inputs into a pandas DataFrame. It mirrors the training format. The model was trained on numerical inputs, so I had to convert categorical choices into numbers. For parking, I mapped "Yes" to one and "No" to zero. That allowed the model to treat it as a binary feature. If I had passed text strings, the model would not understand. By building the DataFrame with exact column names, I ensured compatibility between training and prediction.

---

### Running Predictions

```python
if st.button("Predict Station Type"):
    prediction = model.predict(data)[0]
    st.success(f"The predicted station type is: {prediction}")
```

### Explanation

This block wrapped the prediction process. The condition checks if the user pressed the button. This prevents predictions from running every time the user changes input values. Once pressed, the model’s predict function is called. It takes the DataFrame as input and produces an array. I selected the first element because there was only one row of input. The result is then shown using st.success, which highlights the output visually. This makes the result clear and distinct.

---

## models/station_classifier.joblib

This file is the serialized form of the trained model. I built and trained the classifier using scikit-learn outside this repository. Once the model reached good accuracy, I saved it using joblib. The advantage of this approach is efficiency. When I deploy the app, I do not need to retrain the model. Instead, I just load the trained object. This reduces computation cost and speeds up response time. It also makes the project portable, as anyone with the repository can load the model and get predictions.

---

## Technical Reflection

While constructing the project, I paid close attention to separation of concerns. The requirements file ensures reproducibility. The app file combines interface and logic. The models folder holds the binary artifact. This separation makes it easier to explain, maintain, and expand. The design is minimal but practical. Each part contributes to the whole in a unique way. This modularity also allowed me to expand this blog post with detailed explanations.

---

## Explanations of Helpers and Libraries

It is useful to take a deeper look at why each function and library was valuable in this specific context.

- **Streamlit number_input**: It created validated input fields where I could define ranges. This avoided the need for manual input checking. It also made the app interactive and responsive. Without it, I would have needed to write custom HTML forms.

- **Streamlit selectbox**: It provided a clear way for categorical input. The dropdown ensured that only allowed options could be selected. This simplified the preprocessing step since I could map fixed text values to numbers.

- **Pandas DataFrame constructor**: This function allowed me to pass a dictionary of values and convert them instantly into structured data. It guaranteed column alignment. That meant I could trust the model to interpret the input exactly as intended.

- **Joblib load**: This helper made loading the binary model almost instant. It ensured compatibility with the model trained using scikit-learn. It also supported compression, which helps when storing large models.

- **Model predict**: The core function that uses the trained classifier to generate an output. It expects structured numerical data. In this app, it returned the type of station. Without this single function, the app would not deliver value.

Each of these helpers seems small but they form the critical chain from input to output. Breaking any link would make the project incomplete.

---

## Deeper Dive Into Model Lifecycle

When I first designed the classifier, the initial challenge was deciding what features to include. In this app, I kept four simple features: platform length, number of tracks, daily passengers, and parking availability. Each of these plays a role in defining station type. Longer platforms usually indicate intercity stations, multiple tracks suggest higher capacity, and large passenger counts often point toward major hubs.

I trained the model using scikit-learn because it provided ready access to classification algorithms such as decision trees, random forests, and logistic regression. The choice of algorithm is flexible, but the way I stored the model with joblib meant that I could swap models later without changing the app code. This decoupling of training and deployment was intentional. It means the app acts as a thin client that just loads any compatible classifier.

---

## Importance of Data Preprocessing

Preprocessing is often overlooked, but it was critical here. The model only understands numbers, so converting categorical data was essential. For example, the field "has_parking" could not remain as text. I had to transform it into binary format. This ensured the model could process it mathematically. Without preprocessing, the model would either throw an error or misinterpret the feature. Structuring inputs into a DataFrame was also part of preprocessing.

This demonstrates that even in a small project, the reliability of predictions depends on careful alignment between training and inference data formats. Ignoring preprocessing steps leads to inconsistencies that make results unreliable. This is why the app dedicated a clear block to constructing the DataFrame.

---

## Why Streamlit Was Ideal

I could have built this project using Flask or Django, but I chose Streamlit for several reasons. Streamlit allowed me to focus on the machine learning logic instead of web frameworks. The syntax is clean and concise, which makes it easy to explain to others. I could create a functional web app with only a few lines of code. The integration with widgets like number_input and selectbox gave me everything I needed for interactivity. This made the project deployable within hours instead of weeks.

---

## Expanding the Prediction Section

The prediction section may look small, but it represents the core value of the app. The condition that checks for button press ensures controlled execution. Without this, predictions would run on every parameter change, leading to unnecessary computation and user confusion. The predict function itself encapsulates years of research and implementation within scikit-learn. By calling it, I leveraged a powerful abstraction that handled all mathematical calculations, from probability distributions to decision making.

The presentation of results using st.success is not just cosmetic. It improves user experience by showing the answer in a highlighted format. This ensures that the prediction does not get lost in the rest of the interface. A clear design decision like this demonstrates how user experience can be improved with small touches.

---

## Design Choices in Project Organization

I kept the repository minimal because it reduces cognitive load for new users. Larger repositories with many files can overwhelm beginners. By separating code, dependencies, and model into clear files, I made the project educational. A person reading the repository learns the role of each component quickly. The models folder communicates that all trained objects should be stored there. The requirements file communicates that reproducibility is valued.

This type of organization makes it easier to extend the project later. For example, if I wanted to add new features like station region or ticketing counters, I could just add input fields and retrain the model. The structure already supports such growth.

---

## Reflections on Libraries

Every library in this project had a very specific role. It is worth expanding on how they integrate with each other.

- Streamlit does not process data directly. It only provides input methods. However, the inputs it gathers are passed into pandas structures. This shows the relationship between libraries.
- Pandas does not make predictions. It only prepares the data. Once the DataFrame is ready, joblib passes it to the loaded model.
- Joblib itself is not an algorithm.
- Scikit-learn is the engine behind the classifier. It handles feature splits, probability computations, and model training. Even though it is not visible in app.py, its influence is everywhere because the saved model was trained by it.

These libraries are complementary. None of them alone could deliver the project. Together, they formed a pipeline from user input to meaningful output.

---

## Broader Applications

Although I designed this app for classifying station types, the same design can be applied elsewhere. For example, similar apps could classify types of buildings, categorize retail outlets, or analyze transport hubs. The pattern remains the same: collect features, preprocess into numerical form, load a pre-trained model, and predict. Streamlit provides the interface, pandas structures the data, joblib loads the model, and scikit-learn ensures predictions are valid.
Such generalization is the hallmark of good design. I did not hardcode assumptions about trains. Instead, I structured the app so that with minor modifications, it could serve other classification tasks.

---

## Thoughts

Building this app taught me that even small projects can highlight important practices in machine learning deployment. I learned to value environment reproducibility through requirements.txt. I learned that separating model training from deployment simplifies the workflow. I also realized the importance of user interface in making a project accessible. A model that stays in a notebook does not reach users. Wrapping it in Streamlit makes it interactive and impactful.

---

## Training Overview

Before the app could exist, I had to train the model. The process began with collecting data about different stations. Each station had numerical and categorical attributes. The goal was to assign them to a type: small stop, medium hub, or large terminal. Using scikit-learn, I split the dataset into training and testing sets.

Once the model showed stable accuracy on the test set, I decided to export it with joblib. That is how the file station_classifier.joblib was created. Saving the model meant the training pipeline could be separate from the deployment pipeline. This separation of stages is a hallmark of real-world machine learning projects. Training can happen offline with large datasets, while deployment focuses on lightweight prediction.

---

## Role of Functions

- **st.number_input**: This function ensures structured numeric input. By defining min and max values, it enforces constraints. This prevents invalid values from reaching the model, which protects prediction quality.
- **st.selectbox**: This function enforces controlled choices. By only offering yes or no, it removes ambiguity. This consistency is vital when encoding into numbers.
- **pd.DataFrame**: This function constructs a structured dataset. It ensures that column names match exactly what the model expects. Without this alignment, predictions would fail.
- **joblib.load**: This function restores the model to memory. It reconstructs the scikit-learn object exactly as it was during training. It is fast and reliable.
- **model.predict**: This is the gateway to the classifier. It applies the learned parameters to new data. It is the point where input becomes output.

Every one of these functions contributes to a chain. Removing one link breaks the pipeline. That is why each deserves explicit recognition.

---

## Why I Documented Everything

When I first started building machine learning projects, I underestimated the importance of documentation. Over time, I realized that projects become more valuable when they are explained. Code may run, but without explanation, it loses context. By writing this blog, I ensured that every helper and function has meaning attached to it. Future readers, including myself, can return and understand the reasoning. This is also why I expanded this blog to a long form.

Documentation also builds confidence for recruiters, collaborators, and stakeholders. It shows that the project is not just a technical experiment but a structured artifact ready for presentation.

---

## Practical Lessons Learned

One lesson I learned was the value of simplicity. By keeping the project small and modular, I reduced complexity. Another lesson was the power of preprocessing. Even one categorical field like parking needed careful handling. Ignoring such details could cause mispredictions. A third lesson was the strength of libraries. Each one—streamlit, pandas, joblib, scikit-learn—saved me from writing large amounts of code. Leveraging them is not laziness but efficiency.

The act of creating an interface was as important as the model itself. Without it, the project would remain hidden in a notebook. With it, the project became usable and shareable.

---

## Conclusion

The AI Station Type Classifier represents the intersection of curiosity and execution. It started with an observation about stations, turned into a model through scikit-learn, was saved with joblib, structured with pandas, and deployed with Streamlit. Each piece fit like a puzzle. By documenting every function, block, and helper, I made the project transparent. This blog became more than a guide; it became an extended narrative about building and explaining.
