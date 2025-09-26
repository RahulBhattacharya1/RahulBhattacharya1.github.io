---
layout: default
title: "Creating my AI Water Potability Classifier"
date: 2025-07-02 13:28:49
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-water-potability-classifier.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_water_potability_classifier
featured: true
---

There was a time when I started doubting the quality of the water supplied in my neighborhood. It did not taste the same, and I wondered if it was safe for consumption. That concern made me realize how valuable it would be to have a small system that could instantly evaluate water potability. I imagined using measurable parameters like pH, hardness, turbidity, and other chemical indicators to produce a reliable assessment. This thought planted the seed for the project that became my Water Potability Classifier. Dataset used [here](https://www.kaggle.com/datasets/adityakadiwal/water-potability).

The motivation was not purely academic. It was about practical assurance. I wanted to create a tool that could process numerical values entered by any user and output whether the water is safe to drink. At the same time, I wanted it to be intuitive enough so that anyone could access it through a simple web interface. Machine learning provided the predictive backbone, while Streamlit gave me the platform to share it easily. What began as a personal curiosity evolved into a deployable application with meaningful use cases.

---

## Repository Structure

When I uploaded the project to GitHub for deployment, the repository contained three essential files and folders:

- **app.py** – The Streamlit application containing all logic for inputs, predictions, and interface.
- **requirements.txt** – The file listing dependencies required for Streamlit Cloud to run the app smoothly.
- **models/water_model.pkl** – The pre-trained machine learning model used for classification.

This minimal structure was intentional. Each component has a dedicated purpose, and together they form a fully functional deployment-ready project.

---

## app.py – Main Application

The `app.py` script is the central element of the project. Every interaction flows through this file. Below I explain each code block, its purpose, and how it fits into the overall architecture.

### Importing Libraries

```python
import streamlit as st
import joblib
import numpy as np
```

This block initializes the tools required. Streamlit provides the framework for creating web applications with minimal code. Joblib allows for efficient serialization and deserialization of machine learning models, which is why the `.pkl` file can be used directly. NumPy offers robust numerical handling, which is critical because the model expects data in array format. Without these imports, the rest of the code would not run.

---

### Loading the Model

```python
@st.cache_resource
def load_model():
    model_data = joblib.load("models/water_model.pkl")
    return model_data
```

This helper function loads the pre-trained classifier. The `@st.cache_resource` decorator ensures the model is cached after the first load. This prevents repetitive file reads, which saves time and resources. By isolating the loading process into its own function, I ensured clarity and reusability. If in the future I wanted to switch to another model or path, I would only need to modify this single block.

---

### Collecting Inputs

```python
def get_user_input():
    ph = st.number_input("pH value", min_value=0.0, max_value=14.0, value=7.0)
    hardness = st.number_input("Hardness", min_value=0.0, value=100.0)
    solids = st.number_input("Total Dissolved Solids", min_value=0.0, value=500.0)
    chloramines = st.number_input("Chloramines", min_value=0.0, value=5.0)
    sulfate = st.number_input("Sulfate", min_value=0.0, value=300.0)
    conductivity = st.number_input("Conductivity", min_value=0.0, value=400.0)
    organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=10.0)
    trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=50.0)
    turbidity = st.number_input("Turbidity", min_value=0.0, value=4.0)
    features = np.array([[ph, hardness, solids, chloramines, sulfate,
                          conductivity, organic_carbon, trihalomethanes, turbidity]])
    return features
```

This function organizes user input gathering. Each water quality parameter is captured through a dedicated input box. The use of reasonable defaults allows even inexperienced users to test the application. By the end, the data is collected into a two-dimensional array because scikit-learn models expect input in that shape. Centralizing input logic in one helper avoids repetition and ensures consistency in how features are prepared.

---

### Prediction Logic

```python
def make_prediction(model, features):
    prediction = model.predict(features)
    return prediction[0]
```

This is the simplest but most crucial block. It hands over the prepared features to the model and receives a prediction. Returning only the first element ensures the output is a clean integer rather than a full array. Encapsulating prediction into its own function provides modularity, so if prediction rules change later, I will not need to restructure the entire application.

---

### Main App Workflow

```python
def main():
    st.title("Water Potability Classifier")
    st.write("Enter water quality parameters to check potability.")
    model = load_model()
    features = get_user_input()
    if st.button("Predict"):
        result = make_prediction(model, features)
        if result == 1:
            st.success("The water is safe for drinking.")
        else:
            st.error("The water is not safe for drinking.")
```

This function acts as the orchestrator. It sets the page title and description, loads the model, collects features, and executes predictions upon button click. The conditional ensures that results appear only after user interaction. Depending on the prediction, different visual messages are displayed, guiding the user toward a clear interpretation of the outcome.

---

### Entry Point

```python
if __name__ == "__main__":
    main()
```

This statement ensures the script executes only when run directly. It is a standard practice in Python to prevent accidental execution when modules are imported elsewhere. Here, it ensures the Streamlit app initializes only in the intended scenario.

---

## Supporting Files

### requirements.txt

```text
streamlit
joblib
numpy
scikit-learn
```

This file is indispensable for deployment. Streamlit Cloud reads it and installs the listed libraries automatically. The inclusion of scikit-learn guarantees compatibility since the trained model depends on that framework. A missing entry here would break the environment setup, rendering the app unusable.

---

### models/water_model.pkl

The `.pkl` file is the trained machine learning model. It is not human-readable but holds the learned relationships from training data. It transforms the input values into a binary classification: potable or not potable. Without this model, the app is just an interface with no intelligence.

---

## Detailed Reflections on Functions

- **load_model()**: Acts as the bridge between stored intelligence and runtime usage. Without caching, performance would degrade noticeably.  
- **get_user_input()**: Functions like an interface builder. It abstracts away Streamlit widgets and organizes them into structured input for the model.  
- **make_prediction()**: Keeps classification logic isolated, maintaining separation of concerns.  
- **main()**: Serves as the conductor that brings all components together.  

Each of these functions performs a single responsibility. This modular approach improves maintainability and readability, aligning with good software engineering practices.

---

## Broader Use Cases

This project can help communities monitor water quality quickly. Schools could integrate it into science courses to demonstrate applied machine learning. Local organizations could use it as an initial screening tool before sending samples for laboratory analysis. On a personal level, it shows how everyday concerns can be addressed with data and accessible interfaces.

The simplicity of the Streamlit interface ensures that even non-technical users can benefit. With mobile access, the classifier can be used in the field with minimal effort. Expanding the model to include more parameters or probabilistic outputs would further enhance its usefulness.

---

## Technical Lessons Learned

Developing this project reinforced several technical lessons. First, the importance of modular code cannot be overstated. Separating helpers like `get_user_input()` simplified both understanding and future modifications. Second, dependency management through `requirements.txt` is critical for reproducibility. Third, caching the model significantly improved user experience by reducing wait times. Finally, deploying on Streamlit Cloud proved that small projects can reach a wide audience with minimal setup.


---

## Parameter-by-Parameter Breakdown

Each input parameter is not arbitrary. Every feature contributes to the prediction.

### pH Value
The pH scale runs from 0 to 14. Neutral water is close to 7. Very low or very high pH values can be unsafe for consumption. By including this input, the model captures acidity or alkalinity which has direct implications for potability.

### Hardness
Water hardness refers to the concentration of calcium and magnesium salts. Hard water is not always unsafe, but extreme hardness may indicate contamination or poor quality. The model uses this measurement to learn correlations with unsafe water.

### Total Dissolved Solids (TDS)
This measures the combined content of all inorganic and organic substances. High values often indicate pollution. The model benefits from this input because dissolved solids strongly impact taste and health.

### Chloramines
These are disinfectants used to treat water. While they prevent microbial growth, excessive levels are unsafe. Including this parameter helps the model detect potential over-treatment.

### Sulfate
Sulfates are naturally occurring but harmful in excess. High sulfate levels can cause digestive issues. This feature provides an important chemical indicator.

### Conductivity
This measures water’s ability to conduct electricity, which is influenced by ion concentration. Conductivity correlates with dissolved substances, making it a useful proxy.

### Organic Carbon
This measures the presence of organic matter. High values may support bacterial growth. The model interprets this as a potential health risk.

### Trihalomethanes
These are byproducts of water chlorination. They are linked with health risks at higher levels. Including them strengthens the predictive capacity for treated water.

### Turbidity
This measures the cloudiness of water. Clear water is not always safe, but high turbidity often indicates unsafe conditions. This is a visually relatable parameter that also carries predictive weight.

---

## Training the Model

Although the training process is not included in the repository, it was an essential step. I used a dataset containing thousands of water quality samples with corresponding labels for potability. Using scikit-learn, I trained a classification model that could learn from these features. The choice of algorithm was based on performance during experiments. After training, I saved the model with joblib, making it portable for use in the Streamlit app.

The separation of training from deployment is intentional. Training can be resource-intensive and is usually performed once. Deployment, on the other hand, requires fast predictions, which is exactly what the stored pickle file provides.

---

## User Experience Design

Streamlit was chosen because it allows technical solutions to be shared quickly. Each input widget was chosen with usability in mind. Number inputs are simple and prevent invalid entries. Default values reassure users by showing reasonable starting points. The button-based workflow makes it clear when a prediction is being requested. Color-coded messages give instant interpretation without requiring the user to understand numerical outputs.

This design philosophy ensures accessibility. The app is not only technically correct but also approachable by anyone.

---

## Possible Extensions

1. **Probability Scores**: Instead of binary safe/unsafe, I could extend the prediction function to output probability percentages.  
2. **Data Visualization**: Adding charts showing how inputs compare with safe ranges would enhance interpretability.  
3. **Mobile Optimization**: Streamlit is already responsive, but further tuning could improve mobile usability.  
4. **Expanded Datasets**: Incorporating data from different regions would make the model more generalizable.  
5. **Multi-Class Predictions**: Future versions could classify water into categories such as excellent, acceptable, or unsafe.

---

## Lessons Beyond Code

Working on this project also highlighted non-technical lessons. Documentation is as important as code because it ensures clarity for future users or contributors. Simplicity often leads to robustness. By resisting the temptation to add too many features at once, I ended up with a tool that is reliable and understandable. Finally, projects rooted in real concerns often feel more rewarding because they solve problems that resonate personally.

---

## Streamlit Widgets in Detail

Each widget in Streamlit was carefully chosen for clarity. For this project, I used `st.number_input`. This widget forces inputs to remain numeric, avoiding invalid text entries. The ability to set minimum and maximum values prevents unrealistic numbers from being entered. For example, pH cannot be negative or above 14, so constraining values enforces realistic input.

Defaults provide another advantage. They show an example scenario immediately when the app loads. A user unfamiliar with water chemistry can still press predict and see a result. This encourages exploration without intimidation.

---

## Caching Explained

Streamlit offers several caching decorators. I used `@st.cache_resource` for loading the model. Without caching, every user interaction would reload the model, slowing performance. Caching ensures the heavy lifting is done only once. For small projects, this may not seem critical, but when scaling to many users, the performance difference becomes significant. This also reduces unnecessary disk reads, improving efficiency.

---

## Deployment Considerations

Deploying the app required more than writing code. I had to ensure the repository was lightweight, well-structured, and reproducible. Streamlit Cloud automatically builds the environment from `requirements.txt`, so every dependency needed to be included. The model file had to remain under the size limits imposed by the hosting platform. GitHub integration simplified version control, allowing updates to be reflected quickly.

---

## Handling Edge Cases

While developing, I considered unusual scenarios. For example, what if all values are set to zero? The model still returns a classification, though it may be unreliable because such combinations are unrealistic. By providing sensible default values, I reduced the chance of meaningless input. Another edge case is when a user deliberately enters extreme values, like maximum conductivity with minimum pH. The model still predicts, but results should be interpreted with caution.

---

## Performance Considerations

Even though the model is small, optimizing responsiveness matters. NumPy arrays ensure fast numeric handling. Joblib loading is efficient compared to alternatives. Keeping functions short and focused avoids overhead. Streamlit’s lightweight interface ensures predictions appear in real time. These considerations make the difference between an application that feels sluggish and one that feels smooth.

---

## Environmental Impact

Access to clean water is a global challenge. While this project does not replace professional testing, it offers awareness. By demonstrating how machine learning can highlight unsafe patterns, it can encourage communities to take water quality seriously. Small tools like this can spark conversations about environmental health and motivate further action.

---

## Educational Applications

Teachers can use this project in data science courses to show practical machine learning. Students can learn how raw data transforms into features, how a model interprets them, and how predictions are delivered through an interface. The modular code structure makes it easy to assign exercises, such as adding new parameters or adjusting prediction outputs. This project, therefore, doubles as both a practical solution and a learning resource.

---

## Future Research Directions

1. **Integration with IoT Sensors**: Connecting the app to live sensors could automate inputs and provide continuous monitoring.  
2. **Cloud-Based Training Pipelines**: Instead of training locally, future models could be trained with cloud platforms, ensuring scalability.  
3. **Explainability**: Incorporating SHAP or LIME to explain why the model predicts unsafe conditions could increase trust.  
4. **Localization**: Adapting the app to different languages and regional standards would broaden accessibility.  
5. **Integration with Alerts**: Linking predictions to SMS or email notifications could provide proactive warnings to users.

---

## Final Reflections

Creating this project was an exercise in translating personal concern into practical application. Each block of code was written with clarity in mind. Every supporting file had a defined role. By balancing technical depth with accessibility, I built a project that is both functional and understandable. It demonstrates how even small steps in machine learning can lead to impactful outcomes when guided by real-world motivation.

---

## Closing Thoughts

This project shows how machine learning can intersect with environmental and public health concerns. It highlights the importance of modular coding, reproducible environments, and thoughtful user interface design. Most importantly, it reflects how personal motivation can evolve into a deployable tool with potential social benefit. The Water Potability Classifier is not just an academic exercise but a demonstration of how curiosity, structure, and accessibility can converge into meaningful technology.

---

## Conclusion

The Water Potability Classifier started with a personal concern but evolved into a deployable application. The repository structure was intentionally simple, with every file serving a clear purpose. The code inside `app.py` demonstrates how thoughtful structuring leads to clarity, performance, and extensibility. By breaking down each function and supporting file, I have shown how even small projects can embody good engineering practices. This project reflects the intersection of curiosity, practical need, and technical execution, resulting in a meaningful tool that others can use and extend.
