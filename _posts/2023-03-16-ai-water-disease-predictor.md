---
layout: default
title: "Creating my AI Water Disease Predictor"
date: 2023-03-16 14:35:52
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/water_disease.webp
thumbnail_mobile: /assets/images/water_disease_sq.webp
demo_link: https://rahuls-ai-water-disease-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_water_disease_predictor
---

It began with a personal observation during a visit to a rural area where clean drinking water was not consistently available. The water often looked cloudy and sometimes even smelled unusual. People in the area frequently complained of stomach problems, and I realized these illnesses were not random but closely linked to the water they were drinking. That moment stayed with me and eventually shaped my desire to create something meaningful. Dataset used [here](https://www.kaggle.com/datasets/khushikyad001/water-pollution-and-disease).

Another reason behind this project was my interest in seeing how a small trained model could be deployed in a real-world accessible platform like Streamlit. The challenge was not only about building a machine learning model but also about packaging it in a way that anyone could use it by just opening a simple webpage. The thought of creating something that connects science, health, and technology in a practical manner made this project exciting.

---

## Project Repository Structure

The repository structure is simple but very deliberate. Every file has its place, and each one has an important role in making the project run smoothly.

```
ai_water_disease_predictor-main/
│── README.md
│── app.py
│── requirements.txt
│── models/
    └── diarrhea_model.pkl
```

1. **README.md**: Provides a guide and context for the project. Anyone visiting the repository can immediately understand the purpose and usage steps.
2. **app.py**: This is the main Streamlit application file where I coded the interface, inputs, and predictions.
3. **requirements.txt**: Lists all required dependencies for deployment on Streamlit Cloud.
4. **models/diarrhea_model.pkl**: The trained machine learning model stored using joblib, which is loaded inside the app.

---

## README.md Explained

The README file gives a quick introduction to what the app does. It also explains how the trained model is connected with the deployed app.

```python
# AI Diarrheal Disease Predictor

## Goal
This app predicts diarrheal disease cases per 100,000 people based on **water quality and sanitation factors**.

## How It Works
- Train model in **Google Colab** using `colab_training.ipynb`
- Save `diarrhea_model.pkl` and upload it to `models/`
- Deploy on **Streamlit Cloud** via GitHub

## Folder Structure
```
This block is not executable code but structured text. It outlines the core goal of the project: predicting diarrheal disease cases using water quality and sanitation parameters. The steps tell me to train the model in Colab, save it as a pickle file, and upload it to the repository. This ensures a separation between training and deployment, which is a common practice when working with free cloud-based setups.

---

## app.py - Full Breakdown

The `app.py` file is the main driver of the project. It builds the user interface, collects inputs, loads the trained model, and displays predictions.

### Importing Dependencies

```python
import streamlit as st
import pandas as pd
import joblib
```
This block imports three critical libraries. `streamlit` provides the web interface, allowing sliders, number inputs, and buttons to exist without writing frontend code. `pandas` is necessary for building structured data frames out of user inputs, which are later fed into the model. `joblib` is the library used to load the saved model file efficiently.

### Loading the Model

```python
model = joblib.load("models/diarrhea_model.pkl")
```
This single line loads the trained machine learning model. It reads the pickle file stored inside the models folder and assigns it to the variable `model`. Without this step, the app would not have the intelligence required to predict disease cases.

### Adding Title and Description

```python
st.title("💧 AI Diarrheal Disease Predictor")
st.write("Predict diarrheal disease cases per 100,000 people using water quality and sanitation factors.")
```
The `st.title` method sets a large heading for the app, making it immediately clear to the user what this tool is. The `st.write` method provides a one-line explanation that gives quick context. These lines make the interface more user-friendly and approachable.

### Input Section Header

```python
st.header("Input Water & Sanitation Parameters")
```
This header introduces the section where the user will provide all input values. It improves readability by separating input fields from the title and prediction sections.

### Collecting User Inputs

```python
contaminant = st.number_input("Contaminant Level (ppm)", 0.0, 20.0, 5.0)
ph = st.number_input("pH Level", 0.0, 14.0, 7.0)
turbidity = st.number_input("Turbidity (NTU)", 0.0, 10.0, 3.0)
oxygen = st.number_input("Dissolved Oxygen (mg/L)", 0.0, 15.0, 5.0)
nitrate = st.number_input("Nitrate Level (mg/L)", 0.0, 100.0, 10.0)
lead = st.number_input("Lead Concentration (µg/L)", 0.0, 50.0, 5.0)
bacteria = st.number_input("Bacteria Count (CFU/mL)", 0.0, 200.0, 50.0)
clean_water = st.slider("Access to Clean Water (% of Population)", 0, 100, 70)
sanitation = st.slider("Sanitation Coverage (% of Population)", 0, 100, 60)
```
Each `st.number_input` and `st.slider` line creates an interactive input field. These allow users to set numeric values within defined ranges. For example, contaminant level accepts values between 0.0 and 20.0 with a default of 5.0. Sliders are used for percentage values like clean water access and sanitation coverage because they feel more natural when working with percentages. Collectively, these inputs simulate the environmental and infrastructural factors that affect waterborne diseases.

### Prediction Block

```python
if st.button("Predict"):
    input_df = pd.DataFrame([[
        contaminant, ph, turbidity, oxygen, nitrate,
        lead, bacteria, clean_water, sanitation
    ]], columns=[
        "Contaminant Level (ppm)", "pH Level", "Turbidity (NTU)",
        "Dissolved Oxygen (mg/L)", "Nitrate Level (mg/L)",
        "Lead Concentration (µg/L)", "Bacteria Count (CFU/mL)",
        "Access to Clean Water (% of Population)",
        "Sanitation Coverage (% of Population)"
    ])

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Diarrheal Cases: {prediction:.2f} per 100,000 people")
```
This conditional block is executed when the user clicks the "Predict" button. Inside the condition, I first create a pandas DataFrame using all the collected input values. This DataFrame has one row with all the parameters aligned with column names. The reason for using a DataFrame is that the model was trained on tabular data, so it expects the same format during prediction. Once the DataFrame is ready, the model predicts the disease cases, and the result is displayed using `st.success`. This makes the output visually distinct and easy to read.

---

## requirements.txt Explained

The requirements file lists all dependencies. Streamlit Cloud uses this to install the correct versions before launching the app.

```python
streamlit==1.37.1
scikit-learn==1.5.1
numpy==2.0.2
pandas==2.2.2
joblib==1.4.2
```
- **streamlit**: The web framework that powers the entire interface.
- **scikit-learn**: Provides the machine learning algorithms used to train the model before saving it.
- **numpy**: Supplies numerical computation functions used under the hood by pandas and scikit-learn.
- **pandas**: Handles structured data and is critical for transforming user inputs into the right form.
- **joblib**: Manages model serialization and loading.

Each package is pinned to a version to avoid compatibility issues when deploying. This practice ensures that the same environment works locally and in the cloud.

---

## The Model File - diarrhea_model.pkl

The model file is not code, but it is central to the project. It contains the trained regression or classification logic that interprets water quality and sanitation factors. Without this file, the app cannot make predictions. It was trained in Google Colab using a separate notebook (`colab_training.ipynb`). After training, I exported it with joblib and placed it in the models directory. This separation of training and deployment means the web app remains light and focused.

---

## Deployment Process

The app is deployed on Streamlit Cloud. The process is straightforward:
1. Push the repository to GitHub.
2. Connect the repository with Streamlit Cloud.
3. Streamlit Cloud reads `requirements.txt`, installs dependencies, and runs `app.py`.
4. The model file inside the models directory is loaded during runtime.

This setup allows anyone to open the link and interact with the predictor in real time.

---

## Reflections

Building this project gave me clarity about how to connect raw data science with user-facing applications. Each function and file plays a specific role. Streamlit simplified the process, pandas handled the structure, joblib made the model reusable, and scikit-learn provided the training backbone. The greatest takeaway was learning how to separate training from deployment in a resource-efficient way. It shows that even lightweight projects can have meaningful impact when structured thoughtfully.

---

## Conclusion

This project was inspired by real observations about water quality and its health impacts. It transformed into a Streamlit app that predicts diarrheal disease cases from water parameters. By walking through each file, function, and dependency, I have shown how the pieces fit together. The hope is that this post helps others see the value in combining data science with simple, accessible deployment platforms. It also reminds me that projects do not need to be huge to be useful, they just need to solve a clear problem.


---

## Scientific Relevance of Each Input Parameter

I wanted the predictor to be more than a random guesser, so each parameter I included has scientific grounding. This section dives into why each variable matters and how it connects to waterborne disease prediction.

### Contaminant Level (ppm)
Contaminant level refers to dissolved pollutants in the water. High contaminant levels can indicate the presence of harmful chemicals or industrial waste. Chronic exposure to contaminants raises the probability of gastrointestinal diseases. By allowing a user to set this parameter, the model captures a critical health factor.

### pH Level
The pH level of water measures its acidity or alkalinity. Water that is too acidic or too alkaline can corrode pipes, dissolve metals, and harm intestinal health. Neutral pH is considered safer, and this parameter provides a measurable indicator of water balance. A deviation here can serve as an early red flag for disease risk.

### Turbidity (NTU)
Turbidity measures the cloudiness of water. Higher turbidity usually means suspended particles, which can harbor bacteria and pathogens. Clear water is not always safe, but cloudy water often signals a risk. Including this parameter allows the model to capture indirect signs of contamination.

### Dissolved Oxygen (mg/L)
Dissolved oxygen reflects the balance of oxygen in water, often tied to aquatic ecosystem health. Extremely low oxygen levels may mean organic pollution, which indirectly signals higher microbial activity. This parameter ensures that subtle ecological factors are considered in prediction.

### Nitrate Level (mg/L)
Excessive nitrates in water often come from fertilizers or sewage. High nitrate levels can cause health issues like methemoglobinemia and also support bacterial growth. Tracking nitrate concentration helps the predictor capture agricultural and sewage-linked risks.

### Lead Concentration (µg/L)
Lead is a toxic heavy metal that seeps into water through corroded pipes. Long-term exposure causes serious health problems, particularly in children. Even though lead does not directly cause diarrheal diseases, it contributes to weakened immunity. That is why this predictor considers it.

### Bacteria Count (CFU/mL)
Bacteria count is a direct health indicator. A high colony count usually means fecal contamination, which is one of the strongest triggers for diarrheal outbreaks. Including this parameter ensures the model connects directly with the root cause of disease.

### Access to Clean Water (%)
This percentage reflects how many people in a population have direct access to safe water. Lower coverage often translates into reliance on unsafe sources. This factor is more social than chemical, but it dramatically affects disease risk.

### Sanitation Coverage (%)
Sanitation coverage represents how well toilets and waste systems are available in the area. Poor sanitation correlates strongly with diarrheal diseases. By combining this with clean water access, the model captures infrastructure-based risk.

---

## Training in Google Colab

Although the deployed app does not show training code, the model was trained in Google Colab. These were the steps I followed.

1. I uploaded a dataset with water quality and disease incidence values.
2. I used `pandas` to clean the dataset and prepare features.
3. I selected algorithms from scikit-learn such as Linear Regression and Random Forests to compare performance.
4. After evaluation, I picked the one with better accuracy and generalization.
5. I trained the model with cross-validation to avoid overfitting.
6. I saved the final model with `joblib.dump()` as `diarrhea_model.pkl`.

The training notebook stayed outside the repository to keep the repo light. Only the final model was stored in the `models` folder.

---

## Saving and Uploading the Model

```python
from sklearn.ensemble import RandomForestRegressor
import joblib

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "diarrhea_model.pkl")
```
This is a simplified representation of the training process. The `RandomForestRegressor` was one of the models tested. After training, I saved the fitted model as `diarrhea_model.pkl`. Uploading this file into the GitHub repository under `models/` made it available to the deployed app.

---

## Deployment Troubleshooting

Deploying on Streamlit Cloud was not without issues. Initially, the app failed because I forgot to pin the dependency versions. The error logs pointed to mismatched versions of scikit-learn and joblib. Fixing this required updating `requirements.txt` with exact versions. Another issue came from the path to the model file. On my local environment, relative paths worked differently. I had to ensure that the model path matched the repository structure precisely.

---

## Limitations of the Current Approach

While the app works, there are several limitations worth noting:

1. **Data quality**: The training dataset was not large and came from limited regions. Predictions may not generalize worldwide.
2. **Simplistic model**: The RandomForestRegressor provides reasonable accuracy but is not the most advanced model possible.
3. **Static model**: The model does not learn continuously. Updating it requires retraining in Colab and re-uploading the pickle file.
4. **Interface simplicity**: The app does not visualize trends or provide uncertainty ranges for predictions.

---

## Future Improvements

There are clear ways to enhance this project in the future:

1. Add a data upload option for users to test with their own CSVs.
2. Introduce confidence intervals with predictions.
3. Deploy an updated version that includes charts for water quality trends.
4. Explore transfer learning techniques for global generalization.
5. Automate model retraining with new datasets.

---

## Extended Reflections

One of the most valuable lessons was learning how to move from experimentation to deployment. In earlier projects, I trained models but never thought about making them available for others. Streamlit changed that. It taught me that accessibility can be more impactful than raw accuracy. Another reflection is that constraints such as GitHub’s 25 MB file limit forced me to be efficient. Instead of uploading bulky datasets, I focused on only keeping the essential model file.

---

## Closing Thoughts

At the end of this journey, the project stands as proof of concept that machine learning can be applied to practical health problems in resource-friendly ways. It is not perfect, but it is meaningful. By breaking down the repository, explaining each file, expanding on every code block, and reflecting on the process, I hope this post serves both as documentation and as inspiration for similar ideas.


---

## Walkthrough Case Study

To demonstrate the working of the app, I created a scenario with realistic values. Suppose we have a community with the following water and sanitation conditions:

- Contaminant Level: 7.5 ppm
- pH Level: 6.5
- Turbidity: 5.0 NTU
- Dissolved Oxygen: 4.0 mg/L
- Nitrate Level: 25.0 mg/L
- Lead Concentration: 12.0 µg/L
- Bacteria Count: 80 CFU/mL
- Access to Clean Water: 55%
- Sanitation Coverage: 45%

When these values are entered into the app, the model processes them as a single row DataFrame and generates a predicted number of diarrheal cases per 100,000 people. The result provides a quantitative risk estimate that can guide interventions. For example, low sanitation coverage combined with high bacterial counts leads to a higher prediction. This case study shows how multiple parameters interact and emphasizes the importance of infrastructure.

---

## Detailed Training Steps in Colab

To expand more on the training process in Colab, I followed these detailed steps:

1. **Data Import and Cleaning**
   I loaded CSV datasets with columns like contaminant level, pH, turbidity, oxygen, nitrates, lead, bacteria, clean water access, sanitation coverage, and recorded disease incidence. Using pandas, I dropped rows with missing values and standardized column names.

2. **Exploratory Data Analysis**
   I generated histograms, correlation matrices, and scatter plots to understand relationships. Bacteria count and sanitation coverage showed the strongest correlation with disease rates.

3. **Feature Engineering**
   I normalized continuous variables using StandardScaler from scikit-learn. This ensured features like nitrate and lead, which have different scales, contributed fairly.

4. **Model Selection**
   I tested Linear Regression, Decision Tree, and Random Forest models. Random Forest produced the best results due to its ability to capture non-linear interactions between parameters.

5. **Training**
   I split the data into training and test sets (80/20). I fit the Random Forest with 100 estimators and evaluated the results. The R^2 score on the test set was satisfactory for a lightweight predictor.

6. **Evaluation**
   I compared mean absolute error and root mean squared error across models. Random Forest consistently outperformed others.

7. **Saving**
   After confirming the performance, I saved the model with joblib and downloaded it as `diarrhea_model.pkl` for upload into GitHub.

This process gave me confidence that the model was reasonably accurate before deployment.

---

## Ethical Reflections

Using AI in health-related predictions carries responsibility. This project does not replace medical diagnosis or public health assessments. It serves as an educational and awareness tool. It highlights the connection between water quality and disease risk, but it should not be the sole basis for decision making. Ethical considerations include:

- **Transparency**: Making the model open in GitHub ensures others can inspect it.
- **Limitations Disclosure**: Users are informed that the dataset is limited, and predictions may not apply globally.
- **Responsible Deployment**: The app avoids alarming users by presenting results in a neutral tone with a focus on awareness rather than fear.

In future iterations, I would explore integrating open health datasets and including disclaimers that emphasize professional consultation.

---

## Extended Future Vision

Looking ahead, the project could evolve into a broader public health dashboard. Instead of focusing only on diarrheal disease, it could predict risks for multiple waterborne diseases. By connecting with APIs from organizations like WHO or UNICEF, it could update parameters dynamically. Visualizations could make the tool more intuitive, and geolocation integration could tailor predictions by region. This direction would expand the project into a useful public health awareness platform.

---

## Final Thoughts

Reaching the end of this detailed breakdown, I see how each block of code and each design decision contributed to building the final application. It started from a personal observation, grew through technical effort, and finally resulted in a working deployed tool. By explaining the details of every file, function, and helper, this post documents the journey in full. It reinforces that clear documentation and thoughtful design are as valuable as the code itself.
