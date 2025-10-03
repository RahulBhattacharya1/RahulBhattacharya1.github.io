---
layout: default
title: "Creating my AI Titanic Survival Predictor"
date: 2025-04-30 10:37:14
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/titanic.webp
demo_link: https://rahuls-ai-titanic-survival-prediction.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_titanic_survival_prediction
featured: true
---

The Titanic tragedy has been studied for decades, yet every time I see the passenger records, I feel both the weight of history and the pull of data. Each row in the dataset represents a human life, marked by features like age, gender, class, fare, and survival outcome. I could not help but wonder if survival was random or if patterns existed beneath the surface. Could a model discover relationships hidden in these variables? That thought drove me to create a survival prediction app that merged history with science. Dataset used [here](https://www.kaggle.com/datasets/heptapod/titanic).

Earlier in my career, my work often ended in Jupyter notebooks. I would clean data, train models, visualize results, and stop there. But a model in a notebook has little impact. Nobody else can interact with it easily. The Titanic dataset gave me an opportunity to take the next step: deployment. I wanted to create a web application that allowed anyone to enter details and instantly see a prediction. This project became more than just an exercise in machine learning.

---

## Repository Structure

I kept the repository simple yet powerful. It contained three key elements:

- **app.py**: The Streamlit application file that handled interface, user inputs, transformations, and predictions.
- **requirements.txt**: The dependency file that locked environment versions for reproducibility.
- **models/titanic_model.pkl**: The trained scikit-learn model saved for inference.

This minimal structure was intentional. Too many files create confusion. Too few leave gaps. These three provided a complete cycle: code, environment, and intelligence. Anyone cloning the repository could recreate the app exactly.

---

## requirements.txt in Detail

```python
streamlit>=1.37
scikit-learn==1.6.1
joblib==1.5.2
pandas==2.2.2
numpy==2.0.2
```

This file is deceptively small but deeply important. Streamlit created the UI. Scikit-learn was the framework for training. Joblib serialized the trained model. Pandas managed structured inputs. Numpy powered numerical calculations. Pinned versions were critical. Once, I tried loading a model saved in one version of scikit-learn on another machine with a slightly different version. The app failed. Pinning avoided such failures. This file was not just a dependency list; it was an environment contract.

---

## app.py Details

### Imports

```python
import streamlit as st
import pandas as pd
import joblib
```

This section imported the essentials. Streamlit was the backbone of interaction. Pandas provided tabular structures. Joblib enabled loading the frozen model. Every later function depended on these imports.

### Model Loading

```python
model = joblib.load("models/titanic_model.pkl")
```

This line brought intelligence into the app. Without it, the interface would be hollow. It connected the frontend to the trained model. I tested this carefully to ensure no path issues in deployment.

### Title

```python
st.title("Titanic Survival Predictor")
```

A clear title reassures users. It sets expectations. A vague heading would weaken trust. A precise title makes the purpose obvious.

### Inputs

```python
age = st.number_input("Age", 1, 100, 25)
fare = st.number_input("Fare", 0.0, 600.0, 50.0)
sex = st.selectbox("Sex", ["Male", "Female"])
pclass = st.selectbox("Passenger Class", [1,2,3])
sibsp = st.number_input("Siblings/Spouses aboard", 0, 8, 0)
parch = st.number_input("Parents/Children aboard", 0, 6, 0)
embarked = st.selectbox("Port of Embarkation", [0,1,2])
```

These widgets gave structure to user input. Number ranges prevented nonsense values. Select boxes constrained categorical entries. Default values acted as subtle guides. Each widget aligned with features from the training dataset. This design guaranteed consistency.

### Encoding Helper

```python
sex_val = 1 if sex=="Female" else 0
```

This helper converted text into numeric. The model had been trained on numeric values. Encoding was the bridge between human choice and machine expectation. A single line but a critical role.

### DataFrame Construction

```python
X_new = pd.DataFrame([[age, fare, sex_val, sibsp, parch, pclass, embarked]],
                     columns=["Age","Fare","Sex","sibsp","Parch","Pclass","Embarked"])
```

Here, I assembled inputs into a structured DataFrame. Column names matched exactly what the model expected. Preserving order and names avoided misalignment. Even one mismatch could produce invalid predictions. This block ensured coherence.

### Prediction

```python
if st.button("Predict Survival"):
    pred = model.predict(X_new)[0]
    st.success("Survived" if pred==1 else "Did Not Survive")
```

This completed the loop. The button empowered users to trigger predictions. The model returned a binary result. The conditional translated it into a human message. Streamlit displayed it clearly. This block turned the app from static to interactive.

---

## models/titanic_model.pkl

This file contained the intelligence. It was trained offline, tested, and then frozen using joblib. At 690 KB, it was lightweight yet powerful. It encapsulated feature relationships, probabilities, and learned rules. Without it, the app would just be a shell. By storing it in a models directory, I kept the repository organized.

---

## Data Preprocessing Step by Step

The Titanic dataset is famous but messy. Each column required care:

- **Age**: Many missing values. I used median imputation because age distribution was not symmetrical. Mean would distort results.
- **Fare**: Highly skewed. I applied scaling to normalize it.
- **Sex**: Categorical. I encoded Female as 1 and Male as 0. Consistency was critical.
- **Embarked**: Few missing values. I imputed them with the mode. Encoded ports as 0, 1, 2.
- **Pclass**: Already numeric. Treated as categorical but left as integers.
- **SibSp and Parch**: Represented family connections. I retained them directly.

Each step aligned data with model requirements. Without preprocessing, models would misinterpret inputs.

---

## Training Experiments

I experimented with several models:

1. **Logistic Regression**: Simple, interpretable, but lacked capacity for complex interactions.
2. **Random Forest**: Balanced power and robustness. Handled non-linearity well.
3. **Gradient Boosting**: Higher accuracy but required careful tuning and more compute.
4. **Support Vector Machines**: Struggled with larger feature sets and scaling issues.

Random Forest gave the best balance of accuracy and interpretability. I tuned hyperparameters with cross-validation. Parameters like number of trees, depth, and split criteria were adjusted. The final model performed consistently across folds. I measured not only accuracy but also precision and recall. This ensured balanced performance across classes.

---

## Debugging Deployment Diary

Deployment exposed issues I had not seen in local runs. First, the app failed because Streamlit Cloud used a different scikit-learn version. The model refused to load. Pinning the version fixed it. Then, the app could not find the model file. I had placed it in the wrong folder. Fixing paths solved that. Later, extreme values like fare=0 or age=100 caused odd predictions. Adding widget constraints solved those. Each issue became a lesson. Deployment is a test of robustness, not just correctness.

---

## Case Studies: Applications Across Industries

### Healthcare

A triage model could request vitals, symptoms, and history. The model could classify urgency levels. Deployed as an app, it could guide nurses in real time. Constraints would ensure realistic ranges. Clear outputs would reduce misinterpretation. The same structure applies.

### Finance

A credit scoring app could ask for income, debts, and credit history. Predictions could classify default risk. Deployed interactively, it would allow bankers to test scenarios. Encoding helpers would map categories like employment type into numeric.

### Retail

A churn prediction app could request purchase frequency, basket size, and recency. Predictions could flag high-risk customers. Managers could test different customer profiles. Clear outputs could guide interventions.

### Education

A student success model could ask for grades, attendance, and extracurriculars. Predictions could identify at-risk students. Teachers could act early. The interactive structure would make it accessible to non-technical educators.

### Sports

Match outcome models could request player stats, conditions, and form. Predictions could support strategies. Coaches could simulate different lineups. Deployed apps would bring data to strategy rooms.

### Government

Policy planning models could request demographic data. Predictions could guide allocation of funds. Clear outputs would make the model usable by officials. Transparency and usability would be key.

---

## Few more Reflections

I learned that helpers are not minor details. They are structural anchors. Encoding preserves schema alignment. DataFrame construction ensures consistency. Button checks prevent premature predictions. Every line matters. I also learned that reproducibility saves time. requirements.txt prevents endless debugging. Deployment amplified my respect for details. I realized that trust is built not only by accuracy but by clarity. Usability is as important as algorithms.

---

## Ethical Considerations

Deploying predictive models carries responsibility. Titanic survival is historical, but modern equivalents may impact lives. Bias in training data can propagate into predictions. Transparency is required. Users should know the limitations. Accountability is important. Ethical deployment means honesty about accuracy, fairness, and scope. Even small apps require responsibility.

---

## Personal Journey

This project shifted my perspective. I began it with curiosity about history but ended it with lessons about practice. I discovered that machine learning is not finished at training. Deployment is equally vital. I discovered that usability requires empathy. Defaults, ranges, and clear outputs matter. I realized that reproducibility is not optional. requirements.txt is essential. I learned to view small helpers as structural choices. This project connected history, data science, and user experience for me.

---

## Future Work

The app could evolve. Adding probability scores would enrich feedback. Visualizations could show distributions. SHAP explanations could reveal feature influences. Batch predictions could serve classrooms. APIs could expand usage into other systems. Mobile layouts could improve accessibility. Each extension builds on the same foundation.


---


## Futuristic Views

As I think about this project, I realize it is more than just Titanic survival. It is a pattern of how humans and machines interact. In the future, I see predictive systems woven into everyday life. The structure I used here will not vanish; it will evolve. We will still gather inputs, process them, and return predictions. But the inputs will be richer, the models deeper, and the outputs more meaningful. I can imagine apps where a person speaks naturally, and the system interprets and predicts instantly.

In my own practice, I see this project as a seed. Today it predicts survival on Titanic data. Tomorrow, it could power healthcare triage, financial inclusion, or educational planning. I believe in building tools that scale beyond curiosity. I want my work to live longer than a notebook cell. The act of deployment is the act of giving data science a voice. When I deploy, I invite others to interact. That dialogue shapes the future of machine learning.

Looking further ahead, I see deployment becoming even more automated. Systems will generate environments on demand, heal version conflicts, and adapt interfaces automatically. The friction I faced with requirements.txt or path mismatches will fade. But the responsibility will not fade. With more power comes more accountability. I will need to consider fairness, transparency, and trust more deeply. This project reminds me that models shape perceptions, and perceptions shape actions.

I imagine a time when prediction apps are as common as calculators. Just as calculators turned arithmetic into a universal tool, predictive apps will make probabilistic reasoning accessible. Anyone will be able to test scenarios. A teacher, a doctor, a banker, or a policymaker will enter data and receive predictions instantly. The question will not be whether to deploy but how responsibly to deploy. That thought drives me to sharpen my practice now.

In first person, I admit this project made me think about legacy. I asked myself: when I build a small Titanic app, what am I really building? I am building a pattern that repeats. I am training myself to think about data, model, environment, and user as one whole. The future belongs to those who master this cycle. I want to be among them. I want to build tools that last, that scale, and that respect the people who use them.

---

## Closing Thoughts

The Titanic Survival Prediction App may look small, but it represents a full cycle of machine learning. Data cleaning, preprocessing, training, testing, deployment, debugging, and reflection were all part of it. It showed me that impact requires sharing. A model in a notebook has no life. A deployed app brings data to people. This project connected history with technology and taught me lessons that go far beyond the Titanic dataset. It was about the discipline of turning analysis into accessible tools.

---
