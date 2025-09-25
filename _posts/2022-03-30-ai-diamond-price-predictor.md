---
layout: default
title: "Creating my AI Diamond Price Predictor"
date: 2022-03-30 11:43:26
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-diamond-price-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_diamond_price_predictor
featured: true
---

There was a moment in a jewelry shop that stayed with me. Two diamonds were placed side by side in a display tray. They were the same size, both sparkling under the lights, and to my eyes they appeared nearly identical. Yet the price tags showed values that were vastly different. The salesperson explained with terms like brilliance, fire, and proportions, but the words felt abstract. I walked away wondering why something that should be quantifiable felt so unclear. Dataset used [here](https://www.kaggle.com/datasets/shivam2503/diamonds).

When I gained more experience in data science, I realized I could turn that confusion into a project. I could take data on diamond attributes and train models to predict their prices. My aim was not to replace expertise but to provide transparency. Building the system also gave me the chance to create an end-to-end machine learning project that included data preprocessing, training, artifact management, and deployment.

---

## Repository Structure

I wanted the repository to be complete so that anyone could run the app without missing parts. Here is what it contains:

- **app.py**: The Streamlit application script. It integrates preprocessing, models, and user interface logic.
- **diamonds.csv**: The dataset. It has thousands of diamonds with features like carat, cut, color, clarity, depth, table, and price.
- **requirements.txt**: Dependency file. It lists the Python packages required to run the app.
- **models/diamond_price_regressor.skops**: Serialized regression model artifact.
- **models/diamond_price_range_classifier.skops**: Serialized classification model artifact.
- **models/input_schema.json**: Schema file containing numeric and categorical feature definitions.

By including these, the project remains reproducible and portable.

---

## Imports and Configuration

```python
import json
import os
import traceback
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diamond Price Intelligence")
```

This code sets up the environment. JSON handles schema files. OS manages file paths. Traceback captures errors. Pandas processes the dataset. Streamlit provides the web interface. The page configuration sets a descriptive title for the app window.

---

## Conditional Import of Skops

```python
USE_SKOPS = True
try:
    import skops.io as sio
except Exception:
    USE_SKOPS = False
```

This block ensures that model loading will not break. Skops is safer than pickle. If the import fails, the app sets `USE_SKOPS` to False and can fall back to training models. This increases robustness and portability.

---

## Machine Learning Imports

```python
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
```

This block brings in scikit-learn utilities. Splitting, preprocessing, pipelines, imputers, and encoders form the backbone of data preparation. Gradient boosting algorithms provide strong predictive power. Metrics validate model quality.

---

## Schema Definition

```python
NUMERIC = ["carat", "depth", "table", "x", "y", "z"]
CATEG = ["cut", "color", "clarity"]
CUT_CATS = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_CATS = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_CATS = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
```

This block defines features explicitly. Numeric ones are continuous measurements. Categorical ones are qualities with specific orders. Explicit ordering prevents inconsistencies when models are retrained or reloaded.

---

## Preprocessor Function

```python
def build_preprocessor():
    num_tr = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_tr = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore",
                                 categories=[CUT_CATS, COLOR_CATS, CLARITY_CATS]))
    ])
    return ColumnTransformer([("num", num_tr, NUMERIC),
                              ("cat", cat_tr, CATEG)])
```

This function builds a preprocessing pipeline. Numeric features are imputed with medians. Categorical ones are imputed with the most frequent label and encoded with one-hot encoding. The `handle_unknown` argument ensures unseen categories do not break the pipeline.

---

## Model Loading Function

```python
@st.cache_resource(show_spinner=False)
def load_or_train_models():
    schema = {
        "numeric_features": NUMERIC,
        "categorical_features": CATEG,
        "cut_categories": CUT_CATS,
        "color_categories": COLOR_CATS,
        "clarity_categories": CLARITY_CATS,
    }
    reg, clf = None, None
    errors = []

    if USE_SKOPS:
        try:
            reg = sio.load("models/diamond_price_regressor.skops", trusted=True)
            clf = sio.load("models/diamond_price_range_classifier.skops", trusted=True)
        except Exception as e:
            errors.append(str(e))

    return reg, clf, schema, errors
```

This function loads pre-trained models if possible. If errors occur, they are captured. Streamlit caching ensures the function does not rerun unnecessarily.

---

## Training Functions

```python
def train_regressor(X, y):
    preprocessor = build_preprocessor()
    reg = Pipeline([
        ("pre", preprocessor),
        ("model", HistGradientBoostingRegressor())
    ])
    reg.fit(X, y)
    return reg

def train_classifier(X, y):
    preprocessor = build_preprocessor()
    clf = Pipeline([
        ("pre", preprocessor),
        ("model", HistGradientBoostingClassifier())
    ])
    clf.fit(X, y)
    return clf
```

The regression training function builds a preprocessing step and a regressor, fits it, and returns it. The classification training function does the same with a classifier. Separating them improves modularity.

---

## Streamlit UI: Sidebar

The sidebar contains controls for user interaction. Users can upload their own datasets. If no dataset is uploaded, the default dataset is used. They can also choose whether they want regression or classification predictions. Having these options makes the app flexible and interactive.

---

## Streamlit UI: Data Preview

Data is displayed with `st.dataframe`. This allows users to verify that their file has been read correctly. It reassures them that their input is in the right format. Visual previews reduce errors and confusion.

---

## Streamlit UI: Predictions

Predictions are displayed on the main screen. Regression outputs a numeric price. Classification outputs a price range. The results are shown clearly with Streamlit functions like `st.write` or `st.success`. This makes the predictions easy to understand.

---

## Error Handling

Traceback captures detailed errors. Streamlit displays them in the interface. Users see a red error message rather than a crash. This improves usability and transparency.

---

## Evaluation Metrics

The models are evaluated with three metrics. Mean Absolute Error shows the average difference between predicted and actual prices. R-squared shows how much variance is explained by the model. Accuracy shows how often the classifier is correct. These metrics ensure models are reliable.

---

## Dataset Exploration and Insights

The dataset has over 50,000 entries. Carat weight strongly correlates with price. Cut, color, and clarity also matter. Analysis shows that diamonds with the same weight but better cut and clarity cost more. Correlation studies confirmed carat as the strongest predictor. However, categorical qualities add complexity, making machine learning valuable.

---

## Serialization with Skops

Skops saves models safely. It is designed for scikit-learn and avoids risks present in pickle. It also preserves metadata. By using skops, I ensured reproducibility and safety across environments.

---

## Deployment Guide

Deployment was done in three ways:

1. **Local Deployment**: Using `streamlit run app.py`. This was ideal for testing.
2. **Streamlit Cloud**: Linking the GitHub repository to Streamlit Cloud. It automatically built and hosted the app.
3. **Hugging Face Spaces**: Uploading the repository to Hugging Face. It provided free hosting and integration with ML tools.

Each method had benefits. Local was fast for development. Streamlit Cloud was simple for sharing. Hugging Face added ecosystem benefits.

---

## Reproducibility and Version Control

The project was designed for reproducibility. All files were stored in GitHub. Requirements.txt listed dependencies. Models were serialized. Schema definitions ensured consistency. Version control tracked changes. This setup makes results reproducible anywhere.

---

## Reflections: How This Project Can Help

### Supporting Buyers

For buyers, this project provides transparency. Instead of relying only on salespeople, they can enter diamond attributes and see a price prediction. It empowers them to negotiate with confidence.

### Helping Sellers

For sellers, the tool provides consistency. It helps them justify pricing decisions to customers with data-driven evidence. It builds trust in the sales process.

### Education

For students, the dataset and app become learning tools. They can see how categorical and numeric features influence outcomes. It demonstrates how data science connects to real-world markets.

### Portfolio Value

For me, this project is part of my professional portfolio. It shows my ability to design and implement a full pipeline, from preprocessing to deployment. It is a demonstration of technical and design skills.

---

## Lessons Learned

The project reinforced many lessons. Preprocessing consistency is critical. Serialization choices affect safety and usability. Error handling is not optional. Deployment platforms each have strengths and weaknesses. Streamlit accelerates prototyping but also introduces constraints. Building this project made me think about the full lifecycle of machine learning applications, not just accuracy.

---

## Possibilities

The project can be expanded in several directions:

- **Explainability**: Adding SHAP values to show feature contributions.
- **Integration**: Connecting with live APIs for market data.
- **Scaling**: Supporting larger datasets and distributed training.
- **Visualization**: Building dashboards to show trends and insights.
- **Automation**: Adding CI/CD pipelines for seamless deployment.

These improvements would make the tool even more powerful and informative.

---

## Broad Strokes

### Enhancing Transparency in Pricing

The jewelry industry often feels opaque to buyers. Prices can differ widely for diamonds that appear similar. This project shows how data can provide clarity. By presenting predictions based on measurable features, it gives buyers more confidence.

### Empowering Decision Making

Buyers can use this tool to understand what drives price differences. Sellers can use it to explain pricing more clearly. Students can use it to explore real-world data. Professionals can use it to showcase technical skills. This diversity of impact makes the project valuable.

### Educational Applications

Educators can use this app as a teaching tool. It demonstrates how machine learning models are trained, evaluated, and deployed. It also shows how preprocessing affects results. Students can experiment with uploading new datasets to see changes in predictions.

---

## Practical Scenarios

### Buyers Comparing Diamonds

A buyer is presented with two diamonds that look identical. By entering their attributes into the app, they see why the prices differ. They can negotiate with sellers more confidently. The app helps them avoid confusion.

### Sellers Explaining Prices

A seller wants to build trust with customers. They use the tool to show how features like cut and clarity affect price. Customers see evidence, not just words. Trust grows, and transactions feel fairer.

### Students Learning Data Science

A student wants to understand preprocessing. They load the dataset, change features, and see new predictions. They learn how imputation and encoding matter. The app becomes a live lab for experimentation.

### Researchers Extending the Project

A researcher wants to explore explainability. They add SHAP values to the system. They show feature contributions. The app becomes a research tool, not just a product.

---

## The Wider Impact of Predictive Tools

### Building Confidence in Complex Markets

Complex markets like jewelry often intimidate consumers. Buyers walk into showrooms unsure of whether prices reflect quality or branding. A tool like this project offers a second opinion. By entering measurable attributes, users receive predictions that are consistent and logical. This reduces uncertainty and builds confidence.

### Encouraging Data-Driven Culture

The project highlights how data-driven thinking can transform decision making. Instead of relying solely on tradition, pricing becomes a matter of measurable factors. This cultural shift encourages organizations to embrace evidence rather than intuition. The app is an example of how accessible tools can foster such changes.

### Increasing Accessibility of Knowledge

Knowledge about diamonds has often been restricted to experts. This project makes some of that knowledge available to everyone. By offering an interactive interface, it lowers the barrier to entry. Anyone with basic information about a diamond can access predictive insights. Accessibility of this kind democratizes specialized knowledge.

### Supporting Transparent Negotiations

Negotiation is easier when both sides have access to the same information. Buyers equipped with this tool can ask better questions. Sellers can use it to validate pricing. Both parties benefit from transparency. The project therefore has value not only as a technical exercise but as a social contribution.

---

## Imagining Practical Applications

### Training Workshops for Students

Instructors can use the app in classrooms. They can walk students through dataset exploration, preprocessing, and predictions. The visual and interactive design keeps students engaged. This makes abstract concepts more concrete.

### Consulting for Small Jewelers

Small jewelers without access to sophisticated analytics could benefit. They can use the tool to guide pricing strategies. It reduces reliance on guesswork and provides a data-backed perspective. Over time, it can help them remain competitive.

### Research on Consumer Behavior

Researchers studying consumer trust can integrate the app into experiments. They can measure how buyers respond when shown data-driven evidence. This creates opportunities for behavioral studies that combine economics and data science.

### Expanding into Related Markets

The same concept could be extended to related markets. For example, predicting prices of gemstones, watches, or even artwork. The methodology remains the same: combine features with models to estimate value. This shows how the project has scalability beyond diamonds.

### Professional Development Programs

Professionals in training programs can fork the repository. They can modify pipelines, add new models, or adjust the interface. This hands-on work demonstrates their skills. It provides them with practical evidence of end-to-end project capability.

---

## Reflection

When I built this project, I expected to learn about model pipelines and Streamlit deployment. What I did not expect was how it would change my perspective on data science. It showed me that machine learning is not only about accuracy metrics but also about communication and trust. Every decision—from choosing imputers to designing the interface—affects how others perceive the results. I learned that transparency is as important as technical performance.

The broader lesson is that tools like this can extend beyond their technical boundaries. They can shape how people think, how they negotiate, and how they trust one another. That realization makes me see data science as both a technical and a social discipline. It is not enough to train models; one must also design experiences that people can understand and use.

---

## Reflections: Long-Term Views of the Project

### Sustainability of Predictive Systems

A project like this has long-term value because it shows how predictive systems can sustain relevance. Diamonds may remain luxury items, but the methodology of linking attributes to prices applies to countless industries. Cars, houses, or collectibles all follow the same principle. Features influence value, and models can learn those patterns.

### Demonstrating Responsible AI

In discussions about artificial intelligence, responsibility often centers around fairness and explainability. This project, though simple, embodies that spirit. It uses interpretable features and transparent preprocessing. It highlights the importance of safety by using skops instead of unsafe formats. By embedding responsibility in its design, it shows how even small projects can follow ethical guidelines.

### Inspiring Confidence in Technology

For many people, machine learning feels abstract. Seeing it work on a familiar product like diamonds bridges that gap. The project demonstrates that machine learning is not magic but logic applied to data. This inspires confidence among users and helps them see technology as approachable.

### Broadening Perspectives on Data

Working on this project taught me that data is not just numbers. Each feature—carat, cut, clarity—tells a story about rarity, craftsmanship, and market demand. Treating these attributes as variables does not reduce their meaning; it makes their impact visible. That shift in perspective was one of the deepest lessons I took from the experience.

---

## Use Cases: Expanding the Vision

### Academic Research in Economics

Economists can use the app to study pricing strategies. They can test how much weight markets place on physical attributes versus branding. This makes the app a research instrument, not just a consumer tool.

### Empowering Online Marketplaces

Online platforms could integrate such a predictor to give instant price estimates to sellers. It reduces disputes and aligns expectations. By adding transparency, it could improve customer satisfaction and reduce returns.

### Government and Regulatory Bodies

Agencies monitoring luxury markets could use such tools to detect anomalies. If a seller consistently lists items far above predicted values, regulators could investigate. This use case shows how machine learning can support fairness in markets.

### Extending to Insurance Valuations

Insurance companies often need to estimate replacement value for jewelry. This tool could support faster and more consistent valuations. It reduces subjectivity and speeds up claim processes. It also ensures fairness for policyholders.

### Personal Financial Planning

Individuals planning major purchases could use this tool. They could enter attributes of diamonds they consider buying. The predictions would guide them on whether the price is reasonable. This use case shows personal-level impact.

---

## Closing Reflections

At its heart, this project is about clarity. It is about turning data into insight and using insight to reduce confusion. It is about building trust through transparency. The lessons extend beyond diamonds. They apply to how we build systems, how we communicate results, and how we use technology to empower people. I believe that is why this project matters—not just as code but as a philosophy for how data science can serve society.

---
