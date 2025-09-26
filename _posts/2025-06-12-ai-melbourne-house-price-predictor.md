---
layout: default
title: "Building my AI Melbourne House Price Predictor"
date: 2025-06-12 14:27:31
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-melbourne-house-price-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_melbourne_house_price_predictor
featured: true
---

The Melbourne House Price Predictor originated from a simple observation: property prices often appear inconsistent. Two houses with the same number of rooms might differ greatly in value. Some smaller houses near central Melbourne were priced higher than much larger homes further away. These inconsistencies suggested the presence of complex interactions between variables. I wanted to see if machine learning could reveal these hidden relationships. Training a regression model was the first step, but the real challenge was building an application that others could use reliably. Dataset used [here](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot).




This project therefore became more than a machine learning experiment. It evolved into a case study in deployment. I had to think about frozen dependencies, environment documentation, modular design, usability, reproducibility, and scaling. This blog provides an exhaustive breakdown of the files, the functions, the reasoning behind design choices, and lessons that apply broadly.



## requirements.txt


Dependencies required for the app to run are pinned here.


```python
streamlit>=1.37
scikit-learn==1.6.1
joblib==1.5.2
pandas==2.2.2
numpy==2.0.2

```


The requirements file ensures consistency across machines. Streamlit powers the web interface, pandas structures input data, scikit-learn provides tools for the model, and joblib handles model serialization. By freezing versions, I prevent issues from changes introduced in library updates.




In deployment, this file is essential. Without it, cloud platforms might pull newer versions of libraries, leading to unpredictable behavior. In collaborative environments, requirements.txt eliminates ambiguity, ensuring that every contributor works under the same conditions.



## models/ENV_VERSIONS.txt


This file captures the full environment used during training.


```python
sklearn==1.6.1
pandas==2.2.2
numpy==2.0.2
joblib==1.5.2

```


This snapshot ensures that retraining can reproduce identical results. It records the exact versions of all installed packages. This is particularly important for libraries like scikit-learn, which occasionally change defaults. With this file, future training runs can be recreated precisely.




This file is not used by the app itself but serves as embedded documentation. Including it reflects a professional approach to reproducibility and long-term maintainability.



## app.py


The core logic of the application is defined in app.py.


```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Melbourne House Price Predictor", layout="centered")

@st.cache_resource(show_spinner=False)
def load_model():
    model_path = Path("models/house_price_model.joblib.xz")
    if not model_path.exists():
        st.error("Model file not found at 'models/house_price_model.joblib.xz'. Upload your trained model to the models/ folder.")
        st.stop()
    return joblib.load(model_path)

st.title("Melbourne House Price Predictor")
st.markdown(
    "Predict estimated sale price based on property details. "
    "The model was trained on the Melbourne housing dataset."
)

pipe = load_model()

# Defaults (safe starting points; actual values can be adjusted by the user)
numeric_defaults = {
    "Rooms": 3,
    "Bedroom2": 3,
    "Bathroom": 2,
    "Car": 1,
    "Landsize": 150.0,
    "BuildingArea": 120.0,
    "YearBuilt": 1975.0,
    "Distance": 8.0,
    "Postcode": 3000.0,
    "Lattitude": -37.81,
    "Longtitude": 144.96,
    "Propertycount": 4000.0,
    "SaleYear": 2017.0,
    "SaleMonth": 5.0
}

categorical_defaults = {
    "Type": "h",                       # h=house, u=unit, t=townhouse
    "Method": "S",                     # common sale methods in dataset
    "SellerG": "Jellis",
    "Suburb": "Richmond",
    "Regionname": "Northern Metropolitan",
    "CouncilArea": "Yarra"
}

with st.form("predict_form"):
    st.subheader("Enter Property Details")

    col1, col2 = st.columns(2)
    with col1:
        Rooms = st.number_input("Rooms", min_value=1, max_value=12, value=numeric_defaults["Rooms"], step=1)
        Bedroom2 = st.number_input("Bedrooms", min_value=0, max_value=12, value=int(numeric_defaults["Bedroom2"]), step=1)
        Bathroom = st.number_input("Bathrooms", min_value=0, max_value=12, value=int(numeric_defaults["Bathroom"]), step=1)
        Car = st.number_input("Car Spaces", min_value=0, max_value=12, value=int(numeric_defaults["Car"]), step=1)
        Landsize = st.number_input("Land size (m²)", min_value=0.0, value=float(numeric_defaults["Landsize"]))
        BuildingArea = st.number_input("Building area (m²)", min_value=0.0, value=float(numeric_defaults["BuildingArea"]))
        YearBuilt = st.number_input("Year built", min_value=1800.0, max_value=2035.0, value=float(numeric_defaults["YearBuilt"]))

    with col2:
        Distance = st.number_input("Distance to CBD (km)", min_value=0.0, value=float(numeric_defaults["Distance"]))
        Postcode = st.number_input("Postcode", min_value=3000.0, max_value=3999.0, value=float(numeric_defaults["Postcode"]))
        Lattitude = st.number_input("Latitude", value=float(numeric_defaults["Lattitude"]))
        Longtitude = st.number_input("Longitude", value=float(numeric_defaults["Longtitude"]))
        Propertycount = st.number_input("Property count (suburb)", min_value=0.0, value=float(numeric_defaults["Propertycount"]))
        SaleYear = st.number_input("Sale year", min_value=2000.0, max_value=2035.0, value=float(numeric_defaults["SaleYear"]))
        SaleMonth = st.number_input("Sale month", min_value=1.0, max_value=12.0, value=float(numeric_defaults["SaleMonth"]))

    st.markdown("### Categorical Fields")
    col3, col4, col5 = st.columns(3)
    with col3:
        Type = st.selectbox("Type", options=["h", "u", "t"], index=0)
        Method = st.selectbox("Method", options=["S", "SP", "PI", "VB", "SA", "PN"], index=0)
    with col4:
        SellerG = st.text_input("Seller", value=categorical_defaults["SellerG"])
        Suburb = st.text_input("Suburb", value=categorical_defaults["Suburb"])
    with col5:
        Regionname = st.text_input("Region name", value=categorical_defaults["Regionname"])
        CouncilArea = st.text_input("Council area", value=categorical_defaults["CouncilArea"])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Build a single-row input matching training features. The pipeline handles imputation/encoding.
    row = {
        "Rooms": Rooms,
        "Bedroom2": Bedroom2,
        "Bathroom": Bathroom,
        "Car": Car,
        "Landsize": Landsize,
        "BuildingArea": BuildingArea,
        "YearBuilt": YearBuilt,
        "Distance": Distance,
        "Postcode": Postcode,
        "Lattitude": Lattitude,
        "Longtitude": Longtitude,
        "Propertycount": Propertycount,
        "SaleYear": SaleYear,
        "SaleMonth": SaleMonth,
        "Type": Type,
        "Method": Method,
        "SellerG": (SellerG or "Unknown").strip(),
        "Suburb": (Suburb or "Unknown").strip(),
        "Regionname": (Regionname or "Unknown").strip(),
        "CouncilArea": (CouncilArea or "Unknown").strip()
    }

    df_infer = pd.DataFrame([row])

    with st.spinner("Predicting"):
        try:
            pred = float(pipe.predict(df_infer)[0])
            st.success(f"Estimated Price: ${pred:,.0f}")
            st.caption("Estimate based on historical patterns from the Melbourne dataset.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.divider()
st.markdown(
    "- Ensure your model file is at `models/house_price_model.joblib.xz`.\n"
    "- If you change training library versions, update `requirements.txt` to match."
)

```


### Imports and Initialization


The script begins with imports for Streamlit, pandas, and joblib. The trained model is then loaded into memory. This ensures responsiveness since the model is available for predictions without being reloaded each time.


### Function `load_model`


The `load_model` function is an example of modular programming. Each function handles a specific task: collecting user inputs, converting them into a structured format, or generating predictions.




By isolating responsibilities, debugging becomes easier. If an error occurs in input collection, it is confined to one function rather than spread across the program. This approach also simplifies testing since each function can be tested independently.




This design mirrors best practices from production systems. Well-structured functions improve readability, collaboration, and long-term maintenance.



## Simplified Flow in Pseudo-code


To understand the logic, consider this simplified version:

```python
import streamlit as st
import pandas as pd
import joblib

# Load model once at startup
model = joblib.load('models/house_price_model.joblib.xz')

def collect_user_input():
    rooms = st.slider('Rooms', 1, 10, 3)
    land = st.number_input('Land Size', value=100)
    return pd.DataFrame([[rooms, land]], columns=['Rooms','LandSize'])

def predict_price(df):
    return model.predict(df)

def main():
    st.title('Melbourne House Price Predictor')
    input_df = collect_user_input()
    result = predict_price(input_df)
    st.write('Predicted Price:', result)

if __name__ == '__main__':
    main()
```





This captures the essence of the app: load once, gather input, predict, display. The simplicity comes from Streamlit, which handles much of the interface complexity.



## models/house_price_model.joblib.xz


This compressed file contains the trained regression model. The app reloads it for inference only, keeping training separate. This separation is essential for lightweight deployment.




Compression ensures manageable file size. It reduces repository weight and accelerates deployment without affecting accuracy.



## Execution Flow


The app executes in the following order:
1. Import dependencies.
2. Load model into memory.
3. Collect inputs via Streamlit widgets.
4. Package inputs into DataFrame.
5. Generate predictions.
6. Display results.




## Case Study Example


Suppose a user enters 3 rooms and 150 square meters of land. The app constructs a DataFrame with these values, passes it to the model, and displays the prediction. If the user changes to 4 rooms, the prediction updates immediately. This responsiveness highlights the strength of Streamlit in interactive modeling.



## Challenges


Three challenges dominated the project. First, dependency drift. This was solved with requirements.txt and ENV_VERSIONS.txt. Second, model size. Compression solved this. Third, interface design. Balancing enough detail for accuracy without overwhelming users required iteration.



## Design Philosophy


The philosophy guiding this app was clarity, modularity, and reproducibility. Clarity ensures code readability. Modularity ensures maintainability. Reproducibility ensures results can be replicated years later. Together, these principles transform a prototype into a robust project.



## Maintainability and Scaling


Maintaining the app involves monitoring dependencies, validating model outputs, and retraining with updated data when necessary. Scaling could involve containerization with Docker, deployment as an API service, or hosting in cloud platforms with load balancing.



## Monitoring and Logging


In a production setting, monitoring is essential. Inputs and outputs should be logged. Data drift should be detected and flagged. Dashboards can provide visibility, while alerts notify maintainers of anomalies.



## Ethical Considerations


Models that predict prices influence decisions about affordability and investment. It is critical to present outputs as estimates, not guarantees. Fairness and transparency must be prioritized to avoid misuse. This project, while educational, highlights these broader concerns.



## Alternative Designs


Flask or Django could have been used for the interface, but Streamlit enabled rapid development. Combining training and serving was another option, but separation kept the app light and reproducible.



## Broader Applications


The architecture generalizes to car resale predictions, rental values, retail demand forecasting, and healthcare analytics. The only requirement is swapping the model and adjusting input fields. This demonstrates the reusability of the framework.



## Best Practices


1. Separate training and deployment.
2. Compress models.
3. Pin dependencies.
4. Keep the interface intuitive.
5. Document the environment.




## Future Work


Planned work includes integrating interpretability methods like SHAP, adding real-time data feeds, containerizing the app, and deploying on enterprise cloud environments. These steps would transition the project from prototype to production system.



## Conclusion


The Melbourne House Price Predictor represents the full lifecycle of a machine learning project. Every file contributes to reproducibility and usability: requirements.txt freezes dependencies, ENV_VERSIONS.txt captures the training environment, app.py drives the logic, and the joblib file stores the trained model. The project demonstrates that successful machine learning depends not only on accuracy but also on deployment, maintainability, and responsibility.
