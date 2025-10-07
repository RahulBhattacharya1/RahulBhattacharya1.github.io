---
layout: default
title: "Building my AI Sales Revenue Predictor"
date: 2023-07-27 12:33:51
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/sales.webp
thumbnail_mobile: /assets/images/sales_rev_sq.webp
demo_link: https://rahuls-ai-sales-revenue-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_sales_revenue_predictor
---

There are moments in work where I feel overwhelmed by patterns that I notice but cannot measure. One afternoon while reviewing sales logs, I realized I was always estimating revenues based on unit price and quantity. I found myself repeatedly doing rough math in my head and often misjudging the results. That experience led me to imagine that I needed a small intelligent tool that could automate revenue predictions. It did not need to be a full enterprise system. I simply wanted a focused application that could take three order inputs and return an estimated revenue instantly. Dataset used [here](https://www.kaggle.com/datasets/sadiqshah/bike-sales-in-europe).

That personal moment of friction guided me to design a compact but powerful machine learning web app. I wanted something that I could run in the browser, share with colleagues, and also keep in my personal portfolio. I also wanted the entire solution to be easy to replicate. That meant packaging the code, dependencies, and model into a form that can be uploaded to GitHub. The goal was not just to solve my own challenge but also to showcase how quickly a small predictive system can be deployed with modern tools.

---

## Project Structure

When I built the project, I organized it into three main components. Each component plays a distinct role in how the application works:

1. **app.py** – The main Streamlit application that defines the interface and links together input, processing, and output. This is the file that executes when the app is launched in the browser.
2. **requirements.txt** – The file that specifies all necessary Python packages. By listing dependencies here, anyone can recreate the environment without guesswork.
3. **models/sales_revenue_predictor.pkl** – The trained machine learning model stored in serialized form. This file holds the logic learned during training and is loaded when the app runs.

By structuring the files in this way, I ensured clarity. Anyone cloning the repository can see exactly where the app starts, what dependencies are required, and where the trained model is stored.

---

## File: requirements.txt

The `requirements.txt` file is short but critical. It contains the following lines:

```python
streamlit
pandas
scikit-learn
joblib
```

### Explanation of Each Package

- **streamlit**: This library is the backbone of the user interface. It allows me to create an interactive web app without needing to write complex HTML, CSS, or JavaScript. Streamlit takes care of rendering text, inputs, and buttons so I can focus on logic.
- **pandas**: This package provides data manipulation features. Even though my app uses small inputs, I still wanted to store them in a DataFrame because the model was trained with pandas. Using pandas ensures the input structure matches what the model expects.
- **scikit-learn**: This is the library that was used to train the predictive model. Although the training process is not part of the app itself, scikit-learn defines the model object stored in the pickle file. Without this package, the app could not interpret the model.
- **joblib**: This is a utility library for saving and loading models. It allows me to load the `.pkl` file back into memory quickly. Joblib handles compression and efficiency better than the standard pickle module when working with scikit-learn objects.

Listing these four packages in `requirements.txt` ensures anyone can install the exact environment with a single command: `pip install -r requirements.txt`. This makes the app portable.

---

## File: app.py

This is the main driver of the application. Below is the complete code:

```python
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/sales_revenue_predictor.pkl")

st.title("Sales Revenue Predictor")

st.write("Enter order details below to predict revenue:")

# Inputs
quantity = st.number_input("Order Quantity", min_value=1, max_value=100)
unit_price = st.number_input("Unit Price", min_value=1, max_value=10000)
unit_cost = st.number_input("Unit Cost", min_value=1, max_value=10000)

# Prediction
if st.button("Predict Revenue"):
    input_df = pd.DataFrame([[quantity, unit_price, unit_cost]], 
                            columns=["Order_Quantity", "Unit_Price", "Unit_Cost"])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Revenue: ${prediction:.2f}")
```

---

### Code Walkthrough

The file starts with three imports:

- **import streamlit as st**: This line brings in the Streamlit library and assigns it the alias `st`. All interface commands like `st.title`, `st.number_input`, and `st.success` use this alias. Without this import, none of the interface would render.
- **import pandas as pd**: This imports pandas under the alias `pd`. The app uses pandas to create a DataFrame structure for inputs. The model expects its features in tabular form, so pandas is the natural choice.
- **import joblib**: This imports joblib so I can reload the trained model. The model was saved earlier as a `.pkl` file and joblib knows how to restore it.

---

### Loading the Model

```python
model = joblib.load("models/sales_revenue_predictor.pkl")
```

This line loads the serialized model into memory. During training, I saved the estimator using `joblib.dump`. Here I call `joblib.load` to reverse the process. The returned object is a scikit-learn model that understands `.predict()`. This design allows me to keep training separate from deployment. I only need the trained artifact when serving predictions.

---

### Application Title and Instructions

```python
st.title("Sales Revenue Predictor")
st.write("Enter order details below to predict revenue:")
```

These lines control the header of the app. The first call renders a bold title at the top. The second call shows an explanatory message beneath it. I wanted users to immediately understand what the app does and how to use it. Streamlit makes it easy to add text without HTML tags.

---

### User Inputs

```python
quantity = st.number_input("Order Quantity", min_value=1, max_value=100)
unit_price = st.number_input("Unit Price", min_value=1, max_value=10000)
unit_cost = st.number_input("Unit Cost", min_value=1, max_value=10000)
```

Here I created three numeric input widgets. Each widget appears as a box with up and down arrows. The arguments `min_value` and `max_value` protect against invalid inputs. For quantity, I limited values between 1 and 100 because orders should be realistic. For price and cost, I allowed a larger range since product values can vary. Using `st.number_input` ensures the values returned are already numeric and safe for calculations.

---

### Prediction Section

```python
if st.button("Predict Revenue"):
    input_df = pd.DataFrame([[quantity, unit_price, unit_cost]], 
                            columns=["Order_Quantity", "Unit_Price", "Unit_Cost"])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Revenue: ${prediction:.2f}")
```

This block executes only if the user clicks the button. The conditional `if st.button("Predict Revenue"):` acts as a trigger. Inside the block, I first create a DataFrame from the inputs. The DataFrame matches the same column names used during training: `Order_Quantity`, `Unit_Price`, and `Unit_Cost`. Passing the input in this exact structure avoids mismatches.

Once the DataFrame is created, the code calls `model.predict(input_df)`. The model returns an array of predictions. Since I only passed one row, I access the first element using `[0]`. Finally, I display the result using `st.success`, which renders the output in a highlighted style.

---

## File: sales_revenue_predictor.pkl

The `.pkl` file is not human-readable, but it is essential. It stores the learned parameters of the machine learning model. During training, the algorithm studied patterns between order quantity, unit price, unit cost, and revenue. It then encoded those rules into numbers, coefficients, and structures. Joblib compresses all of this into a single binary file.

At runtime, when `joblib.load` is called, the file is deserialized back into a model object. This means I do not need to retrain every time I want to predict. The heavy lifting was done once, and now the app benefits from instant predictions. The `.pkl` file is what makes the system efficient and reusable.

---

## Deployment to GitHub

To make this work on GitHub Pages or Streamlit Cloud, I uploaded the following files:

- `app.py` in the root folder so the platform knows where to start.
- `requirements.txt` alongside it to handle dependencies.
- `models/` directory containing `sales_revenue_predictor.pkl`.

This structure ensures that when someone clones the repository, they have everything needed to run the app. Streamlit Cloud will automatically read `requirements.txt` and install the libraries before running `app.py`.

---

## Reflections on Design

The simplicity of this project hides the power of its structure. By combining clear organization with modern Python libraries, I built something that feels both light and functional. The modular separation of code, dependencies, and model artifact reduces confusion. The use of Streamlit reduces the need for front-end work. The reliance on pandas ensures data integrity. The presence of scikit-learn allows predictive capability. The joblib integration makes the system portable.

What excites me most is how fast the system responds. As soon as values are entered and the button is clicked, the model produces a revenue estimate. This responsiveness demonstrates how machine learning can support decision making in small but impactful ways. It shows that predictive tools do not need to be large enterprise projects. Even a compact app like this can save time and reduce mistakes.

---

## Training the Model

Although the training script is not included in this repository, I want to explain the general process that produced `sales_revenue_predictor.pkl`. The model was trained with historical sales data where each row represented an order. The features used were `Order_Quantity`, `Unit_Price`, and `Unit_Cost`. The target variable was total revenue, computed as `(Unit_Price - Unit_Cost) * Order_Quantity`. By feeding many examples of such data into a regression model, the algorithm learned how these inputs relate to revenue outcomes.

I chose to use a regression estimator from scikit-learn because it provides a clean API and interpretable behavior. The model could have been a LinearRegression, a RandomForestRegressor, or another regressor type. What matters is that the estimator captured consistent patterns from the training dataset. Once the model was trained and evaluated, I used `joblib.dump` to serialize it. That produced the `.pkl` file that is now bundled with the app.

This separation of training and deployment is deliberate. It ensures that the prediction app remains lightweight. Users of the app do not need to see or understand the training process. They only need access to the final model artifact that produces predictions quickly.

---

## Why Use Pandas for a Single Row

At first glance, one might think it is unnecessary to create a DataFrame when there is only one row of input. However, the design choice has deeper reasons. Models trained with pandas expect their features in the same column order and format. If I were to pass only a list or array, I would risk misalignment. By explicitly constructing a DataFrame with column names, I guarantee that the model receives inputs in the same way it was trained.

This consistency is not trivial. Many machine learning bugs arise when training and inference data are not aligned. By always wrapping input inside a DataFrame with the correct headers, I reduce risk. It also makes it easier to extend the system later if I want to support batch predictions. With pandas, the jump from one row to multiple rows requires no code change.

---

## Streamlit User Experience

Streamlit was chosen not only for simplicity but also for clarity. I wanted the interface to feel professional while requiring minimal development overhead. The widgets such as `st.number_input` are powerful because they enforce data type and constraints. If a user tries to input an invalid value, the widget simply prevents it. This reduces error handling in code and makes the app more robust.

Another benefit of Streamlit is its reactive nature. Whenever an input changes, the script reruns. This design keeps state consistent and eliminates the need for complex event handlers. By combining a button trigger with input widgets, I gained full control over when predictions are made. The result is an experience that feels both responsive and controlled.

---

## Why Joblib Instead of Pickle

It is worth discussing why I used joblib rather than the default pickle module. Joblib is optimized for objects containing large numpy arrays, which are common in machine learning models. It compresses these arrays efficiently and reloads them faster. While pickle could technically handle the model, joblib is considered best practice for scikit-learn artifacts. By using joblib, I not only improved performance but also ensured better compatibility.

Joblib also makes it easy to switch between compressed and uncompressed storage. This can save disk space when models become large. In my case, the model file is small, but using joblib prepares the project for scalability. If I later train a more complex estimator with larger data, the same approach will still work.

---

## Error Handling Considerations

One design improvement I considered was error handling. Currently, the app assumes the model loads successfully and predictions always work. In production systems, it would be wise to wrap the loading step in a try-except block. That way, if the model file were missing or corrupted, the app could display a friendly error instead of crashing. Similarly, I could validate inputs further to ensure logical consistency, such as price always being greater than cost.

For this portfolio project, I kept the code concise to emphasize clarity. However, these considerations are important if I were to extend the app for broader use. It demonstrates awareness of the difference between prototypes and production systems.

---

## Deployment on Streamlit Cloud

Deploying this project was straightforward. I pushed the repository to GitHub and connected it to Streamlit Cloud. The platform automatically read `requirements.txt` and installed the dependencies. When it detected `app.py`, it launched the app. This seamless process shows the power of reproducible environments. Without specifying dependencies, the deployment could have failed due to version mismatches.

Another benefit of Streamlit Cloud is that it automatically provides a shareable link. I can send that link to anyone, and they can interact with the app instantly. No local installation is needed. This ease of sharing turns my small project into a public demonstration of applied machine learning.

---

## Extensibility

The design of this app allows for natural extensions. For example, I could add more input fields such as discount rate, shipping cost, or tax. I could also visualize the predicted revenue using charts. Streamlit has built-in charting functions that integrate with pandas. By plotting revenue against input parameters, users could gain deeper insights. I could even allow users to upload CSV files with multiple rows and get predictions for all orders at once.

Each extension would follow the same principle: inputs handled through widgets or file uploads, structured into pandas DataFrames, passed to the trained model, and outputs displayed in a clear format. The modular nature of the code makes these enhancements easy to add without rewriting core logic.

---

## Lessons Learned

From this project I learned the importance of structure, reproducibility, and simplicity. I discovered how even a minimal set of dependencies can enable powerful functionality. I realized that deploying a machine learning app does not require heavy infrastructure. With modern tools, it is possible to turn an idea into a working application within a short time.

Most importantly, I learned to always respond to small personal frictions with creativity. Instead of tolerating repeated calculations, I chose to build. That mindset can turn annoyances into portfolio pieces that showcase initiative and technical ability.

---


## Future Improvements

There are many ways I can take this project further. One option is to integrate authentication so that only approved users can access the predictor. Another option is to log all predictions into a database so that decision makers can analyze usage patterns. I could also add monitoring to track the accuracy of the model over time. If real-world data starts to drift, the model might lose accuracy, and retraining would be required.

Another area of improvement would be user experience. Right now, the output is a single number. I could enhance it by showing confidence intervals or comparative scenarios. For example, the app could display what revenue would look like if quantity increased by 10 percent or if cost dropped by 5 percent. These additional insights would make the tool more actionable for business decisions.

---

## Broader Context

What this project illustrates is not only a prediction tool but a demonstration of the modern data workflow. Data is collected, a model is trained, the model is saved, and a small web app deploys it. This pipeline condenses the essence of applied machine learning. It bridges the gap between abstract algorithms and tangible results. It shows how skills in Python, data science, and deployment can converge into a working product.

By keeping the system compact, I also highlighted accessibility. Anyone with basic Python knowledge can follow the repository and run the app. The barrier to entry is low, and the potential impact is meaningful. This balance of simplicity and usefulness is what excites me about projects like this.

---

## Conclusion

This project began with a simple personal frustration. Instead of tolerating the repetitive calculations, I turned that frustration into motivation. I built a solution that was practical, deployable, and shareable. Every file serves a clear purpose, every block of code contributes to the workflow, and every library plays a role in keeping the app concise.

Through this build I learned once again that the best ideas often come from small annoyances. By responding with creativity and structure, I was able to design an application that predicts sales revenue reliably. This tool not only solves my problem but also stands as an example in my portfolio. It illustrates how ideas, code, and design come together to form something useful.

---
