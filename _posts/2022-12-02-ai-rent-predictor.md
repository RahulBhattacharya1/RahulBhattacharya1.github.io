---
layout: default
title: "Building my AI Rent Predictor"
date: 2022-12-02 17:45:51
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/rent.webp
thumbnail_mobile: /assets/images/rent_predict_sq.webp
demo_link: https://rahuls-ai-rent-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_rent_predictor
featured: true
synopsis: Here I develop a rent prediction tool using demographic and regional factors such as household income, ownership rates, and population. It estimates typical rental costs across cities, providing individuals with quick insights to guide relocation decisions and housing affordability planning.
custom_snippet: true
custom_snippet_text: Predicts regional rent using demographics, aiding relocation and housing decisions. 
---

A few months ago I was reading about how rent prices have been rising across many regions. 
At that moment I thought about how useful it would be if people could quickly estimate the median rent in a region based on simple demographic features. 
I had once seen datasets containing details like household income, family income, ownership rates, and population. 
Those variables gave me the idea that I could build a predictor that helps answer a question many people ask before moving to a new city: *what would be the typical rent there?*
This personal curiosity pushed me to create a small end-to-end project where I trained a model, saved it, and then built a Streamlit app to serve predictions interactively. Dataset used [here](https://www.kaggle.com/datasets/goldenoakresearch/us-acs-mortgage-equity-loans-rent-statistics).

The entire project ended up being structured into three main files that I had to upload to my repository: 
- `app.py` which contains the Streamlit app logic
- `requirements.txt` which lists the packages needed for deployment
- `models/rent_predictor.joblib` which is the trained machine learning model file. 
In this blog I will go over each file, explain the purpose, and then dive deep into every code block so that the logic is clear and transparent.

---

## The Core Application: app.py

The heart of the project lies in `app.py`. This script is the entry point for the Streamlit app. 
Let us first look at the entire content of this file.

```python
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/rent_predictor.joblib")

st.title("AI Rent Price Predictor")

st.write("Predict median rent for a region based on demographics and income")

# Input fields
hi_mean = st.number_input("Average Household Income", min_value=0, value=60000)
family_mean = st.number_input("Average Family Income", min_value=0, value=70000)
pct_own = st.slider("Percent Home Ownership", 0.0, 1.0, 0.5)
married = st.slider("Percent Married", 0.0, 1.0, 0.5)
pop = st.number_input("Population", min_value=0, value=10000)

# Predict
if st.button("Predict Rent"):
    features = [[hi_mean, family_mean, pct_own, married, pop]]
    prediction = model.predict(features)[0]
    st.success(f"Predicted Median Rent: ${prediction:,.0f}")
```

### Explaining the Imports

The first three lines import the external libraries.

```python
import streamlit as st
import pandas as pd
import joblib
```

Here `streamlit` is the core library that powers the interactive web interface. I imported it with an alias `st` so that every function call can be short and clear. 
Next, `pandas` is imported as `pd`. Even though I do not use heavy dataframes here, I like to keep pandas ready because sometimes preprocessing or additional analysis requires it. 
Finally, `joblib` is imported since that is the library used to serialize and deserialize trained models. 
When I trained the rent predictor model earlier, I saved it using `joblib`. Now in this script I need to load it back into memory.

### Loading the Model

The next block handles the actual model loading.

```python
model = joblib.load("models/rent_predictor.joblib")
```

This single line is critical. It tells Python to read the `rent_predictor.joblib` file located in the `models` directory. 
Joblib efficiently loads scikit-learn models back into usable form. Without this step the app would not be able to generate any predictions. 
I placed the model file inside a `models` folder to keep the project tidy. 
This also separates the heavy binary model file from the light application code.

### Setting the Title and Introduction Text

After loading the model I added a title and a description to the app.

```python
st.title("AI Rent Price Predictor")
st.write("Predict median rent for a region based on demographics and income")
```

The first line sets the main header on the Streamlit page. The second line adds a one-sentence explanation that appears right below the title. 
I always think it is important for any application to tell the user what the tool does right at the beginning. 
This reduces confusion and sets context before the inputs appear.

### Collecting User Inputs

The next section defines how users will provide input features.

```python
hi_mean = st.number_input("Average Household Income", min_value=0, value=60000)
family_mean = st.number_input("Average Family Income", min_value=0, value=70000)
pct_own = st.slider("Percent Home Ownership", 0.0, 1.0, 0.5)
married = st.slider("Percent Married", 0.0, 1.0, 0.5)
pop = st.number_input("Population", min_value=0, value=10000)
```

Each of these lines corresponds to one input field in the web interface. 
I used `st.number_input` for numerical values that are naturally entered as whole numbers, such as household income, family income, and population. 
For percentages I used `st.slider` because a slider gives a more intuitive way to select values between 0 and 1. 
By providing default values like `60000` or `0.5`, I ensured that the interface has sensible starting points, so users can see a prediction quickly even without modifying all values.

### Predicting Rent

Finally comes the prediction logic.

```python
if st.button("Predict Rent"):
    features = [[hi_mean, family_mean, pct_own, married, pop]]
    prediction = model.predict(features)[0]
    st.success(f"Predicted Median Rent: ${prediction:,.0f}")
```

This block begins with a conditional statement. The `st.button("Predict Rent")` creates a button on the interface. 
The `if` checks whether the user clicked it. 
If the button is clicked, the code inside the block runs. 
I created a list of features in the same order as the model expects: household income, family income, percent ownership, percent married, and population. 
I wrapped them inside another list because scikit-learn expects a 2D array for predictions. 
Then I called `model.predict(features)` which returns an array of predictions. Since there is only one row of input, I accessed the first element with `[0]`. 
Finally, I used `st.success` to display the predicted rent in a styled message box. 
I formatted the prediction as a dollar value with commas for readability.

---

## The Dependencies: requirements.txt

The second file in the repository is `requirements.txt`. This file is small but essential. 
It lists the packages that must be installed for the app to run. When deploying to Streamlit Cloud or another environment, the system reads this file and installs the dependencies automatically.

Here are the contents.

```text
streamlit
pandas
scikit-learn
joblib
```

- `streamlit`: required to build and run the web interface.  
- `pandas`: useful for handling data and ensures compatibility if any preprocessing is added later.  
- `scikit-learn`: needed because the model was trained with this library. Even though the code does not directly import it, loading the joblib model internally requires the scikit-learn classes.  
- `joblib`: handles loading of the saved model file.  

The file is simple, but without it the deployment platform would not know what libraries to install. This is why every project needs to have this file clearly defined.

---

## The Model File: models/rent_predictor.joblib

The third important element is the model itself. This is not a script but a serialized binary file. 
When I trained the model offline I used scikit-learn and saved it with joblib. 
The file ended up being around twelve megabytes in size because it stored all the trained parameters. 
I decided to keep it in a `models` folder so that the project directory remains organized. 
Users of the repository do not need to open this file, but the app requires it at runtime. 
In effect, this file is the brain behind the predictions, while `app.py` is the body that interacts with the user.

---

## How the Whole System Works Together

When the repository is uploaded to GitHub and deployed through Streamlit, the system follows a simple flow. 
First, the platform installs all packages listed in `requirements.txt`. Next, `app.py` is executed, which loads the model file. 
Streamlit then generates the web interface automatically based on the code. Users can visit the app URL, enter values in the input fields, and press the button. 
The app collects the numbers, sends them to the model, and displays the predicted rent. 
Each component plays a specific role, and together they form a complete pipeline from input to prediction.

---

## Breakdown of Code Sections

### Why I Chose Streamlit

I decided to use Streamlit because it allows building a front-end interface without needing deep knowledge of HTML, CSS, or JavaScript. 
Streamlit functions map directly to input components such as sliders, text inputs, and buttons. 
This makes it perfect for data science projects where the goal is to quickly share results with end users. 
In this project, I knew that my audience would not want to run Python scripts manually. They would prefer a clickable web app. 
That reason alone made Streamlit the most natural choice.

Another point worth mentioning is that Streamlit automatically reloads the script when code changes. 
During development this saved me a lot of time. I could adjust the labels, test the sliders, or tweak the defaults and see the results instantly. 
This kind of rapid feedback loop improves productivity and makes the building process enjoyable. 
I have worked with Flask and Django before, but they require templates and routing. Streamlit lets me skip all of that overhead.

### More on Model Loading with Joblib

The choice of joblib for saving and loading the model was deliberate. 
I could have used pickle, which is Python's built-in serialization module, but joblib is more efficient for objects containing large numpy arrays. 
Scikit-learn models often include arrays of coefficients and parameters. Joblib stores them in a compressed binary format that loads faster. 
That becomes especially important when deploying because users expect quick responses. 
If the model took several seconds to load, the first request might fail. By using joblib I avoided this problem.

The other advantage is that joblib is widely used in the Python data science ecosystem. 
This means future contributors to the project will not struggle with compatibility. 
The file extension `.joblib` also makes it clear that this file is a serialized model. 
That improves readability of the repository.

### Input Design Decisions

Each input element in the interface was chosen carefully. 
For example, `st.number_input` provides a small box where users can type or adjust values with arrow buttons. 
This works well for income values, which may not be intuitive to represent with a slider. 
By contrast, `st.slider` is ideal for percentages because dragging a bar from 0 to 1 gives a better sense of scale. 
These decisions were not random; they directly affect usability. 
When building apps, small design choices like these determine whether the tool feels intuitive.

Default values also serve a purpose. 
When a user first opens the app, they immediately see numbers in the fields. 
This encourages them to try the button and see a result right away. 
If every input started empty, the app might feel broken or confusing. 
That initial experience matters because it keeps users engaged.

### The Prediction Step in Detail

The prediction step deserves extra explanation. 
The `features` variable is constructed as a two-dimensional list: `[[hi_mean, family_mean, pct_own, married, pop]]`. 
It might seem unnecessary to wrap the features in another list, but this matches the expected input shape of scikit-learn models. 
They expect a matrix of rows and columns, where each row is one observation and each column is one feature. 
If I passed a one-dimensional list, the model would throw an error. 
By keeping the correct structure, I ensured that the code runs smoothly every time.

The output of `model.predict(features)` is also an array, even if it contains only one number. 
That is why I accessed `[0]`. Without indexing, the variable would be a numpy array, which might print awkwardly. 
By extracting the scalar, I can format it as a clean number. 
Finally, the f-string formatting `f"Predicted Median Rent: ${prediction:,.0f}"` rounds the number to zero decimal places and adds commas. 
This transforms a raw number like `1895.23421` into something more readable like `1,895`. 
It is a small detail but adds a professional touch.

### Handling Edge Cases

One of the strengths of this app is that the input widgets enforce valid ranges. 
For example, household income cannot be negative because I set `min_value=0`. 
The same goes for population. These constraints prevent users from accidentally entering nonsensical data. 
Similarly, the sliders for ownership and marriage percentages are bounded between 0.0 and 1.0. 
This makes sure the model never sees values outside the expected range. 
In practice, adding such safeguards reduces errors and improves stability.

---

## Deployment Considerations

When I uploaded the repository to GitHub and connected it to Streamlit Cloud, the platform automatically detected `requirements.txt`. 
It created a Python environment and installed all the listed packages. 
Once the environment was ready, it executed `app.py`. Streamlit then created a secure URL for the app. 
Anyone with the link could now interact with the rent predictor in their browser.

One challenge was the model file size. At over twelve megabytes, it is much larger than the code. 
This meant that the repository itself became heavy. I learned that GitHub has a file size limit of 25 MB per file, so I was safe. 
Had the model been larger, I might have needed Git LFS or another storage solution. 
It reminded me that deployment is not just about code but also about managing assets responsibly.

Another deployment detail is the fact that scikit-learn versions matter. 
If the model was trained on one version and loaded on another, errors could appear. 
That is why I pinned scikit-learn in `requirements.txt`. 
It ensures that the environment running the app has the same library versions as the training environment. 
This consistency prevents compatibility issues.

---

## Lessons Learned from the Project

Looking back, the biggest lesson was the importance of simplicity. 
A predictive model on its own is not useful unless people can interact with it. 
By using Streamlit, I bridged the gap between technical code and a user-friendly interface. 
The inputs were designed to be natural, the predictions were formatted to look professional, and the output was instant. 
These qualities made the app more approachable.

Another lesson was about reproducibility. 
Uploading all three files—`app.py`, `requirements.txt`, and `rent_predictor.joblib`—meant that anyone could clone the repository and deploy the app themselves. 
This creates transparency. Others can verify the results, extend the model, or repurpose the interface for their own datasets. 
In data science, reproducibility is a sign of quality work.

---

## Possible Extensions

While the current app is functional, there are many ways to extend it. 
One option is to add more features, such as unemployment rate or education levels. 
These could improve prediction accuracy. 
Another option is to let users upload a CSV file with multiple regions and get predictions for all of them at once. 
That would turn the app into a batch predictor rather than a single-point estimator.

Visualizations could also enhance the experience. 
For example, showing a histogram of predicted rents or a scatter plot comparing inputs to outputs. 
Streamlit makes it easy to integrate plots from matplotlib or plotly. 
Such visual elements would help users understand trends, not just single numbers.

Finally, deploying the model with an API backend could make it more scalable. 
For small usage Streamlit works fine, but for production-grade traffic an API served with FastAPI or Flask might be more robust. 
The Streamlit interface could then consume the API. 
This separation of concerns is common in professional deployments.

---

## Reflections

This project was born out of curiosity but grew into a complete end-to-end system. 
It combined model training, serialization, application coding, dependency management, and deployment. 
Every file in the repository played a role, from the heavy model file to the tiny requirements list. 
The success of the project shows that building useful applications does not require complex infrastructure. 
With the right tools, even a single developer can create and share something valuable.

I now see the app not just as a rent predictor but as a template. 
It can inspire other small projects where a trained model is shared with the world through a simple interface. 
That perspective makes the effort worthwhile and leaves me motivated to build more such tools.

---


## Appendix: Detailed Role of Each Dependency

### Streamlit
Streamlit is the backbone of the application. 
Without it there would be no interface and the model would remain hidden in code. 
What makes Streamlit unique is that it abstracts away the complexity of web development. 
In other frameworks you need to define routes, templates, and static files. 
In Streamlit you just write Python scripts and they immediately turn into apps. 
This project is a clear example of that simplicity. 
The entire interface is less than fifty lines of code yet it is fully interactive.

### Pandas
Pandas might seem optional in this version because the app does not explicitly use dataframes. 
However, I included it because I know that any extension of the app will likely require structured data. 
For example, if users upload a CSV file of multiple regions, pandas will be the tool to read it and transform it into features. 
Keeping it in the requirements ensures that the app is ready for such upgrades. 
It also signals to collaborators that the project is designed with data handling in mind.

### Scikit-learn
Even though the script does not import scikit-learn directly, the model depends on it internally. 
When joblib deserializes the file, it reconstructs the scikit-learn estimator that was saved earlier. 
If the library is missing or incompatible, the loading will fail. 
That is why scikit-learn must be present in the environment. 
It also allows me to retrain or update the model without changing the environment drastically. 
The presence of scikit-learn thus secures the future of the project.

### Joblib
Joblib is the tool that makes model persistence simple. 
It is optimized for numerical data and compresses large arrays efficiently. 
In this project, it reduced both the size of the model file and the time required to load it. 
Another important aspect is that joblib is reliable. 
It is maintained and widely used, so I can trust that it will work for years to come. 
Its role may seem small, but without it there would be no way to move the model from the training phase to the deployment phase.

---

## Broader Perspective

Projects like this remind me that applied machine learning is not just about algorithms. 
It is about creating tools that connect data, models, and people. 
By packaging the predictor into a Streamlit app I crossed that bridge. 
I took something abstract, like regression coefficients, and turned it into something practical: a rent estimate. 
That transformation is what excites me about building such projects.

In the end, the AI Rent Price Predictor is both a demonstration and a foundation. 
It demonstrates how to combine Python libraries to solve a small problem. 
It also provides a foundation for larger applications that can expand in scope and scale. 
That dual nature makes it valuable not only to me but to anyone who reads the code and decides to extend it. 
I believe that is the true impact of sharing work publicly.

---

## Closing Thoughts

This project taught me that even a small dataset and a lightweight model can become useful when paired with a simple interface. 
Building the Streamlit app forced me to think about user experience, not just the model accuracy. 
Adding elements like sliders and formatted output makes the app approachable for non-technical users. 
The key lesson was that data science does not stop at building a model. Sharing it in an accessible form matters just as much.

---
