---
layout: default
title: "Building my AI Book Rating Predictor"
date: 2025-02-04 17:32:45
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/book_rating.webp
thumbnail_mobile: /assets/images/book_rating_sq.webp
demo_link: https://rahuls-ai-book-rating-predictor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_book_rating_predictor
---

The idea for this project came from a simple frustration while browsing online bookstores. I often came across books with an overwhelming number of ratings and reviews. Some books had thousands of reviews but still averaged only a moderate score, while others with fewer reviews managed to hold stellar ratings. I began to wonder if I could build a small application that would let me experiment with predicting the average rating of a book based only on basic measurable features. I imagined this as a learning exercise that could connect model training with deployment. Dataset used [here](https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks).

The personal push came when I tried to organize my own reading list. I realized that if I could have a quick predictor to guess a potential rating before committing to read a book, it would save me time and add an element of curiosity. That thought encouraged me to pick a dataset, train a regression model, and wrap it into a simple user‑facing application. The journey was both instructive and satisfying because it taught me the entire flow from data to model to an interactive interface.

---

## File Structure Overview

When I packaged the project, I uploaded three main items into GitHub for hosting and later deployment:

- `app.py`: the main Streamlit application file that defines the interface and the prediction flow.
- `requirements.txt`: the dependency list that ensures consistent environment setup when others run the project.
- `models/rating_model.pkl`: the serialized model trained earlier and stored for loading inside the application.

Each of these files has a role that ties the project together. Without the application file, there would be no interface. Without the requirements file, installation would be inconsistent. Without the model file, the prediction pipeline would not function. I will now walk through each file, then go block by block into the main application.

---

## requirements.txt

The requirements file is one of the simplest but most important files. It looks like this:

```python
streamlit>=1.37
scikit-learn==1.6.1
joblib==1.5.2
pandas==2.2.2
numpy==2.0.2
```

This file locks down the dependencies. I set a minimum version for Streamlit to make sure the interactive widgets run properly. For scikit‑learn, joblib, pandas, and numpy I pinned exact versions because I wanted reproducibility across systems. By specifying these versions, I could avoid the classic problem of code working on my machine but breaking on someone else’s machine. This step, while short in lines, has a major effect on project stability.

---

## app.py

Now I move to the heart of the project: the application script. The full content is below:

```python
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/rating_model.pkl")

st.title("📚 Book Rating Predictor")

# User inputs
pages = st.number_input("Number of Pages", min_value=1, max_value=2000, value=300)
ratings_count = st.number_input("Ratings Count", min_value=0, value=500)
reviews_count = st.number_input("Text Reviews Count", min_value=0, value=50)

# Predict
if st.button("Predict Rating"):
    input_df = pd.DataFrame([[pages, ratings_count, reviews_count]], 
                            columns=["  num_pages", "ratings_count", "text_reviews_count"])
    pred = model.predict(input_df)[0]
    st.success(f"Predicted Average Rating: {round(pred,2)} ⭐")
```

### Import Statements

The imports at the top bring in the three tools that the application needs: Streamlit for the interface, pandas for handling tabular data, and joblib for loading the saved model. Each import is minimal but necessary. I avoided unused imports to keep the script clean. Streamlit drives the inputs and outputs, pandas prepares the DataFrame for the model, and joblib ensures that the pre‑trained model can be loaded without retraining.

### Loading the Model

```python
model = joblib.load("models/rating_model.pkl")
```

This line loads the pre‑trained model into memory. It points to the `models` directory and retrieves the `rating_model.pkl` file. By doing this at the start, I ensured that the model is ready as soon as the application runs. It also avoids repeated reloads each time a prediction is requested. This simple command is the bridge between training done earlier and live usage inside the app.

### Streamlit Title

```python
st.title("📚 Book Rating Predictor")
```

This sets the title at the top of the web interface. Even though it is a single line, it provides the first impression to users. Streamlit makes it trivial to brand the application. I used a simple title that states what the app does. It signals clearly that the tool is about predicting ratings for books.

### User Input Widgets

```python
pages = st.number_input("Number of Pages", min_value=1, max_value=2000, value=300)
ratings_count = st.number_input("Ratings Count", min_value=0, value=500)
reviews_count = st.number_input("Text Reviews Count", min_value=0, value=50)
```

Here I created three numeric input widgets. Each one has constraints to prevent invalid values. For number of pages, the minimum is one and the maximum is two thousand, covering most realistic books. For ratings count and text reviews count, the minimum is zero because a book might have no prior reviews. I also provided default values to give users a starting point. These widgets form the main interface for interaction, and they are crucial because the model depends entirely on these inputs.

### Prediction Block

```python
if st.button("Predict Rating"):
    input_df = pd.DataFrame([[pages, ratings_count, reviews_count]], 
                            columns=["  num_pages", "ratings_count", "text_reviews_count"])
    pred = model.predict(input_df)[0]
    st.success(f"Predicted Average Rating: {round(pred,2)} ⭐")
```

This is the action block. The condition checks if the user presses the button. Once the button is clicked, the application builds a DataFrame with a single row containing the inputs. The columns are aligned with what the model expects. This alignment is critical, because if the column names or order mismatched, the prediction would fail. The prediction is then computed, rounded to two decimal places, and displayed using Streamlit’s success message. This ensures feedback is immediate and clear.

---

## Detailed Functionality Walkthrough

Even though the application is small, each part hides useful lessons:

- **Conditional Check**: The `if st.button("Predict Rating"):` line is a gatekeeper. Without the click, the prediction does not run, saving compute and avoiding confusion.  
- **DataFrame Construction**: The conversion of raw input values into a DataFrame is the connector between UI and model. It transforms numbers into the structured format that the model was trained on.  
- **Prediction and Display**: The `model.predict` step is where machine learning enters the picture. Everything before it is setup, and everything after it is presentation. This block shows how a machine learning model can integrate into a few lines of code for end‑users.

---

## The Model File

The model file `models/rating_model.pkl` is not visible in text form here because it is a binary file. But it represents the heart of the project. It stores the coefficients and learned parameters from training. By saving it in a pickle format, I could avoid retraining each time the app starts. This decision illustrates the separation between training and inference. Training is heavy and requires a dataset, but inference can be light and fast when parameters are saved.

---

## Reflections on Design Choices

I deliberately kept the application lightweight. There are no external APIs or large data fetches. The predictor runs entirely on local inputs, making it fast and independent of network delays. The use of Streamlit made the interface trivial to build, and joblib made model loading straightforward. The model is abstracted away from the user, who only interacts with a clean interface.

One subtle design choice was to give meaningful default values for inputs. This allowed the interface to produce a prediction even if a user did not adjust anything. That made testing and demonstrations much smoother. Another choice was to round predictions to two decimals, which balanced clarity with precision. A raw floating value with many digits would look unpolished, while rounding created a neat display.

---

## Use Cases and Extensions

Although this project started as a personal experiment, it can be extended. For example, if I add more features such as author popularity, publication year, or genre, the predictor could improve. I could also expose the application through an API, letting other systems call the predictor programmatically. Another extension could be building a comparison view, where two books are entered side by side and their predicted ratings are shown together.

There is also potential to add interpretability features. For instance, I could use SHAP values to show which input contributed most to the prediction. That would make the tool not only predictive but also explanatory. These extensions highlight how even a simple project can evolve into a larger tool.

---

## Deployment Considerations

To deploy this on a free service like Streamlit Cloud, I had to ensure that the repository included all required files. The `requirements.txt` ensured dependencies installed automatically. The `app.py` provided the entry point, and the `models` folder held the trained model. By uploading this to GitHub and linking it to Streamlit Cloud, I could run the application live. This deployment model is convenient because it avoids local setup for users. They simply click a link and interact.

---

## Lessons Learned

Working on this project reinforced several lessons:

- Small projects can still illustrate the full cycle of machine learning application development.  
- File organization and dependency pinning prevent headaches later.  
- Clear user interfaces matter even when the model behind the scenes is the main attraction.  
- The divide between training and inference should be explicit, with saved models bridging the gap.  

These lessons extend to larger projects as well. Building in a modular and reproducible way scales better than a quick script.

---

## Technical Deep Dive

### Why Streamlit

Choosing Streamlit was intentional. Many frameworks exist for building interfaces, but Streamlit balances simplicity with power. It allows data scientists to move quickly from a notebook-style workflow to a deployed app. The ability to define inputs in one line and outputs in another makes it perfect for projects like this. In this context, Streamlit gave me slider and number input widgets without writing HTML or JavaScript.

### Why Pandas

Even though only a single row of input is used, pandas was still the right tool. Machine learning models in scikit-learn expect tabular inputs. By creating a DataFrame, I matched the interface exactly. If in the future I want to extend the application to batch prediction, pandas makes that trivial. Using a DataFrame also enforces explicit column names, which prevents the subtle but serious bug of feature misalignment. This design choice keeps the pipeline stable as features evolve.

### Why Joblib

The choice of joblib for model persistence was also deliberate. While pickle is the default serializer in Python, joblib is better suited for scikit-learn objects. It handles numpy arrays efficiently and reduces file size. When I trained the model earlier, I saved it with joblib so that loading here would be seamless. This shows how early design decisions during training affect deployment later. Every decision carries forward.

### Input Validation

One key aspect in any user-facing tool is validation. The min and max values in number inputs provide basic validation. They stop users from entering negative values or absurdly high counts. Without these constraints, predictions could become meaningless. In production, validation might also include checking input ranges against statistical distributions from training data. While I kept it simple here, the pattern can scale.

### Output Formatting

The final display of prediction was formatted with rounding and a star symbol. This was not cosmetic only. Rounding makes the result digestible. The star gives a familiar association with ratings. In building interfaces, such small touches affect usability. If a user sees a floating-point number without context, it may confuse them. By shaping the result, I turned a raw value into an experience.

---

## Broader Implications

This project, though small, touches on several broader themes in applied machine learning:

- **Trust in Predictions**: By limiting inputs to interpretable features, I allowed users to understand why a prediction comes out as it does.  
- **Reproducibility**: Pinning versions in requirements ensures that results tomorrow match results today.  
- **Separation of Concerns**: Training was done offline. Deployment is clean and only concerned with inference.  
- **User-Centric Design**: Even the most advanced model fails if the interface is poor. Streamlit bridges this gap.  

These themes repeat in professional settings. Many large systems are built on the same principles, only at larger scale.

---

## Future Roadmap

I see multiple paths for this project’s growth:

1. **Feature Expansion**: Adding variables like author reputation, language, publication date, or genre.  
2. **Model Improvement**: Testing algorithms beyond linear regression, such as gradient boosting or neural networks.  
3. **User Interface Enhancements**: Allowing sliders for input, dropdowns for categorical fields, or plots of predicted rating distributions.  
4. **Data Visualization**: Showing how predicted rating changes when inputs vary, creating sensitivity analysis charts.  
5. **Integration**: Connecting to live book databases or APIs to auto-fetch inputs based on ISBN.  

Each of these steps would push the project closer to a production-level system.

---

## Reflections on Learning Journey

This project was not only about building a predictor. It was about practicing an end-to-end cycle. From idea to code, from local environment to public deployment, every step mattered. I learned the importance of environment isolation, the ease of deploying on cloud platforms, and the responsibility that comes with even small predictions. Models can mislead if poorly communicated. Thus, clarity of output is as important as accuracy of model.

---

## Conclusion: Why This Project Matters

At its core, this predictor demonstrates how ideas can move from frustration to functional tool. I had a personal need, translated it into a modeling problem, implemented it in Python, and shared it through Streamlit. The files may be small, but the lessons are big. They cover dependencies, code clarity, model persistence, validation, presentation, and deployment.

I see this project as more than code. It is an artifact of learning, curiosity, and design thinking. It shows that building useful AI does not always require massive data or deep networks. Sometimes it requires only a clear problem statement, careful design, and a willingness to share. This mindset will guide me as I create more ambitious applications in the future.

---


## Points from my Notes

### Step 1: Install Dependencies

Before running the app, the environment must match what was used in development. That is why `requirements.txt` exists. Running `pip install -r requirements.txt` ensures every package is installed in the correct version. This step is simple but vital. Many beginner projects fail at this point because they assume global installations will work. By making this explicit, I reduced barriers for others.

### Step 2: Run the Application

Launching the app is as simple as `streamlit run app.py`. This command boots the Streamlit server and opens a local web interface. The fact that such a command exists illustrates why Streamlit is so powerful. A few lines of code turn into a web application without additional configuration. For learners, this is empowering. For developers, it shortens development cycles.

### Step 3: Enter Values

Once the app runs, users see the number input fields. This design is important because it transforms raw numbers into an interactive experience. Instead of editing code or files, users simply type or click arrows. It removes friction and opens access to non-technical users.

### Step 4: Generate Prediction

Clicking the button triggers the conditional block. This introduces a human-centered event loop. Nothing happens until the user decides. When the button is clicked, the pipeline moves from idle to active. This creates a natural rhythm for interaction.

### Step 5: Interpret the Output

The final message uses `st.success`. This style choice is deliberate. A success message shows a green box, which visually signals positivity. It creates a feeling that the operation was smooth. The rounded prediction with a star makes the result both clear and friendly. This is subtle user experience design embedded in code.

---

## Alternative Approaches

While building this project, I thought about alternatives:

- **Frameworks**: I could have used Flask or Django for the interface, but that would require HTML templates and more boilerplate. Streamlit reduced complexity.  
- **Serialization**: I could have used pickle, but joblib was more efficient for scikit-learn models.  
- **Deployment**: I could have used Docker or Heroku, but Streamlit Cloud offered simplicity and direct integration with GitHub.  

Each choice was weighed against the goal: keep it lightweight and reproducible. This trade-off analysis is part of good engineering practice.

---

## Educational Value

This project is also a teaching artifact. If I show it to others, it can serve as an example of the machine learning workflow. It demonstrates how to organize files, declare dependencies, build an interface, validate inputs, and think about deployment. It is a complete but accessible case study. Anyone can clone the repository, install dependencies, and run the app in minutes. That accessibility is a feature, not an accident.

---

## Closing Thoughts

This Book Rating Predictor is modest in scope but rich in teaching value. It brought together elements of modeling, packaging, and deployment in a single pipeline. It showed me that even small datasets and simple models can lead to interactive applications that feel alive. I also realized that the clarity of presentation often matters more to users than the complexity of algorithms.

As I continue building more projects, this one remains a reference point. It reminds me of the importance of not only training models but also thinking about how they are consumed. Software is about delivering experiences, and this project delivered a small but meaningful one. It also confirmed that the path from data to interface is not as long as it seems if broken down into well‑organized steps.

---

## Final Reflection

Reaching this point reminded me that many great projects begin as small personal experiments. What starts with curiosity can become a structured application. By writing down the process in detail, I not only documented it for others but also deepened my own understanding. Documentation is not a side task. It is part of the engineering process. This blog post is my way of making sure the learning stays visible.

---
