---
layout: default
title: "Building my AI Airline Tweet Sentiment Classifier"
date: 2022-04-02 14:33:18
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/airline_tweet.webp
thumbnail_mobile: /assets/images/airline_tweet_sq.webp
demo_link: https://rahuls-ai-airline-tweet-sentiment-classifier.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_airline_tweet_sentiment_classifier
---

There are times when I scroll through online platforms and see streams of opinions. Many of these are about airlines. Some are positive, while others are filled with frustration. I remember a situation when a delayed flight caused confusion for many passengers. Reading their posts gave me an idea. What if I could build a tool that reads these texts and predicts whether the sentiment is positive, neutral, or negative? That thought stayed in my mind. I wanted something simple, clear, and useful. It became the seed for this project. Dataset used [here](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment).

The project was not just about predicting emotions. It was about learning how to connect data science with real experiences. Every time I built a part of this app, I imagined how an airline support team could use it. Instead of scanning thousands of tweets, they could glance at predicted sentiments. This could save time, highlight common pain points, and improve customer experience. That personal connection gave me energy to complete every small step of development.

---

## Project Overview

This application is a sentiment classifier built with Streamlit. It uses a machine learning model trained on airline tweets. The trained model and vectorizer were saved in pickle files. The application loads these files and makes predictions on new tweets entered by the user. The design is simple but effective. It includes a web interface, backend logic, and saved model files.

The main files I uploaded to GitHub for this project were:

- **app.py**: The core Streamlit application file where everything comes together.
- **requirements.txt**: A list of dependencies needed to run the project without errors.
- **models/sentiment_model.pkl**: The trained sentiment analysis model.
- **models/vectorizer.pkl**: The text vectorizer that converts raw text into features.

Every file plays a role in making the entire system work as a pipeline. I will now explain them in detail, block by block.

---

## The Application File: app.py

The file `app.py` is the entry point for the Streamlit app. It contains all the logic needed to load the model, process input, and display predictions. Let me show the complete file first.

```python
import streamlit as st
import joblib

# Load models
vectorizer = joblib.load("models/vectorizer.pkl")
model = joblib.load("models/sentiment_model.pkl")

st.title("Airline Tweet Sentiment Classifier")

tweet = st.text_area("Enter a tweet:")

if st.button("Predict Sentiment"):
    vec = vectorizer.transform([tweet])
    pred = model.predict(vec)[0]
    st.write(f"Predicted Sentiment: **{pred}**")
```

### Imports Section

The first part of the file imports two important libraries.

```python
import streamlit as st
import joblib
```

Streamlit is the library that allows me to create a web application with very little code. I chose it because it is lightweight and easy to deploy. Joblib is a library that helps in saving and loading Python objects, especially machine learning models and vectorizers. Without joblib, I would have to retrain or redefine the model every time the app runs. That would be inefficient and slow.

---

### Loading Pretrained Objects

```python
vectorizer = joblib.load("models/vectorizer.pkl")
model = joblib.load("models/sentiment_model.pkl")
```

Here I am loading two files. The vectorizer is a helper that transforms raw text into numerical features. This step is essential because machine learning models cannot work directly with plain words. Instead, words are converted into feature vectors that represent counts or importance. The model is the actual trained classifier. It takes those feature vectors and predicts a sentiment label. By loading both, I recreate the training pipeline during runtime without needing the training code.

This design is practical. It separates the training process from prediction. The app can be deployed with only the vectorizer and the model, making it lightweight.

---

### Application Title

```python
st.title("Airline Tweet Sentiment Classifier")
```

This block creates a title at the top of the application page. It sets the tone for users and makes the purpose of the app clear. Streamlit functions like `st.title` are helpful because they instantly convert Python code into user interface elements. It requires no HTML or CSS coding. The goal was to keep it simple and readable.

---

### Input Section

```python
tweet = st.text_area("Enter a tweet:")
```

This block creates a text area where users can paste or type a tweet. Streamlit returns the input as a string that can be processed further. The reason I used a text area instead of a text input is that tweets vary in length. Some are short, but others are longer. A text area provides flexibility.

---

### Button and Prediction Logic

```python
if st.button("Predict Sentiment"):
    vec = vectorizer.transform([tweet])
    pred = model.predict(vec)[0]
    st.write(f"Predicted Sentiment: **{pred}**")
```

This conditional block is where the actual interaction happens. When the user clicks the button, the code inside runs. The input tweet is passed to the vectorizer, which converts it into numerical form. That result is stored in `vec`. The vectorized text is then fed into the model for prediction. The model outputs a label such as positive, neutral, or negative. I only need the first element because predictions are returned as arrays. Finally, I display the predicted sentiment using `st.write`. This makes the application interactive and responsive.

The conditional `if st.button` ensures the prediction only happens when the button is pressed. This design prevents the app from predicting continuously while the user is typing. It adds control to the workflow.

---

## Requirements File: requirements.txt

The second important file is `requirements.txt`. This file lists all the Python packages that need to be installed. It ensures the app runs smoothly on different machines or cloud platforms. Here is the content:

```python
streamlit>=1.37
scikit-learn==1.6.1
joblib==1.5.2
pandas==2.2.2
numpy==2.0.2
```

Each package has a role. Streamlit creates the user interface. Scikit-learn is the library used to build and train the model. Joblib is required for loading the saved objects. Pandas helps in managing datasets during preprocessing. Numpy is the backbone for handling numerical arrays. By fixing versions, I ensure that the code will run consistently across environments. This practice avoids errors caused by version mismatches.

---

## Model Files

The models folder contains two files: `sentiment_model.pkl` and `vectorizer.pkl`. These are the outputs of training that I stored for later use.

- **vectorizer.pkl**: This file contains the fitted vectorizer. It knows the vocabulary and how to transform any text into feature vectors. Without it, I could not guarantee the new tweets are processed the same way as the training data.
- **sentiment_model.pkl**: This is the trained machine learning model. It was trained on labeled airline tweets. It can classify whether a tweet is negative, neutral, or positive. Saving the model allows me to reuse it without retraining.

These files are binary and not meant to be read by humans. They are essential because they encapsulate knowledge gained during training.

---

## Reflections and Use Cases

Building this project taught me more than technical details. It showed me how simple tools can turn into meaningful solutions. The application may look small, but the thought process behind it was serious. It combined data preprocessing, model training, persistence of objects, and a web interface into one package. That integration gave me confidence to handle larger projects in the future.

The possible use cases extend beyond airlines. Any company that faces a flood of customer opinions could benefit. Hotels, e-commerce platforms, or even public services could adapt the same idea. With minimal adjustments, the model can be retrained on different datasets. That flexibility makes it powerful.

---

## Deployment Notes

When I pushed this project to GitHub, I had to upload:

- app.py in the root folder.
- requirements.txt in the root folder.
- The models folder containing the pickle files.

After uploading, I linked the GitHub repo to Streamlit Cloud. The platform automatically read the requirements file and installed dependencies. The app ran successfully in the browser. This confirmed that the structure was correct and portable.

---

## Conclusion

This project started from a personal observation. It ended as a working application that predicts sentiments from airline tweets. Every file had a role. Every code block added value. Writing about it allows me to reflect on both the journey and the outcome. What mattered most was not the complexity, but the clarity. The simplicity of this project is what makes it effective. I see it as a foundation on which I can build more advanced applications in the future.


---

## Expanded Breakdown of Each Component

### Why I Chose Streamlit

Streamlit was not the only option. There are many frameworks such as Flask or Django. But Streamlit is built with data scientists in mind. It reduces the effort to transform Python scripts into web apps. With one line like `st.text_area`, I get a ready-to-use component. With Flask, I would have to write HTML, CSS, and handle routing. That would increase complexity. My aim was clarity, not heavy architecture. This reasoning shaped the decision from the start.

### Why Joblib Matters

Some might ask why I did not use pickle directly. While pickle works, joblib is more efficient with large NumPy arrays. Machine learning models often contain such arrays. Joblib compresses and optimizes storage. It is widely used in scikit-learn projects. Using joblib keeps the project aligned with industry practices. It also makes loading models faster, which improves user experience.

### The Vectorizer Role

The vectorizer is critical. It maps words into numbers. In my training phase, it learned the vocabulary of airline tweets. Words like "delay", "lost", or "thanks" are associated with patterns of usage. When I load the vectorizer here, it guarantees the same transformation logic is applied to new tweets. If I ignored this and built a fresh vectorizer, the results would not match. The model would misinterpret inputs. So persisting the vectorizer was not optional. It was required.

### The Model’s Job

The classifier itself is the brain. I trained it using scikit-learn, and the algorithm could have been Logistic Regression, Naive Bayes, or Support Vector Machine. Each algorithm has its strengths. Logistic Regression is interpretable. Naive Bayes is fast with text. SVM can be powerful with the right kernel. I experimented and saved the best performer. The saved file is the result of those trials. Loading it here gives me the same performance every time. That reliability is what makes deployment possible.

---

## Detailed Explanations of Code Flow

### Initialization

When the app starts, the imports and model loading happen first. This ensures everything is ready before any user interaction. If the files are missing, joblib will raise an error. This taught me the importance of keeping file paths consistent. On GitHub, the relative path `models/vectorizer.pkl` works because I preserved folder names.

### User Interaction

The text area stays empty until a user types something. Streamlit updates the variable instantly when text is entered. This dynamic binding is powerful. It feels natural to use, like filling a box on a webpage.

### Prediction Trigger

The button acts as a gate. Without it, the prediction would run continuously with every keystroke. That would waste resources. By using `if st.button`, I gain control. It also creates a clear action for users. They know when to expect results.

### Display Output

The last line uses `st.write`. It displays both text and formatting. The use of `**{pred}**` makes the sentiment bold. This improves readability. Users can quickly see the result without scanning a paragraph.

---

## Broader Reflections

### Handling Edge Cases

One challenge was dealing with empty input. If the user presses the button without entering text, the vectorizer will throw an error. To fix this, I considered adding a simple check:

```python
if tweet.strip() == "":
    st.warning("Please enter text before predicting.")
else:
    vec = vectorizer.transform([tweet])
    pred = model.predict(vec)[0]
    st.write(f"Predicted Sentiment: **{pred}**")
```

This extra block prevents errors and guides the user. It may seem small, but user experience depends on such details.

### Improving Accuracy

The current model works but has limitations. Tweets can include sarcasm, slang, or emojis. These are hard for traditional models to interpret. If I wanted to improve, I could explore deep learning. Libraries like TensorFlow or PyTorch could bring neural networks into the project. But that would increase complexity. For this phase, scikit-learn was enough. It balanced simplicity and performance.

### Lessons from Deployment

Deploying to Streamlit Cloud taught me about dependency management. If versions in `requirements.txt` do not match, the app might crash. Fixing versions was not just best practice. It was survival. I learned to always test locally with the same versions before pushing to GitHub.

---

## Use Cases Beyond Airlines

Sentiment analysis has wide applications. Companies want to know what customers feel. With this structure, anyone can swap the dataset and retrain. For example:

- A hotel chain could analyze guest reviews.
- A retailer could study product feedback.
- A news outlet could track public mood on events.

The workflow remains the same. Replace training data, retrain the model, and save new pickle files. The app does not change. This separation of training and inference makes the design flexible.

---

## Step-by-Step Reflection on Files

1. **app.py**  
   This is the heart of the system. It defines how the user interacts with the classifier. It also glues together the vectorizer and model.

2. **requirements.txt**  
   This acts as a contract between environments. It ensures consistency across local, GitHub, and cloud deployments.

3. **models/vectorizer.pkl**  
   This file is the memory of how text is turned into numbers. It makes sure predictions use the same language representation as training.

4. **models/sentiment_model.pkl**  
   This file is the intelligence. It contains the mathematical rules that decide if text is positive, neutral, or negative.

Together, these files form a pipeline. Input text goes in, predictions come out. The structure is clean and reusable.

---

## Closing Thoughts

The journey of creating this app was not just technical. It was also about discipline. Each file taught me something. Each block of code reminded me that even small steps matter. By breaking the system down, I gained clarity. By deploying it, I gained confidence. Looking back, I see how a personal observation turned into a practical tool. That transformation is what excites me about building more projects in the future.


---

## Behind the Scenes of Training

While the deployed app only shows predictions, much work happens before reaching this stage. Training a sentiment model involves several steps:

1. **Collecting Data**  
   I started with a dataset of airline tweets. These tweets were labeled as positive, neutral, or negative. Labels are critical because they act as ground truth. Without labels, supervised learning would not be possible.

2. **Cleaning Text**  
   Tweets often contain noise such as links, mentions, hashtags, or random punctuation. I removed these elements. Cleaning ensures the model focuses on meaningful words.

3. **Vectorization**  
   Using scikit-learn’s CountVectorizer or TfidfVectorizer, I transformed words into numeric form. Each tweet became a vector. The vectorizer learned the vocabulary during training and saved it for future use.

4. **Training the Model**  
   I experimented with different algorithms. Logistic Regression gave me a balance between speed and accuracy. Other models were tested but did not perform as consistently.

5. **Evaluation**  
   I split the dataset into training and testing sets. Accuracy and F1-score were calculated. These metrics guided me in choosing the best model. Once satisfied, I saved the trained model using joblib.

6. **Saving Objects**  
   Both the trained model and the fitted vectorizer were saved. This separation ensured that new predictions use the exact same transformations as the training phase.

---

## Detailed Explanations of Design Choices

### Why Separate Training and Deployment

Keeping training outside the deployed app was intentional. Training requires heavy computation and is not needed by the end user. By focusing the app only on prediction, I reduced complexity and improved performance. Users only interact with lightweight components.

### Why Save Vectorizer Separately

The vectorizer and model are distinct objects. If I only saved the model, I would lose the transformation logic. By saving both, I preserved the entire pipeline. This design avoids mismatches and ensures reproducibility.

### Why Use Relative Paths

In `app.py`, I used relative paths such as `"models/vectorizer.pkl"`. This was important for GitHub and cloud deployment. Absolute paths would fail because directory structures differ. Relative paths guarantee portability.

---

## Improving the Application

### Adding More Features

The current app only classifies one tweet at a time. I could expand it to accept CSV uploads. That way, entire datasets could be analyzed in bulk. Streamlit supports file uploads easily. The prediction loop would then process rows of data.

### Better Visualization

At present, results are displayed as text. I could add bar charts or pie charts to show sentiment distribution. Streamlit supports `st.bar_chart` and `st.pyplot`. These additions would make insights clearer for business users.

### Error Handling

Improved error handling would make the app more robust. For example, handling network issues, missing files, or unexpected input formats. Adding try-except blocks could enhance reliability.

---

## Reflections on Learning

This project gave me confidence in multiple skills. I practiced text preprocessing, model training, file handling, and web deployment. Each step felt like a mini milestone. The small victories accumulated into a working application. I realized that learning is often about building end-to-end systems rather than isolated scripts.

---

## Potential Business Value

Airlines deal with thousands of tweets daily. Manually reading them is not scalable. An automated system like this could help teams triage issues quickly. Negative tweets could be flagged for immediate attention. Positive tweets could be used in marketing campaigns. Neutral tweets could be ignored or tracked. This classification provides value by saving time and focusing resources.

The same logic can be applied in customer support centers. Agents could prioritize responses based on sentiment. This improves service quality and reduces frustration.

---

## Personal Reflection on the Journey

When I started, I doubted whether a small project like this mattered. But I saw how every block of code had meaning. The imports showed me how libraries are powerful. Loading models showed me the value of persistence. Writing a title taught me about clarity. The button logic taught me about interaction. Even saving dependencies taught me about discipline. Each lesson added up. The project grew from lines of code into a full narrative of learning.

---

## Next Steps

If I continue, I would like to explore:

- **Deep Learning Approaches**: Using LSTM or Transformer-based models for better accuracy on complex tweets.
- **Language Support**: Expanding beyond English to cover global airlines.
- **Dashboard Integration**: Embedding results into dashboards for leadership teams.
- **Real-Time Monitoring**: Connecting to Twitter APIs for live analysis.

These steps would extend the project beyond a demo into a production-ready system.

---

## Final Conclusion

The airline tweet sentiment classifier may appear small, but it represents a complete pipeline. From data collection to model training, from saving objects to deployment, it covers the cycle of machine learning projects. It demonstrates that even a personal observation can inspire technical solutions. By explaining each code block, I showed how clarity is built piece by piece. The journey taught me patience, discipline, and creativity. I now view every tweet differently, not just as words, but as signals of sentiment that can be captured, analyzed, and understood.

---
