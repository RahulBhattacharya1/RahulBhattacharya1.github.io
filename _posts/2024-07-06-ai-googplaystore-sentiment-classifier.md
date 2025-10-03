---
layout: default
title: "Building my AI Google Play Review Sentiment Classifier"
date: 2024-07-06 10:22:31
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/play.webp
demo_link: https://rahuls-ai-googplaystore-sentiment-classifier.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_googleplaystore_sentiment_classifier
featured: true
---

This project emerged from my own experience of browsing reviews on the Google Play Store. I frequently noticed that many reviews were emotionally charged but not easy to interpret quickly. A review might start with a complaint but end on a positive note, or it could contain praise followed by criticism about a small feature. As I read more of these, I realized that manual scanning was inefficient. I needed a tool that could summarize the underlying sentiment at a glance. Dataset used [here](https://www.kaggle.com/datasets/lava18/google-play-store-apps).

I decided that a sentiment classifier could solve this need. Rather than skimming hundreds of reviews, I wanted to type or paste one into an application and instantly receive a classification. This tool would be practical not just for me but for developers monitoring app feedback. A simple positive or negative flag, along with model confidence, could help prioritize responses. That vision is what led me to design this project, which now exists as an interactive Streamlit app.

## requirements.txt

This file defines the environment needed to run the application. I will show its content and then expand on every dependency.

```python
streamlit>=1.37
scikit-learn==1.6.1
joblib==1.5.2
pandas==2.2.2
numpy==2.0.2

```

The first line specifies `streamlit>=1.37`. Streamlit is the lightweight framework that allows me to convert Python scripts into shareable web apps with minimal overhead. By setting a minimum version, I ensure features like `st.cache_resource` are available. The second line fixes `scikit-learn==1.6.1`. This is crucial because the saved model pipeline was trained under that version. If another version were used, the serialization format might not match, and loading could fail.

The third line lists `joblib==1.5.2`. Joblib is the tool used to save and load the trained pipeline. It handles numpy arrays efficiently and allows complex scikit-learn objects to be restored. The fourth dependency is `pandas==2.2.2`. Even though the current script does not call pandas directly, the model pipeline may rely on it internally, particularly if data transformations were part of the training process. Finally, `numpy==2.0.2` ensures compatibility for numeric operations.

## app.py

This file is the executable heart of the project. It contains the Streamlit logic for the interface, model loading, and prediction flow.

```python
import streamlit as st
import joblib

@st.cache_resource(show_spinner=False)
def load_pipeline():
    # Single portable artifact – no separate vectorizer/model
    return joblib.load("models/sentiment_pipeline.joblib")

st.title("Google Play Review Sentiment Classifier")

try:
    pipe = load_pipeline()
except Exception as e:
    st.error(
        "Failed to load the model pipeline. "
        "Ensure 'models/sentiment_pipeline.joblib' exists and matches the library versions."
    )
    st.exception(e)
    st.stop()

txt = st.text_area("Enter a review to analyze:")
if st.button("Predict"):
    review = txt.strip()
    if not review:
        st.warning("Please enter a review.")
    else:
        pred = pipe.predict([review])[0]
        proba = None
        # If classifier supports predict_proba, show confidence
        if hasattr(pipe, "predict_proba"):
            import numpy as np
            proba = pipe.predict_proba([review])[0]
            conf = float(np.max(proba))
            st.write(f"**Predicted Sentiment:** {pred}")
            st.write(f"Confidence: {conf:.2f}")
        else:
            st.write(f"**Predicted Sentiment:** {pred}")

```


### Imports and Setup

The first lines import Streamlit and joblib. These are essential because Streamlit powers the interface and joblib restores the saved pipeline. Without these imports, the file could not execute its core functionality. I deliberately keep the import section minimal to avoid unnecessary dependencies.

### load_pipeline Function

The definition of `load_pipeline` is decorated with `@st.cache_resource(show_spinner=False)`. This decorator plays an important role. By caching the loaded model, I avoid repeated disk reads every time the user presses the Predict button. Caching reduces latency and improves efficiency. The argument `show_spinner=False` suppresses the default loading spinner, which keeps the interface cleaner. Inside the function, I call `joblib.load("models/sentiment_pipeline.joblib")`.

The function therefore serves as a helper that abstracts the model loading process. It hides complexity from the rest of the script. If I later change the model file path, I only need to modify this function.

### Title Element

The line `st.title("Google Play Review Sentiment Classifier")` gives a clear identity to the application. A title is more than decoration. It sets user expectations immediately. Without a title, the interface would feel incomplete or confusing.

### Try-Except for Model Loading

The code then enters a try-except block to handle potential failures when calling `load_pipeline()`. This is a defensive programming technique. If the model file is missing or incompatible, the exception is caught. Inside the except block, I call `st.error` to display a readable message. Then I show the full exception using `st.exception(e)`. This is extremely helpful for debugging in deployment environments where logs may not be easily accessible. Finally, `st.stop()` halts execution. This block transforms a potential silent crash into a clear, user-facing explanation.

### Text Area Input

The widget `st.text_area("Enter a review to analyze:")` invites the user to provide input. This widget is crucial because sentiment analysis cannot proceed without text. By providing a multi-line area, I allow users to paste full reviews instead of restricting them to one line.

### Predict Button Logic

The conditional `if st.button("Predict"):` defines the event handler. When clicked, it triggers the inner logic. Inside, I strip whitespace from the input using `review = txt.strip()`. This step is subtle but important. Stripping whitespace prevents accidental inputs like a space or newline from being misinterpreted. The next block checks `if not review:`. If true, I warn the user with `st.warning("Please enter a review.")`. This validation ensures the pipeline only processes meaningful text.

The line `pred = pipe.predict([review])[0]` requests a prediction. Wrapping the review inside a list satisfies the expected input format of scikit-learn. Extracting index zero gives the scalar prediction, which is typically a string label such as "positive" or "negative".

### Probability Estimates

The conditional `if hasattr(pipe, "predict_proba"):` checks whether the pipeline supports probability output. Many classifiers, like logistic regression, do, but some, like SVMs with certain kernels, do not. This attribute check prevents errors. If the method exists, I import numpy and call `pipe.predict_proba([review])[0]`. The result is a probability distribution across classes. Taking the maximum value yields a confidence score. This addition is not mandatory, but it enriches the application.

### Display of Results

The call to `st.success(f"Prediction: {pred}")` highlights the result in a green box. The user immediately sees the classification. If probability data is available, I display it using `st.info`. This secondary display offers context. For example, a negative prediction with 0.51 confidence feels weaker than one with 0.98 confidence. Users can then interpret borderline cases with more nuance.


## models/sentiment_pipeline.joblib

This serialized object is the silent core of the application. It contains both preprocessing and classification in one pipeline. The preprocessing step may include tokenization, vectorization, and normalization, while the classifier could be logistic regression, naive Bayes, or another algorithm. Saving them together prevents mismatch. For example, if I trained with TF-IDF vectorization but later attempted to use a plain count vectorizer, the predictions would be meaningless.

Another reason for this approach is portability. Anyone who clones the repository can immediately load the model without retraining. The only requirement is matching library versions. This makes the project reproducible and shareable. I can distribute it as a teaching tool or a lightweight product prototype.


## Reflections, Lessons, and Future Directions

This project may look small in terms of code size, but it demonstrates principles that extend far beyond sentiment classification. Every function, conditional, and helper was carefully chosen to serve both functionality and usability.

One key lesson is the importance of error handling. By wrapping model loading inside a try-except block, I ensured that users receive explanations rather than cryptic failures. This principle applies to any robust software system. Another lesson is the value of abstraction. The `load_pipeline` function isolates the loading logic, making the rest of the script simpler to read and maintain. This separation of concerns is a hallmark of good design.

From a user experience perspective, adding probability estimates provides transparency. A bare label can be misleading, but pairing it with confidence makes the output more trustworthy. I learned that small enhancements like this can significantly change user perception of reliability.

Looking ahead, there are many extensions possible. I could expand the classifier to support multi-class outputs, distinguishing neutral, strongly positive, and strongly negative reviews. I could also integrate explainability libraries such as LIME or SHAP to highlight which words influenced the decision. Another idea is to batch process CSV files of reviews, generating aggregate sentiment dashboards. These directions show how a simple Streamlit app can grow into a more comprehensive analytics platform.

Finally, this project reinforced the importance of documentation. Writing detailed breakdowns like this not only helps others but clarifies my own thinking. By analyzing each code block in context, I see how design decisions interlock to create a cohesive whole. That clarity will guide me in larger and more complex projects in the future.


## Deeper Dive into Each Block

### Why Caching Matters

The decorator `@st.cache_resource` is not just a performance trick. It fundamentally changes how the application behaves under repeated use. Without caching, every button press would reload the model from disk. On large pipelines, this could take several seconds, frustrating the user. With caching, the pipeline is loaded only once per session. The absence of the spinner is also deliberate. I wanted the interface to feel smooth, not cluttered with unnecessary animations.

### Error Handling Philosophy

Error handling in this application reflects a philosophy of graceful failure. Instead of letting Python crash with a stack trace, I intercept problems and translate them into clear messages. This is not only about aesthetics. When users see a red error box with plain English, they understand what went wrong and what to fix. If I simply let the program die, they would be left guessing. This approach mirrors production-grade systems where observability and communication are vital.

### User Input Validation

The input validation step may seem trivial, but it prevents cascading issues. If an empty string or whitespace were passed into the pipeline, the vectorizer would return an empty feature set. That could lead to errors inside scikit-learn or unpredictable outputs. By checking early and issuing a warning, I prevent wasted computation. This is an example of guarding the edges of the system.

### Why Predict Probability is Optional

The conditional check for `predict_proba` demonstrates defensive design. Some classifiers provide probability estimates, but others do not. If I assumed all did, the code would break under certain models. By checking dynamically, I made the script more flexible. This flexibility means I can swap classifiers without rewriting large sections of the code.

## Deployment Considerations

When deploying on Streamlit Cloud, I ensured that the `models` folder was included in the repository. The artifact `sentiment_pipeline.joblib` is loaded relative to the app directory. If this file were missing, the app would fail immediately. To avoid confusion, I added clear error handling. I also pinned dependencies in requirements.txt so that the cloud environment would install matching versions automatically.

Another detail is resource usage. Streamlit apps run in lightweight environments with memory limits. A compact model pipeline is more efficient. That is why I bundled preprocessing and classification together. This keeps runtime memory lower and avoids recomputing vectorizers.

## Testing the Application

Before deploying, I tested the script locally. I ran `streamlit run app.py` and experimented with sample reviews. I tried both positive and negative phrases to confirm that predictions made sense. I also intentionally entered empty strings to confirm that the warning triggered correctly. I even renamed the model file temporarily to see if the error handling displayed as expected. These manual tests reassured me that edge cases were covered.

For automated testing, one could design unit tests that mock the pipeline and verify UI responses. Streamlit has ways to test logic in isolation, but for this project, manual testing sufficed.

## Design Choices

Several design decisions shaped this app. One was to use a single pipeline artifact. Another was to keep the interface minimal. I did not add multiple pages, complex navigation, or unnecessary styling. This keeps the focus on the core function: sentiment classification. Another choice was clarity over brevity. Error messages are verbose, not cryptic. I wanted the user to feel informed rather than confused.

## Educational Value

Although the app is simple, it serves as a teaching tool. Students can see how a machine learning pipeline connects to a web interface. They can inspect requirements.txt to learn about dependency management. They can explore the model artifact to understand serialization. Each file in the repository tells a story about real-world deployment practices.
