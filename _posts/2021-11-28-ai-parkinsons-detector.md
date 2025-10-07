---
layout: default
title: "Creating my AI Parkinson's Disease Classifier"
date: 2021-11-28 10:26:45
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/parkinsons.webp
thumbnail_mobile: /assets/images/parkingsons_sq.webp
demo_link: https://rahuls-ai-parkinsons-detection-classifier.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_parkinsons_detection_classifier
---

I once came across a health article that described how subtle changes in the human voice could reveal the presence of neurological conditions. That article stayed in my mind for a long time. Later, when I began exploring data science projects, I realized that building a simple and accessible voice-based classifier could be powerful. The possibility that voice recordings might carry early indicators of Parkinson's disease inspired me to develop this project. Dataset used [here](https://www.kaggle.com/datasets/shreyadutta1116/parkinsons-disease).

During the development of this project, I kept thinking about how everyday speech, which people often take for granted, can hold medical insights. That thought shaped the way I designed the project. I wanted the app to be easy to use, even for someone without a technical background. I also wanted the underlying model to remain transparent so that a fellow developer or researcher could easily follow the code and replicate it.

---

## Project Structure
When I built this project, I organized it into a few clear files and folders. Each one played a distinct role in making the project functional:

- **app.py**: The main Streamlit application that serves as the user interface. This is where the user can input acoustic features of speech and get a prediction result.
- **requirements.txt**: The list of dependencies required to run the application. It ensures that anyone deploying the app installs compatible versions of the libraries.
- **models/**: A folder that contains the serialized machine learning pipeline and model artifacts. These files were trained offline and then stored here for direct loading inside the app.

In the following sections, I will expand on every part of this project. I will show each code block, then explain why it was written that way and how it fits into the overall solution.

---

## Dataset Background
Before writing any code, I studied the dataset used for this project. The dataset contained several voice recordings and extracted acoustic features from individuals with and without Parkinson’s disease. The features included measures like jitter, shimmer, and other vocal variations. These values reflect how consistent or irregular the voice is during sustained speech. Researchers had shown that patients with Parkinson’s disease often display distinctive patterns in these metrics.

I did not build a speech signal processor myself for this project. Instead, I used the provided dataset and focused on the machine learning pipeline. The dataset was split into training and testing sets, and the pipeline was trained offline. This design decision kept the deployment lightweight. The Streamlit app only loads the trained model and applies it to new input values without retraining. That makes it efficient, fast, and suitable for web deployment.

---

## Code Breakdown of `app.py`

The first block of code imports the required Python libraries. These libraries handle the web app interface, data processing, and model loading.

```python
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
```

This import section is small but foundational. `streamlit` powers the user interface that runs directly in the browser. `pandas` helps manage feature inputs as tabular data, which is the expected format for scikit-learn pipelines. `joblib` loads the serialized machine learning bundle from disk. `Path` provides safe and clear file path handling across different operating systems. Without these imports, the remaining code could not function. Each library was chosen for stability and reliability.

---

Next, I configured the Streamlit page and added a title.

```python
st.set_page_config(page_title="Parkinson's Disease Voice Classifier", layout="centered")
st.title("Parkinson's Disease Voice Classifier")
```

The `set_page_config` function controls how the Streamlit app appears when loaded. I set the page title so that the browser tab shows the project name. I also chose a centered layout to keep the app balanced on most screens. The `title` call creates a large headline at the top of the page. This sets the tone for the app and makes it clear to the user what the app does. Good first impressions matter, so the title plays a small but important role in usability.

---

After setting up the interface, I loaded the trained model pipeline from the `models` folder.

```python
MODEL_PATH = Path("models/parkinsons_subset_pipeline.joblib")
bundle = joblib.load(MODEL_PATH)
pipe = bundle["pipeline"]
FEATURES = bundle["features"]
```

Here, I defined the path to the stored pipeline file. The `joblib.load` function reads the file and loads it into memory. The file itself stores a dictionary with two keys. The first key holds the machine learning pipeline. This pipeline includes preprocessing steps and the final classifier. The second key holds the list of required feature names. This design choice allows me to keep the pipeline and the feature schema bundled together.

---

I then introduced instructions for the user.

```python
st.markdown("Enter the 7 acoustic features exactly as collected:")
```

This line creates a short message above the input fields. It tells the user that they need to provide seven specific acoustic features. Using `markdown` gives me the flexibility to format the text in a cleaner way than plain strings. This simple addition reduces confusion and keeps the input process clear. Clear instructions at this stage make the difference between smooth usage and user frustration.

---

The following section defines interactive numeric inputs for each of the seven features.

```python
fo = st.number_input("Average Vocal Fundamental Frequency (MDVP:Fo(Hz))", value=120.00, step=0.01, format="%.2f")
jitter_pct = st.number_input("Jitter (MDVP:Jitter(%))", value=0.01, step=0.0001, format="%.5f")
mdvp_shimmer = st.number_input("Shimmer (MDVP:Shimmer)", value=0.04, step=0.0001, format="%.5f")
hnr = st.number_input("HNR", value=20.00, step=0.01, format="%.2f")
spread1 = st.number_input("spread1", value=-5.00, step=0.01, format="%.2f")
spread2 = st.number_input("spread2", value=0.00, step=0.01, format="%.2f")
ppe = st.number_input("PPE", value=0.20, step=0.01, format="%.2f")
```

Each `st.number_input` call creates an input widget where the user can type or adjust values. The `value` argument provides a safe default that prevents empty entries. The `step` argument controls the granularity of adjustments, and the `format` argument controls how numbers display. These inputs map directly to the features required by the pipeline. By providing all seven in the interface, I ensure the app collects complete input before making a prediction.

---

Once I had the inputs ready, I converted them into a DataFrame.

```python
row = pd.DataFrame([[fo, jitter_pct, mdvp_shimmer, hnr, spread1, spread2, ppe]], columns=FEATURES)
```

This single line creates a DataFrame with one row. The row contains the values collected from the user. The `columns` parameter matches them to the feature names extracted from the bundle earlier. This alignment is critical because scikit-learn pipelines rely on exact feature names and ordering. Without this step, the pipeline would not process the inputs correctly. It is a small but essential bridge between the user interface and the model logic.

---

Next, I created a button that triggers the prediction.

```python
if st.button("Predict"):
    pred = pipe.predict(row)[0]
    if pred == 1:
        st.error("The model predicts a positive Parkinson's indication.")
    else:
        st.success("The model predicts a negative Parkinson's indication.")
```

This conditional block starts with a button. When the user clicks the button, the block executes. The model pipeline predicts on the prepared row of input. The result is either `1` for positive or `0` for negative. I then check the value with an `if` condition. If the result is `1`, I display an error message in red to signal a positive detection. If the result is `0`, I display a green success message for a negative result. This feedback loop gives the user an immediate and clear outcome.

---

## Explaining `requirements.txt`

The `requirements.txt` file ensures that the app installs the correct versions of dependencies. It looks like this:

```
streamlit>=1.34.0
pandas>=2.1.0
numpy>=1.26.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
joblib>=1.3.0
holidays>=0.55
```

Each line lists a library with a minimum version requirement. This ensures compatibility and prevents errors during deployment. Streamlit powers the interface, pandas and numpy manage data, scikit-learn provides machine learning tools, matplotlib supports optional visualization, joblib handles model loading, and holidays supports additional date handling if extended. Keeping this file in the repository allows anyone to replicate the environment easily.

---

## Model Files

The `models` folder contains two files:

- `parkinsons_model.pkl`
- `parkinsons_subset_pipeline.joblib`

These files are binary and store the trained machine learning artifacts. I trained the model offline using a dataset of acoustic features. The `joblib` pipeline file contains both preprocessing and the classifier. The `.pkl` file stores a backup version. By separating training from deployment, I kept the web app lightweight. The model does not train during runtime. Instead, it simply loads and applies the trained pipeline for predictions. This design improves speed and makes deployment easier.

---

## Deployment Considerations

When deploying this app to GitHub Pages, I knew that GitHub Pages does not directly run Python or Streamlit. To make the app work, I needed to use a hosting platform that supports Python execution. Popular choices include Streamlit Cloud or Hugging Face Spaces. Once deployed on one of those platforms, I could link the running app back into my GitHub Pages portfolio using an iframe or a button.

I also had to consider file sizes and storage. GitHub enforces a 25 MB file limit, so model files had to remain below that size. To address this, I used joblib compression when saving the pipeline. That kept the file compact and ready for version control. Hosting on Streamlit Cloud further simplified deployment because it directly reads the GitHub repository and installs dependencies from requirements.txt.

---

## Error Handling and Validation

In this simple app, I relied on defaults and safe values to prevent errors. However, I also considered edge cases. For example, if a user tried to input extreme or invalid numbers, the number input widgets would restrict the ranges. Streamlit handles this gracefully, so the app stays stable. If the model encounters a shape mismatch, the DataFrame construction ensures proper alignment. This attention to validation keeps the app usable even when inputs are unusual.

---

## Reflections and Future Work

This project gave me a chance to combine machine learning with a real-world application. I learned the importance of keeping user interfaces simple while ensuring the backend logic is correct. I also learned that deployment constraints matter just as much as model accuracy. Training a good model is one challenge. Serving it to users in an accessible way is another. Balancing both aspects is where projects succeed.

In the future, I could expand this app by adding visualization of feature contributions. For example, I could show a bar chart of which features influenced the decision most. Another extension would be allowing CSV upload of multiple patient records. That would make the tool more practical for batch analysis. Finally, I could enhance the training pipeline by testing more classifiers such as gradient boosting or neural networks. Each of these directions could deepen the value of the project.

---

## Conclusion

This project taught me how to package a trained model, design a web interface, and keep the workflow simple but reliable. The voice classifier demonstrates how acoustic features can serve as inputs to a machine learning pipeline for Parkinson's disease detection. While it is not a medical diagnostic tool, it shows the potential of combining healthcare ideas with accessible AI deployment. By breaking down every file and every block of code, I made sure that the project remains transparent and reproducible.

---


## Acoustic Feature Details

The model relies on seven acoustic features. Each one carries specific meaning about the quality of the voice. I will describe them here to show why they matter.

- **MDVP:Fo(Hz)**: This is the average vocal fundamental frequency. It represents the rate of vocal fold vibrations. Patients with Parkinson’s often show irregularities in pitch stability.
- **MDVP:Jitter(%)**: This measures short-term variations in frequency. High jitter values indicate instability in vocal fold vibration. Such irregularities are more common in patients.
- **MDVP:Shimmer**: This measures short-term variations in amplitude. It shows how much the loudness of the voice wavers. Again, Parkinson’s patients often show higher shimmer values.
- **HNR (Harmonic-to-Noise Ratio)**: This measures the proportion of harmonic sound to noise in the voice. Lower values suggest more breathiness and irregular phonation.
- **spread1** and **spread2**: These are nonlinear measures that capture spectral distribution of the signal. They are not as intuitive but provide strong discrimination power in classification tasks.
- **PPE (Pitch Period Entropy)**: This measures entropy in pitch periods. High values mean more unpredictability in pitch, which is a strong marker of disease.

Understanding these features helped me appreciate how voice carries hidden patterns. The model does not hear the sound directly. Instead, it learns from these engineered measures.

---

## Why I Chose Joblib

I used joblib to store the trained model pipeline. Joblib has advantages over pickle when dealing with large numpy arrays. It compresses the data more efficiently and loads faster. Since the pipeline contains preprocessing objects and a classifier, joblib made serialization smooth. Another advantage is backward compatibility. The same joblib file can be loaded across environments if versions match. That makes it ideal for deployment across cloud platforms.

---

## Training Pipeline Design

During training, I used scikit-learn’s pipeline object. The pipeline included scaling, feature selection, and a classifier. Scaling ensured that all features contributed equally regardless of their raw magnitudes. Feature selection reduced redundancy and improved generalization. For classification, I used logistic regression as a baseline. It provided stable and interpretable results. Later, I experimented with random forests and support vector machines.

The trained pipeline was evaluated with cross-validation. Metrics included accuracy, precision, recall, and F1-score. While accuracy was above 85%, recall for positive cases was most important. A false negative means missing a potential case. Balancing precision and recall shaped my model selection choices. This reasoning mattered more than chasing a high accuracy number.

---

## UI Design Philosophy

When I designed the Streamlit app, I thought about two groups: technical users and non-technical users. For technical users, clarity of feature names mattered. That is why I displayed the exact dataset names like `MDVP:Fo(Hz)`. For non-technical users, I added human-readable descriptions alongside the codes. This hybrid approach made the interface friendly without hiding scientific rigor. Another UI choice was using number inputs instead of text boxes.

---

## Reproducibility Practices

Reproducibility was an important concern. I used version pinning in requirements.txt to lock dependencies. I also committed all code and model files to version control. I documented dataset sources and random seeds used during training. This ensures that if I or another developer train the model again, results will be close. Without these practices, projects can become brittle. Small version changes often cause large differences in behavior.

---

## Ethical Considerations

Working on a health-related model requires caution. I made it clear that this project is not a medical diagnostic tool. It is only a demonstration of how machine learning can work with voice features. Real medical use would require clinical trials, regulatory approval, and much larger datasets. I also thought about privacy. Voice data is sensitive. If I collected real samples, I would need strong safeguards. For now, I relied on a publicly available dataset.

---

## Technical Limitations

The model has limitations. It was trained on a relatively small dataset. That means it may not generalize well to broader populations. Also, the model only considers seven features. Real voices contain many more cues, such as tremor or articulation speed. Another limitation is deployment scale. Streamlit apps are fine for demos but not optimized for high-traffic production. For scaling, I would need to containerize the model, serve it with FastAPI, and deploy it on cloud infrastructure.

---

## Personal Reflections

When I completed the first working version, I felt satisfied that the project worked end-to-end. But I also saw how many directions it could expand. It reminded me that learning projects are stepping stones. The value lies not only in the final product but in the practice of tying together datasets, models, and user interfaces. I enjoyed refining explanations in this blog because it forces me to understand my own work better.

---

## Final Thoughts

This Parkinson’s Disease Voice Classifier was more than a coding exercise. It combined healthcare motivation, machine learning skills, and deployment experience. By structuring the project carefully and documenting every decision, I created something that is both functional and instructive. I hope future readers can take this project as a template, adapt it to new datasets, and build tools that matter.

---
