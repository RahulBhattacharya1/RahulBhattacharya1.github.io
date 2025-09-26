---
layout: default
title: "Building my AI Fire Extinguisher Predictor"
date: 2025-05-28 10:21:17
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-fire-extinguisher-prediction.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_fire_extinguisher_prediction
featured: true
---

The idea for this project began during a safety demonstration where I observed how traditional extinguishers suppress flames. Water and chemicals are effective, but they often damage electronics or leave residue. That experience led me to wonder whether a different physical principle could be harnessed. I discovered research on acoustic fire suppression, which suggested that sound waves can disrupt combustion. This concept fascinated me because it combined physics, engineering, and machine learning. I imagined building a simple predictor that lets anyone explore how sound and airflow could extinguish a fire. Dataset used [here](https://www.kaggle.com/datasets/muratkokludataset/acoustic-extinguisher-fire-dataset).

I wanted the project to go beyond theory. My goal was to make it interactive, reproducible, and deployable. That meant carefully structuring code, packaging the model, and using Streamlit for an intuitive web interface. The result is a lightweight tool that turns a complex scientific idea into an approachable app. This blog documents every file, every block of code, and the design choices behind them.

---

## requirements.txt

This file locked down the environment. I froze dependencies to avoid surprises when running the app on another machine or cloud service.

```python
streamlit>=1.37
scikit-learn==1.6.1
joblib==1.5.2
pandas==2.2.2
numpy==2.0.2
```

- **streamlit**: powers the web interface. Without it, there would be no way to interact visually.  
- **scikit-learn**: the model was trained in this ecosystem, so the runtime environment must match it.  
- **joblib**: required to reload the serialized model. It ensures the artifact remains usable after training.  
- **pandas**: transforms raw inputs into tabular form. It keeps feature ordering and names consistent.  
- **numpy**: underpins the numerical operations used by both pandas and scikit-learn.  

Pinning exact versions ensured compatibility. Even a small version mismatch could cause errors when loading the model or displaying widgets.

---

## app.py Walkthrough

This file was the heart of the project. It orchestrated model loading, user input collection, preprocessing, prediction, and feedback display.

### Code Block 1: Imports

```python
import streamlit as st
import pandas as pd
import joblib
```

This block imported the three essential libraries. Streamlit created the interactive dashboard. Pandas gave me a structured way to manage tabular input. Joblib allowed deserialization of the saved model. Each import aligned with a distinct role: interface, data handling, and machine learning. Keeping imports at the top provided clarity and a quick overview of dependencies.

### Code Block 2: Load Model

```python
# Load model
model = joblib.load("models/fire_model.pkl")
```

Here I restored the trained model. I kept it in a dedicated `models` folder to separate code from artifacts. Joblib’s load function was efficient for large arrays, which made it ideal for machine learning objects. By loading the model once at startup, I avoided repeated disk reads, improving responsiveness. This line bridged training and deployment, making the app functional.

### Code Block 3: Title

```python
st.title("🔥 Acoustic Fire Extinguisher Predictor")
```

This single line set the identity of the app. The Streamlit title widget placed a clear heading at the top. I chose wording that balanced technical detail with approachability. A descriptive title ensured users understood the focus immediately. Even though small, this block influenced user trust and engagement.

### Code Block 4: User Inputs

```python
size = st.number_input("Fire Size (1=small, 2=medium, 3=large)", min_value=1, max_value=3)
fuel = st.selectbox("Fuel Type", ["gasoline"])  # extend as per dataset
distance = st.number_input("Distance (cm)", min_value=1)
desibel = st.number_input("Sound Intensity (dB)", min_value=50, max_value=150)
airflow = st.number_input("Airflow", min_value=0.0)
frequency = st.number_input("Frequency (Hz)", min_value=20, max_value=200)
```

This section defined all interactive controls. Each widget mapped directly to a model feature.

- **size**: captured the scale of the fire. A simple integer range kept it interpretable.  
- **fuel**: categorical input. At this stage only gasoline was supported, but the selectbox design allowed expansion.  
- **distance**: represented how far the acoustic source was placed. Negative values were prevented with a minimum.  
- **desibel**: intensity of sound waves. I constrained it to realistic dB ranges to avoid nonsense input.  
- **airflow**: measured environmental factors like ventilation. A floating number captured subtle variations.  
- **frequency**: frequency of sound waves. The range of 20–200 Hz aligned with feasible acoustic suppression research.  

These widgets transformed abstract variables into user-friendly controls. Valid ranges encoded real-world constraints, preventing unrealistic scenarios.

### Code Block 5: Fuel Encoding

```python
# Encode fuel (use same encoder as training)
fuel_map = {"gasoline": 0}
fuel_encoded = fuel_map[fuel]
```

Machine learning models cannot interpret strings. They require numeric representations. During training, I encoded gasoline as 0, so I mirrored the same encoding here. This mapping preserved consistency. If I expanded the dataset with more fuels, I would extend this dictionary. Despite its small size, this helper was crucial for aligning inference with training.

### Code Block 6: Data Preparation

```python
input_data = pd.DataFrame([[size, fuel_encoded, distance, desibel, airflow, frequency]],
                          columns=["SIZE","FUEL","DISTANCE","DESIBEL","AIRFLOW","FREQUENCY"])
```

This block created a one-row DataFrame with the same schema used during training. Explicitly naming columns prevented mismatches. Scikit-learn estimators rely on column ordering, so this ensured correct alignment. Pandas also provided clarity: I could log or display inputs easily if needed. Structuring inputs this way emphasized reproducibility.

### Code Block 7: Prediction

```python
prediction = model.predict(input_data)[0]
```

This was the core computational step. I passed the DataFrame into the model, which returned an array of predictions. Since only one row was provided, I extracted the first element. The model output was binary: `0` meant the fire was extinguished, `1` meant it persisted. This call unified training knowledge with real-time input.

### Code Block 8: Feedback

```python
if prediction == 0:
    st.success("✅ Fire extinguished successfully!")
else:
    st.error("❌ Fire not extinguished.")
```

This conditional closed the loop. Users received immediate feedback in natural language. Success and error messages provided visual cues with colors. I mapped binary results into human terms, building trust and usability. Without this block, the app would feel abstract. With it, predictions became clear outcomes.

---

## models/fire_model.pkl

The model was stored in the `models` folder. I trained it with scikit-learn and exported it with joblib. The file contained not just coefficients, but the full preprocessing pipeline. Keeping it separate from code emphasized modularity. Anyone could replace the model file while keeping the app logic unchanged. This separation followed best practices of reproducible machine learning.

---

## Deployment Considerations

I designed the repository with deployment in mind. By keeping files minimal, I ensured it could be hosted on Streamlit Cloud or similar services. The structure was straightforward:

- `requirements.txt` defined the environment.  
- `app.py` contained the logic.  
- `models/fire_model.pkl` held the trained artifact.  

This separation mirrored professional software practices. The repository could be cloned, dependencies installed, and app launched with `streamlit run app.py`. No hidden steps were required.

---

## Lessons Learned

1. **Reproducibility matters**: pinning versions saved hours of debugging.  
2. **Simplicity wins**: fewer files and fewer lines made the project easy to maintain.  
3. **User experience counts**: carefully chosen widgets improved trust.  
4. **Structure enables scaling**: separating models from code prepared the project for growth.  
5. **Clarity builds confidence**: documenting every decision made the project transparent.  

---

## Reflections and Future Directions

This project showed me how even a small app can teach big lessons. By combining physics-inspired ideas with machine learning, I built something approachable yet meaningful. The process reinforced that technical strength alone is not enough. Presentation, reproducibility, and user experience are equally important. The deliberate structure made the app not only functional but also educational.

Looking ahead, I see opportunities to expand the project. More fuel types could be added. Visualization of parameter sweeps could reveal thresholds. Integration with real-time sensors could make it practical for experiments. Each extension would build on the strong foundation established here. The project began with curiosity, and it can grow into a valuable tool for learning and exploration.

---


## Training Background

Before deployment, the model had to be trained. I collected a dataset of simulated fire scenarios with features like size, fuel type, distance, decibel level, airflow, and frequency. I used scikit-learn classifiers to learn the relationship between these inputs and outcomes. Joblib was used to export the trained estimator. This step highlighted the importance of consistency: the encoding of features during training had to be replicated exactly in app.py for reliable inference.

The training process also required careful handling of categorical variables. Since only gasoline was included at first, the encoding map was simple. For future extensions, one-hot encoding or label encoding could be applied. Another challenge was balancing classes. Extinguished versus not extinguished cases had to be represented fairly, otherwise the model would bias toward the majority class. These lessons shaped how I structured the final application.

---

## Error Handling and Validation

One improvement I considered was better error handling. Currently, if the model file was missing, joblib.load would throw an exception. Adding a check with os.path.exists would allow me to display a user-friendly error message. This kind of validation enhances robustness. Similarly, input validation could be expanded. For example, ensuring airflow values are within expected physical ranges would make the app more realistic.

Such safeguards matter when moving from prototype to production. They protect against edge cases and prevent confusing failures. By building them early, I could reduce support needs later. In a safety-related domain like fire suppression, reliability is not optional.

---

## Deployment in Streamlit Cloud

Deploying to Streamlit Cloud required minimal effort because of the clean repository structure. I pushed the project to GitHub, connected it to Streamlit Cloud, and specified Python 3.10 as the runtime. Streamlit automatically installed dependencies from requirements.txt. Within minutes, the app was live. This demonstrated the value of disciplined packaging.

I also tested local deployment using `streamlit run app.py`. This worked seamlessly, confirming that the environment was correctly pinned. Having both local and cloud options ensured flexibility. For demonstrations, the cloud option was ideal. For development, local runs gave faster feedback.

---

## Design Trade-offs

Every project involves trade-offs. Here, I chose simplicity over sophistication. For example, I used a single selectbox for fuel type rather than implementing dynamic encoding. This kept the interface minimal but limited flexibility. I also stored the model as a joblib file instead of exploring ONNX export. That decision made the system Python-dependent but easier to integrate with scikit-learn.

Another trade-off was visualization. I did not include charts or plots, even though they could provide richer feedback. The reason was to keep performance snappy and avoid clutter. Each trade-off was deliberate. Documenting them here is part of making the design transparent.

---

## Broader Implications

This project represents more than a single app. It demonstrates a workflow that can be applied broadly. The sequence of training, serializing, packaging, and deploying is common across many machine learning projects. By mastering it in a small domain, I built skills transferable to larger domains like healthcare, finance, or logistics. The emphasis on reproducibility and user experience makes this workflow sustainable.

The acoustic fire extinguisher idea itself has intriguing possibilities. Imagine testing various frequencies to identify suppression thresholds. Imagine combining models with real-time sensors to adjust sound dynamically. These scenarios show how a simple app can inspire further research.

---


## Repository Structure in Detail

The repository followed a minimal yet professional layout. This was intentional. Too many files overwhelm new users. Too few files obscure structure. I aimed for balance.

- `app.py`: the application logic, integrating input, preprocessing, prediction, and feedback.  
- `requirements.txt`: pinned dependencies for reproducibility.  
- `models/fire_model.pkl`: serialized model artifact, separated from code.  

This structure mirrors professional templates. By segregating code and models, I reduced coupling. If I trained a new model, I would only need to replace the artifact. The app would remain unchanged. That design is scalable, as future models can be dropped in with minimal effort.

I also avoided unnecessary clutter. No unused data files were committed. No redundant scripts were included. This clarity reduced onboarding time for collaborators and demonstrated discipline in design.

---

## Future Extensions

Several directions exist to grow this project. One path is expanding the dataset to include multiple fuel types such as alcohol, diesel, or wood. Each would require additional encoding and retraining. Another path is enhancing the interface with visualizations. For example, plotting suppression probability as a function of frequency would provide insight into acoustic behavior.

Another extension is integrating real-time hardware. Sensors could capture fire size, airflow, and decibels directly. The model could then predict outcomes dynamically. This would bring the project closer to practical application. A further idea is migrating the model to ONNX for cross-platform inference, reducing dependence on scikit-learn.

---

## Comparative Perspective

I compared this project to others I have built. Many machine learning prototypes remain stuck in notebooks. They showcase results but lack deployment. This project stood out because it bridged that gap. By deploying with Streamlit, I made the model accessible to non-technical users. That accessibility is rare but valuable.

Compared to larger production systems, this project is small. Yet it shares the same principles: environment management, modularity, and user experience. It is a microcosm of professional practice. That makes it an effective teaching tool and a demonstration of applied skill.

---

## Extended Reflections

Working on this project deepened my appreciation for simplicity. Machine learning is often associated with complexity, but the hardest challenge is often clarity. By writing clear code, structuring repositories cleanly, and documenting decisions, I made the project resilient. Others can build on it without guessing my intent.

I also learned that communication is part of engineering. This blog post is not an afterthought. It is an essential artifact that completes the cycle. Without explanation, code alone can mislead. With explanation, even small projects become valuable resources.

Finally, this project reminded me why I enjoy applied machine learning. It combines creativity with rigor. It transforms curiosity into tangible tools. It provides opportunities to teach and to learn. That balance of exploration and execution is what sustains my interest long-term.

---


## Deep Dive into Design Choices

### Widget Ranges
I carefully chose ranges for each input widget. Fire size was limited to three discrete levels because that captured meaningful categories without overwhelming users. Distance was constrained to positive values, since negative distances are nonsensical. Sound intensity was bounded between 50 and 150 dB, aligning with feasible acoustic experiments. Frequency was set between 20 and 200 Hz, covering the range of low-frequency sound waves studied in suppression research. These boundaries guided users toward valid ...
By enforcing ranges, I encoded domain knowledge directly into the interface. It protected against invalid configurations and improved the realism of the simulation. Without such boundaries, the model could be fed meaningless data, undermining trust in the tool.

### Model Storage
Placing the trained model in the `models` folder was more than convenience. It mirrored best practices where artifacts are kept separate from code. This separation clarified responsibilities: code handled logic, artifacts carried learned parameters. It also created flexibility. Replacing the model file required no code changes. This separation is one of the reasons why the repository can scale to more complex experiments.

### Dependency Pinning
The decision to lock versions in requirements.txt was deliberate. Libraries evolve quickly. A new version of scikit-learn may deprecate functions, breaking compatibility. By pinning exact versions, I guaranteed reproducibility. This matters when sharing projects publicly. Without pinned dependencies, collaborators may face installation issues. With them, setup is predictable and smooth.

### Interface Feedback
The choice of Streamlit success and error messages shaped user experience. A numeric result alone would have been cryptic. By mapping outputs into clear language and visual cues, I improved comprehension. Green checkmarks and red crosses carried intuitive meaning across cultures. This design choice turned abstract predictions into meaningful results. It transformed the model into a tool rather than just an algorithm.

---

## Final Thoughts

Completing this project reminded me that clarity is power. A well-structured repository, with pinned dependencies and modular files, has lasting value. Others can clone it, learn from it, and extend it. That openness is part of why I share it on GitHub Pages. It transforms a personal experiment into a resource for the community.

This app may be small, but it demonstrates the full lifecycle of machine learning: from idea to dataset, from training to serialization, from deployment to documentation. Writing this blog is as much part of the project as coding. It captures the reasoning, the trade-offs, and the lessons. That makes the work not only reproducible, but also teachable.
