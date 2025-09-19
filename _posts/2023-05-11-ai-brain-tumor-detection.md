---
layout: default
title: "Building my AI Brain Tumor Detector"
date: 2023-05-11 11:43:26
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/resume.webp
demo_link: https://rahuls-ai-brain-tumor.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_brain_tumor
featured: true
---

It started with a moment that seemed ordinary but stayed with me for a long time. I once attended a technical session where medical images were being discussed. Among them was an MRI scan of a brain. To me, at that time, it looked like a complex blur of shapes and shades. The speaker pointed out subtle patterns and explained how doctors could interpret them to detect early signs of tumors. I was struck not just by the science but by how critical timely recognition was. Dataset used [here](https://www.kaggle.com/datasets/arwabasal/brain-tumor-mri-detection).

That thought stayed with me and slowly turned into a project idea. I wanted to see if I could build something that accepted a brain MRI image and returned a prediction. I wanted it to be accessible through a simple browser without heavy setups. I wanted it to be light enough to run on almost any system, yet still carry the intelligence of a trained deep learning model. This is how the brain tumor classifier project began. I will now explain every file I uploaded to GitHub, and how the app is structured.

---

## Project Files Overview

This project required only a few files, but each one was critical. The entire repository structure was simple:

- `app.py` – the Streamlit application that runs the interface and logic
- `requirements.txt` – a list of dependencies with exact versions
- `README.md` – the project description for GitHub visitors
- `model/model.tflite` – the trained model file in TensorFlow Lite format

By keeping the structure small, the project was easy to maintain. But each of these files carried weight. Without the application file, there would be no interface. Without the requirements file, deployments would break. Without the README file, others would not understand the purpose. And without the model file, the classifier would be empty. Each file was like a pillar of a small structure. Remove one, and the whole thing would fall apart. This clarity is why I documented the role of each.

---

## The README.md File

This file is often overlooked, but it sets the stage for any project. My README was short:

```python
# Brain Tumor Classifier
Upload a brain MRI image and classify as Tumor / No Tumor.
Model: MobileNetV2 transfer learning, exported to TFLite (<25 MB).
```

Although small, this description served an important purpose. It told visitors that the app classified MRI images into two categories: Tumor or No Tumor. It also noted that the underlying model was based on MobileNetV2 transfer learning. This meant I had started from a pre-trained network and fine-tuned it for the dataset. Finally, it mentioned that the model was exported to TensorFlow Lite and kept under 25 MB, making it lightweight and easy to share. A README should be concise but clear, and this one d...

Every time someone lands on a repository, the README is the first thing they see. If it is too long, people may ignore it. If it is too short, people may be confused. I wanted mine to be direct. In three lines, it answered three questions: what the app does, how it works, and what technology it uses. Even though I could have added installation steps, I kept it minimal because deployment on Streamlit Cloud made those unnecessary. In this way, the README balanced clarity with brevity.

---

## The Requirements.txt File – More Than Just a List

The next file in the repository was the `requirements.txt`. Here is the content:

```python
streamlit==1.38.0
Pillow==10.4.0
numpy==1.26.4
tensorflow-cpu==2.19.0
```

This file ensured that anyone who cloned the repository or deployed it on Streamlit Cloud would install the exact versions of libraries required. Each dependency played a clear role. But beyond that, each one carried its own history, purpose, and alternatives.

- **Streamlit**: This was the backbone of the web interface. I chose it because it reduced the complexity of deployment. Alternatives like Flask or Django would have given more control but required far more boilerplate. Streamlit allowed me to focus on the classifier itself, not web infrastructure. It handled routing, session state, and rendering automatically, which saved me many hours.  
- **Pillow**: This library provided robust tools for image manipulation. It could open formats like JPEG and PNG and resize them. Alternatives like OpenCV are more powerful for advanced computer vision tasks, but Pillow was lightweight and sufficient here. It also integrated smoothly with Streamlit because it could handle in-memory file uploads without extra conversion.  
- **Numpy**: Almost every machine learning pipeline uses Numpy arrays. I needed it for normalization and reshaping. Without Numpy, I would have to rely on slower manual loops. It also acted as the bridge between Pillow images and TensorFlow tensors. Its consistency across libraries made it the safe choice.  
- **TensorFlow CPU**: I chose the CPU version because it runs on nearly any machine, including free Streamlit Cloud servers. A GPU version would be faster but unnecessary for single inferences. This decision kept the project simple and accessible. By using the CPU version, I reduced deployment complexity and avoided incompatibility issues on cloud servers.

Specifying versions was also critical. Machine learning libraries evolve quickly, and breaking changes are common. By fixing versions, I ensured that the app would run the same way tomorrow as it did today. Without version pinning, even a minor update could silently break preprocessing or prediction functions. Reproducibility is the hidden strength of a well-written `requirements.txt` file.

---

## The App.py File – Breaking It Down

The real heart of the project was `app.py`. This was the file that created the web application. I will go through it step by step, explaining every block, every function, and every decision. Each function will be expanded not just with what it does, but also why it was written in that exact way.

### Imports and Constants

```python
import os, io, numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

IMG_SIZE = 224
```

This was the first block of the file. It imported all the external libraries that would be used throughout the program. The `os` and `io` modules handled file paths and data streams. Numpy gave me fast and efficient array operations. Pillow’s Image class made it easy to manipulate uploaded images. Streamlit brought the interactive web interface, and TensorFlow loaded the model. At the end of this block, I defined a constant `IMG_SIZE` with the value 224. This was the size that MobileNetV2 expects. All in...

This block may look like boilerplate, but it set the stage. Without a defined `IMG_SIZE`, there would be a risk of inconsistencies across functions. By defining it once, I ensured every image was resized consistently. This also made the code easier to update. If I wanted to experiment with another model that required 299x299 inputs, I could change the constant in one place, and the entire app would adapt.

### Configuring the Interface

```python
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("Brain Tumor Classifier")
st.write("Upload a brain MRI image. The model predicts Tumor vs No Tumor.")
```

This block was about the look and feel of the interface. With `st.set_page_config`, I gave the page a title and told Streamlit to center the layout. The `st.title` call placed a large heading at the top of the app. The `st.write` call gave users a short instruction. These lines may seem small, but they guided the entire user experience. They made the app approachable and clear from the very first view. A person landing on this page would know what the app does and what they should do.

The reason I chose `layout="centered"` was deliberate. Streamlit allows other layouts, but centering the content gave the app a balanced appearance on most screens. Small design choices like this determine whether users feel comfortable engaging with an app. A good technical project can still fail if its interface feels rough. This section was where design met function.

### Loading the Model

```python
@st.cache_resource(show_spinner=False)
def load_interpreter():
    model_path = os.path.join("model", "model.tflite")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

interpreter, input_details, output_details = load_interpreter()
```

This was one of the most important blocks. The `load_interpreter` function was responsible for loading the TensorFlow Lite model. I used `st.cache_resource` so that the model would be loaded only once and reused across sessions. Without caching, Streamlit would reload the model every time the page refreshed, which would waste time. Inside the function, I built the path to the model file. Then I created a TensorFlow Lite interpreter using that file. The call to `allocate_tensors` prepared memory for input...

The return values were just as important as the interpreter itself. I returned not only the interpreter but also the input and output details. These details described the expected shapes and data types. Without them, it would be guesswork to feed inputs or extract predictions. This design decision saved me from hardcoding indices or shapes. Instead, the app adapted automatically if the model structure changed. This was a subtle but powerful example of making code robust to future changes.

### Preprocessing the Image

```python
def preprocess(img):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img).astype("float32") / 255.
    arr = np.expand_dims(arr, axis=0)
    return arr
```

This helper function ensured that every uploaded image was transformed into the correct format for the model. First, it converted the image to RGB. Brain MRI images may sometimes be grayscale, but converting to RGB made sure the model always received three channels. Next, it resized the image to 224x224, the required input size for MobileNetV2. Then it converted the image into a Numpy array of type float32 and normalized values between 0 and 1. Finally, it expanded the dimensions to add a batch size of o...

The importance of normalization cannot be overstated. Neural networks are sensitive to input ranges. If pixel values remained in the range 0–255, the model would struggle because it was trained on normalized inputs. This small line of dividing by 255 carried huge weight in prediction accuracy. Likewise, expanding the dimensions from `(224,224,3)` to `(1,224,224,3)` made sure the batch dimension was present. Models expect batches, even if there is only one image. This helper was small but it made the ent...

### Making Predictions

```python
def predict(img):
    arr = preprocess(img)
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output
```

The `predict` function wrapped the inference process. It called the `preprocess` function to prepare the input. Then it set the input tensor using the details collected when the model was loaded. The `invoke` call ran the model and produced outputs. Finally, the function retrieved the output tensor, which contained the raw prediction values. This function was neat because it separated concerns. Instead of mixing preprocessing and prediction inside one block, I made them modular. This made debugging and t...

Another detail worth noting was the threshold applied later in the code. The raw model output was a probability value. If it was above 0.5, I labeled the image as Tumor. If it was below, I labeled it as No Tumor. This threshold could be adjusted if the model was retrained or if sensitivity needed to be tuned. The predict function itself stayed neutral—it just produced values. The decision-making logic lived outside. This separation of computation from interpretation made the design clearer.

### Handling Uploaded Files

```python
uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    preds = predict(image)
    label = "Tumor" if preds[0][0] > 0.5 else "No Tumor"
    st.subheader(f"Prediction: {label}")
```

This was the part of the app that connected the user with the model. The `st.file_uploader` created a button for users to upload an image file. I limited the types to JPG and PNG formats. If a file was uploaded, the code opened it using Pillow and displayed it back to the user. Showing the uploaded image built trust by confirming what file was selected. Then the app displayed “Classifying...” while it processed the image. The `predict` function was called, and the result was stored in `preds`. I applied ...

This block was where everything came together. The interface, preprocessing, prediction, and output display all met here. The conditional check `if uploaded_file is not None` was essential. Without it, the app would crash when no file was uploaded. It may look like a simple guard clause, but it prevented runtime errors. Good error handling in simple terms often comes down to checks like this.

---

## The Model File – Inside TensorFlow Lite

The model was stored in `model/model.tflite`. This file was the product of training. I had fine-tuned a MobileNetV2 model on a dataset of brain MRI images. Once trained, I converted the model to TensorFlow Lite format. This conversion reduced its size and made it more efficient. The final file was under 25 MB, which was important because GitHub has file size limits and Streamlit runs more smoothly with lightweight models. This file contained the intelligence of the project. Without it, the app could not ...

TensorFlow Lite conversion itself was a critical step. A large TensorFlow model may be hundreds of megabytes and too heavy for deployment. By converting to TFLite, I compressed weights and optimized layers for inference. The process typically involved calling TensorFlow’s converter API, which transformed the graph and applied optimizations. This made the model faster to load and run, especially on devices without GPUs. Choosing this path meant I could deploy the classifier on Streamlit Cloud without worr...

---

## Deployment Steps – From Local to Cloud

To make this project accessible, I had to deploy it. Here were the steps I followed in detail:

1. **Creating the repository**: I made a new GitHub repository and uploaded all files. I kept the structure clean so Streamlit Cloud would detect the app correctly.  
2. **Verifying dependencies**: I checked that `requirements.txt` had exact versions. Streamlit Cloud installs these automatically, and missing one would cause deployment failures.  
3. **Linking GitHub to Streamlit Cloud**: I logged into Streamlit Cloud, clicked “New app,” and connected my GitHub account. I selected the repository and branch that contained the project.  
4. **Choosing entrypoint**: Streamlit Cloud looked for `app.py` by default. Since I used that filename, there was no need for extra configuration.  
5. **First launch**: The platform installed all dependencies, set up the environment, and launched the app. This took a few minutes the first time.  
6. **Sharing the link**: Once running, I received a unique URL. Anyone with the link could upload an MRI image and test the classifier.

The deployment may sound simple, but each step mattered. If the repository had the wrong structure, Streamlit would not detect the app. If the requirements file was incomplete, the app would fail to start. If the model was too large, GitHub would reject it. Deployment is not just about pushing files—it is about respecting the constraints of each platform. This is why I paid close attention at every stage.

---

## Design Decisions and Reflections

Every project involves decisions. For this one, I had to decide between multiple frameworks, models, and workflows.

- **Why MobileNetV2?** It is lightweight and optimized for mobile and embedded devices. I could have chosen ResNet or EfficientNet, but those would have been larger and slower. MobileNetV2 balanced accuracy with speed. Its depthwise separable convolutions reduced computation without destroying performance.  
- **Why TensorFlow Lite?** Standard TensorFlow models are too large for GitHub’s 25 MB file size limit. TensorFlow Lite made the model compact enough while still retaining performance. It also opened the possibility of running the model on mobile devices in the future.  
- **Why Streamlit?** Alternatives like Flask or FastAPI would require HTML templates and routing logic. Streamlit let me focus on machine learning while still giving a professional interface. It also gave me built-in widgets, image display, and layout tools.  
- **Why CPU version of TensorFlow?** Many users do not have GPUs, and free platforms often limit GPU access. By using CPU-only TensorFlow, I ensured accessibility for everyone. The inference was slightly slower, but still fast enough for single image predictions.

These decisions shaped the project into something that was not just functional but also practical. It was not about building the most complex system but about building something that works reliably.

---

## Conclusion

This project began with an idea inspired by a real-world challenge. It became a working app because I broke it down into manageable parts. The `README.md` explained the purpose. The `requirements.txt` handled dependencies. The `app.py` tied together the interface and the model. The `model.tflite` file carried the intelligence. By explaining each file and block of code, I can now look back and see how every part mattered.

I learned how important preprocessing is for reliable predictions. I saw the value of caching in reducing load times. I realized how Streamlit made deployment almost effortless. Most of all, I experienced how a trained deep learning model can be turned into something anyone can use through a browser. That is the beauty of combining machine learning with accessible tools. This project taught me not only about coding but also about design, deployment, and clarity. It is a reminder that even simple structur...

