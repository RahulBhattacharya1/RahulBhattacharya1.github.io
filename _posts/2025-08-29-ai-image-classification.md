---
layout: default
title: "My Custom Image Classification with PyTorch and Gradio"
date: 2025-08-29 23:37:33
categories: [ai]
tags: [ai,art,classification]
thumbnail: /assets/images/ai_image_classifier.webp
thumbnail_mobile: /assets/images/ai_image_classifier_sq.webp
demo_link: https://rahulbhattacharya-rahuls-image-classifier.hf.space/
github_link: https://github.com/RahulBhattacharya1/ai_image_classification
featured: true
---

I’ve built and deployed an **image classifier** on [Hugging Face Spaces](https://rahulbhattacharya-rahuls-image-classifier.hf.space) that lets you upload a picture and get predictions in real time. This project combines **PyTorch**, **Torchvision models**, and **Gradio** to create a smooth end-to-end pipeline — from training a custom model to deploying it with an interactive UI.

---

<iframe
	src="https://rahulbhattacharya-rahuls-image-classifier.hf.space"
style="width:100%;height:820px;border:0;border-radius:12px;overflow:hidden"></iframe>
# 🖼️ Rahul’s Image Classifier – Build, Train, and Deploy with Hugging Face


## 🚀 The Idea

The goal is simple:  
- Train a model on your own dataset.  
- Save it with labels.  
- Deploy it to Hugging Face Spaces with a user-friendly interface.  

If no custom model is uploaded, the app falls back to **ImageNet** with a pretrained ResNet50 so you always get a working demo.

---

## 🛠️ The App (app.py)

The app is powered by **Gradio Blocks**. Here’s what happens under the hood:

### Loading Models
```python
def build_model(model_name: str, num_classes: int):
    if model_name == "resnet50":
        m = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_f = m.fc.in_features
        m.fc = torch.nn.Linear(in_f, num_classes)
        tfm = ResNet50_Weights.DEFAULT.transforms()
    # similar setup for efficientnet_b0 and vit_b_16
```

- Supports **ResNet50**, **EfficientNet-B0**, and **ViT-B16**.  
- The final classification layer is replaced with a new one matching your dataset classes.  
- Each model comes with its own preprocessing transforms.  

### Custom vs. Fallback
```python
CUSTOM_MODEL, CUSTOM_TFM, CUSTOM_CLASSES = load_custom()
if CUSTOM_MODEL is None:
    IMAGENET_MODEL, IMAGENET_TFM, IMAGENET_CLASSES = load_imagenet_fallback()
```

- If you provide `model.pth` and `labels.json`, the app loads your custom model.  
- Otherwise, it defaults to **ImageNet demo mode**.  

### Prediction
```python
def predict(img: Image.Image, top_k: int = 3):
    x = tfm(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        topk = torch.topk(probs, k=min(top_k, probs.numel()))
    return {classes[i]: float(topk.values[i]) for i in range(len(topk.indices))}
```

- Takes an uploaded image, applies transforms, runs inference, and returns top-K predictions with probabilities.  

### Gradio Interface
```python
with gr.Blocks(title="Image Classifier (Transfer Learning)") as demo:
    gr.Markdown("# Rahul's Image Classifier")
    inp = gr.Image(type="pil", label="Upload image")
    k   = gr.Slider(1, 5, value=3, step=1, label="Top-K")
    out = gr.Label(num_top_classes=5, label="Predictions")
    btn.click(fn=predict, inputs=[inp, k], outputs=[out])
```

- Upload an image, choose top-K predictions, and click **Predict**.  
- The predictions appear instantly in a neat label box.  

---

## 🎓 Training the Model (train.py)

Before deploying, I trained my own model with a flexible script.

### Build Model
```python
model, train_tfms, val_tfms = build_model(args.model_name, num_classes)
```
- Chooses architecture (`resnet50`, `efficientnet_b0`, `vit_b_16`).  
- Applies normalization and data augmentation (resize, crop, flips).  

### Training Loop
```python
for epoch in range(1, args.epochs+1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device)
    va_loss, va_acc = evaluate(model, val_loader, device)
    if va_acc > best_acc:
        torch.save(model.state_dict(), best_path)
```

- Uses **AdamW optimizer** and **CrossEntropy loss**.  
- Saves the best model based on validation accuracy.  

### Saving Metadata
```python
with open(os.path.join(args.out_dir, "labels.json"), "w") as f:
    json.dump({"model_name": args.model_name, "classes": classes}, f, indent=2)
```

- Stores the model type and class labels so `app.py` knows how to rebuild it.  

---

## ✨ Why This Project Matters

- **Transfer Learning**: Reuses pretrained models for fast, accurate classification.  
- **Flexible Deployment**: Works with your own custom dataset or defaults to ImageNet.  
- **Interactive AI**: Gradio + Hugging Face Spaces makes it simple for anyone to test AI in the browser.  
