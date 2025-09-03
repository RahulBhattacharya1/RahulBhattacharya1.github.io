---
layout: default
title: "The Future of Art Is Here — And I Built It with AI"
date: 2025-08-25 18:22:33
categories: [ai]
tags: [chat,bot,nlp]
thumbnail: /assets/images/ai_conversational.webp
featured: true
---

I recently created a Hugging Face Space called **Rahul’s AI Art Generator**, which lets anyone turn simple text prompts into beautiful AI-generated images. It’s built with **Gradio** for the user interface and **Stable Diffusion** for the image generation model. Let me walk you through how it works and what’s happening behind the scenes.

---

<iframe
	src="https://rahulbhattacharya-rahuls-ai-art-generator.hf.space"
style="width:100%;height:820px;border:0;border-radius:12px;overflow:hidden"></iframe>

## 🚀 The Core Idea

The app uses [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1), a powerful open-source text-to-image model. You type a description—like *“sunset over Chicago skyline in watercolor”*—and the AI paints it for you. Hugging Face Spaces makes it easy to share such apps with the world, and Gradio provides the interactive web interface.

---

## 🛠️ The Code Explained

Here’s the breakdown of my `app.py`:

```python
import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

MODEL_ID = "stabilityai/stable-diffusion-2-1"

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")
```

- **`StableDiffusionPipeline`** loads the model from Hugging Face.  
- The model runs on GPU (`cuda`) if available, otherwise CPU.  
- Using `torch.float16` helps speed up inference on supported GPUs.

---

### The Generate Function

```python
def generate(prompt, guidance_scale, steps, seed):
    generator = torch.Generator(device=pipe.device).manual_seed(int(seed)) if seed is not None else None
    image = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=steps, generator=generator).images[0]
    return image
```

- **`prompt`**: Your text description.  
- **`guidance_scale`**: Controls how closely the image follows the prompt (higher = more literal).  
- **`steps`**: Number of refinement steps during generation (more steps = better quality, slower).  
- **`seed`**: Optional number to reproduce the same image output.  

The function returns the generated image.

---

### Building the Interface with Gradio

```python
with gr.Blocks(title="Rahul's AI Art Generator") as demo:
    gr.Markdown("# 🎨 Rahul's AI Art Generator\nType a prompt and generate an image.")
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="Prompt", value="sunset over Chicago skyline in watercolor")
            guidance = gr.Slider(1.0, 15.0, value=7.5, label="Guidance scale")
            steps = gr.Slider(10, 50, value=30, step=1, label="Steps")
            seed = gr.Number(label="Seed (optional)", value=42, precision=0)
            btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=3):
            out = gr.Image(label="Result", height=512)

    btn.click(generate, [prompt, guidance, steps, seed], out)
```

- I used **Gradio Blocks** to design a clean UI.  
- On the left side, users enter a prompt, tweak sliders for guidance/steps, and set a seed.  
- On the right side, the generated image is displayed at **512px** height.  
- When the **Generate** button is clicked, it calls the `generate` function.

Finally:

```python
if __name__ == "__main__":
    demo.launch()
```

This launches the Gradio app inside Hugging Face Space.

---

## ✨ Why This Matters

- **Accessibility**: Anyone, without coding skills, can now play with AI image generation.  
- **Customization**: Sliders for guidance, steps, and seed give users control over the results.  
- **Reproducibility**: Using seeds ensures the same input always produces the same artwork.  

---
