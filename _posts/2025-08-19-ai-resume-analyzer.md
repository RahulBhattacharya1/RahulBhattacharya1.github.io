---
layout: default
title: "Building My AI Resume Analyzer with Gradio + Transformers"
date: 2025-08-25 18:22:00
thumbnail: /assets/images/resume.webp
categories: [ai]
tags: [transformer]
featured: true
---

I recently built Rahul’s AI Resume Analyzer, a tool that compares a resume against a job description and highlights missing keywords while measuring semantic fit. Let me walk you through the code that powers it.

First, I listed dependencies in requirements.txt: Gradio for UI, sentence-transformers for embeddings, scikit-learn for TF-IDF and cosine similarity, PyPDF for PDF parsing, and pandas/numpy/torch for data handling.

<div class="hf-embed" markdown="1">
  <iframe
    src="https://rahulbhattacharya-rahuls-ai-resume-analyzer.hf.space/?__theme=light"
    title="Rahul's AI Resume Analyzer"
    loading="lazy"
    allow="camera; microphone; clipboard-read; clipboard-write; fullscreen; autoplay"
    style="width:100%;height:820px;border:0;border-radius:12px;overflow:hidden">
  </iframe>
  <noscript>
    <p><a href="https://huggingface.co/spaces/RahulBhattacharya/Rahuls_AI_Resume_Analyzer">Open Rahul's AI Resume Analyzer</a></p>
  </noscript>
</div>

In app.py, I began with imports: core libraries like os, re, io, json, and string for utilities; numpy and pandas for data manipulation; and Gradio for the interface. From ML packages, I used TfidfVectorizer for keyword ranking, cosine_similarity for text similarity, and SentenceTransformer for embeddings.

I defined configuration constants like the embedding model (all-MiniLM-L6-v2), number of keywords, and similarity thresholds. Then I wrote text utilities (clean_text, normalize_tokens, split_sentences) to preprocess both resumes and job descriptions. Functions like read_pdf and read_text extract raw text.

The core logic lives in analyze(). It embeds the resume and job description, ranks job-specific terms with TF-IDF, checks which ones are “covered” or “missing,” and suggests bullet points to strengthen the resume.

Finally, I wired it all into a Gradio Blocks UI, so users can upload resumes, paste JDs, and instantly see missing skills, strong matches, and suggestions.
