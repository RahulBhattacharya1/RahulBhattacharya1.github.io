---
layout: default
title: "My AI Text to Slide Generator"
date: 2021-09-06 14:12:43
categories: [ai]
tags: [ai,art,stable diffusion]
thumbnail: /assets/images/ai_text_to_slides.webp
---

This project turns free-form text into a branded slide deck using **Streamlit**, **OpenAI**, or an offline rule-based generator, and converts the result into a downloadable PowerPoint using `python-pptx`.

## What this app does

- Accepts a topic/paragraph and a brand color.  
- Generates a deck JSON (with titles and bullets) using either OpenAI or an offline rule-based generator.  
- Scores the result for basic quality.  
- Converts the JSON into a downloadable PowerPoint (.pptx).

## File 1 — app.py

### Imports and configuration

The top imports bring in `streamlit`, typing helpers, `json`, `time`, and utility modules. The app later calls into `utils_pptx.py` to build the PowerPoint.

### 1) Call-rate limiting

```python
COOLDOWN_SECONDS = 8
DAILY_LIMIT = 60
HOURLY_SHARED_CAP = 200

def init_rate_limit_state():
    ...

def can_call_now() -> Tuple[bool, str, int]:
    ...

def record_successful_call():
    ...

def _shared_hourly_counters():
    ...

def _hour_bucket(ts=None):
    ...
```

Ensures API usage is controlled and user-friendly.

### 2) Content shape constraints

```python
MAX_TITLE_WORDS = 12
MIN_CONTENT_SLIDES = 4
MAX_CONTENT_SLIDES = 10
MIN_BULLETS = 3
MAX_BULLETS = 6
MAX_BULLET_WORDS = 16
```

Defines structural rules for slide content.

### 3) Data models

```python
class SlideModel(TypedDict):
    title: str
    bullets: List[str]

class DeckJSON(TypedDict):
    title: str
    subtitle: str
    slides: List[SlideModel]

class Slide(NamedTuple):
    title: str
    bullets: List[str]

class Deck(NamedTuple):
    title: str
    subtitle: str
    slides: List[Slide]
```

Schema for JSON interchange and internal deck representation.

### 4) OpenAI generation

```python
def call_openai(topic: str, model: str, temperature: float, max_tokens: int) -> str:
    # Crafts system prompt and calls OpenAI API
    return response_text
```

Generates a JSON deck from OpenAI models.

### 5) Text utilities

```python
def naive_sentence_split(text: str) -> List[str]:
    ...
def extract_keywords(text: str, k: int = 6) -> List[str]:
    ...
def chunk_list(xs: List[str], n: int) -> List[List[str]]:
    ...
```

Support functions for offline deck generation.

### 6) Offline deck generator

```python
def generate_offline_deck_json(topic: str, brand_hex: str) -> DeckJSON:
    ...
```

Produces a structured deck without external APIs.

### 7) JSON hardening

```python
def coerce_json_block(s: str) -> str:
    ...
```

Extracts and sanitizes valid JSON from model outputs.

### 8) Deck scoring

```python
def score_deck(dj: DeckJSON) -> Tuple[float, dict]:
    ...
```

Computes quality heuristics for slide content.

### 9) Streamlit UI

```python
st.set_page_config(page_title="AI Text → Slides", page_icon="📑", layout="wide")

provider = st.selectbox("Provider", ["OpenAI", "Offline (rule-based)"])
model = st.selectbox("Model (OpenAI)", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])
brand = st.color_picker("Brand color", value="#0F62FE")
temp = st.slider("Creativity (OpenAI)", 0.0, 1.0, 0.4, 0.05)
max_tokens = st.slider("Max tokens (OpenAI)", 256, 2048, 900, 32)

topic = st.text_area("Enter topic or paragraph", height=160)

if st.button("Generate Slides"):
    ...
```

Controls UI, model selection, and output display.

## File 2 — utils_pptx.py

### 1) Color helpers

```python
def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    ...
def apply_brand_to_title(shape, rgb):
    ...
```

Handles brand coloring.

### 2) Slide builders

```python
def add_title_slide(prs, title, subtitle, rgb):
    ...
def add_content_slide(prs, title, bullets):
    ...
```

Adds title and content slides.

### 3) Deck to PPTX

```python
def deck_to_pptx_bytes(deck: Deck, brand_hex: str) -> bytes:
    ...
```

Assembles and exports deck to `.pptx` bytes.

## End-to-end flow

1. User enters text and color.  
2. Deck is generated (OpenAI or offline).  
3. JSON is validated and scored.  
4. Deck is converted into `.pptx`.  
5. User downloads slides.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

For OpenAI mode:

```bash
export OPENAI_API_KEY="sk-..."
```

Offline mode works without internet.
