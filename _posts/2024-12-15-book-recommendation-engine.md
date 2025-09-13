---
layout: default
title: "Creating my AI Book Recommendation Engine"
date: 2024-12-15 09:46:21
categories: [ai]
tags: [python,streamlit,openai]
thumbnail: /assets/images/tictactoe.webp
demo_link: https://rahuls-ai-book-recommendation-engine.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_book_recommendation_engine
featured: true
---

It began with a simple thought. I had been reading books in different genres, but I often felt lost when I wanted to choose my next read. The world of books is wide and deep, and I wanted a guide that could help me find titles matching my taste. That was the moment when I realized I needed a personal recommendation engine. I decided to build one using Python, data, and simple machine learning logic. The result is a working AI Book Recommendation Engine that I deployed as a web application with Streamlit. This post explains every file, every function, and every helper used in the project.

## Files in the Project

The project has four main files and folders that I uploaded to GitHub:

- `README.md`: This is the project description file. It contains instructions about what the app does and how to run it.
- `app.py`: This is the main Python file. It runs the Streamlit web application and contains the logic for recommendations.
- `requirements.txt`: This file lists the dependencies required to run the app. Streamlit and Pandas are the core requirements here.
- `data/books_catalog.csv`: This is the dataset file that contains a list of books with their metadata. It acts as the foundation for recommendations.

Each of these files plays an important role. Without the dataset, the engine would have nothing to recommend. Without the application file, there would be no way for a user to interact. Without requirements, the environment setup would fail. The README ties everything together by guiding the user.

---

## The Main Application File: `app.py`

The `app.py` file is the centerpiece of the project. It contains the full code that powers the web app.

```python
import streamlit as st
import pandas as pd
from pathlib import Path
```

This first block imports the required libraries. `streamlit` creates the user interface, `pandas` helps with data handling, and `pathlib.Path` is used to manage file paths in a clean and portable way. Without these imports, none of the later logic would work.

---

### Setting Up the Streamlit Page

```python
st.set_page_config(page_title="AI Book Recommendation Engine", layout="wide")
st.title("AI Book Recommendation Engine")
```

This block sets the title of the application and defines its layout. The `set_page_config` function prepares the page with a wide layout so that tables and lists look better. The `title` function adds a visible heading at the top of the app. These small steps give structure and clarity to the user interface.

---

### Loading the Dataset

```python
DATA_PATH = Path("data/books_catalog.csv")

@st.cache_data(show_spinner=True)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

books_df = load_data(DATA_PATH)
```

This block loads the dataset from the CSV file. The dataset contains book details such as title, author, and genre. The `@st.cache_data` decorator is important because it prevents reloading the file again and again, which saves time when a user interacts with the app. The helper function `load_data` reads the CSV into a Pandas DataFrame, and this DataFrame is then stored in `books_df`. Without this helper, the data would not be ready for analysis or recommendations.

---

### Displaying the Dataset

```python
st.subheader("Books Catalog")
st.dataframe(books_df.head(20))
```

This block displays the first 20 rows of the dataset as a preview. The `subheader` gives a clear label, while the `dataframe` function shows the actual table inside the app. This helps users confirm what data is available before using the recommendation features.

---

### User Input: Selecting a Genre

```python
genres = books_df["genre"].unique().tolist()
selected_genre = st.selectbox("Choose a Genre", genres)
```

Here the app extracts all unique genres from the dataset. Then it asks the user to select one from a dropdown menu. The `selectbox` widget is a simple way to take structured input. This step allows the user to guide the recommendation engine based on their interest.

---

### Filtering Books by Genre

```python
filtered_books = books_df[books_df["genre"] == selected_genre]
```

This line applies a filter on the DataFrame. Only books that belong to the selected genre are kept. This makes sure that the recommendations remain relevant to the user’s choice.

---

### Displaying Recommendations

```python
st.subheader("Recommended Books")
st.write(filtered_books[["title", "author"]].head(10))
```

This block shows the final recommendations. It takes the filtered DataFrame and selects the first 10 books, showing only their title and author. This keeps the output concise and readable. Without this display step, the user would not see the result of their interaction.

---

## The Dataset File: `data/books_catalog.csv`

The dataset is stored in a CSV file inside the `data` folder. It contains multiple rows of books with fields like:

- `title`: Name of the book
- `author`: Author of the book
- `genre`: Category or type of the book

The dataset is the foundation of the app. It holds the raw material for every recommendation shown to the user. If the dataset is updated with new titles, the recommendation engine becomes richer.

---

## The Requirements File: `requirements.txt`

The `requirements.txt` file lists the Python packages that need to be installed. For this project it looks like this:

```python
streamlit
pandas
```

Streamlit provides the web app framework, while Pandas manages the data. Listing them here ensures that anyone who wants to run the app can install the same versions without conflict. Without this file, the app might fail to run in a new environment.

---

## The README File: `README.md`

The `README.md` file describes the project. It explains what the app does, how to install dependencies, and how to run the project. This file is important for guiding users who open the GitHub repository. Without documentation, others would struggle to understand the purpose or usage of the project.

---

## Conclusion

This project taught me how even a simple dataset can become powerful when paired with the right tools. I learned how to connect Pandas for data, Streamlit for user interface, and caching for efficiency. More than the code, it was about building a smooth flow: load the data, show the data, let the user choose, and then return recommendations. This flow is what makes the project functional and enjoyable.

