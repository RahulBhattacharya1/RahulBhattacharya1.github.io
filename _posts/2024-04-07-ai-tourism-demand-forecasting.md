---
layout: default
title: "Creating my AI Tourism Demand Forecasting App"
date: 2024-04-07 12:36:41
categories: [ai]
tags: [python,streamlit,self-trained]
thumbnail: /assets/images/tourism.webp
thumbnail_mobile: /assets/images/tourism_demand_sq.webp
demo_link: https://rahuls-ai-tourism-demand-forecasting.streamlit.app/
github_link: https://github.com/RahulBhattacharya1/ai_tourism_demand_forecasting
featured: true
synopsis: My forecasting app using European tourism data to predict seasonal demand. It lets users select a country to view history and forecasts side by side. Lightweight preprocessing yields insights for capacity planning, staffing, and marketing decisions.
custom_snippet: true
custom_snippet_text: This app forecasts European seasonal tourism demand and travels.
---

There are times when one observes a repeating pattern in the world and feels compelled to design a tool around it. My motivation for this app came from observing travel numbers that rose and fell with seasons yet were rarely anticipated with enough precision. Businesses often guessed demand instead of forecasting it. That left gaps during busy times and waste during slow times. I imagined a solution where data spoke more clearly. Dataset used [here](https://www.kaggle.com/datasets/gpreda/tourism-trips-in-europe).

This project became my way to bring historical tourism data into a forecasting application. I wanted a user to simply pick a country and immediately see both history and forecasts side by side. The project made use of Streamlit for the interface and simple data manipulation steps that carried significant impact. In this blog, I will walk through every file I uploaded, explain each block of code in detail, and show how the pieces form a coherent forecasting app that can run directly in a browser.

## Requirements File

To guarantee reproducibility, I used a requirements.txt file. This file lists the Python libraries the project depends on.

```python
streamlit
pandas
matplotlib
statsmodels
joblib
```

Each line represents one package. Streamlit is central, because it allows me to serve the app as a web interface without building HTML or JavaScript manually. Pandas enables me to clean, filter, and transform the CSV datasets. Matplotlib provides visualization support if I extend graphs later. Statsmodels adds econometric forecasting utilities. Joblib is present for persisting models if I later expand into saving trained pipelines.
## The Core Application (app.py)

The application file is where all the logic resides. I will paste the entire code and then explain section by section.

```python
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Tourism Forecaster", layout="wide")
st.title("Tourism Demand Forecaster")

@st.cache_data
def load_data():
    hist = pd.read_csv("data/history_clean.csv")
    fcst = pd.read_csv("data/forecasts_next_8_quarters.csv")
    # Normalize quarter strings (sorted)
    hist["quarter"] = pd.PeriodIndex(hist["quarter"], freq="Q").astype(str)
    fcst["quarter"] = pd.PeriodIndex(fcst["quarter"], freq="Q").astype(str)
    return hist, fcst

hist, fcst = load_data()
countries = sorted(hist["geo"].unique().tolist())
country = st.selectbox("Country code", countries, index=0)

h = hist[hist["geo"]==country].sort_values("quarter")
f = fcst[fcst["geo"]==country].sort_values("quarter")

col1, col2 = st.columns(2)
with col1:
    st.subheader("History (last 12 rows)")
    st.dataframe(h.tail(12), use_container_width=True)
with col2:
    st.subheader("Forecast (next 8 quarters)")
    st.dataframe(f, use_container_width=True)

st.subheader("History + Forecast Chart")
plot_df = pd.concat([
    h[["quarter","value"]].rename(columns={"value":"Trips"}).assign(type="History"),
    f[["quarter","yhat"]].rename(columns={"yhat":"Trips"}).assign(type="Forecast")
], ignore_index=True)

# Simple chart without extra libs
chart_df = plot_df.pivot_table(index="quarter", columns="type", values="Trips", aggfunc="first")
st.line_chart(chart_df)

st.caption("Slice used: dest=TOTAL, tinfo=TOTAL, purpose=TOTAL, duration=N_GE1")

```

### Imports and Page Setup
The script starts with imports. Streamlit is imported as `st`, and Pandas as `pd`. Then I configure the app page with `st.set_page_config`. This step sets the page title and layout. The layout is set to wide, which allows tables to take more space across the screen. The page title ensures a clear label in the browser tab. Finally, `st.title` prints a visible heading on the dashboard.

### Loading Data with a Helper Function
The `load_data` function is defined with the decorator `@st.cache_data`. This decorator caches results, meaning that the function does not reload the data every time the app reruns. Inside the function, I read the history CSV and the forecast CSV. After loading, I normalize the quarter columns using Pandas PeriodIndex. This ensures quarters are treated consistently across both datasets. Without this step, sorting could misbehave because strings like '2000Q1' and '2000Q10' would not align properly.

### Data Preparation
Once data is loaded, I extract the list of unique countries from the history dataset. I sort them alphabetically and present them in a selectbox widget. This widget lets the user pick a country code. The index is set to zero, so the first country is selected by default.

After the selection, I filter both the history and forecast datasets by the chosen country. Sorting by quarter ensures chronological order is respected. These filtered frames are then ready for display.

### Display Columns and Tables
I split the interface into two columns using `st.columns(2)`. The first column shows the last twelve rows of the historical dataset. This gives the user recent context. The second column shows the next eight forecasted quarters. Both are displayed with `st.dataframe`, which allows scrolling and dynamic formatting inside Streamlit.

### Adding Headings and Messages
After presenting the two dataframes, the script calls `st.subheader` again to create a section titled 'Tourism Demand Forecasting App'. Under this, more content can be added, including plots or text. This ensures the app feels structured, guiding the user from inputs to outputs.

## Data Files

The `data` folder holds the actual CSV files. These are essential for the application because they provide input and output.

### history_clean.csv
This file stores the historical demand data. The structure is simple but powerful. The 'geo' column encodes the country, the 'quarter' column stores the period, and 'value' is the numeric measure of tourism demand.

```python
geo quarter     value
 AT  2000Q1 4222279.0
 AT  2000Q2 5338091.0
 AT  2000Q3 6411998.0
 AT  2000Q4 3609368.0
 AT  2001Q2 4886350.0
 AT  2001Q3 6558044.0
 AT  2001Q4 4025088.0
 AT  2002Q1 3861365.0
 AT  2002Q2 4881184.0
 AT  2002Q3 6147241.0
 AT  2002Q4 4033876.0
 AT  2003Q1 3715066.0
 AT  2003Q2 4638258.0
 AT  2003Q3 5699122.0
 AT  2003Q4 3702653.0
 AT  2004Q1 3152192.0
 AT  2004Q2 4069131.0
 AT  2004Q3 6022621.0
 AT  2004Q4 3305193.0
 AT  2005Q1 3068438.0
```

These rows show how the data spans across multiple years. The consistency in quarters ensures alignment with forecast data. The dataset is cleaned, meaning missing or malformed rows were already handled before being placed here. This file anchors the forecasting logic, because forecasts without past data would be meaningless.

### forecasts_next_8_quarters.csv
This file contains the predicted demand values. The 'quarter' column again encodes periods, the 'yhat' column stores predictions, and 'geo' marks the country.

```python
quarter         yhat geo
 2012Q1 3.558698e+06  AT
 2012Q2 4.574877e+06  AT
 2012Q3 6.385710e+06  AT
 2012Q4 3.689293e+06  AT
 2013Q1 3.610694e+06  AT
 2013Q2 4.626873e+06  AT
 2013Q3 6.437706e+06  AT
 2013Q4 3.741289e+06  AT
 2012Q1 1.778979e+06  BE
 2012Q2 2.847249e+06  BE
 2012Q3 4.375188e+06  BE
 2012Q4 1.773966e+06  BE
 2013Q1 1.804709e+06  BE
 2013Q2 2.872980e+06  BE
 2013Q3 4.400918e+06  BE
 2013Q4 1.799697e+06  BE
 2012Q1 9.987318e+05  BG
 2012Q2 1.715656e+06  BG
 2012Q3 2.328487e+06  BG
 2012Q4 1.700723e+06  BG
```

Each row here matches a future quarter for a given country. The presence of 'yhat' indicates this file came from a time series model, which could have been generated separately and saved. The app does not retrain models in real time. Instead, it consumes these predictions as inputs. That design keeps the app fast and light. Users are not burdened with model training every time they click a country. Instead, they see ready-to-use numbers.

## Breakdown of Functions and Helpers

### The load_data Function
This helper is small in code but very important in purpose. It reads two CSV files and returns them as Pandas dataframes. The decorator `@st.cache_data` ensures efficiency by avoiding repeated reloads. Without this caching, every rerun of the app would re‑read from disk, which is costly and slow. By caching, Streamlit remembers the output unless the underlying file changes. This improves user experience dramatically.

The function also normalizes the 'quarter' strings. By converting them into PeriodIndex objects and then back to string, I enforce consistent sorting behavior. This detail may look minor, but it prevents incorrect orderings. For example, without normalization, a value like '2000Q10' would be placed right after '2000Q1' instead of after '2000Q9'. Such issues break charts and tables. Thus, this function guarantees both speed and correctness.

### The selectbox Widget
The selectbox is how the user chooses a country. It is populated from unique values of the 'geo' column in the history dataset. Sorting the list makes it easier for a user to locate their country. The index parameter defines the default choice, which here is the first country alphabetically. This widget converts static datasets into an interactive experience. Without it, the app would be locked to one country or would require editing code.

### The use of st.columns
Splitting the screen into two columns makes results easier to compare. The left column contains the recent historical data. The right column shows forecasts. Placing them side by side helps the human eye track patterns more naturally. If they were stacked vertically, the context might be lost. This layout choice is not about code length but about communication clarity.

## Understanding the Conditional Logic
There are no explicit if‑else blocks in this script, but conditions exist implicitly in user interaction. When a user selects a country, the filtering step `hist[hist["geo"]==country]` is executed. This acts as a condition because only matching rows are kept. The same happens for the forecast dataframe. Such filtering is fundamental. Without it, the app would dump all countries at once, which is unreadable. By tying conditionals to user inputs, the app remains relevant and focused.

## Streamlit Components and Their Roles

### st.dataframe
This function is central for presenting data interactively. It renders Pandas dataframes in a scrollable grid with sorting and resizing. Unlike printing raw text, it allows users to navigate the data. In this app, I restrict history to the last 12 rows because showing decades at once would overwhelm the screen. Similarly, I show only the next 8 forecast rows, which aligns with the forecasting horizon. These decisions balance completeness with readability.

### st.subheader and st.title
These functions add structure to the interface. The title is placed at the very top, anchoring the identity of the app. Subheaders divide content into smaller chunks. Without these, the app would appear as a long stream of tables, confusing to follow. The act of labeling sections may not change computation, but it changes perception, and perception determines usability.

## Possible Extensions

This application can serve as a foundation. Future work could add:
- **Visualization**: Line charts that overlay history and forecast, making patterns visible at a glance.
- **Downloads**: Buttons to export filtered datasets for external use.
- **Model Updating**: Integrating time series training directly, so new data automatically refreshes predictions.
- **Comparisons**: Allowing users to select multiple countries and compare them side by side.

By keeping the code modular, such extensions can be added without rewriting core logic.

## Column by Column Data Explanation

### history_clean.csv Columns
- **geo**: This field uses two‑letter or short codes to represent each country. It is essential because filtering relies on it. By encoding country codes, the dataset remains compact while preserving uniqueness.
- **quarter**: This represents time in quarters, like 2000Q1 for first quarter of year 2000. Using quarters instead of months reduces noise while still capturing seasonality.
- **value**: This numeric field holds the measure of demand. In tourism, that might be nights spent, arrivals, or expenditure. It is stored as float to allow large counts without truncation.

### forecasts_next_8_quarters.csv Columns
- **quarter**: Same format as history, ensuring alignment. Consistency here is critical because forecasts must map directly onto past quarters.
- **yhat**: This column stores forecasted values. The name yhat is common in forecasting libraries. It represents the predicted y.
- **geo**: The country code ensures forecasts are matched correctly. Without this, merging history and forecast would become impossible.

## Role of Requirements Libraries

- **Streamlit**: Powers the web app framework. Handles UI elements like selectboxes, tables, and headers. Simplifies deployment since it runs in browser with a single command.
- **Pandas**: Backbone of data handling. Reads CSVs, filters rows, sorts quarters, and ensures correct structure.
- **Matplotlib**: Provides visualization capability. Even though not heavily used in the initial app, it was included anticipating chart extensions.
- **Statsmodels**: Enables time series forecasting techniques. Though not invoked inside app.py, it was likely used when generating forecasts saved into CSV.
- **Joblib**: Useful for saving trained models. It ensures that heavy computations are not repeated. Again, while not used in app.py, it reflects preparation for future enhancements.

## Hypothetical User Scenario

Imagine a tourism analyst opening the app. The analyst selects 'AT' as the country code. Immediately, the left column fills with the most recent twelve quarters of arrivals. The right column shows predictions for the next two years. With both tables visible, the analyst notices that demand peaks every third quarter, aligning with summer. Forecast rows project the continuation of this seasonal cycle. With this information, the analyst prepares staffing schedules, resource allocations, and promotional campaigns.

Another scenario is a policymaker exploring multiple countries one after another. By switching codes in the selectbox, they see different historical shapes and forecast slopes. Some countries show steady growth, while others exhibit volatility. These insights could guide marketing investments or infrastructure planning.

## Code Philosophy

The philosophy behind this project is clarity over complexity. Rather than embedding model training inside the app, I separated concerns. Forecasts are precomputed and served as simple CSVs. The app then remains light, responsive, and stable. This separation mirrors real production systems where training and serving often occur on different schedules and environments.

Another philosophy is transparency. By showing raw tables, I emphasize that forecasts are not abstract numbers but grounded outputs from data. This makes the tool trustworthy. Users can trace patterns from past quarters and judge the credibility of predictions themselves.

## Reflection

Working on this project reinforced the balance between technical depth and accessibility. A forecasting model alone might impress data scientists but leave decision makers behind. An app without sound data might engage users but mislead them. By combining both, I achieved a middle path. The technical rigor of time series forecasting pairs with the usability of Streamlit dashboards. That dual focus ensures the project remains useful, not just demonstrative.

The size of this project also reminded me of scalability. Even a few CSV files can carry years of information. By structuring the pipeline cleanly, I can scale to larger datasets or more countries without major refactoring. This forward‑looking approach is part of why I document every detail here. The next iteration can build on this base rather than restart from scratch.

## Visualizations and Extensions

If I were to extend the app with visualizations, I would create a line chart that overlays history and forecast. The x‑axis would represent quarters, while the y‑axis would show demand values. History would be drawn in one color and forecasts in another. This chart would allow immediate recognition of seasonality, trends, and forecast continuity. Streamlit provides simple chart functions, and Matplotlib or Plotly can be integrated for more control.

Another useful visualization would be bar charts per year, which highlight growth or decline annually. Stacked bars could compare multiple countries. These additions would not only beautify the app but also deepen its analytical value. A table speaks in numbers, but a chart speaks in shapes and slopes. Combining both creates a complete narrative.

### Caching Mechanism in Practice
The use of `@st.cache_data` deserves more emphasis. When Streamlit runs, every interaction reruns the script top to bottom. Without caching, every rerun would reload CSVs from disk. That overhead slows down response. By caching, Streamlit remembers that the same file path produced the same dataframe. It stores it in memory. On rerun, the function call is skipped, and stored values are reused. This mechanism turns the app from sluggish to smooth.

### Deployment Steps
To deploy the app, I followed a simple process. First, I pushed the code and data into a GitHub repository. Then I connected the repo to Streamlit Cloud. With a few clicks, the app was live at a public URL. This ease of deployment highlights why I chose Streamlit. There was no need for manual server configuration. Streamlit Cloud automatically read requirements.txt, built an environment, and launched the app.

Deployment also involves considering data placement. I placed CSV files in a dedicated `data` folder. This makes paths predictable and avoids clutter. Anyone cloning the repo can find data immediately. By maintaining clear structure, collaboration and maintenance become smoother.

### Thoughts
At the end, this project demonstrates more than forecasting. It demonstrates design decisions, usability concerns, and deployment discipline. A working app is the combination of many invisible choices: caching strategies, file structures, dependency management, and interaction flows. Explaining each piece, as I have done here, makes those invisible choices visible. That transparency ensures the project can be maintained, extended, and trusted.

## Reflection and Conclusion

The complete app may appear small in lines of code, but it carries significant weight. The helper function for loading data ensures speed and efficiency. The selectbox widget enables interactivity. The split column layout organizes results in a digestible way. By combining these elements, the app presents forecasts in a clear, structured format.

Looking back, the decision to normalize quarters proved essential. Without it, the chronological order of periods would have broken the sequence. Another key element was caching data. In a dashboard, speed matters. By caching, repeated runs become instant, which improves usability.

If I wanted to extend this project, I could integrate charts that plot historical and forecast values together. That would make patterns more visual. Another option is to allow downloads of filtered subsets so a tourism analyst could export data for a report. The modular design of this script makes such extensions simple.

This project taught me the importance of bridging technical models with human‑friendly interfaces. A model on its own can remain locked away in a notebook. But once paired with an interactive app, it becomes accessible. That transformation, from hidden algorithm to practical tool, is what excites me about building projects like this.
