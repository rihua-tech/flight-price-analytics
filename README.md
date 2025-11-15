# flight-price-analytics
Flight price analytics with Python, SQL, ML, and Power BI — Buy/Wait guidance and forecasting for a travel-planner app.


# Flight Price Analytics – When Should Users Book?

End-to-end project for a **travel-planner app** that uses Python, SQL, ML and Power BI to:

- clean and model daily flight price snapshots,
- analyze price behavior by route and lead time,
- train a **Buy vs Wait** classifier, and
- deliver dashboards and business recommendations.

---

## 1. Business Problem

Users don’t know **when to book**. Fares move with seasonality, lead time, weekends/holidays, and route volatility.  
Without guidance, users overpay or abandon the app, which hurts:

- **conversion**
- **trust**
- **retention**

**Business question**

> How can we use route-level daily price data to forecast near-term moves and give simple, trustworthy **Buy/Wait guidance** that improves conversion and user trust?

High-level framing:

- `01_Business Problem Statement.pdf`

---

## 2. Tech Stack

- **Languages:** Python, SQL, DAX (Power BI)
- **Python:** pandas, NumPy, matplotlib, scikit-learn
- **ML:** logistic regression, random forest, time-based train/test + cross-validation
- **BI & Reporting:** Power BI, Excel, PDF, PowerPoint

---

## 3. Repository Structure

> Filenames may vary slightly depending on your export – adjust as needed.

### Business & Reporting

- `01_Business Problem Statement.pdf` – context, problem, scope, deliverables.
- `02_Flight_Price_Analytics_Business insight.*` – written report (EDA, SQL, dashboard, recommendations).
- `03_Flight-Price-Analytics_Business insight.pptx` – slide deck version of the report.

### Notebooks & Scripts

- `04_Flight Price Analytics.ipynb` – main **EDA + pricing analysis**.
- `05_Wrangling_EDA.ipynb` – data wrangling and sanity checks.
- `06_Forecast_Backtest.ipynb` – forecasting backtests (baselines + ARIMA/Prophet).
- `07_Hypothesis_Tests.ipynb` – weekend vs weekday, short vs long lead-time tests.
- `09_flight_buy_wait_ml.ipynb` – **Buy/Wait ML notebook** (features, models, ROC curves).
- `10_Flight_Price_Buy_Wait_ML.py` – end-to-end Python script for the Buy/Wait model.

### Dashboards

- `08_Flight Price Analytics.pbix` – Power BI report:
  - Route Overview
  - Forecast vs Actual
  - Lead-Time Curves
  - Alerts & Buy/Wait playbook

### Data (sample / anonymized)

- `fares_fact.csv` – main daily flight price fact table (route, snapshot_date, depart_date, price, etc.).
- `forecast_detail.*`, `forecast_summary.*` – backtest exports.
- `hyp_short_vs_long.*`, `hyp_weekend_vs_weekday.*` – hypothesis-test data.

---

## 4. Analysis & Modeling Workflow

### 4.1 Data Preparation

- Merge raw API/CSV exports into `fares_fact`.
- Clean and dedupe records, drop invalid prices, standardize currency.
- Compute `days_to_departure = depart_date – snapshot_date`.
- Add calendar features: **day of week, month, weekend/holiday flags**.

### 4.2 Forecasting & Backtesting

Notebook `06_Forecast_Backtest.ipynb`:

- Baselines: 7-day moving average, seasonal-naive rules.
- Optional ARIMA/Prophet models.
- Rolling backtests by route and lead-time band.
- Metrics: **MAE / MAPE**, plus simple comparison tables.

### 4.3 Hypothesis Testing

Notebook `07_Hypothesis_Tests.ipynb`:

- Compare **weekend vs weekday** search prices.
- Compare **short vs long lead time** (e.g., ≤14 days vs >14 days).
- Use simple tests to see if differences are meaningful.
- Feed results into messaging / UX ideas.

### 4.4 Buy/Wait ML Model (Python, scikit-learn)

Notebook `09_flight_buy_wait_ml.ipynb` and script `10_Flight_Price_Buy_Wait_ML.py`:

1. **Label definition**

   For each route + departure date and snapshot:

   - Look ahead **7 days**.
   - If minimum future price ≤ current_price × (1 – 5%), label as **Wait (1)**.
   - Otherwise label as **Buy (0)**.

2. **Features**

   - `price`
   - `pct_change_7d` – current price vs 7-day rolling mean
   - `rolling_std_7d` – short-term volatility
   - `days_to_departure`
   - `dow`, `month`, `is_weekend`

3. **Time-based split**

   - Sort by `snapshot_date`.
   - First ~80% of dates → **train**, last 20% → **test** (no leakage).

4. **Models**

   - **Logistic Regression** (scaled features, class_weight="balanced").
   - **Random Forest** (class-balanced subsample, 300 trees).

5. **Evaluation**

   - Classification report (precision, recall, F1 for Buy/Wait).
   - **ROC AUC** for both models.
   - ROC curve comparison vs random guess.

6. **Helper function**

   - `buy_or_wait()` wraps the trained model + scaler.
   - Input: a single snapshot’s features.
   - Output: `"Buy"` / `"Wait"` + probability of **Wait**.
   - Ready to feed into app logic or alerts.
     




## 5. How to Run the Buy/Wait ML Script

1. **Clone the repo**

   ```bash
   git clone https://github.com/rihua-tech/flight-price-analytics.git
   cd flight-price-analytics

2. **Create and activate a virtual env**

    ```bash
   python -m venv .venv

   #### Windows
   .venv\Scripts\activate

   #### macOS / Linux
   source .venv/bin/activate


3. **Install dependencies**

   ```bash
   pip install -r requirements.txt

   pip install pandas numpy scikit-learn matplotlib

4. **Run the script**
 
   ```bash
   python 10_Flight_Price_Buy_Wait_ML.py


   ```
   You should see in the console:
   ```
   label distribution
   
   logistic regression metrics + ROC AUC
   
   random forest metrics + ROC AUC
   
   baseline (“always Buy”) comparison

