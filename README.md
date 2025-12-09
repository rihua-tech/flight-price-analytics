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
- `09_flight_buy_wait_ml.py` – refactored end-to-end Buy/Wait ML pipeline (labeling, feature engineering, models, ROC AUC, feature importance).
- `10_Flight_Price_Buy_Wait_ML.ipynb` – earlier Buy/Wait ML notebook (exploratory version, optional).

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

Script 09_flight_buy_wait_ml.py (refactored pipeline) and notebook 10_Flight_Price_Buy_Wait_ML.ipynb:

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
   - Random Forest feature importance plot (which features drive Buy vs Wait).


## 5. How to Run the Buy/Wait ML Script

Clone the repo

git clone https://github.com/rihua-tech/flight-price-analytics.git
cd flight-price-analytics

Create and activate a virtual env

python -m venv .venv

#### Windows
.venv\Scripts\activate

#### macOS / Linux
source .venv/bin/activate

Install dependencies (minimal)

pip install pandas numpy scikit-learn matplotlib

Run the script

python 09_flight_buy_wait_ml.py

You should see in the console:

- basic label distribution (Buy vs Wait)
- baseline (“always Buy”) classification report
- logistic regression metrics + ROC AUC
- random forest metrics + ROC AUC
- a Random Forest feature importance chart popping up


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

   ```
   Or install a minimal set manually:
   
   ```bash
   pip install pandas numpy scikit-learn matplotlib

4. **Run the script**
 
   ```bash
   python 10_Flight_Price_Buy_Wait_ML.py


   ```
   You should see in the console:
   
   label distribution
   
   logistic regression metrics + ROC AUC
   
   random forest metrics + ROC AUC
   
   baseline (“always Buy”) comparison

