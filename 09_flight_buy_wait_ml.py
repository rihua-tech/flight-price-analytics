"""
Flight Price Buy/Wait ML

End-to-end pipeline for the Flight Price Analytics project:

1. Load and clean Travelpayouts-style flight snapshot data
   (e.g., 10 routes x 150+ days of daily API snapshots).
2. Engineer features for lead time, seasonality, and price volatility.
3. Create a Buy/Wait label based on future price drops
   (1 = Wait if a cheaper fare appears within N days, else 0 = Buy).
4. Split train/test by snapshot_date so the model only sees past data.
5. Train and evaluate:
   - Baseline (always Buy)
   - Logistic Regression
   - Random Forest
   using classification reports and ROC AUC.
6. Visualize Random Forest feature importance.

This script is written as a clean, modular portfolio project
for resumes and technical interviews.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# ---------------- CONFIG ---------------- #

# Raw snapshot data (already aggregated to one CSV in this repo)
DATA_PATH = "fares_fact.csv"

# Label rules:
# if within HORIZON_DAYS a cheaper price appears that is at least
# DROP_PCT_THRESHOLD lower than today, label = Wait (1), else Buy (0)
DROP_PCT_THRESHOLD = 0.05  # 5% cheaper
HORIZON_DAYS = 7           # look-ahead horizon in days

# Time-based split: first 80% of snapshot dates for training,
# last 20% for testing
TRAIN_FRACTION = 0.8

# Features used by the models
FEATURE_COLS = [
    "price",
    "pct_change_7d",
    "rolling_std_7d",
    "days_to_departure",
    "dow",
    "month",
    "is_weekend",
]


# --------------- DATA LOADING ------------ #

def load_and_clean(path: str) -> pd.DataFrame:
    """
    Load raw Travelpayouts-style snapshot data and do basic cleaning.
    Expected raw columns (can be adjusted to match your dataset):

    - route          -> renamed to route_id
    - check_date     -> renamed to snapshot_date
    - departure_date -> renamed to depart_date
    - price          -> numeric fare
    """
    df_raw = pd.read_csv(path)

    df = df_raw.rename(
        columns={
            "route": "route_id",
            "check_date": "snapshot_date",
            "departure_date": "depart_date",
        }
    )

    # Parse dates
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["depart_date"] = pd.to_datetime(df["depart_date"])

    # Basic cleaning
    core_cols = ["route_id", "snapshot_date", "depart_date", "price"]
    df = df.dropna(subset=core_cols)
    df = df[df["price"] > 0]

    # Sort by route / departure / snapshot (time series order)
    df = df.sort_values(["route_id", "depart_date", "snapshot_date"]).reset_index(
        drop=True
    )
    return df


# ------------- FEATURE ENGINEERING -------- #

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based and volatility features:

    - days_to_departure: days from snapshot_date to depart_date
    - dow/month: day-of-week, month
    - is_weekend: whether snapshot_date is Sat/Sun
    - rolling_mean_7d, rolling_std_7d: rolling stats by route_id + depart_date
    - pct_change_7d: deviation from rolling mean
    """
    df = df.copy()

    df["days_to_departure"] = (df["depart_date"] - df["snapshot_date"]).dt.days
    df["dow"] = df["snapshot_date"].dt.weekday
    df["month"] = df["snapshot_date"].dt.month
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

    group_cols = ["route_id", "depart_date"]
    df = df.sort_values(group_cols + ["snapshot_date"])

    # Rolling stats within each (route, depart_date) group
    df["rolling_mean_7d"] = df.groupby(group_cols)["price"].transform(
        lambda s: s.rolling(window=7, min_periods=3).mean()
    )
    df["rolling_std_7d"] = df.groupby(group_cols)["price"].transform(
        lambda s: s.rolling(window=7, min_periods=3).std()
    )

    # Deviation from rolling mean（相对均值的偏离）
    df["pct_change_7d"] = (df["price"] - df["rolling_mean_7d"]) / df["rolling_mean_7d"]

    return df


# ------------- LABEL CREATION ------------- #

def _compute_labels_for_group(group: pd.DataFrame) -> pd.DataFrame:
    """
    For a single (route_id, depart_date) group, look forward HORIZON_DAYS
    from each snapshot_date and label:
    - 1 (Wait) if a cheaper price <= current_price * (1 - DROP_PCT_THRESHOLD)
      appears within the horizon
    - 0 (Buy) otherwise
    """
    prices = group["price"].values
    dates = group["snapshot_date"].values
    n = len(group)
    labels = np.zeros(n, dtype=int)

    for i in range(n):
        current_price = prices[i]
        current_date = dates[i]

        horizon_end = current_date + np.timedelta64(HORIZON_DAYS, "D")
        mask = (dates > current_date) & (dates <= horizon_end)

        if mask.any():
            future_min = prices[mask].min()
            if future_min <= current_price * (1 - DROP_PCT_THRESHOLD):
                labels[i] = 1  # Wait
            else:
                labels[i] = 0  # Buy
        else:
            # No future data within horizon -> default Buy
            labels[i] = 0

    group = group.copy()
    group["label_wait"] = labels
    return group


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Buy/Wait labeling per (route_id, depart_date).
    """
    df_labeled = df.groupby(["route_id", "depart_date"], group_keys=False).apply(
        _compute_labels_for_group
    )
    return df_labeled


# ------------- TIME-BASED SPLIT ----------- #

def time_based_split(model_df: pd.DataFrame):
    """
    Split into train/test by snapshot_date so that the model only sees
    'past' snapshots when predicting 'future' ones.
    """
    unique_dates = np.sort(model_df["snapshot_date"].unique())
    split_idx = int(len(unique_dates) * TRAIN_FRACTION)
    train_cutoff = unique_dates[split_idx]

    train_df = model_df[model_df["snapshot_date"] <= train_cutoff]
    test_df = model_df[model_df["snapshot_date"] > train_cutoff]

    return train_df, test_df


# ------------- DATASET STRUCT ------------- #

@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def make_dataset(df: pd.DataFrame) -> Dataset:
    """
    Build scaled train/test arrays for Logistic Regression,
    using a time-based split on snapshot_date.
    """
    model_df = df.dropna(subset=FEATURE_COLS + ["label_wait"]).copy()

    train_df, test_df = time_based_split(model_df)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["label_wait"].values

    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["label_wait"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return Dataset(
        X_train=X_train_scaled,
        y_train=y_train,
        X_test=X_test_scaled,
        y_test=y_test,
        scaler=scaler,
    )


# ------------- MODEL TRAINING ------------- #

def train_log_reg(ds: Dataset) -> LogisticRegression:
    """
    Logistic Regression with balanced classes and L2 regularization.
    """
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=500,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(ds.X_train, ds.y_train)
    return model


def train_random_forest(
    df: pd.DataFrame,
) -> tuple[RandomForestClassifier, np.ndarray, np.ndarray]:
    """
    Random Forest classifier on the same time-based split,
    using unscaled features.
    """
    model_df = df.dropna(subset=FEATURE_COLS + ["label_wait"]).copy()
    train_df, test_df = time_based_split(model_df)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["label_wait"].values

    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["label_wait"].values

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )
    rf.fit(X_train, y_train)
    return rf, X_test, y_test


# ------------- VISUALIZATION -------------- #

def plot_rf_feature_importance(
    model: RandomForestClassifier, feature_names: list[str]
) -> None:
    """
    Plot feature importance for the Random Forest model.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.figure(figsize=(8, 5))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Feature importance")
    plt.title("Random Forest - Feature Importance")
    plt.tight_layout()
    plt.show()


# ------------- MAIN PIPELINE -------------- #

def main() -> None:
    # 1. Load & clean
    print("1) Loading data...")
    df = load_and_clean(DATA_PATH)
    print("Rows after cleaning:", len(df))

    # 2. Feature engineering
    print("2) Engineering features...")
    df = add_features(df)

    # 3. Label creation
    print("3) Creating Buy/Wait labels...")
    df = add_labels(df)
    print("Label distribution (0=Buy, 1=Wait):")
    print(df["label_wait"].value_counts(normalize=True).rename("proportion"))

    # 4. Dataset & baseline + Logistic Regression
    print("\n4) Time-based split & dataset...")
    ds = make_dataset(df)

    print("\n=== Baseline (Always Buy) ===")
    # Baseline: always predict Buy (0)
    y_pred_base = np.zeros_like(ds.y_test)
    print(classification_report(ds.y_test, y_pred_base, target_names=["Buy", "Wait"]))

    print("\n=== Logistic Regression ===")
    log_reg = train_log_reg(ds)
    y_pred_lr = log_reg.predict(ds.X_test)
    y_proba_lr = log_reg.predict_proba(ds.X_test)[:, 1]
    print(classification_report(ds.y_test, y_pred_lr, target_names=["Buy", "Wait"]))
    print("ROC AUC (LogReg):", roc_auc_score(ds.y_test, y_proba_lr))

    # 5. Random Forest + feature importance
    print("\n=== Random Forest ===")
    rf, X_test_rf, y_test_rf = train_random_forest(df)
    y_pred_rf = rf.predict(X_test_rf)
    y_proba_rf = rf.predict_proba(X_test_rf)[:, 1]
    print(classification_report(y_test_rf, y_pred_rf, target_names=["Buy", "Wait"]))
    print("ROC AUC (RandomForest):", roc_auc_score(y_test_rf, y_proba_rf))

    print("\nPlotting Random Forest feature importance...")
    plot_rf_feature_importance(rf, FEATURE_COLS)

    print("\nDone.")


if __name__ == "__main__":
    main()
