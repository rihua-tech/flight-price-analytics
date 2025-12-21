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
from sklearn.metrics import classification_report, roc_auc_score, precision_score


# ---------------- CONFIG ---------------- #

DATA_PATH = "fares_fact.csv"

DROP_PCT_THRESHOLD = 0.05  # 5% cheaper
HORIZON_DAYS = 7           # look-ahead horizon in days

TRAIN_FRACTION = 0.8

FEATURE_COLS = [
    "price",
    "pct_change_7d",
    "rolling_std_7d",
    "days_to_departure",
    "dow",
    "month",
    "is_weekend",
]

ALERT_TOP_K = 0.20  # top 20% most confident BUY/WAIT signals


# --------------- DATA LOADING ------------ #

def load_and_clean(path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(path)

    df = df_raw.rename(
        columns={
            "route": "route_id",
            "check_date": "snapshot_date",
            "departure_date": "depart_date",
        }
    )

    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["depart_date"] = pd.to_datetime(df["depart_date"])

    core_cols = ["route_id", "snapshot_date", "depart_date", "price"]
    df = df.dropna(subset=core_cols)
    df = df[df["price"] > 0]

    df = df.sort_values(["route_id", "depart_date", "snapshot_date"]).reset_index(drop=True)
    return df


# ------------- FEATURE ENGINEERING -------- #

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["days_to_departure"] = (df["depart_date"] - df["snapshot_date"]).dt.days
    df["dow"] = df["snapshot_date"].dt.weekday
    df["month"] = df["snapshot_date"].dt.month
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

    group_cols = ["route_id", "depart_date"]
    df = df.sort_values(group_cols + ["snapshot_date"])

    df["rolling_mean_7d"] = df.groupby(group_cols)["price"].transform(
        lambda s: s.rolling(window=7, min_periods=3).mean()
    )
    df["rolling_std_7d"] = df.groupby(group_cols)["price"].transform(
        lambda s: s.rolling(window=7, min_periods=3).std()
    )

    df["pct_change_7d"] = (df["price"] - df["rolling_mean_7d"]) / df["rolling_mean_7d"]
    return df


# ------------- LABEL CREATION ------------- #

def _compute_labels_for_group(group: pd.DataFrame) -> pd.DataFrame:
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
            labels[i] = 1 if future_min <= current_price * (1 - DROP_PCT_THRESHOLD) else 0
        else:
            labels[i] = 0

    group = group.copy()
    group["label_wait"] = labels
    return group


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df_labeled = df.groupby(["route_id", "depart_date"], group_keys=False).apply(_compute_labels_for_group)
    return df_labeled


# ------------- TIME-BASED SPLIT ----------- #

def time_based_split(model_df: pd.DataFrame):
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
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=500,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(ds.X_train, ds.y_train)
    return model


def train_random_forest(df: pd.DataFrame) -> tuple[RandomForestClassifier, np.ndarray, np.ndarray]:
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


# ------------- ALERT REPORTING (TOP-K) ------------- #

def report_topk_alerts(y_true: np.ndarray, y_prob_wait: np.ndarray, k: float = 0.20) -> None:
    """
    Top-K alerting:
    - WAIT alert = top k most confident WAIT predictions (highest P(Wait))
    - BUY  alert = top k most confident BUY predictions  (highest P(Buy)=1-P(Wait))

    Prints:
    - precision: when we alert, how often we are correct
    - coverage: how often we alert (alert volume)
    """
    y_true = np.asarray(y_true)
    y_prob_wait = np.asarray(y_prob_wait)
    y_prob_buy = 1.0 - y_prob_wait

    thr_wait = np.quantile(y_prob_wait, 1 - k)
    thr_buy = np.quantile(y_prob_buy, 1 - k)

    wait_alert = (y_prob_wait >= thr_wait)
    buy_alert = (y_prob_buy >= thr_buy)

    wait_precision = precision_score((y_true == 1).astype(int), wait_alert.astype(int), zero_division=0)
    buy_precision = precision_score((y_true == 0).astype(int), buy_alert.astype(int), zero_division=0)

    wait_coverage = wait_alert.mean()
    buy_coverage = buy_alert.mean()

    print(f"Top-{int(k*100)}% WAIT alerts: precision={wait_precision:.3f}, coverage={wait_coverage:.3f}")
    print(f"Top-{int(k*100)}% BUY  alerts: precision={buy_precision:.3f}, coverage={buy_coverage:.3f}")


# ------------- VISUALIZATION -------------- #

def plot_rf_feature_importance(model: RandomForestClassifier, feature_names: list[str]) -> None:
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
    print("1) Loading data...")
    df = load_and_clean(DATA_PATH)
    print("Rows after cleaning:", len(df))

    print("2) Engineering features...")
    df = add_features(df)

    print("3) Creating Buy/Wait labels...")
    df = add_labels(df)
    print("Label distribution (0=Buy, 1=Wait):")
    print(df["label_wait"].value_counts(normalize=True).rename("proportion"))

    print("\n4) Time-based split & dataset...")
    ds = make_dataset(df)

    print("\n=== Baseline (Always Buy) ===")
    y_pred_base = np.zeros_like(ds.y_test)
    print(classification_report(ds.y_test, y_pred_base, target_names=["Buy", "Wait"], zero_division=0))

    print("\n=== Logistic Regression ===")
    log_reg = train_log_reg(ds)
    y_pred_lr = log_reg.predict(ds.X_test)
    y_proba_lr = log_reg.predict_proba(ds.X_test)[:, 1]
    print(classification_report(ds.y_test, y_pred_lr, target_names=["Buy", "Wait"], zero_division=0))
    print("ROC AUC (LogReg):", roc_auc_score(ds.y_test, y_proba_lr))
    report_topk_alerts(ds.y_test, y_proba_lr, k=ALERT_TOP_K)

    print("\n=== Random Forest ===")
    rf, X_test_rf, y_test_rf = train_random_forest(df)
    y_pred_rf = rf.predict(X_test_rf)
    y_proba_rf = rf.predict_proba(X_test_rf)[:, 1]
    print(classification_report(y_test_rf, y_pred_rf, target_names=["Buy", "Wait"], zero_division=0))
    print("ROC AUC (RandomForest):", roc_auc_score(y_test_rf, y_proba_rf))
    report_topk_alerts(y_test_rf, y_proba_rf, k=ALERT_TOP_K)

    print("\nPlotting Random Forest feature importance...")
    plot_rf_feature_importance(rf, FEATURE_COLS)

    print("\nDone.")


if __name__ == "__main__":
    main()
