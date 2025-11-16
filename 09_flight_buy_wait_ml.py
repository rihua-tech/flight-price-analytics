"""
Flight Price Buy/Wait ML

End-to-end script:
- Load Travelpayouts-style flight snapshot data
- Engineer features
- Create Buy/Wait label based on future price drops
- Train Logistic Regression & Random Forest
- Print metrics

Adjust DATA_PATH and column names to match your dataset.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# ---------------- CONFIG ---------------- #

DATA_PATH = "data/fares_fact.csv"
DROP_PCT_THRESHOLD = 0.05
HORIZON_DAYS = 7
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

# --------------- DATA LOADING ------------ #

def load_and_clean(path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(path)

    # Rename if needed
    df = df_raw.rename(columns={
        "route": "route_id",
        "check_date": "snapshot_date",
        "departure_date": "depart_date",
    })

    # Parse dates
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["depart_date"] = pd.to_datetime(df["depart_date"])

    # Basic cleaning
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

    df["rolling_mean_7d"] = (
        df.groupby(group_cols)["price"]
          .transform(lambda s: s.rolling(window=7, min_periods=3).mean())
    )
    df["rolling_std_7d"] = (
        df.groupby(group_cols)["price"]
          .transform(lambda s: s.rolling(window=7, min_periods=3).std())
    )

    df["pct_change_7d"] = (df["price"] - df["rolling_mean_7d"]) / df["rolling_mean_7d"]

    return df


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
            if future_min <= current_price * (1 - DROP_PCT_THRESHOLD):
                labels[i] = 1  # Wait
            else:
                labels[i] = 0  # Buy
        else:
            labels[i] = 0  # Default Buy

    group = group.copy()
    group["label_wait"] = labels
    return group


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = (
        df.groupby(["route_id", "depart_date"], group_keys=False)
          .apply(_compute_labels_for_group)
    )
    return df


# ------------- SPLIT & MODELING ---------- #

@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def make_train_test(df: pd.DataFrame) -> Dataset:
    model_df = df.dropna(subset=FEATURE_COLS + ["label_wait"]).copy()

    unique_dates = np.sort(model_df["snapshot_date"].unique())
    split_idx = int(len(unique_dates) * TRAIN_FRACTION)
    train_cutoff = unique_dates[split_idx]

    train_df = model_df[model_df["snapshot_date"] <= train_cutoff]
    test_df = model_df[model_df["snapshot_date"] > train_cutoff]

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

    unique_dates = np.sort(model_df["snapshot_date"].unique())
    split_idx = int(len(unique_dates) * TRAIN_FRACTION)
    train_cutoff = unique_dates[split_idx]

    train_df = model_df[model_df["snapshot_date"] <= train_cutoff]
    test_df = model_df[model_df["snapshot_date"] > train_cutoff]

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


# ------------- MAIN PIPELINE ------------ #

def main():
    print("Loading data...")
    df = load_and_clean(DATA_PATH)
    print("Rows after cleaning:", len(df))

    print("Engineering features...")
    df = add_features(df)

    print("Creating Buy/Wait labels...")
    df = add_labels(df)
    print("Label distribution:\n", df["label_wait"].value_counts(normalize=True))

     # Baseline (always Buy)
    print("\n=== Baseline (Always Buy) ===")
    y_pred_base = np.zeros_like(y_test_rf)
    print(classification_report(y_test_rf, y_pred_base, target_names=["Buy", "Wait"]))

    # Logistic Regression (scaled features)
    print("\n=== Logistic Regression ===")
    ds = make_train_test(df)
    log_reg = train_log_reg(ds)

    y_pred_lr = log_reg.predict(ds.X_test)
    y_proba_lr = log_reg.predict_proba(ds.X_test)[:, 1]

    print(classification_report(ds.y_test, y_pred_lr, target_names=["Buy", "Wait"]))
    print("ROC AUC (LogReg):", roc_auc_score(ds.y_test, y_proba_lr))

    # Random Forest (unscaled features)
    print("\n=== Random Forest ===")
    rf, X_test_rf, y_test_rf = train_random_forest(df)
    y_pred_rf = rf.predict(X_test_rf)
    y_proba_rf = rf.predict_proba(X_test_rf)[:, 1]

    print(classification_report(y_test_rf, y_pred_rf, target_names=["Buy", "Wait"]))
    print("ROC AUC (RandomForest):", roc_auc_score(y_test_rf, y_proba_rf))

    print("\nDone.")


if __name__ == "__main__":
    main()
