"""
Train ML models for the Value-Based Care Command Center.
Fixed version: Sav_rate removed from features to prevent leakage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    mean_absolute_error,
    r2_score,
)
from xgboost import XGBClassifier, XGBRegressor

ML_DIR = Path(__file__).parent
ROOT = ML_DIR.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_engineering import prepare_features, create_target_at_risk, create_target_savings

DATA_DIR = ROOT / "data"
MODEL_DIR = ML_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_all_years() -> pd.DataFrame:
    frames = []
    for path in sorted(DATA_DIR.glob("aco_scorecard_*.csv")):
        try:
            year = int(path.stem.split("_")[-1])
            df = pd.read_csv(path)
            df["performance_year"] = year
            frames.append(df)
            print(f"  Loaded {path.name}: {len(df)} rows")
        except Exception as e:
            print(f"  Skipped {path.name}: {e}")

    if not frames:
        main = DATA_DIR / "aco_scorecard.csv"
        if main.exists():
            df = pd.read_csv(main)
            df["performance_year"] = 2024
            frames.append(df)
            print(f"  Loaded fallback aco_scorecard.csv: {len(df)} rows")

    if not frames:
        raise FileNotFoundError("No scorecard CSV files found in data/")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows after combining: {len(combined)}")
    return combined


def train():
    print("=" * 60)
    print("VBC ML Training Pipeline (Leakage Fixed)")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    df = load_all_years()

    df = df[df["contract_outcome"].isin(["Strong Performer", "On Track", "At Risk"])].copy()
    print(f"  Rows with valid outcome: {len(df)}")

    print("\n[2/5] Preparing features (Sav_rate removed)...")
    X, encoders = prepare_features(df)
    y_risk = create_target_at_risk(df)
    y_savings = create_target_savings(df)

    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Features used: {list(X.columns)}")
    print(f"  At Risk rate: {y_risk.mean():.1%}")

    print("\n[3/5] Splitting data...")
    X_train, X_test, y_risk_train, y_risk_test, y_sav_train, y_sav_test = train_test_split(
        X, y_risk, y_savings, test_size=0.25, random_state=42, stratify=y_risk
    )

    print("\n[4/5] Training At-Risk Classifier (XGBoost)...")
    clf = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_risk_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_risk_test, y_prob)
    acc = accuracy_score(y_risk_test, y_pred)

    print(f"  AUC:      {auc:.3f}")
    print(f"  Accuracy: {acc:.3f}")
    print("\n  Classification Report:")
    print(classification_report(y_risk_test, y_pred, target_names=["Not At Risk", "At Risk"]))

    print("\n[5/5] Training Savings Rate Regressor (XGBoost)...")
    reg = XGBRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
    )
    reg.fit(X_train, y_sav_train)

    y_sav_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_sav_test, y_sav_pred)
    r2 = r2_score(y_sav_test, y_sav_pred)

    print(f"  MAE: {mae:.2f} percentage points")
    print(f"  R²:  {r2:.3f}")

    print("\nSaving models and artifacts...")
    joblib.dump(clf, MODEL_DIR / "at_risk_classifier.joblib")
    joblib.dump(reg, MODEL_DIR / "savings_regressor.joblib")
    joblib.dump(encoders, MODEL_DIR / "encoders.joblib")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_columns.joblib")

    metrics = {
        "classifier": {
            "auc": round(float(auc), 4),
            "accuracy": round(float(acc), 4),
            "at_risk_rate_in_data": round(float(y_risk.mean()), 4),
        },
        "regressor": {
            "mae": round(float(mae), 3),
            "r2": round(float(r2), 4),
        },
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": list(X.columns),
        "note": "Sav_rate removed from features to prevent data leakage",
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModels saved to: {MODEL_DIR}")
    print("  - at_risk_classifier.joblib")
    print("  - savings_regressor.joblib")
    print("  - encoders.joblib")
    print("  - feature_columns.joblib")
    print("  - metrics.json")
    print("\nTraining complete!")


if __name__ == "__main__":
    train()