"""
Prediction helper for the VBC dashboard.
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import sys

import joblib
import numpy as np
import pandas as pd

ML_DIR = Path(__file__).parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from feature_engineering import prepare_features

MODEL_DIR = Path(__file__).parent / "models"


@lru_cache(maxsize=1)
def _load_artifacts():
    clf_path = MODEL_DIR / "at_risk_classifier.joblib"
    reg_path = MODEL_DIR / "savings_regressor.joblib"
    enc_path = MODEL_DIR / "encoders.joblib"
    feat_path = MODEL_DIR / "feature_columns.joblib"

    if not clf_path.exists() or not reg_path.exists():
        return None

    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)
    encoders = joblib.load(enc_path) if enc_path.exists() else {}
    feature_cols = joblib.load(feat_path) if feat_path.exists() else None

    return {
        "classifier": clf,
        "regressor": reg,
        "encoders": encoders,
        "feature_cols": feature_cols,
    }


def models_available() -> bool:
    return (MODEL_DIR / "at_risk_classifier.joblib").exists()


def _savings_trajectory(predicted_savings) -> str:
    """
    Bucket the regressor's raw prediction into a directional trajectory
    instead of presenting it as a precise point estimate.

    R^2 is 0.28 and MAE is ~2.8pp (see ml/models/metrics.json) — not
    accurate enough to justify a single decimal number.
    """
    if pd.isna(predicted_savings):
        return "Unknown"
    if predicted_savings >= 5:
        return "High"
    if predicted_savings >= 0:
        return "Moderate"
    return "Low"


def predict_for_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    artifacts = _load_artifacts()
    if artifacts is None:
        out = df.copy()
        out["risk_probability"] = np.nan
        out["risk_score"] = np.nan
        out["risk_level"] = "Unknown"
        out["predicted_savings"] = np.nan
        out["savings_trajectory"] = "Unknown"
        return out

    X, _ = prepare_features(df, fit_encoders=artifacts["encoders"])

    if artifacts["feature_cols"] is not None:
        for col in artifacts["feature_cols"]:
            if col not in X.columns:
                X[col] = 0
        X = X[artifacts["feature_cols"]]

    risk_prob = artifacts["classifier"].predict_proba(X)[:, 1]
    pred_savings = artifacts["regressor"].predict(X)

    out = df.copy()
    out["risk_probability"] = risk_prob
    out["risk_score"] = (risk_prob * 100).round(1)
    out["predicted_savings"] = pred_savings.round(2)

    def _level(score):
        if pd.isna(score):
            return "Unknown"
        if score >= 70:
            return "High"
        if score >= 40:
            return "Medium"
        return "Low"

    out["risk_level"] = out["risk_score"].apply(_level)
    out["savings_trajectory"] = out["predicted_savings"].apply(_savings_trajectory)
    return out


def predict_for_row(row: pd.Series) -> dict:
    df = pd.DataFrame([row])
    result = predict_for_dataframe(df).iloc[0]

    return {
        "risk_probability": float(result["risk_probability"]) if pd.notna(result["risk_probability"]) else None,
        "risk_score": float(result["risk_score"]) if pd.notna(result["risk_score"]) else None,
        "risk_level": result["risk_level"],
        "predicted_savings": float(result["predicted_savings"]) if pd.notna(result["predicted_savings"]) else None,
        "savings_trajectory": result["savings_trajectory"],
    }


def get_model_metrics() -> dict | None:
    metrics_path = MODEL_DIR / "metrics.json"
    if not metrics_path.exists():
        return None
    import json
    with open(metrics_path) as f:
        return json.load(f)


def get_shap_explanation(row: pd.Series, top_n: int = 8) -> dict | None:
    """
    Return SHAP values for a single ACO so we can explain the risk score.
    """
    artifacts = _load_artifacts()
    if artifacts is None:
        return None

    df = pd.DataFrame([row])
    X, _ = prepare_features(df, fit_encoders=artifacts["encoders"])

    if artifacts["feature_cols"] is not None:
        for col in artifacts["feature_cols"]:
            if col not in X.columns:
                X[col] = 0
        X = X[artifacts["feature_cols"]]

    try:
        import shap
        explainer = shap.TreeExplainer(artifacts["classifier"])
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]

        feature_names = list(X.columns)
        pairs = list(zip(feature_names, shap_vals))
        pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:top_n]

        return {
            "features": [p[0] for p in pairs],
            "values": [float(p[1]) for p in pairs],
            "base_value": float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value),
        }
    except Exception as e:
        print(f"SHAP error: {e}")
        return None