"""
Shared feature engineering for VBC ML models.
Final fixed version: removed Sav_rate AND cost_variance_pct to prevent leakage.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

# Removed Sav_rate and cost_variance_pct to avoid leakage
NUMERIC_FEATURES = [
    "QualScore",
    "inpatient_variance_pct",
    "outpatient_variance_pct",
    "physician_variance_pct",
    "snf_variance_pct",
    "er_visit_variance_pct",
    "er_to_admit_variance_pct",
    "admission_variance_pct",
    "providers_per_1000",
    "N_AB",
]

OPTIONAL_FEATURES = [
    "Perc_Dual",
    "Perc_LTI",
]

CATEGORICAL_FEATURES = [
    "Current_Track",
]


def prepare_features(df: pd.DataFrame, fit_encoders: dict | None = None) -> tuple[pd.DataFrame, dict]:
    from sklearn.preprocessing import LabelEncoder

    data = df.copy()
    encoders = fit_encoders or {}

    for col in NUMERIC_FEATURES + OPTIONAL_FEATURES:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val if pd.notna(median_val) else 0)
        else:
            data[col] = 0.0

    if "Current_Track" in data.columns:
        if "Current_Track" not in encoders:
            le = LabelEncoder()
            data["Current_Track_encoded"] = le.fit_transform(
                data["Current_Track"].astype(str).fillna("Unknown")
            )
            encoders["Current_Track"] = le
        else:
            le = encoders["Current_Track"]
            known = set(le.classes_)
            data["Current_Track_encoded"] = data["Current_Track"].astype(str).fillna("Unknown").apply(
                lambda x: le.transform([x])[0] if x in known else -1
            )
    else:
        data["Current_Track_encoded"] = 0

    feature_cols = NUMERIC_FEATURES + OPTIONAL_FEATURES + ["Current_Track_encoded"]
    feature_cols = [c for c in feature_cols if c in data.columns]

    X = data[feature_cols].copy()
    X = X.fillna(0)

    return X, encoders


def create_target_at_risk(df: pd.DataFrame) -> pd.Series:
    return (df["contract_outcome"] == "At Risk").astype(int)


def create_target_savings(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["Sav_rate"], errors="coerce").fillna(0)