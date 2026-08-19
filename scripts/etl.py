"""
ETL script for the Value-Based Care Command Center.

Works with ANY MSSP Performance Year (2013–2024+).

Two ways to load data:

  1) Local CSV (recommended when you already downloaded the file):
       python scripts/etl.py --csv "path/to/Performance_Year_...2013....csv"

  2) CMS API (needs internet):
       python scripts/etl.py
       # or pass --api URL

Output:
    data/aco_scorecard.parquet
    data/aco_scorecard.csv

Usage examples:
    python scripts/etl.py --csv data/raw_2013.csv
    python scripts/etl.py --csv "PY 2023 ACO Results PUF.csv"
    python scripts/etl.py --api https://data.cms.gov/data-api/v1/dataset/UUID/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Default API = PY 2022 (has Current_Track as a single column)
DEFAULT_API_URL = (
   " https://data.cms.gov/data-api/v1/dataset/156c00e2-ab42-4923-b54f-09c031f5f28d/data"
)

SUPPRESSED_MARKERS = {"*", "-", "", None, "N/A", "n/a", "NA", "^", "~"}

PERCENT_COLS = [
    "Sav_rate", "MinSavPerc", "QualScore", "FinalShareRate",
    "Perc_Dual", "Perc_CovDiag", "Perc_CovEpisode", "Perc_LTI",
]

NUMERIC_COLS = [
    "N_AB", "BnchmkMinExp", "GenSaveLoss", "EarnSaveLoss",
    "UpdatedBnchmk", "HistBnchmk", "ABtotBnchmk", "ABtotExp",
    "Per_Capita_Exp_TOTAL_PY",
    "CapAnn_INP_All", "CapAnn_OPD", "CapAnn_PB", "CapAnn_SNF",
    "CapAnn_HHA", "CapAnn_HSP", "CapAnn_DME", "CapAnn_AmbPay",
    "ADM", "ADM_S_Trm", "P_EDV_Vis", "P_EDV_Vis_HOSP",
    "P_CT_VIS", "P_MRI_VIS", "P_EM_Total", "P_EM_PCP_Vis",
    "P_EM_SP_Vis", "P_Nurse_Vis", "P_SNF_ADM", "SNF_LOS", "SNF_PayperStay",
    "N_PCP", "N_Spec", "N_NP", "N_PA","N_CNS", "N_AB_Year_PY",
    "Measure_479", "Measure_484",
    "QualityID_318", "QualityID_110", "QualityID_226",
    "QualityID_113", "QualityID_112", "QualityID_438", "QualityID_370",
    "QualityID_001_WI", "QualityID_236_WI",
    "CAHPS_1", "CAHPS_2", "CAHPS_3", "CAHPS_4", "CAHPS_5",
    "CAHPS_6", "CAHPS_7", "CAHPS_8", "CAHPS_9", "CAHPS_11",
]


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
def fetch_from_api(api_url: str, use_cache: bool = False) -> list:
    cache_path = DATA_DIR / "raw_aco_data.json"
    if use_cache and cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    print(f"Fetching from {api_url} ...")
    resp = requests.get(api_url, timeout=90)
    resp.raise_for_status()
    records = resp.json()
    print(f"Fetched {len(records)} ACO records")
    with open(cache_path, "w") as f:
        json.dump(records, f)
    return records


def load_csv(path) -> pd.DataFrame:
    path = Path(path)
    print(f"Loading local CSV: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# 2. NORMALIZE COLUMN NAMES (works across 2013–2024)
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {
    "ACO_Num": "ACO_ID",
    "ACO_NUM": "ACO_ID",
    "ACO_NAME": "ACO_Name",
    "ACO_name": "ACO_Name",
    "Start_Date": "Current_Start_Date",
    "Sav_Rate": "Sav_rate",
    "EarnShrSavings": "EarnSaveLoss",
    "Track1": "Current_Track_1",
    "Track2": "Current_Track_2",
    "capann_hsp": "CapAnn_HSP",
    "capann_snf": "CapAnn_SNF",
    "capann_opd": "CapAnn_OPD",
    "capann_pb": "CapAnn_PB",
    "CapAnn_ambpay": "CapAnn_AmbPay",
    "capann_hha": "CapAnn_HHA",
    "capann_dme": "CapAnn_DME",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        if col in COLUMN_ALIASES:
            target = COLUMN_ALIASES[col]
            if target not in df.columns:
                rename[col] = target
    if rename:
        print(f"  Renamed columns: {rename}")
        df = df.rename(columns=rename)
    return df


# ---------------------------------------------------------------------------
# 3. CLEAN + DERIVE MISSING FIELDS
# ---------------------------------------------------------------------------
def _to_float(val):
    if val in SUPPRESSED_MARKERS:
        return np.nan
    if isinstance(val, str):
        s = val.strip()
        if s in SUPPRESSED_MARKERS:
            return np.nan
        s = s.replace("%", "").replace(",", "").replace("$", "").strip()
        if s in ("", "-"):
            return np.nan
        val = s
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def _is_flag_on(val) -> bool:
    if pd.isna(val):
        return False
    s = str(val).strip()
    return s in ("1", "1.0", "True", "true", "Y", "Yes")


def derive_current_track(df: pd.DataFrame) -> pd.Series:
    if "Current_Track" in df.columns:
        return df["Current_Track"].astype(str).str.strip().replace({"nan": "Unknown"})

    def from_row(row):
        for letter in "ABCDE":
            col = f"Current_BASIC_{letter}"
            if col in row.index and _is_flag_on(row.get(col)):
                return letter
        if "Current_ENHANCED" in row.index and _is_flag_on(row.get("Current_ENHANCED")):
            return "EN"
        if "Current_Track_1_Plus" in row.index and _is_flag_on(row.get("Current_Track_1_Plus")):
            return "1+"
        if "Current_Track_1" in row.index and _is_flag_on(row.get("Current_Track_1")):
            return "1"
        if "Current_Track_2" in row.index and _is_flag_on(row.get("Current_Track_2")):
            return "2"
        if "Current_Track_3" in row.index and _is_flag_on(row.get("Current_Track_3")):
            return "3"
        if "Track2" in row.index and _is_flag_on(row.get("Track2")):
            return "2"
        if "Track1" in row.index and _is_flag_on(row.get("Track1")):
            return "1"
        # 2013 file only has Track2 flag; everyone else was Track 1
        if "Track2" in row.index or "Current_Track_2" in row.index:
            return "1"
        return "Unknown"

    print("  Deriving Current_Track from flag columns ...")
    return df.apply(from_row, axis=1)


def derive_risk_model(df: pd.DataFrame) -> pd.Series:
    if "Risk_Model" in df.columns:
        return df["Risk_Model"]
    two_sided = {"2", "3", "1+", "EN", "C", "D", "E"}
    return df["Current_Track"].apply(
        lambda t: "Two-Sided" if str(t) in two_sided else "One-Sided"
    )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    for col in PERCENT_COLS + NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_to_float)

    flag_cols = ["Met_QPS", "SNF_Waiver", "Met_30pctl", "DisAffQual", "Adv_Pay"]
    for col in flag_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Current_Track"] = derive_current_track(df)

    if "Risk_Model" not in df.columns:
        df["Risk_Model"] = derive_risk_model(df)

    # Per_Capita_Exp_TOTAL_PY (missing in 2013)
    if "Per_Capita_Exp_TOTAL_PY" not in df.columns:
        if "ABtotExp" in df.columns and "N_AB" in df.columns:
            print("  Deriving Per_Capita_Exp_TOTAL_PY = ABtotExp / N_AB ...")
            df["Per_Capita_Exp_TOTAL_PY"] = df["ABtotExp"] / df["N_AB"].replace(0, np.nan)
        else:
            df["Per_Capita_Exp_TOTAL_PY"] = np.nan

    # UpdatedBnchmk (missing in 2013)
    if "UpdatedBnchmk" not in df.columns:
        if "ABtotBnchmk" in df.columns and "N_AB" in df.columns:
            print("  Deriving UpdatedBnchmk = ABtotBnchmk / N_AB ...")
            df["UpdatedBnchmk"] = df["ABtotBnchmk"] / df["N_AB"].replace(0, np.nan)
        else:
            df["UpdatedBnchmk"] = np.nan

    # Sav_rate
        # Sav_rate
    if "Sav_rate" not in df.columns:
        if "BnchmkMinExp" in df.columns and "ABtotBnchmk" in df.columns:
            print("  Deriving Sav_rate from BnchmkMinExp / ABtotBnchmk ...")
            df["Sav_rate"] = (
                df["BnchmkMinExp"] / df["ABtotBnchmk"].replace(0, np.nan) * 100
            )
        else:
            df["Sav_rate"] = np.nan
    else:
        # Clean and fix scale
        df["Sav_rate"] = pd.to_numeric(df["Sav_rate"], errors="coerce")
        med = df["Sav_rate"].median()

        if pd.notna(med):
            if abs(med) < 1.5:
                # Values are fractions (e.g. 0.05) → convert to percent
                print("  Scaling Sav_rate from fraction → percent ...")
                df["Sav_rate"] = df["Sav_rate"] * 100
            elif abs(med) > 50:
                # Values are 100× too large (common in 2014 & 2015)
                print("  Sav_rate looks 100x too large — correcting by dividing by 100 ...")
                df["Sav_rate"] = df["Sav_rate"] / 100
    # EarnSaveLoss
    if "EarnSaveLoss" not in df.columns:
        if "EarnShrSavings" in df.columns:
            df["EarnSaveLoss"] = df["EarnShrSavings"].apply(_to_float)
        else:
            df["EarnSaveLoss"] = np.nan

    # QualScore (2013 is 0/1)
    if "QualScore" in df.columns:
        med = df["QualScore"].median()
        if pd.notna(med) and med <= 1.5:
            print("  QualScore looks like 0/1 flag; mapping 1->85, 0->50 ...")
            df["QualScore"] = df["QualScore"].map({1: 85.0, 0: 50.0}).fillna(50.0)

    # Met_QPS
    if "Met_QPS" not in df.columns:
        if "QualScore" in df.columns:
            df["Met_QPS"] = (df["QualScore"] >= 60).astype(float)
        else:
            df["Met_QPS"] = np.nan

    if "CapAnn_OPD" not in df.columns:
        df["CapAnn_OPD"] = np.nan

    if "ACO_ID" not in df.columns and "ACO_Num" in df.columns:
        df["ACO_ID"] = df["ACO_Num"]
    if "ACO_Name" not in df.columns and "ACO_NAME" in df.columns:
        df["ACO_Name"] = df["ACO_NAME"]

    # -----------------------------------------------------------------------
    # Population & Risk Metrics Derivations
    # -----------------------------------------------------------------------
    # Perc_Dual
    if "Perc_Dual" not in df.columns or df["Perc_Dual"].isna().all():
        if "N_AB_Year_Dual_PY" in df.columns and "N_AB_Year_PY" in df.columns:
            df["Perc_Dual"] = (df["N_AB_Year_Dual_PY"] / df["N_AB_Year_PY"].replace(0, np.nan)) * 100
        elif "N_AB_Year_AGED_Dual_PY" in df.columns and "N_AB_Year_PY" in df.columns:
            df["Perc_Dual"] = (df["N_AB_Year_AGED_Dual_PY"] / df["N_AB_Year_PY"].replace(0, np.nan)) * 100
    else:
        df["Perc_Dual"] = pd.to_numeric(df["Perc_Dual"], errors="coerce")
        med = df["Perc_Dual"].median()
        if pd.notna(med) and 0 < med <= 1.0:
            df["Perc_Dual"] = df["Perc_Dual"] * 100

    # Perc_LTI
    if "Perc_LTI" in df.columns:
        df["Perc_LTI"] = pd.to_numeric(df["Perc_LTI"], errors="coerce")
        med_lti = df["Perc_LTI"].median()
        if pd.notna(med_lti) and 0 < med_lti <= 1.0:
            df["Perc_LTI"] = df["Perc_LTI"] * 100

    # Perc_Disability
    if "Perc_Disability" not in df.columns:
        if "N_AB_Year_DIS_PY" in df.columns and "N_AB_Year_PY" in df.columns:
            df["Perc_Disability"] = (df["N_AB_Year_DIS_PY"] / df["N_AB_Year_PY"].replace(0, np.nan)) * 100
        elif "N_Ben_Age_0_64" in df.columns and "N_AB" in df.columns:
            df["Perc_Disability"] = (df["N_Ben_Age_0_64"] / df["N_AB"].replace(0, np.nan)) * 100

    # Perc_ESRD
    if "Perc_ESRD" not in df.columns:
        if "N_AB_Year_ESRD_PY" in df.columns and "N_AB_Year_PY" in df.columns:
            df["Perc_ESRD"] = (df["N_AB_Year_ESRD_PY"] / df["N_AB_Year_PY"].replace(0, np.nan)) * 100

    # Perc_Age_75plus
    if "Perc_Age_75plus" not in df.columns:
        if "N_Ben_Age_75_84" in df.columns and "N_Ben_Age_85plus" in df.columns and "N_AB" in df.columns:
            df["Perc_Age_75plus"] = ((df["N_Ben_Age_75_84"].fillna(0) + df["N_Ben_Age_85plus"].fillna(0)) / df["N_AB"].replace(0, np.nan)) * 100

    # Weighted Average CMS-HCC Risk Score
    risk_cols = [
        ("CMS_HCC_RiskScore_AGND_PY", "N_AB_Year_AGED_NonDual_PY"),
        ("CMS_HCC_RiskScore_AGDU_PY", "N_AB_Year_AGED_Dual_PY"),
        ("CMS_HCC_RiskScore_DIS_PY", "N_AB_Year_DIS_PY"),
        ("CMS_HCC_RiskScore_ESRD_PY", "N_AB_Year_ESRD_PY"),
    ]
    weighted_sum = pd.Series(0.0, index=df.index)
    total_weight = pd.Series(0.0, index=df.index)
    has_any_risk = False
    for r_col, w_col in risk_cols:
        if r_col in df.columns and w_col in df.columns:
            r_val = pd.to_numeric(df[r_col], errors="coerce").fillna(0)
            w_val = pd.to_numeric(df[w_col], errors="coerce").fillna(0)
            weighted_sum += r_val * w_val
            total_weight += w_val
            has_any_risk = True
    if has_any_risk:
        df["Avg_HCC_Risk_Score"] = (weighted_sum / total_weight.replace(0, np.nan)).round(3)

    return df


# ---------------------------------------------------------------------------
# 4. ENRICH
# ---------------------------------------------------------------------------
def _pct_variance(df, actual_col, peer_col, min_peer_value=300):
    """Compute % variance vs peer average, but return NaN when the peer
    average is too small to divide by reliably — a near-zero denominator
    turns a small dollar difference into a meaningless huge percentage."""
    if actual_col not in df.columns or peer_col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    peer = df[peer_col].where(df[peer_col].abs() >= min_peer_value, np.nan)
    return (df[actual_col] - peer) / peer * 100

def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Per_Capita_Exp_TOTAL_PY" in df.columns and "UpdatedBnchmk" in df.columns:
        df["cost_variance_pct"] = (
            (df["Per_Capita_Exp_TOTAL_PY"] - df["UpdatedBnchmk"])
            / df["UpdatedBnchmk"].replace(0, np.nan)
            * 100
        )
        df["over_benchmark_flag"] = df["cost_variance_pct"] > 0
    else:
        df["cost_variance_pct"] = np.nan
        df["over_benchmark_flag"] = False

    peer_group = df.groupby("Current_Track", dropna=False)
    peer_group_size = df.groupby("Current_Track", dropna=False)["Current_Track"].transform("size")
    df["peer_group_size"] = peer_group_size
    peer_map = {
        "Per_Capita_Exp_TOTAL_PY": "peer_avg_per_capita_exp",
        "CapAnn_INP_All": "peer_avg_inpatient_exp",
        "CapAnn_OPD": "peer_avg_outpatient_exp",
        "CapAnn_PB": "peer_avg_physician_exp",
        "CapAnn_SNF": "peer_avg_snf_exp",
        "ADM": "peer_avg_admissions",
        "P_EDV_Vis": "peer_avg_er_visits",
        "P_EDV_Vis_HOSP": "peer_avg_er_to_admit",
        "QualScore": "peer_avg_quality_score",
    }
    for src, peer in peer_map.items():
        if src in df.columns:
            df[peer] = peer_group[src].transform("mean")
        else:
            df[peer] = np.nan

    df["inpatient_variance_pct"] = _pct_variance(df, "CapAnn_INP_All", "peer_avg_inpatient_exp")
    df["outpatient_variance_pct"] = _pct_variance(df, "CapAnn_OPD", "peer_avg_outpatient_exp")
    df["physician_variance_pct"] = _pct_variance(df, "CapAnn_PB", "peer_avg_physician_exp")
    df["snf_variance_pct"] = _pct_variance(df, "CapAnn_SNF", "peer_avg_snf_exp")
    df["er_visit_variance_pct"] = _pct_variance(df, "P_EDV_Vis", "peer_avg_er_visits")
    df["er_to_admit_variance_pct"] = _pct_variance(df, "P_EDV_Vis_HOSP", "peer_avg_er_to_admit")
    df["admission_variance_pct"] = _pct_variance(df, "ADM", "peer_avg_admissions")
    # Suppress variance for tracks with too few peers to be a reliable
    # comparison group (small n makes both the mean and the % swing wildly).
    variance_cols = [
        "inpatient_variance_pct", "outpatient_variance_pct", "physician_variance_pct",
        "snf_variance_pct", "er_visit_variance_pct", "er_to_admit_variance_pct",
        "admission_variance_pct",
    ]
    unreliable = df["peer_group_size"] < 5
    for col in variance_cols:
        df.loc[unreliable, col] = np.nan
        # Cap extreme variance outliers to keep charts and comparisons legible
        df[col] = df[col].clip(lower=-100, upper=200)
        
    if "QualScore" in df.columns:
        df["quality_tier"] = pd.cut(
            df["QualScore"],
            bins=[-1, 60, 75, 85, 101],
            labels=["Needs Improvement", "Fair", "Good", "Excellent"],
        )
    else:
        df["quality_tier"] = "Unknown"

    def outcome_bucket(row):
        sav = row.get("Sav_rate", np.nan)
        met = row.get("Met_QPS", np.nan)
        if pd.isna(sav):
            return "Unknown"
        if sav >= 5 and (met == 1 or pd.isna(met)):
            return "Strong Performer"
        if sav >= 0:
            return "On Track"
        return "At Risk"

    df["contract_outcome"] = df.apply(outcome_bucket, axis=1)

    driver_cols = {
        "Inpatient": "inpatient_variance_pct",
        "Outpatient": "outpatient_variance_pct",
        "Physician/Supplier": "physician_variance_pct",
        "SNF": "snf_variance_pct",
    }

    def top_driver(row):
        vals = {k: row[v] for k, v in driver_cols.items() if pd.notna(row.get(v))}
        if not vals:
            return "Unknown"
        return max(vals, key=vals.get)

    df["top_cost_driver"] = df.apply(top_driver, axis=1)
    # Provider workforce capacity: providers per 1,000 assigned beneficiary-years
    provider_cols = ["N_PCP", "N_Spec", "N_NP", "N_PA", "N_CNS"]
    have_cols = [c for c in provider_cols if c in df.columns]
    if have_cols and "N_AB_Year_PY" in df.columns:
        df["providers_per_1000"] = (
            df[have_cols].sum(axis=1, min_count=1)
            / df["N_AB_Year_PY"].replace(0, np.nan)
            * 1000
        )
    else:
        df["providers_per_1000"] = np.nan
    return df


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------
def run(csv_path=None, api_url=None, use_cache=False):
    if csv_path:
        df = load_csv(csv_path)
    else:
        url = api_url or DEFAULT_API_URL
        records = fetch_from_api(url, use_cache=use_cache)
        df = pd.DataFrame(records)

    df = clean_data(df)
    df = enrich_data(df)

    out_parquet = DATA_DIR / "aco_scorecard.parquet"
    out_csv = DATA_DIR / "aco_scorecard.csv"
    try:
        df.to_parquet(out_parquet, index=False)
    except Exception as e:
        print(f"  (parquet skipped: {e})")
    df.to_csv(out_csv, index=False)

    print(f"\nSaved {len(df)} ACO records to:")
    print(f"  {out_csv}")
    if out_parquet.exists():
        print(f"  {out_parquet}")
    print(f"Columns: {len(df.columns)}")
    print(f"\nTrack distribution:\n{df['Current_Track'].value_counts()}")
    print(f"\nOutcome distribution:\n{df['contract_outcome'].value_counts()}")
    return df


def main():
    parser = argparse.ArgumentParser(description="MSSP ACO ETL (any year 2013-2024+)")
    parser.add_argument("--csv", type=str, default=None, help="Path to local Performance Year CSV")
    parser.add_argument("--api", type=str, default=None, help="CMS data-api URL")
    parser.add_argument("--cache", action="store_true", help="Use cached raw JSON (API mode)")
    args = parser.parse_args()
    run(csv_path=args.csv, api_url=args.api, use_cache=args.cache)


if __name__ == "__main__":
    main()