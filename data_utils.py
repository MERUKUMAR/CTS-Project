"""Shared helpers for loading the ACO scorecard data across all app pages."""

from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"
SCRIPTS_DIR = Path(__file__).parent / "scripts"

YEAR_API_UUID = {
    2024: "d33cf946-28cd-4479-b55e-73024331f4ca",
    2023: "7082a8f1-6d51-4723-853d-086bf254f5fb",
    2022: "a5d74ce2-ba38-47be-8523-146e4ad41832",
    2021: "bd6b766f-6fa3-43ae-8e9a-319da31dc374",
    2020: "8f073013-9db0-4b12-9a34-5802bdabbdfe",
    2019: "9c3a4c69-7d00-4307-9b6f-a080dc90417e",
    2018: "80c86127-8839-4f35-b87b-aa37664afd19",
    2017: "3b306450-1836-417b-b779-7d70fd2fc734",
    2016: "a290fdd3-976a-4fc9-9139-a98193b3af82",
    2015: "156c00e2-ab42-4923-b54f-09c031f5f28d",
    2014: "0ef9b1e2-e23b-4a01-921c-1ac7290c814b",
    2013: "bc90f498-76f4-4e75-8225-8aae30336059",
}

YEAR_CSV_URL = {
    2024: "https://data.cms.gov/sites/default/files/2026-07/fb6ba14b-3450-47c2-8ff5-d1f2a5bdb3e3/PY_Financial_and_Quality_Results_2024_revised%202026_07_17.csv",
    2023: "https://data.cms.gov/sites/default/files/2024-10/7d0067f6-55c1-4121-bcad-a4b7b45defb1/PY%202023%20ACO%20Results%20PUF.csv",
    2022: "https://data.cms.gov/sites/default/files/2024-03/2489bcc5-3a6e-446a-bdd1-11a4cce17137/Performance_Year_Financial_and_Quality_Results_PUF_2022_01_01.csv",
}

SUPPORTED_YEARS = sorted(set(YEAR_API_UUID) | set(YEAR_CSV_URL), reverse=True)
DEFAULT_YEAR = 2022


def _year_path(year: int) -> Path:
    return DATA_DIR / f"aco_scorecard_{year}.csv"


def is_valid_scorecard_file(path: Path) -> bool:
    """Check if a file exists, is not an unresolved Git LFS pointer, and has valid ACO columns."""
    if not path.exists() or not path.is_file():
        return False
    try:
        # LFS pointers are ~130 bytes; valid data files are much larger
        if path.stat().st_size < 500:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline()
                if "git-lfs" in first_line or "oid sha256" in first_line:
                    return False
        # Verify valid scorecard columns
        header = pd.read_csv(path, nrows=1)
        return ("ACO_ID" in header.columns or "ACO_Name" in header.columns or "ACO_Num" in header.columns)
    except Exception:
        return False


def _process_raw_json_fallback(target_year: int = 2022) -> pd.DataFrame | None:
    """If raw_aco_data.json exists, process it via ETL to generate valid CSVs."""
    raw_json_path = DATA_DIR / "raw_aco_data.json"
    if not raw_json_path.exists() or raw_json_path.stat().st_size < 1000:
        return None
    try:
        etl = _load_etl_module()
        with open(raw_json_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        if not records:
            return None
        df = pd.DataFrame(records)
        df = etl.clean_data(df)
        df = etl.enrich_data(df)
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(_year_path(target_year), index=False)
        df.to_csv(DATA_DIR / "aco_scorecard.csv", index=False)
        return df
    except Exception:
        return None


def list_cached_years() -> list[int]:
    years = []
    for p in DATA_DIR.glob("aco_scorecard_*.csv"):
        try:
            year = int(p.stem.split("_")[-1])
            if is_valid_scorecard_file(p):
                years.append(year)
        except ValueError:
            continue
    # If no cached year CSVs exist yet, check if raw_aco_data.json is available
    if not years:
        raw_json_path = DATA_DIR / "raw_aco_data.json"
        if raw_json_path.exists() and raw_json_path.stat().st_size > 1000:
            years.append(2022)
    return sorted(set(years), reverse=True)


def get_selectable_years() -> list[int]:
    return sorted(set(SUPPORTED_YEARS) | set(list_cached_years()), reverse=True)


def _load_etl_module():
    etl_path = SCRIPTS_DIR / "etl.py"
    spec = importlib.util.spec_from_file_location("vbc_etl", etl_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vbc_etl"] = mod
    spec.loader.exec_module(mod)
    return mod


@st.cache_data(show_spinner=False)
def _read_csv_cached(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


def load_scorecard(year: int | None = None) -> pd.DataFrame:
    if year is not None:
        path = _year_path(year)
        if is_valid_scorecard_file(path):
            return _read_csv_cached(str(path))

    # fallback to general single file
    for name in ["aco_scorecard_full.csv", "aco_scorecard.csv"]:
        path = DATA_DIR / name
        if is_valid_scorecard_file(path):
            return _read_csv_cached(str(path))

    # Attempt to process raw_aco_data.json if present
    df = _process_raw_json_fallback(target_year=year if year is not None else 2022)
    if df is not None and not df.empty:
        return df

    st.error("No valid scorecard data found. Select a performance year in the sidebar and click 'Fetch & process'.")
    return pd.DataFrame()


def load_aco_trend(aco_id) -> pd.DataFrame:
    """Load this ACO's key metrics across every cached performance year."""
    rows = []
    for year in list_cached_years():
        path = _year_path(year)
        if not is_valid_scorecard_file(path):
            continue
        try:
            df = _read_csv_cached(str(path))
            if "ACO_ID" not in df.columns:
                continue
            match = df[df["ACO_ID"] == aco_id]
            if match.empty:
                continue
            r = match.iloc[0]
            rows.append({
                "Year": year,
                "Savings Rate (%)": r.get("Sav_rate"),
                "Cost vs Benchmark (%)": r.get("cost_variance_pct"),
                "Quality Score (%)": r.get("QualScore"),
                "Earned Savings/Losses ($)": r.get("EarnSaveLoss"),
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Year")


def fetch_and_process_year(year: int) -> pd.DataFrame:
    etl = _load_etl_module()
    df = None
    errors = []

    uuid = YEAR_API_UUID.get(year)
    if uuid:
        api_url = f"https://data.cms.gov/data-api/v1/dataset/{uuid}/data"
        try:
            records = etl.fetch_from_api(api_url, use_cache=False)
            df = pd.DataFrame(records)
        except Exception as e:
            errors.append(f"API: {e}")

    if df is None or df.empty:
        csv_url = YEAR_CSV_URL.get(year)
        if csv_url:
            try:
                df = pd.read_csv(csv_url, low_memory=False)
            except Exception as e:
                errors.append(f"CSV: {e}")

    if df is None or df.empty:
        raise RuntimeError(f"Could not download PY {year}. " + "; ".join(errors))

    df = etl.clean_data(df)
    df = etl.enrich_data(df)

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(_year_path(year), index=False)
    df.to_csv(DATA_DIR / "aco_scorecard.csv", index=False)  # keep general file updated too
    return df


def render_year_selector() -> int:
    years = get_selectable_years()
    cached = set(list_cached_years())

    if "performance_year" not in st.session_state:
        st.session_state.performance_year = max(cached) if cached else (2022 if 2022 in years else DEFAULT_YEAR)

    with st.sidebar:
        st.markdown("### 📅 Performance Year")
        selected = st.selectbox(
            "Year",
            options=years,
            index=years.index(st.session_state.performance_year)
            if st.session_state.performance_year in years else 0,
            key="year_selector_widget",
        )
        st.session_state.performance_year = selected

        if is_valid_scorecard_file(_year_path(selected)):
            st.success(f"PY {selected} data is ready.")
        else:
            st.warning(f"PY {selected} not cached yet.")
            if st.button(f"⬇️ Fetch & process PY {selected}", type="primary", use_container_width=True):
                with st.spinner(f"Downloading PY {selected}..."):
                    try:
                        df = fetch_and_process_year(selected)
                        st.cache_data.clear()
                        st.success(f"Loaded {len(df)} ACOs for PY {selected}.")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

    return int(st.session_state.performance_year)


def load_national_context() -> dict:
    path = DATA_DIR / "national_context.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


# Helpers
def fmt_dollars(val) -> str:
    if pd.isna(val):
        return "—"
    return f"${val:,.0f}"


def fmt_pct(val, decimals=1) -> str:
    if pd.isna(val):
        return "—"
    return f"{val:.{decimals}f}%"


def fmt_variance(val) -> str:
    if pd.isna(val):
        return "—"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.1f}%"


OUTCOME_COLORS = {
    "Strong Performer": "🟢",
    "On Track": "🟡",
    "At Risk": "🔴",
    "Unknown": "⚪",
}

TIER_COLORS = {
    "Excellent": "🟢",
    "Good": "🟡",
    "Fair": "🟠",
    "Needs Improvement": "🔴",
}