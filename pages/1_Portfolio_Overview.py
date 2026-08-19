import streamlit as st

import pandas as pd
from data_utils import load_scorecard, render_year_selector, OUTCOME_COLORS, TIER_COLORS, load_national_context

try:
    from ml.predict import predict_for_dataframe, models_available, get_model_metrics
    ML_AVAILABLE = models_available()
except Exception:
    ML_AVAILABLE = False

st.set_page_config(page_title="Portfolio Overview", page_icon="📋", layout="wide")
st.title("📋 Portfolio Overview")
st.caption("All ACOs under contract — filter and sort to find where to focus")

year = render_year_selector()
df = load_scorecard(year)
if df.empty:
    st.stop()

if ML_AVAILABLE:
    df = predict_for_dataframe(df)
else:
    st.info(
        "ℹ️ ML models not trained yet. Run `python ml/train_models.py` to enable "
        "Risk Score and Predicted Savings columns.",
        icon="ℹ️",
    )

col1, col2, col3, col4 = st.columns(4)
with col1:
    tracks = st.multiselect("Contract Track", sorted(df["Current_Track"].dropna().unique()))
with col2:
    outcomes = st.multiselect("Outcome", sorted(df["contract_outcome"].dropna().unique()))
with col3:
    drivers = st.multiselect("Top Cost Driver", sorted(df["top_cost_driver"].dropna().unique()))
with col4:
    if ML_AVAILABLE and "risk_level" in df.columns:
        risk_levels = st.multiselect(
            "ML Risk Level",
            options=["High", "Medium", "Low"],
            default=[],
        )
    else:
        risk_levels = []

filtered = df.copy()
if tracks:
    filtered = filtered[filtered["Current_Track"].isin(tracks)]
if outcomes:
    filtered = filtered[filtered["contract_outcome"].isin(outcomes)]
if drivers:
    filtered = filtered[filtered["top_cost_driver"].isin(drivers)]
if risk_levels and "risk_level" in filtered.columns:
    filtered = filtered[filtered["risk_level"].isin(risk_levels)]

st.caption(f"Showing {len(filtered)} of {len(df)} ACOs")

base_cols = [
    "ACO_Name", "Current_Track", "N_AB", "Sav_rate", "QualScore",
    "contract_outcome", "cost_variance_pct", "top_cost_driver",
]
# Some performance years (e.g. 2018-2021) don't have every column CMS
# added in later years. Only keep columns that actually exist this year
# so the page never crashes on a missing one.
base_cols = [c for c in base_cols if c in filtered.columns]

rename_map = {
    "ACO_Name": "ACO",
    "Current_Track": "Track",
    "N_AB": "Beneficiaries",
    "Sav_rate": "Savings Rate (%)",
    "QualScore": "Quality Score (%)",
    "contract_outcome": "Outcome",
    "cost_variance_pct": "Cost vs Benchmark (%)",
    "top_cost_driver": "Top Cost Driver",
}

if ML_AVAILABLE and "risk_score" in filtered.columns:
    base_cols += ["risk_score", "risk_level"]
    rename_map.update({
        "risk_score": "ML Risk Score",
        "risk_level": "ML Risk Level",
    })

display_df = filtered[base_cols].rename(columns=rename_map)
display_df["Outcome"] = display_df["Outcome"].apply(
    lambda x: f"{OUTCOME_COLORS.get(x, '')} {x}"
)

if "ML Risk Level" in display_df.columns:
    risk_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    display_df["ML Risk Level"] = display_df["ML Risk Level"].apply(
        lambda x: f"{risk_colors.get(x, '')} {x}" if pd.notna(x) else x
    )

sort_options = [
    "Savings Rate (%)",
    "Quality Score (%)",
    "Cost vs Benchmark (%)",
    "Beneficiaries",
]
sort_options = [c for c in sort_options if c in display_df.columns]
if "ML Risk Score" in display_df.columns:
    sort_options = ["ML Risk Score"] + sort_options
    
    
sort_col = st.selectbox("Sort by", sort_options, index=0)
ascending = st.toggle("Ascending", value=False)
display_df = display_df.sort_values(sort_col, ascending=ascending)

st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)

st.caption(
    "💡 Tip: sort by 'ML Risk Score' descending to find ACOs most likely to become "
    "At Risk — these should be your priority conversations."
)

if ML_AVAILABLE:
    metrics = get_model_metrics()
    with st.expander("ℹ️ About the ML Risk Model"):
        if metrics:
            st.markdown(
                f"""
                **Model:** XGBoost Classifier + Regressor trained on historical CMS MSSP data.

                **Classifier (At-Risk prediction)**
                - AUC: **{metrics['classifier']['auc']}**
                - Accuracy: **{metrics['classifier']['accuracy']}**

                **Regressor (Predicted Savings)**
                - Mean Absolute Error: **{metrics['regressor']['mae']}** percentage points
                - R²: **{metrics['regressor']['r2']}**

                **How to use**
                - **Risk Score 70–100** → High priority for intervention
                - **Risk Score 40–69** → Medium – monitor closely
                - **Risk Score 0–39** → Low risk
                """
            )
        else:
            st.write("Model metrics not available.")