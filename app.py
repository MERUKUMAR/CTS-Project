import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import duckdb

from pathlib import Path
from datetime import date

from data_utils import (
    load_scorecard,
    render_year_selector,
    OUTCOME_COLORS,
    TIER_COLORS,
    load_national_context,
    load_aco_trend,
    fmt_dollars,
    fmt_pct,
    fmt_variance,
    DEFAULT_YEAR,
    list_cached_years,
)

# ============================================================
# PATH CONFIGURATION FOR GEOPROVIDER PARQUET DATA
# ============================================================

BASE_DIR = Path(__file__).parent
GEO_DATA_DIR = BASE_DIR / "GeoProviderService"
PROVIDER_PARQUET = (GEO_DATA_DIR / "Medicare_By_Provider_Cleaned.parquet").resolve().as_posix()
PROVIDER_SERVICE_PARQUET = (GEO_DATA_DIR / "Medicare_By_Provider_and_Service_Cleaned.parquet").resolve().as_posix()
GEO_PARQUET = (GEO_DATA_DIR / "Medicare_Geo_Service_Cleaned.parquet").resolve().as_posix()

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VBC Command Center & Provider Analytics",
    page_icon="🏥",
    layout="wide",
)

# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
    /* Hide default Streamlit multi-page sidebar navigation */
    div[data-testid="stSidebarNav"] {
        display: none;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    section[data-testid="stSidebar"] {
        min-width: 260px;
        max-width: 260px;
    }

    .vbc-section {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-top: 12px;
        margin-bottom: 6px;
    }

    .vbc-divider {
        border-top: 1px solid rgba(148, 163, 184, 0.25);
        margin: 12px 0;
    }

    /* ---- Header banner ---- */
    .dashboard-header {
        padding: 1.6rem 2rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #0f2545 0%, #1f4e8c 55%, #2b7bd1 100%);
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .dashboard-header h1 {
        font-size: 1.9rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #ffffff;
    }
    .dashboard-header p {
        font-size: 0.95rem;
        color: #cfe0f7;
        opacity: 0.95;
        margin-bottom: 0;
    }

    /* ---- Metric cards ---- */
    div[data-testid="stMetric"] {
        background: linear-gradient(160deg, #1a2740 0%, #131d30 100%);
        border: 1px solid #2a3a58;
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #9fb0c9 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700;
    }

    /* ---- Tabs ---- */
    div[data-baseweb="tab-list"] {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 8px !important;
        row-gap: 10px !important;
        background-color: #131d30;
        border: 1px solid #2a3a58;
        border-radius: 12px;
        padding: 10px;
    }
    button[data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.86rem;
        padding: 10px 16px !important;
        margin: 0 !important;
        color: #b6c2d6;
        white-space: nowrap;
        border-radius: 8px !important;
        flex: 0 0 auto;
    }
    button[aria-selected="true"] {
        background-color: #1f2f4a !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(43, 123, 209, 0.25);
    }
    div[data-baseweb="tab-highlight"],
    div[data-baseweb="tab-border"] {
        display: none !important;
        height: 0 !important;
        background: transparent !important;
    }

    /* ---- Section card wrapper ---- */
    .section-card {
        background: linear-gradient(160deg, #1a2740 0%, #131d30 100%);
        border-radius: 16px;
        padding: 1.4rem 1.6rem 0.6rem 1.6rem;
        border: 1px solid #2a3a58;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.15rem;
    }
    .section-sub {
        font-size: 0.85rem;
        color: #9fb0c9;
        margin-bottom: 0.8rem;
    }

    /* ---- Expander ---- */
    div[data-testid="stExpander"] {
        background-color: #131d30;
        border: 1px solid #2a3a58;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CAHPS RADAR CHART
# ============================================================

def create_cahps_radar(df, selected_aco_id):
    """Build CAHPS radar chart only if CAHPS columns exist."""

    # All possible CAHPS columns CMS has used over the years
    cahps_candidates = [
        "CAHPS_1",
        "CAHPS_2",
        "CAHPS_3",
        "CAHPS_4",
        "CAHPS_5",
        "CAHPS_6",
        "CAHPS_7",
        "CAHPS_8",
        "CAHPS_9",
        "CAHPS_11",
    ]

    # Keep only columns that actually exist
    cahps_cols = [
        c for c in cahps_candidates
        if c in df.columns
    ]

    # No CAHPS columns in this year's dataset
    if not cahps_cols:
        return None

    # Filter selected ACO
    temp_df = df[
        df["ACO_ID"] == selected_aco_id
    ].copy()

    if temp_df.empty:
        return None

    # Human-readable labels
    labels = {
        "CAHPS_1": "Getting Timely Care",
        "CAHPS_2": "Provider Communication",
        "CAHPS_3": "Patient Rating",
        "CAHPS_4": "Access to Specialists",
        "CAHPS_5": "Health Promotion",
        "CAHPS_6": "Shared Decision Making",
        "CAHPS_7": "Health Status",
        "CAHPS_8": "Courteous Staff",
        "CAHPS_9": "Care Coordination",
        "CAHPS_11": "Stewardship of Resources",
    }

    values = []
    names = []

    for col in cahps_cols:
        val = temp_df[col].iloc[0]
        if pd.notna(val):
            try:
                values.append(float(val))
                names.append(labels.get(col, col))
            except (ValueError, TypeError):
                continue

    if not values:
        return None

    # Close radar shape
    values_closed = values + [values[0]]
    names_closed = names + [names[0]]

    # Create radar
    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=names_closed,
            fill="toself",
            name="CAHPS Scores",
        )
    )

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(
                    size=12,
                    color="#e5e7eb"
                ),
                gridcolor="rgba(148,163,184,0.4)",
                linecolor="rgba(148,163,184,0.6)",
            ),
            angularaxis=dict(
                tickfont=dict(
                    size=12,
                    color="#f1f5f9"
                ),
                gridcolor="rgba(148,163,184,0.3)",
                linecolor="rgba(148,163,184,0.5)",
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        title=dict(
            text="CAHPS Patient Experience",
            font=dict(
                color="#f1f5f9",
                size=16
            )
        ),
        height=450,
        margin=dict(
            t=50,
            b=40,
            l=80,
            r=80
        ),
    )

    return fig


# ============================================================
# SHARED PERFORMANCE YEAR STATE
# ============================================================

def get_current_year():
    """Use the selected performance year without changing page calculations."""
    if "performance_year" not in st.session_state:
        cached = list_cached_years()
        st.session_state.performance_year = (
            max(cached) if cached else DEFAULT_YEAR
        )
    return int(st.session_state.performance_year)


# ============================================================
# ACO APP
# ============================================================

def render_aco_app(year):
    # ========================================================
    # HEADER
    # ========================================================

    st.title("🏥 Value-Based Care Command Center")
    st.caption("Payer-facing analytics for ACO contracts + Hospital Safety (HVBP)")

    # ========================================================
    # REFRESH BUTTON
    # ========================================================

    col_btn, col_info = st.columns([1, 4])

    with col_btn:
        if st.button(
            "🔄 Refresh data",
            help=(
                "Click after running scripts/etl.py "
                "so the dashboard picks up the new CSV"
            )
        ):
            st.cache_data.clear()
            st.rerun()

    with col_info:
        st.caption(
            "After you change the API year and run "
            "`python scripts/etl.py`, click **Refresh data** here."
        )

    # ========================================================
    # LOAD SCORECARD
    # ========================================================

    df = load_scorecard(year)

    if df.empty or "ACO_Name" not in df.columns:
        st.warning(
            f"No valid ACO scorecard data available for "
            f"performance year {year}. Please select a year and click 'Fetch & process' in the sidebar."
        )
        st.stop()

    # ========================================================
    # PATIENT EXPERIENCE — CAHPS RADAR
    # ========================================================

    st.divider()
    st.subheader("Patient Experience Analysis")

    # --------------------------------------------------------
    # ACO SELECTOR
    # --------------------------------------------------------

    aco_options = (
        df[
            ["ACO_ID", "ACO_Name"]
        ]
        .drop_duplicates()
        .sort_values("ACO_Name")
    )

    if aco_options.empty:
        st.warning("No ACO information is available.")
    else:
        selected_aco_name = st.selectbox(
            "Select an ACO to analyze",
            aco_options["ACO_Name"].tolist()
        )

        selected_aco_id = aco_options.loc[
            aco_options["ACO_Name"] == selected_aco_name,
            "ACO_ID"
        ].iloc[0]

        # ----------------------------------------------------
        # CAHPS RADAR
        # ----------------------------------------------------

        fig = create_cahps_radar(
            df,
            selected_aco_id
        )

        if fig is not None:
            st.plotly_chart(
                fig,
                use_container_width=True
            )
        else:
            st.warning(
                f"CAHPS data is not available for "
                f"{selected_aco_name} in performance year {year}."
            )

    # ========================================================
    # DASHBOARD DESCRIPTION
    # ========================================================

    st.markdown(
        """
        This tool tracks **cost, quality, utilization, and patient
        experience performance** for provider groups (ACOs) under
        value-based contracts, identifies **what's driving savings
        or losses**, and recommends actions for the next provider
        review meeting.

        **Navigate using the sidebar:**

        - **Portfolio Overview** — see all contracted ACOs at a glance
        - **Contract Scorecard** — drill into a single ACO's
          cost / quality / utilization
        - **Driver Analysis** — see what's driving each ACO's
          cost variance vs peers
        - **Recommendations** — get a payer-ready meeting brief
          for any ACO
        - **Hospital Safety (HVBP)** — Hospital Value-Based
          Purchasing safety KPIs and network targeting
        - **GeoProvider Service** — 9 Medicare claims-based analyses across provider cost, markup, and geography
        """
    )

    # ========================================================
    # PORTFOLIO SUMMARY
    # ========================================================

    st.divider()
    st.subheader("Portfolio Overview")

    col1, col2, col3, col4 = st.columns(4)

    # --------------------------------------------------------
    # ACO COUNT
    # --------------------------------------------------------

    col1.metric(
        "ACOs in Portfolio",
        len(df)
    )

    # --------------------------------------------------------
    # STRONG PERFORMERS
    # --------------------------------------------------------

    if "contract_outcome" in df.columns:
        strong_performers = int(
            (
                df["contract_outcome"]
                == "Strong Performer"
            ).sum()
        )
    else:
        strong_performers = 0

    col2.metric(
        "Strong Performers",
        strong_performers
    )

    # --------------------------------------------------------
    # AT RISK
    # --------------------------------------------------------

    if "contract_outcome" in df.columns:
        at_risk = int(
            (
                df["contract_outcome"]
                == "At Risk"
            ).sum()
        )
    else:
        at_risk = 0

    col3.metric(
        "At Risk",
        at_risk
    )

    # --------------------------------------------------------
    # TOTAL SAVINGS
    # --------------------------------------------------------

    if "EarnSaveLoss" in df.columns:
        total_savings = (
            pd.to_numeric(
                df["EarnSaveLoss"],
                errors="coerce"
            )
            .fillna(0)
            .sum()
        )
    else:
        total_savings = 0

    col4.metric(
        "Total Earned Savings",
        f"${total_savings:,.0f}"
    )

    # ========================================================
    # CONTRACT OUTCOME SUMMARY
    # ========================================================

    st.divider()
    st.subheader("Quick look — contract outcomes")

    for outcome in [
        "Strong Performer",
        "On Track",
        "At Risk"
    ]:
        if "contract_outcome" not in df.columns:
            continue

        subset = df[
            df["contract_outcome"] == outcome
        ]

        if len(subset) == 0:
            continue

        with st.expander(
            f"{OUTCOME_COLORS.get(outcome, '')} "
            f"{outcome} ({len(subset)} ACOs)"
        ):
            display_columns = [
                "ACO_Name",
                "Current_Track",
                "Sav_rate",
                "QualScore",
                "top_cost_driver"
            ]

            available_columns = [
                col
                for col in display_columns
                if col in subset.columns
            ]

            display_df = subset[
                available_columns
            ].copy()

            display_df = display_df.rename(
                columns={
                    "ACO_Name": "ACO",
                    "Current_Track": "Track",
                    "Sav_rate": "Savings Rate (%)",
                    "QualScore": "Quality Score (%)",
                    "top_cost_driver": "Top Cost Driver",
                }
            )

            if "Savings Rate (%)" in display_df.columns:
                display_df = display_df.sort_values(
                    "Savings Rate (%)",
                    ascending=False
                )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

    # ========================================================
    # HOSPITAL SAFETY — HVBP
    # ========================================================

    st.divider()
    st.subheader("🏥 Hospital Safety — HVBP")

    st.info(
        "Hospital Value-Based Purchasing (HVBP) analysis "
        "is included as a separate layer of the VBC Command "
        "Center. Hospital safety KPIs can be used to identify "
        "high-risk hospitals and support payer network targeting."
    )

    # ========================================================
    # DATA SOURCE
    # ========================================================

    st.caption(
        "Data sources: CMS Medicare Shared Savings Program (ACO) "
        "+ CMS Hospital Value-Based Purchasing (HVBP) + CMS Medicare Part B Claims. "
        "This tool builds analytics on top of CMS published results."
    )


# ============================================================
# PORTFOLIO OVERVIEW
# ============================================================

def render_portfolio_overview(year):
    st.title("📋 Portfolio Overview")
    st.caption("All ACOs under contract — filter and sort to find where to focus")

    df = load_scorecard(year)
    if df.empty or "Current_Track" not in df.columns:
        st.warning("No valid ACO scorecard data available. Please select a year and click 'Fetch & process' in the sidebar.")
        st.stop()

    try:
        from ml.predict import predict_for_dataframe, models_available, get_model_metrics
        ML_AVAILABLE = models_available()
    except Exception:
        ML_AVAILABLE = False

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


# ============================================================
# CONTRACT SCORECARD
# ============================================================

def render_contract_scorecard(year):
    st.title("📊 Contract Scorecard")
    st.caption("Drill into a single ACO's cost, quality, and utilization performance")

    df = load_scorecard(year)
    if df.empty or "ACO_Name" not in df.columns:
        st.warning("No valid ACO scorecard data available. Please select a year and click 'Fetch & process' in the sidebar.")
        st.stop()

    aco_name = st.selectbox("Select an ACO", sorted(df["ACO_Name"].unique()))
    row = df[df["ACO_Name"] == aco_name].iloc[0]

    # --- Header ---------------------------------------------------------------
    outcome = row["contract_outcome"]
    st.subheader(f"{OUTCOME_COLORS.get(outcome, '')} {aco_name}")

    # Rev_Exp_Cat only exists in later years (2018+); fall back safely
    rev_exp = row["Rev_Exp_Cat"] if "Rev_Exp_Cat" in row.index and pd.notna(row.get("Rev_Exp_Cat")) else "N/A"
    n_ab = int(row["N_AB"]) if pd.notna(row.get("N_AB")) else 0
    track = row.get("Current_Track", "Unknown")
    risk = row.get("Risk_Model", "Unknown")

    st.caption(
        f"Track {track} · {risk} · "
        f"{n_ab:,} assigned beneficiaries · {rev_exp}"
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Savings Rate", fmt_pct(row["Sav_rate"]))
    col2.metric("Quality Score", fmt_pct(row["QualScore"]), help=f"Tier: {row['quality_tier']}")
    col3.metric("Earned Savings/Losses", fmt_dollars(row["EarnSaveLoss"]))

    st.divider()

    # --- Cost: actual vs benchmark ---------------------------------------------
    st.markdown("### 💰 Cost — Actual vs Benchmark")
    c1, c2 = st.columns([1, 1])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Benchmark", "Actual Spend", "Peer Average"],
            y=[row["UpdatedBnchmk"], row["Per_Capita_Exp_TOTAL_PY"], row["peer_avg_per_capita_exp"]],
            marker_color=["#94a3b8", "#ef4444" if row["cost_variance_pct"] > 0 else "#22c55e", "#3b82f6"],
            text=[fmt_dollars(row["UpdatedBnchmk"]), fmt_dollars(row["Per_Capita_Exp_TOTAL_PY"]), fmt_dollars(row["peer_avg_per_capita_exp"])],
            textposition="outside",
        ))
        fig.update_layout(
            title="Per-Capita Spend ($)",
            yaxis_title="$ per beneficiary",
            height=350,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cost_breakdown = pd.DataFrame({
            "Category": ["Inpatient", "Outpatient", "Physician/Supplier", "SNF"],
            "This ACO": [row["CapAnn_INP_All"], row["CapAnn_OPD"], row["CapAnn_PB"], row["CapAnn_SNF"]],
            "Peer Avg": [row["peer_avg_inpatient_exp"], row["peer_avg_outpatient_exp"],
                         row["peer_avg_physician_exp"], row["peer_avg_snf_exp"]],
        })
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="This ACO", x=cost_breakdown["Category"], y=cost_breakdown["This ACO"], marker_color="#6366f1"))
        fig2.add_trace(go.Bar(name="Peer Avg", x=cost_breakdown["Category"], y=cost_breakdown["Peer Avg"], marker_color="#cbd5e1"))
        fig2.update_layout(
            title="Spend by Care Setting ($ per capita)",
            barmode="group",
            height=350,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # --- Quality ---------------------------------------------------------------
    st.markdown("### ⭐ Quality")
    q1, q2 = st.columns(2)
    q1.metric("Met Quality Standard", "Yes ✅" if row.get("Met_QPS") == 1 else "No ❌")
    _m479 = row.get("Measure_479") if "Measure_479" in row.index else None
    q2.metric(
        "30-Day Readmission Rate",
        fmt_pct(_m479 * 100 if pd.notna(_m479) else None),
        help="Lower is better",
    )

    st.divider()

    # --- Utilization -------------------------------------------------------
    st.markdown("### 🏥 Utilization")
    u1, u2, u3, u4, u5 = st.columns(5)

    def _util(col, var_col=None):
        val = row.get(col) if col in row.index else None
        label = f"{val:.0f}" if pd.notna(val) else "—"
        kwargs = {}
        if var_col and var_col in row.index:
            kwargs["delta"] = fmt_variance(row.get(var_col))
            kwargs["delta_color"] = "inverse"
        return label, kwargs

    lab, kw = _util("ADM", "admission_variance_pct")
    u1.metric("Inpatient Admission /1,000", lab, **kw)
    lab, kw = _util("P_EDV_Vis", "er_visit_variance_pct")
    u2.metric("ER Visits /1,000", lab, **kw)
    lab, kw = _util("P_EDV_Vis_HOSP", "er_to_admit_variance_pct")
    u3.metric("ER→Admit /1,000", lab, **kw)
    lab, kw = _util("P_EM_PCP_Vis")
    u4.metric("Primary Care Visits /1,000", lab, **kw)
    _ppt = row.get("providers_per_1000") if "providers_per_1000" in row.index else None
    u5.metric(
        "Providers /1,000 Beneficiaries",
        f"{_ppt:.1f}" if pd.notna(_ppt) else "—",
        help="(PCPs + Specialists + NPs + PAs + CNSs) / assigned beneficiary-years × 1,000.",
    )
    st.caption(
        "Deltas show variance vs. peer average within the same contract track. "
        "Red = higher utilization than peers (usually unfavorable for cost)."
    )
    st.divider()
    st.markdown("### 📈 Performance Trend — Across Contract Years")

    trend_df = load_aco_trend(row.get("ACO_ID"))

    if len(trend_df) < 2:
        st.caption(
            "Only one cached performance year is available for this ACO. "
            "Fetch additional years from the sidebar to see a multi-year trend."
        )
    else:
        t1, t2 = st.columns(2)

        with t1:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=trend_df["Year"], y=trend_df["Savings Rate (%)"],
                mode="lines+markers", name="Savings Rate (%)", line=dict(color="#22c55e"),
            ))
            fig_trend.add_trace(go.Scatter(
                x=trend_df["Year"], y=trend_df["Cost vs Benchmark (%)"],
                mode="lines+markers", name="Cost vs Benchmark (%)", line=dict(color="#ef4444"),
            ))
            fig_trend.add_hline(y=0, line_color="#94a3b8", line_dash="dot")
            fig_trend.update_layout(
                title=dict(text="Savings Rate & Cost Variance Over Time", y=0.97, yanchor="top"),
                xaxis_title="Performance Year",
                height=380,
                margin=dict(t=80, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0.5, xanchor="center"),
            )
            fig_trend.update_xaxes(dtick=1)
            st.plotly_chart(fig_trend, use_container_width=True)

        with t2:
            fig_qual = go.Figure()
            fig_qual.add_trace(go.Scatter(
                x=trend_df["Year"], y=trend_df["Quality Score (%)"],
                mode="lines+markers", name="Quality Score (%)", line=dict(color="#6366f1"),
            ))
            fig_qual.update_layout(
                title="Quality Score Over Time",
                xaxis_title="Performance Year", height=350, margin=dict(t=40, b=20),
            )
            fig_qual.update_xaxes(dtick=1)
            st.plotly_chart(fig_qual, use_container_width=True)

        st.dataframe(trend_df, use_container_width=True, hide_index=True)


# ============================================================
# DRIVER ANALYSIS
# ============================================================

def render_driver_analysis(year):
    st.title("🔍 Driver Analysis")
    st.caption("What's actually driving this ACO's cost variance vs. peers + ML risk explanation")

    df = load_scorecard(year)
    if df.empty or "ACO_Name" not in df.columns:
        st.warning("No valid ACO scorecard data available. Please select a year and click 'Fetch & process' in the sidebar.")
        st.stop()

    try:
        from ml.predict import predict_for_row, get_shap_explanation, models_available
        ML_AVAILABLE = models_available()
    except Exception:
        ML_AVAILABLE = False

    aco_name = st.selectbox("Select an ACO", sorted(df["ACO_Name"].unique()))
    row = df[df["ACO_Name"] == aco_name].iloc[0]

    # ------------------------------------------------------------------
    # Cost variance chart
    # ------------------------------------------------------------------
    drivers = pd.DataFrame({
        "Driver": ["Inpatient", "Outpatient", "Physician/Supplier", "SNF", "ER Visits", "ER→Admit", "Admissions"],
        "Variance vs Peers (%)": [
            row["inpatient_variance_pct"], row["outpatient_variance_pct"],
            row["physician_variance_pct"], row["snf_variance_pct"],
            row["er_visit_variance_pct"], row["er_to_admit_variance_pct"],
            row["admission_variance_pct"],
        ],
    }).sort_values("Variance vs Peers (%)", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=drivers["Driver"],
        x=drivers["Variance vs Peers (%)"],
        orientation="h",
        marker_color=["#ef4444" if v > 0 else "#22c55e" for v in drivers["Variance vs Peers (%)"]],
        text=[fmt_variance(v) for v in drivers["Variance vs Peers (%)"]],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color="#94a3b8")
    fig.update_layout(
        title=f"{aco_name} — Variance vs Peer Average, by Category",
        xaxis_title="% above (red) / below (green) peer average",
        height=450,
        margin=dict(t=50, b=20, l=10, r=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ How peer variance is calculated"):
        st.markdown(
            f"""
            Peer comparisons group ACOs **by contract track only** (e.g., Track 1,
            Track 2) — this ACO's peer group has **{int(row['peer_group_size'])} ACOs**.

            This does **not** control for region, ACO size, or population risk
            profile — two ACOs in the same track can have very different patient
            panels and cost structures. Variance vs. peers should be read as a
            **starting signal for investigation**, not a definitive judgment of
            performance. Always cross-check a flagged driver against population
            context (dual-eligible %, LTI %) before concluding it reflects
            controllable spend.
            """
        )

    st.divider()
    st.subheader(f"🎯 Top cost driver: {row['top_cost_driver']}")

    driver_variance_map = {
        "Inpatient": row["inpatient_variance_pct"],
        "Outpatient": row["outpatient_variance_pct"],
        "Physician/Supplier": row["physician_variance_pct"],
        "SNF": row["snf_variance_pct"],
    }
    top_val = driver_variance_map.get(row["top_cost_driver"])

    if pd.notna(top_val) and top_val > 0:
        st.error(
            f"**{row['top_cost_driver']} spend is {top_val:.0f}% above peer average** — "
            f"this is the single largest contributor to cost variance for this ACO."
        )
    elif pd.notna(top_val):
        st.success(
            f"**{row['top_cost_driver']} spend is {abs(top_val):.0f}% below peer average** — "
            f"this ACO's largest relative cost category is actually a strength."
        )

    # ------------------------------------------------------------------
    # ML Risk Explanation with SHAP
    # ------------------------------------------------------------------
    if ML_AVAILABLE:
        st.divider()
        st.subheader("🤖 Why is this ACO predicted as High/Medium/Low Risk?")

        ml_result = predict_for_row(row)
        risk_score = ml_result.get("risk_score")
        risk_level = ml_result.get("risk_level", "Unknown")

        col1, col2 = st.columns(2)
        col1.metric("ML Risk Score", f"{risk_score:.1f}" if risk_score is not None else "—")
        col2.metric("Risk Level", risk_level)

        shap_data = get_shap_explanation(row, top_n=8)

        if shap_data:
            name_map = {
                "QualScore": "Quality Score",
                "inpatient_variance_pct": "Inpatient Variance",
                "outpatient_variance_pct": "Outpatient Variance",
                "physician_variance_pct": "Physician Variance",
                "snf_variance_pct": "SNF Variance",
                "er_visit_variance_pct": "ER Visit Variance",
                "er_to_admit_variance_pct": "ER→Admit Variance",
                "admission_variance_pct": "Admission Variance",
                "providers_per_1000": "Providers per 1,000",
                "N_AB": "Number of Beneficiaries",
                "Perc_Dual": "% Dual Eligible",
                "Perc_LTI": "% Long-Term Institutionalized",
                "Current_Track_encoded": "Contract Track",
            }

            features = [name_map.get(f, f) for f in shap_data["features"]]
            values = shap_data["values"]

            fig_shap = go.Figure()
            fig_shap.add_trace(go.Bar(
                y=features[::-1],
                x=values[::-1],
                orientation="h",
                marker_color=["#ef4444" if v > 0 else "#22c55e" for v in values[::-1]],
                text=[f"{v:+.3f}" for v in values[::-1]],
                textposition="outside",
            ))
            fig_shap.add_vline(x=0, line_color="#94a3b8")
            fig_shap.update_layout(
                title="Top factors increasing (red) or decreasing (green) the Risk Score",
                xaxis_title="SHAP value (impact on risk probability)",
                height=420,
                margin=dict(t=50, b=20, l=10, r=10),
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            top_positive = [(f, v) for f, v in zip(features, values) if v > 0][:3]
            top_negative = [(f, v) for f, v in zip(features, values) if v < 0][:2]

            if top_positive:
                reasons = ", ".join([f"**{f}**" for f, _ in top_positive])
                st.info(f"📌 Main reasons increasing risk: {reasons}")
            if top_negative:
                reasons = ", ".join([f"**{f}**" for f, _ in top_negative])
                st.success(f"✅ Factors helping reduce risk: {reasons}")
        else:
            st.warning("Could not generate SHAP explanation for this ACO.")
    else:
        st.info("Train the ML models first (`python ml/train_models.py`) to see risk explanation.")

    # ------------------------------------------------------------------
    # Population context
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Population context")
    c1, c2 = st.columns(2)

    def _extract_pct(val):
        if pd.isna(val) or val is None:
            return None
        try:
            v = float(str(val).replace("%", "").replace(",", "").strip())
            return v if v > 1.0 else v * 100 if v > 0 else v
        except Exception:
            return None

    perc_dual = _extract_pct(row.get("Perc_Dual"))
    if perc_dual is None:
        n_dual = row.get("N_AB_Year_Dual_PY")
        n_tot = row.get("N_AB_Year_PY") if pd.notna(row.get("N_AB_Year_PY")) else row.get("N_AB")
        if pd.notna(n_dual) and pd.notna(n_tot) and float(n_tot) > 0:
            perc_dual = (float(n_dual) / float(n_tot)) * 100
        elif pd.notna(row.get("N_AB_Year_AGED_Dual_PY")) and pd.notna(n_tot) and float(n_tot) > 0:
            perc_dual = (float(row.get("N_AB_Year_AGED_Dual_PY")) / float(n_tot)) * 100

    perc_lti = _extract_pct(row.get("Perc_LTI"))

    c1.metric(
        "% Dual-Eligible Beneficiaries",
        f"{perc_dual:.1f}%" if pd.notna(perc_dual) else "—",
        help="Dual-eligible (Medicare + Medicaid) beneficiaries are typically higher-cost, higher-need.",
    )
    c2.metric(
        "% Long-Term Institutionalized",
        f"{perc_lti:.1f}%" if pd.notna(perc_lti) else "—",
        help="A high share signals a much sicker/costlier population — important context, not an excuse.",
    )

    if pd.notna(perc_dual) and perc_dual > 20:
        st.info(
            "⚠️ This ACO's population skews notably higher-need (dual-eligible / "
            "institutionalized) than typical peers. Cost variance should be read "
            "alongside this context, not in isolation."
        )


# ============================================================
# RECOMMENDATIONS
# ============================================================

def render_recommendations(year):
    st.title("✅ Recommendations & Meeting Brief")
    st.caption("Rule-based + ML-powered recommended actions, and a one-page brief for the provider review meeting")

    df = load_scorecard(year)
    if df.empty or "ACO_Name" not in df.columns:
        st.warning("No valid ACO scorecard data available. Please select a year and click 'Fetch & process' in the sidebar.")
        st.stop()

    try:
        from ml.predict import predict_for_row, models_available, get_model_metrics
        ML_AVAILABLE = models_available()
    except Exception:
        ML_AVAILABLE = False

    aco_name = st.selectbox("Select an ACO", sorted(df["ACO_Name"].unique()))
    row = df[df["ACO_Name"] == aco_name].iloc[0]

    ml_result = None
    if ML_AVAILABLE:
        ml_result = predict_for_row(row)

        st.subheader("🤖 ML Risk Prediction")
        m1, m2 = st.columns(2)

        risk_score = ml_result.get("risk_score")
        risk_level = ml_result.get("risk_level", "Unknown")

        risk_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢", "Unknown": "⚪"}

        m1.metric(
            "ML Risk Score",
            f"{risk_score:.1f}" if risk_score is not None else "—",
            help="Probability of becoming At Risk (0–100). Higher = more urgent.",
        )
        m2.metric(
            "Risk Level",
            f"{risk_colors.get(risk_level, '')} {risk_level}",
        )

        if risk_level == "High":
            st.error(
                f"**High Risk Alert:** This ACO has a {risk_score:.0f}% probability of becoming At Risk. "
                "Prioritize this contract for intervention."
            )
        elif risk_level == "Medium":
            st.warning(
                f"**Medium Risk:** This ACO has a {risk_score:.0f}% probability of becoming At Risk. "
                "Monitor closely and address key drivers."
            )
        else:
            st.success(
                f"**Low Risk:** This ACO currently shows low probability ({risk_score:.0f}%) of becoming At Risk."
            )

        st.divider()
    else:
        st.info(
            "ℹ️ ML models not trained yet. Run `python ml/train_models.py` to enable "
            "risk prediction and smarter recommendations.",
            icon="ℹ️",
        )

    def build_recommendations(row, ml_result=None) -> list[dict]:
        recs = []

        def variance(col):
            v = row.get(col)
            return v if pd.notna(v) else None

        inp = variance("inpatient_variance_pct")
        snf = variance("snf_variance_pct")
        er = variance("er_visit_variance_pct")
        er_admit = variance("er_to_admit_variance_pct")
        pb = variance("physician_variance_pct")
        admissions = variance("admission_variance_pct")

        if snf is not None and snf > 25:
            recs.append({
                "priority": "High",
                "area": "Skilled Nursing Facility",
                "finding": f"SNF spend is {snf:.0f}% above peer average.",
                "action": "Review SNF length-of-stay and discharge planning practices; "
                          "consider tighter SNF network management or a preferred SNF partnership "
                          "with strong readmission performance.",
                "source": "Rule",
            })

        if er is not None and er > 20:
            recs.append({
                "priority": "High",
                "area": "Emergency Department Utilization",
                "finding": f"ER visits are {er:.0f}% above peer average"
                           + (f", with ER-to-admission conversion {er_admit:.0f}% above peers." if er_admit and er_admit > 20 else "."),
                "action": "Launch care management outreach for high-ER-utilizing beneficiaries; "
                          "expand same-day/urgent primary care access to divert avoidable ER visits.",
                "source": "Rule",
            })

        if inp is not None and inp > 20:
            recs.append({
                "priority": "High" if inp > 40 else "Medium",
                "area": "Inpatient Utilization",
                "finding": f"Inpatient spend is {inp:.0f}% above peer average.",
                "action": "Audit admission patterns for ambulatory-sensitive conditions; "
                          "strengthen transitional care management to reduce avoidable admissions.",
                "source": "Rule",
            })

        if pb is not None and pb < -15:
            recs.append({
                "priority": "Medium",
                "area": "Physician/Supplier Services",
                "finding": f"Physician/supplier spend is {abs(pb):.0f}% below peer average.",
                "action": "Confirm this reflects efficient care coordination and not under-coding "
                          "or access barriers — verify with a quick chart audit sample.",
                "source": "Rule",
            })

        if admissions is not None and admissions > 20:
            recs.append({
                "priority": "Medium",
                "area": "Hospital Admissions",
                "finding": f"Admission rate is {admissions:.0f}% above peer average.",
                "action": "Evaluate primary care panel capacity and after-hours access — "
                          "gaps here often correlate with higher admission rates.",
                "source": "Rule",
            })

        if row.get("QualScore") is not None and pd.notna(row["QualScore"]) and row["QualScore"] < 75:
            recs.append({
                "priority": "Medium",
                "area": "Quality Performance",
                "finding": f"Quality score of {row['QualScore']:.1f}% is below the 'Good' tier threshold.",
                "action": "Prioritize quality measure reporting completeness and targeted "
                          "improvement on lowest-performing individual measures.",
                "source": "Rule",
            })

        if row.get("Perc_Dual") is not None and pd.notna(row["Perc_Dual"]) and row["Perc_Dual"] > 20:
            recs.append({
                "priority": "Low",
                "area": "Population Risk Context",
                "finding": f"{row['Perc_Dual']:.0f}% of the population is dual-eligible — "
                           "meaningfully higher-need than typical peers.",
                "action": "Factor population complexity into performance conversations; "
                          "consider whether risk adjustment and care management resourcing "
                          "match this population's needs.",
                "source": "Rule",
            })

        if ml_result and ml_result.get("risk_score") is not None:
            risk_score = ml_result["risk_score"]
            risk_level = ml_result["risk_level"]

            if risk_level == "High":
                recs.append({
                    "priority": "High",
                    "area": "ML Early Warning",
                    "finding": f"Machine learning model assigns a {risk_score:.0f}% probability "
                               f"that this ACO will become At Risk.",
                    "action": "Schedule an urgent performance review meeting. Focus on the top cost "
                              "drivers and quality gaps identified above. Consider increased care "
                              "management support or contract adjustment discussions.",
                    "source": "ML",
                })
            elif risk_level == "Medium":
                recs.append({
                    "priority": "Medium",
                    "area": "ML Monitoring Alert",
                    "finding": f"Model predicts a moderate ({risk_score:.0f}%) chance of becoming At Risk.",
                    "action": "Increase monitoring frequency. Review utilization trends quarterly "
                              "and address any emerging cost drivers before they escalate.",
                    "source": "ML",
                })

            if risk_level == "High":
                for rec in recs:
                    if rec["priority"] == "Medium":
                        rec["priority"] = "High"
                        rec["finding"] += " (Priority raised due to high ML risk score.)"

        if not recs:
            recs.append({
                "priority": "Low",
                "area": "Overall",
                "finding": "No major cost or quality outliers detected vs. peers.",
                "action": "Maintain current performance; monitor for emerging trends next period.",
                "source": "Rule",
            })

        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        recs.sort(key=lambda r: priority_order[r["priority"]])
        return recs

    recommendations = build_recommendations(row, ml_result)

    priority_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    source_badge = {"Rule": "📋 Rule", "ML": "🤖 ML"}

    for rec in recommendations:
        with st.container(border=True):
            badge = source_badge.get(rec.get("source", "Rule"), "")
            st.markdown(
                f"**{priority_colors[rec['priority']]} {rec['priority']} priority — {rec['area']}**  "
                f"<span style='font-size:0.8em; color:gray'>{badge}</span>",
                unsafe_allow_html=True,
            )
            st.write(f"**Finding:** {rec['finding']}")
            st.write(f"**Recommended action:** {rec['action']}")

    st.divider()

    st.subheader("📄 Provider Review Meeting Brief")

    brief_lines = [
        f"# Provider Review Brief — {aco_name}",
        f"*Generated {date.today().isoformat()}*",
        "",
        f"**Contract Track:** {row['Current_Track']} ({row.get('Risk_Model', 'N/A')})  ",
        f"**Assigned Beneficiaries:** {int(row['N_AB']):,}  ",
        f"**Contract Outcome:** {row['contract_outcome']}",
    ]

    if ml_result and ml_result.get("risk_score") is not None:
        brief_lines += [
            "",
            "## ML Risk Assessment",
            f"- **Risk Score:** {ml_result['risk_score']:.1f} / 100 ({ml_result['risk_level']})",
        ]

    brief_lines += [
        "",
        "## Financial Performance",
        f"- Savings Rate: {fmt_pct(row['Sav_rate'])}",
        f"- Earned Savings/Losses: {fmt_dollars(row['EarnSaveLoss'])}",
        f"- Per-Capita Spend vs Benchmark: {fmt_variance(row['cost_variance_pct'])}",
        "",
        "## Quality Performance",
        f"- Quality Score: {fmt_pct(row['QualScore'])} ({row.get('quality_tier', 'N/A')})",
        f"- Met Quality Performance Standard: {'Yes' if row.get('Met_QPS') == 1 else 'No'}",
        "",
        "## Key Drivers",
        f"- Top Cost Driver: **{row['top_cost_driver']}**",
        f"- Inpatient vs peers: {fmt_variance(row['inpatient_variance_pct'])}",
        f"- Outpatient vs peers: {fmt_variance(row['outpatient_variance_pct'])}",
        f"- SNF vs peers: {fmt_variance(row['snf_variance_pct'])}",
        f"- ER visits vs peers: {fmt_variance(row['er_visit_variance_pct'])}",
        f"- ER-to-admission vs peers: {fmt_variance(row['er_to_admit_variance_pct'])}",
        "",
        "## Recommended Actions",
    ]

    for rec in recommendations:
        source_tag = f"[{rec.get('source', 'Rule')}]"
        brief_lines.append(
            f"- **[{rec['priority']}] {rec['area']}** {source_tag}: {rec['finding']} → {rec['action']}"
        )

    brief_text = "\n".join(brief_lines)
    st.markdown(brief_text)

    st.download_button(
        "⬇️ Download brief as Markdown",
        data=brief_text,
        file_name=f"{aco_name.replace(' ', '_')}_meeting_brief.md",
        mime="text/markdown",
    )

    if ML_AVAILABLE:
        metrics = get_model_metrics()
        with st.expander("ℹ️ About the ML Model used on this page"):
            if metrics:
                st.markdown(
                    f"""
                    This page combines **transparent business rules** with an **XGBoost model** 
                    trained on historical CMS Medicare Shared Savings Program data.

                    **Model Performance**
                    - At-Risk Classifier AUC: **{metrics['classifier']['auc']}**
                    - Savings Rate MAE: **{metrics['regressor']['mae']}** pp

                    High ML Risk Score automatically raises the priority of existing recommendations
                    and adds an early-warning action item.
                    """
                )


# ============================================================
# HOSPITAL OVERVIEW — HOSPITAL VISUALIZATION
# ============================================================

@st.cache_data
def load_and_process_hospital_data():
    DATA_PATH = Path(__file__).parent / "data" / "calculated_hvbp_data.csv"
    df_raw = pd.read_csv(DATA_PATH)

    # Apply the strict 1612-row completeness filter
    hai_numbered_cols = [
        'HAI-1 Measure Score', 'HAI-2 Measure Score', 'HAI-3 Measure Score',
        'HAI-4 Measure Score', 'HAI-5 Measure Score', 'HAI-6 Measure Score'
    ]

    has_sep1 = df_raw['SEP-1 Measure Score'].notna()
    has_ssi = df_raw['Combined SSI Measure Score'].notna()
    has_min_3_hais = df_raw[hai_numbered_cols].notna().sum(axis=1) >= 3

    df = df_raw[has_sep1 & has_ssi & has_min_3_hais].copy()

    # Calculate KPI 1: Comprehensive Hospital Safety Score (CHSS)
    all_hai_cols = hai_numbered_cols + ['Combined SSI Measure Score', 'SEP-1 Measure Score']
    df['CHSS'] = df[all_hai_cols].mean(axis=1)

    # Calculate KPI 4: Continuous Improvement Rate
    improvement_cols = [c for c in df.columns if 'Improvement Points' in c]
    df['Improvement Rate'] = df[improvement_cols].mean(axis=1)

    return df


def render_hospital_overview():
    st.title("🏥 Hospital Value-Based Purchasing (HVBP) Safety Dashboard")
    st.markdown("Visualizing Comprehensive Safety KPIs for America's top reporting hospitals.")

    df = load_and_process_hospital_data()

    # --- 2. TOP LEVEL METRICS ---
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Facilities Evaluated", f"{len(df):,}")
    col2.metric("National Avg Safety Score", f"{df['CHSS'].mean():.3f} / 1.0")
    col3.metric("National Avg Improvement Rate", f"{df['Improvement Rate'].mean():.3f} / 1.0")
    st.markdown("---")

    # --- 3. ROW 1 CHARTS ---
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("1.Hospital Safety Score  Distribution")
        st.markdown("Displays the spread of overall facility safety. A normal bell curve indicates a standard distribution of safety quality.")

        st.info("""
        **💡 How to read this chart (Network Tiering):**
        * Think of this score like a hospital's **'Safety GPA'**.
        * Payers use this to build Narrow Networks: Bars pushed further to the right (closer to 1.0) represent elite, highly safe hospitals that should be given preferred network status.
        """)

        fig_chss = px.histogram(
            df, x='CHSS', nbins=20,
            color_discrete_sequence=['#2E86C1'],
            labels={'CHSS': 'Safety Score (Max 1.0)'}
        )
        fig_chss.update_layout(bargap=0.1)
        st.plotly_chart(fig_chss, use_container_width=True)

    with chart_col2:
        st.subheader("2. SEP vs SSI Matrix")
        st.markdown("Evaluates whether safety is systemic. Top right indicates excellence in both Emergency/ICU and Surgery.")

        st.info("""
        **💡 How to read this chart (Targeted Negotiations):**
        * **Top-Right (High ER, High OR):** 🌟 Excellent safety in both environments.
        * **Bottom-Right (High ER, Low OR):** 🏥 High risk for costly post-surgical complications.
        * *(Note: Dots form a grid because scores are on a 10-point scale, causing stacking).*
        """)

        fig_matrix = px.scatter(
            df, x='SEP-1 Measure Score', y='Combined SSI Measure Score',
            opacity=0.5, color_discrete_sequence=['#E74C3C'],
            trendline="ols",
            hover_data=['Facility Name', 'State']
        )
        fig_matrix.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig_matrix, use_container_width=True)

    # --- 4. ROW 2 CHARTS ---
    st.markdown("---")
    st.subheader("3. State-Level Safety Performance Index")
    st.markdown("Compares the average Comprehensive Hospital Safety Score across states.")

    st.info("""
    **💡 How to read this map (Risk Forecasting):**
    * **Darker/Brighter Colors:** Highlight regions where hospitals generally perform exceptionally well.
    * **Lighter/Faded Colors:** Highlight high-risk regions where patient complications may be more frequent, requiring premium pricing adjustments.
    * **Interact:** Hover your mouse over any state to see its exact average safety score.
    """)

    state_avg = df.groupby('State')['CHSS'].mean().reset_index()

    fig_map = px.choropleth(
        state_avg,
        locations='State',
        locationmode="USA-states",
        color='CHSS',
        scope="usa",
        color_continuous_scale="Viridis",
        labels={'CHSS':'Avg Safety Score'}
    )
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")
    st.subheader("4. National Improvement Rate Distribution")
    st.markdown("Displays the volume of hospitals based on their effort to rapidly improve safety protocols.")

    st.warning("""
    **⚠️ Important Note on Improvement Scores:**
    Scores represent a government report card grade for effort, not an infection drop percentage. For example, a score of **0.80** means the hospital earned an **80/100 grade on their improvement evaluation**, *not* that their infections went down by 80%.
    """)

    st.info("""
    **💡 How to read this Bar Chart (Targeting the Overachievers):**
    * **The Bottom (X-Axis):** The hospital's grade on their improvement effort (0.0 to 1.0).
    * **The Height (Y-Axis):** How many hospitals in the country received that exact grade.
    * **The Strategy:** The massive spike in the middle shows that most hospitals are making "average" progress. Payers should look at the tiny bars on the far right—these are the rare, highly motivated facilities they should target for bonus contracts.
    """)

    fig_improve = px.histogram(
        df, x='Improvement Rate', nbins=30,
        color_discrete_sequence=['#27AE60'],
        labels={'Improvement Rate': 'Avg Improvement Score (Max 1.0)'}
    )
    fig_improve.update_layout(bargap=0.1, yaxis_title="Count of Hospitals")
    st.plotly_chart(fig_improve, use_container_width=True)

    # --- 5. INTERACTIVE DATA TABLE (COMMAND CENTER) ---
    st.markdown("---")
    st.subheader("🔎 Command Center: Contract Target Finder")
    st.markdown("Filter the national database to identify specific high-effort hospitals for Value-Based Care contracts or safe hospitals for Network Tiering.")

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        search_mode = st.selectbox(
            "1. Select Search Goal:",
            options=[
                "Search based on Improvement Score (Shared-Savings)",
                "Search based on Safety Score (Network Tiering)"
            ]
        )

    with filter_col2:
        if search_mode == "Search based on Improvement Score (Shared-Savings)":
            min_score = st.slider(
                "2. Minimum Improvement Score:",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.05
            )
        else:
            min_score = st.slider(
                "2. Minimum Safety Score:",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.05
            )

    if search_mode == "Search based on Improvement Score (Shared-Savings)":
        filtered_df = df[df['Improvement Rate'] >= min_score]
        sorted_df = filtered_df.sort_values(by='Improvement Rate', ascending=False)
    else:
        filtered_df = df[df['CHSS'] >= min_score]
        sorted_df = filtered_df.sort_values(by='CHSS', ascending=False)

    st.success(f"🎯 **Target List Generated:** Found {len(sorted_df)} individual hospitals nationwide matching your criteria.")

    display_cols = ['Facility Name', 'State', 'Improvement Rate', 'CHSS', 'SEP-1 Measure Score', 'Combined SSI Measure Score']
    st.dataframe(sorted_df[display_cols].round(3), use_container_width=True)


# ============================================================
# PERFORMANCE YEAR
# ============================================================

def render_performance_year():
    st.title("📅 Performance Year")
    st.caption("Select the CMS performance year used by the ACO analytics.")

    render_year_selector()
    current_year = get_current_year()

    st.divider()
    st.info(f"Current performance year: {current_year}")



# ============================================================
# GEOPROVIDER SERVICE DATA LAYER & HELPERS
# ============================================================

@st.cache_resource
def get_connection():
    return duckdb.connect()

@st.cache_data(show_spinner=False)
def run_query(sql: str) -> pd.DataFrame:
    con = get_connection()
    return con.execute(sql).fetchdf()

def section_header(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">{title}</div>
            <div class="section-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

PLOT_BG = "#131d30"
PAPER_BG = "#131d30"
GRID_COLOR = "#2a3a58"

def style_fig(fig):
    if not fig.layout.title or not fig.layout.title.text:
        fig.update_layout(title_text="")
    fig.update_layout(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(color="#dbe3f0"),
        title_font=dict(color="#ffffff"),
        legend=dict(font=dict(color="#dbe3f0")),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=GRID_COLOR)
    return fig

STATE_REFERENCE = [
    ('AL', 'Alabama'), ('AK', 'Alaska'), ('AZ', 'Arizona'), ('AR', 'Arkansas'), ('CA', 'California'),
    ('CO', 'Colorado'), ('CT', 'Connecticut'), ('DE', 'Delaware'), ('DC', 'District of Columbia'),
    ('FL', 'Florida'), ('GA', 'Georgia'), ('HI', 'Hawaii'), ('ID', 'Idaho'), ('IL', 'Illinois'),
    ('IN', 'Indiana'), ('IA', 'Iowa'), ('KS', 'Kansas'), ('KY', 'Kentucky'), ('LA', 'Louisiana'),
    ('ME', 'Maine'), ('MD', 'Maryland'), ('MA', 'Massachusetts'), ('MI', 'Michigan'), ('MN', 'Minnesota'),
    ('MS', 'Mississippi'), ('MO', 'Missouri'), ('MT', 'Montana'), ('NE', 'Nebraska'), ('NV', 'Nevada'),
    ('NH', 'New Hampshire'), ('NJ', 'New Jersey'), ('NM', 'New Mexico'), ('NY', 'New York'),
    ('NC', 'North Carolina'), ('ND', 'North Dakota'), ('OH', 'Ohio'), ('OK', 'Oklahoma'), ('OR', 'Oregon'),
    ('PA', 'Pennsylvania'), ('RI', 'Rhode Island'), ('SC', 'South Carolina'), ('SD', 'South Dakota'),
    ('TN', 'Tennessee'), ('TX', 'Texas'), ('UT', 'Utah'), ('VT', 'Vermont'), ('VA', 'Virginia'),
    ('WA', 'Washington'), ('WV', 'West Virginia'), ('WI', 'Wisconsin'), ('WY', 'Wyoming')
]
STATE_ABBR_MAP = {full: abbr for abbr, full in STATE_REFERENCE}


# ============================================================
# GEOPROVIDER SERVICE DASHBOARD (7 TAB ANALYSES)
# ============================================================

def render_geoprovider_service():
    st.markdown(
        """
        <div class="dashboard-header">
            <h1>🩺 CMS Medicare Provider & Payer Intelligence</h1>
            <p>Payment integrity, risk-adjusted provider tiering, chargemaster ratios, and site-of-service optimization.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_labels = [
        "🏥 Site-of-Service Shift",
        "📈 Markup Ratio",
        "🔬 Low-Value Care Screening",
        "💊 Drug vs Medical Spend",
        "🎯 Provider Risk Quadrants",
        "🌾 Rural & Equity Disparities",
        "🗺️ Regional Price Variation"
    ]
    tabs = st.tabs(tab_labels)

    # ================================================================================
    # TAB 1 — Site of Service Shift (Facility vs Non-Facility)
    # ================================================================================
    with tabs[0]:
        section_header(
            "Site-of-Service Optimization & Outpatient Arbitrage",
            "Select clinical specialty to audit Top 10 procedures and calculate 30% volume shift savings into non-facility settings."
        )
        
        try:
            specialties_sos = run_query(f"""
                SELECT DISTINCT Specialty_Type 
                FROM read_parquet('{PROVIDER_SERVICE_PARQUET}')
                WHERE Entity_Type = 'I' AND Specialty_Type IS NOT NULL
                ORDER BY Specialty_Type;
            """)['Specialty_Type'].tolist()
            
            default_spec_idx = specialties_sos.index("Ophthalmology") if "Ophthalmology" in specialties_sos else 0
            sel_sos_spec = st.selectbox("Select Clinical Specialty:", specialties_sos, index=default_spec_idx, key="sos_spec")
            
            query_sos = f"""
                WITH code_summary AS (
                    SELECT 
                        HCPCS_Code, HCPCS_Desc, Place_Of_Srvc_Code,
                        CAST(SUM(Total_Services) AS BIGINT) AS Srvc_Count,
                        AVG(Avg_Medicare_Allowed_Amt) AS Avg_Allowed_Amt
                    FROM read_parquet('{PROVIDER_SERVICE_PARQUET}')
                    WHERE Specialty_Type = '{sel_sos_spec}' AND Is_Part_B_Drug = 'N'
                    GROUP BY HCPCS_Code, HCPCS_Desc, Place_Of_Srvc_Code
                ),
                pivoted AS (
                    SELECT 
                        f.HCPCS_Code, f.HCPCS_Desc,
                        CAST(f.Srvc_Count AS BIGINT) AS Hospital_Cases, 
                        CAST(o.Srvc_Count AS BIGINT) AS Office_Cases,
                        f.Avg_Allowed_Amt AS Facility_Cost, 
                        o.Avg_Allowed_Amt AS Office_Cost,
                        (f.Avg_Allowed_Amt - o.Avg_Allowed_Amt) AS Unit_Cost_Gap,
                        (f.Srvc_Count * 0.30 * (f.Avg_Allowed_Amt - o.Avg_Allowed_Amt)) AS Projected_Savings
                    FROM (SELECT * FROM code_summary WHERE Place_Of_Srvc_Code = 'F') f
                    INNER JOIN (SELECT * FROM code_summary WHERE Place_Of_Srvc_Code = 'O') o
                        ON f.HCPCS_Code = o.HCPCS_Code
                    WHERE f.Avg_Allowed_Amt > o.Avg_Allowed_Amt AND f.Srvc_Count >= 50
                )
                SELECT * FROM pivoted ORDER BY Projected_Savings DESC LIMIT 10;
            """
            df_sos = run_query(query_sos)
            
            if df_sos.empty:
                st.warning("No dual-setting (Facility + Non-Facility) procedures found for this specialty.")
            else:
                df_sos = df_sos.sort_values(by="Projected_Savings", ascending=True)
                df_sos['Hospital_Cases'] = df_sos['Hospital_Cases'].astype(int)
                df_sos['Label'] = df_sos['HCPCS_Code'] + " - " + df_sos['HCPCS_Desc'].str.slice(0, 30) + "..."
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Top Opportunity Procedures", f"{len(df_sos)}")
                c2.metric("Hospital Caseload Available", f"{int(df_sos['Hospital_Cases'].sum()):,}")
                c3.metric("Projected 30% Shift Savings", f"${df_sos['Projected_Savings'].sum():,.2f}")
                
                fig_sos = make_subplots(
                    rows=1, cols=2, shared_yaxes=True,
                    subplot_titles=("Unit Cost Differential (Facility vs. Office)", "Projected Savings (30% Hospital Shift)"),
                    horizontal_spacing=0.12
                )
                
                fig_sos.add_trace(
                    go.Bar(
                        y=df_sos['Label'], x=df_sos['Facility_Cost'], name="Facility (Hospital/HOPD)",
                        orientation='h', marker=dict(color='#d9534f'),
                        customdata=df_sos[['Hospital_Cases', 'Unit_Cost_Gap']],
                        hovertemplate="<b>%{y}</b><br>Facility Cost: $%{x:,.2f}<br>Hospital Volume: %{customdata[0]:,}<br>Cost Gap: $%{customdata[1]:,.2f}<extra></extra>"
                    ), row=1, col=1
                )
                fig_sos.add_trace(
                    go.Bar(
                        y=df_sos['Label'], x=df_sos['Office_Cost'], name="Non-Facility (Office/ASC)",
                        orientation='h', marker=dict(color='#337ab7'),
                        hovertemplate="<b>%{y}</b><br>Office Cost: $%{x:,.2f}<extra></extra>"
                    ), row=1, col=1
                )
                fig_sos.add_trace(
                    go.Bar(
                        y=df_sos['Label'], x=df_sos['Projected_Savings'], name="Projected Savings ($)",
                        orientation='h', marker=dict(color='#2ca02c'),
                        customdata=df_sos[['Hospital_Cases']],
                        hovertemplate="<b>%{y}</b><br>Projected Savings: $%{x:,.2f}<br>Available Cases: %{customdata[0]:,}<extra></extra>"
                    ), row=1, col=2
                )
                
                fig_sos.update_layout(
                    barmode='group', template="plotly_dark", height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
                    margin=dict(l=10, r=10, t=50, b=80)
                )
                fig_sos.update_xaxes(title_text="Avg Medicare Allowed Amount ($)", row=1, col=1)
                fig_sos.update_xaxes(title_text="Annual Shift Savings ($)", row=1, col=2)
                
                st.plotly_chart(style_fig(fig_sos), use_container_width=True)
                with st.expander("View Underlying Specialty SOS Data"):
                    st.dataframe(df_sos, use_container_width=True)
        except Exception as e:
            st.error(f"Error in Site-of-Service Module: {e}")


    # ================================================================================
    # TAB 2 — Chargemaster Markup Multiples
    # ================================================================================
    with tabs[1]:
        section_header(
            "Commercial Chargemaster Markup Multiples & IDR Defense",
            "Audit submitted chargemaster billed charges vs. Medicare allowable rates across clinical specialties."
        )
        
        try:
            specialties_mk = run_query(f"""
                SELECT DISTINCT Specialty_Type 
                FROM read_parquet('{PROVIDER_SERVICE_PARQUET}')
                WHERE Entity_Type = 'I' AND Specialty_Type IS NOT NULL
                ORDER BY Specialty_Type;
            """)['Specialty_Type'].tolist()
            
            sel_mk_spec = st.selectbox("Select Specialty for Chargemaster Multiples:", specialties_mk, index=0, key="mk_spec")
            
            query_mk = f"""
                SELECT 
                    HCPCS_Code, HCPCS_Desc,
                    CAST(SUM(Total_Services) AS BIGINT) AS Services_Analyzed,
                    AVG(Avg_Submitted_Charge) AS Avg_Billed_Charge,
                    AVG(Avg_Medicare_Allowed_Amt) AS Avg_Allowed_Amt,
                    ROUND(AVG(Avg_Submitted_Charge) / NULLIF(AVG(Avg_Medicare_Allowed_Amt), 0), 2) AS Markup_Ratio
                FROM read_parquet('{PROVIDER_SERVICE_PARQUET}')
                WHERE Specialty_Type = '{sel_mk_spec}' AND Avg_Medicare_Allowed_Amt > 5.0
                GROUP BY HCPCS_Code, HCPCS_Desc
                HAVING SUM(Total_Services) >= 50
                ORDER BY Markup_Ratio DESC
                LIMIT 10;
            """
            df_mk = run_query(query_mk)
            
            if df_mk.empty:
                st.warning("No procedure data found meeting the volume criteria for this specialty.")
            else:
                df_mk = df_mk.sort_values(by="Markup_Ratio", ascending=True)
                df_mk['Services_Analyzed'] = df_mk['Services_Analyzed'].astype(int)
                df_mk['Label'] = df_mk['HCPCS_Code'] + " - " + df_mk['HCPCS_Desc'].str.slice(0, 30) + "..."
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Services Analyzed", f"{int(df_mk['Services_Analyzed'].sum()):,}")
                c2.metric("Specialty Peak Markup", f"{df_mk['Markup_Ratio'].max():.2f}x")
                c3.metric("Specialty Median Markup", f"{df_mk['Markup_Ratio'].median():.2f}x")
                
                fig_mk = make_subplots(
                    rows=1, cols=2, shared_yaxes=True,
                    subplot_titles=("Billed Submitted Charge vs. Medicare Allowed ($)", "Chargemaster Multiple Markup (Ratio)"),
                    horizontal_spacing=0.12
                )
                
                fig_mk.add_trace(
                    go.Bar(
                        y=df_mk['Label'], x=df_mk['Avg_Billed_Charge'], name="Billed Charge (Gross)",
                        orientation='h', marker=dict(color='#f59e0b'),
                        hovertemplate="Billed Charge: $%{x:,.2f}<extra></extra>"
                    ), row=1, col=1
                )
                fig_mk.add_trace(
                    go.Bar(
                        y=df_mk['Label'], x=df_mk['Avg_Allowed_Amt'], name="Medicare Allowed Rate",
                        orientation='h', marker=dict(color='#3b82f6'),
                        hovertemplate="Allowed Rate: $%{x:,.2f}<extra></extra>"
                    ), row=1, col=1
                )
                fig_mk.add_trace(
                    go.Bar(
                        y=df_mk['Label'], x=df_mk['Markup_Ratio'], name="Markup Multiple",
                        orientation='h', marker=dict(color='#ec4899'),
                        customdata=df_mk[['Services_Analyzed']],
                        hovertemplate="<b>%{y}</b><br>Markup: %{x:.2f}x<br>Services Analyzed: %{customdata[0]:,}<extra></extra>"
                    ), row=1, col=2
                )
                
                fig_mk.update_layout(
                    height=480, template="plotly_dark",
                    legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
                    margin=dict(l=10, r=10, t=50, b=80)
                )
                fig_mk.update_xaxes(title_text="Amount ($)", row=1, col=1)
                fig_mk.update_xaxes(title_text="Markup Multiple (Ratio)", row=1, col=2)
                
                st.plotly_chart(style_fig(fig_mk), use_container_width=True)
                with st.expander("View Underlying Markup Data"):
                    st.dataframe(df_mk, use_container_width=True)
        except Exception as e:
            st.error(f"Error in Markup Multiples Module: {e}")


    # ================================================================================
    # TAB 3 — Low-Value Care Diagnostic Screening
    # ================================================================================
    with tabs[2]:
        section_header(
            "Low-Value Care Screening & Diagnostic Overutilization Intensity",
            "Screen provider re-ordering intensity against the 1.0x parity baseline across clinical diagnostic categories."
        )
        
        CLINICAL_DOMAINS = {
            "Routine Laboratory & Blood Tests": {'36415': 'Routine Venipuncture', '80053': 'Comprehensive Metabolic Panel', '85025': 'Complete Blood Count', '80061': 'Lipid Panel', '84443': 'TSH Thyroid Test', '81003': 'Automated Urinalysis'},
            "Cardiology Diagnostics": {'93000': 'Electrocardiogram (EKG)', '93306': 'Echocardiography Complete', '93015': 'Cardiovascular Stress Test', '93224': 'Holter Monitor ECG', '93350': 'Echocardiography Stress', '93010': 'EKG Tracing Only'},
            "Spine & Advanced Imaging": {'72148': 'MRI Lumbar Spine w/o Dye', '70450': 'CT Head/Brain w/o Dye', '73721': 'MRI Knee Joint w/o Dye', '72141': 'MRI Cervical Spine w/o Dye', '74177': 'CT Abdomen & Pelvis w/ Dye', '71250': 'CT Thorax w/o Dye'},
            "Pulmonary & General X-Ray": {'71045': 'Chest X-Ray Single View', '71046': 'Chest X-Ray 2 Views', '94010': 'Routine Spirometry', '94060': 'Bronchodilation Spirometry', '73030': 'X-Ray Shoulder Complete', '73560': 'X-Ray Knee 1-2 Views'}
        }
        
        sel_domain = st.selectbox("Select Diagnostic Clinical Domain:", list(CLINICAL_DOMAINS.keys()), index=0)
        target_dict = CLINICAL_DOMAINS[sel_domain]
        codes_sql = ", ".join([f"'{c}'" for c in target_dict.keys()])
        
        query_lvc = f"""
            SELECT 
                NPI,
                Provider_Last_or_Org_Name || ', ' || Provider_First_Name AS Provider_Name,
                Specialty_Type, State, HCPCS_Code,
                Total_Beneficiaries, Total_Services, Services_Per_Beneficiary,
                ROUND(Total_Services * Avg_Medicare_Allowed_Amt, 2) AS Total_Reimbursement
            FROM read_parquet('{PROVIDER_SERVICE_PARQUET}')
            WHERE Entity_Type = 'I'
              AND HCPCS_Code IN ({codes_sql})
              AND Total_Beneficiaries >= 30
              AND Services_Per_Beneficiary BETWEEN 1.0 AND 8.5
            ORDER BY Total_Beneficiaries DESC
            LIMIT 25000;
        """
        try:
            df_lvc = run_query(query_lvc)
            if df_lvc.empty:
                st.warning("No diagnostic data found for selected clinical domain.")
            else:
                df_lvc['Service_Label'] = df_lvc['HCPCS_Code'].map(lambda c: f"{c}: {target_dict.get(str(c), '')}")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Providers Screened (Benes ≥ 30)", f"{df_lvc['NPI'].nunique():,d}")
                c2.metric("Aggregate Services / Beneficiary", f"{df_lvc['Services_Per_Beneficiary'].mean():.2f}x")
                c3.metric("Total Reimbursement Screened", f"${df_lvc['Total_Reimbursement'].sum():,.0f}")
                
                fig_lvc = px.box(
                    df_lvc, x="Service_Label", y="Services_Per_Beneficiary", color="Service_Label",
                    points="outliers", hover_name="Provider_Name",
                    hover_data={
                        "Service_Label": False, "Specialty_Type": True, "State": True,
                        "Total_Beneficiaries": ":,", "Total_Services": ":,",
                        "Services_Per_Beneficiary": ":.2f", "Total_Reimbursement": ":$,.2f"
                    },
                    labels={"Service_Label": "Diagnostic Code", "Services_Per_Beneficiary": "Services per Beneficiary (Times/Patient/Year)"},
                    template="plotly_dark", height=600
                )
                fig_lvc.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="1.0x Parity Baseline (Zero Repeat Testing)", annotation_position="bottom right")
                fig_lvc.update_layout(showlegend=False, margin=dict(l=10, r=10, t=30, b=70))
                st.plotly_chart(style_fig(fig_lvc), use_container_width=True)
                with st.expander("View Low-Value Care Outlier Table"):
                    st.dataframe(df_lvc, use_container_width=True)
        except Exception as e:
            st.error(f"Error in Low-Value Care Module: {e}")


    # ================================================================================
    # TAB 4 — Drug vs Medical Spend Decomposition
    # ================================================================================
    with tabs[3]:
        section_header(
            "Part B Drug vs. Medical Service Decomposition (CPT vs. J-Codes)",
            "Isolate clinic-administered specialty injectable drugs from physician labor and drill down into top procedural spend drivers."
        )
        
        query_tree = f"""
            SELECT 
                Specialty_Type,
                COUNT(DISTINCT NPI) AS Total_Providers,
                SUM(Med_Standardized_Amt) AS Total_Medical_Spend,
                SUM(Drug_Standardized_Amt) AS Total_Drug_Spend,
                SUM(Total_Medicare_Standardized_Amt) AS Total_Spend,
                ROUND(SUM(Drug_Standardized_Amt) * 100.0 / NULLIF(SUM(Total_Medicare_Standardized_Amt), 0), 2) AS Pct_Drug_Spend
            FROM read_parquet('{PROVIDER_PARQUET}')
            WHERE Entity_Type = 'I'
              AND Specialty_Type IS NOT NULL
              AND Specialty_Type NOT IN ('All Other Clinicians', 'Undefined')
              AND Total_Medicare_Standardized_Amt > 0
            GROUP BY Specialty_Type
            HAVING SUM(Total_Medicare_Standardized_Amt) >= 50000000
            ORDER BY Total_Spend DESC
            LIMIT 15;
        """
        try:
            df_tree = run_query(query_tree)
            tot_med = df_tree['Total_Medical_Spend'].sum()
            tot_drg = df_tree['Total_Drug_Spend'].sum()
            tot_all = df_tree['Total_Spend'].sum()
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Analyzed Part B Spend", f"${tot_all:,.0f}")
            c2.metric("Medical Services Share (CPT)", f"{(tot_med / tot_all * 100):.1f}%")
            c3.metric("Part B Drug Share (J-Codes)", f"{(tot_drg / tot_all * 100):.1f}%")
            
            treemap_labels = ["Total Part B Spend"]
            treemap_parents = [""]
            treemap_values = [tot_all]
            treemap_hover = ["Macro Summary"]
            treemap_colors = [50]
            
            for _, row in df_tree.iterrows():
                spec = row['Specialty_Type']
                treemap_labels.append(spec)
                treemap_parents.append("Total Part B Spend")
                treemap_values.append(row['Total_Spend'])
                treemap_hover.append(f"Total Spend: ${row['Total_Spend']:,.2f}<br>Drug Share: {row['Pct_Drug_Spend']:.1f}%")
                treemap_colors.append(row['Pct_Drug_Spend'])

                treemap_labels.append(f"{spec} - Medical")
                treemap_parents.append(spec)
                treemap_values.append(row['Total_Medical_Spend'])
                treemap_hover.append(f"Medical Spend: ${row['Total_Medical_Spend']:,.2f}")
                treemap_colors.append(0)

                treemap_labels.append(f"{spec} - Part B Drugs")
                treemap_parents.append(spec)
                treemap_values.append(row['Total_Drug_Spend'])
                treemap_hover.append(f"Drug Spend: ${row['Total_Drug_Spend']:,.2f}")
                treemap_colors.append(100)
                
            col_left, col_right = st.columns([3, 2])
            with col_left:
                fig_tr = go.Figure(
                    go.Treemap(
                        labels=treemap_labels, parents=treemap_parents, values=treemap_values,
                        branchvalues="total", hovertext=treemap_hover, hoverinfo="text+value",
                        marker=dict(
                            colors=treemap_colors,
                            colorscale=[[0.0, "#337ab7"], [1.0, "#d9534f"]],
                            cmid=50, line=dict(width=1.5, color="#0f1a2e")
                        ),
                        textfont=dict(color="#ffffff", size=13),
                        root_color="#1f2f4a"
                    )
                )
                fig_tr.update_layout(template="plotly_dark", height=540, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(style_fig(fig_tr), use_container_width=True)
                
            with col_right:
                st.subheader("🔍 Specialty HCPCS Drill-Down")
                drill_spec = st.selectbox("Select Specialty to Inspect Top Billed Codes:", df_tree['Specialty_Type'].tolist(), key="drill_tree_spec")
                
                query_drill = f"""
                    SELECT 
                        HCPCS_Code, HCPCS_Desc,
                        CASE WHEN Is_Part_B_Drug = 'Y' THEN 'Part B Drug (J-Code)' ELSE 'Medical Service (CPT)' END AS Category,
                        SUM(Total_Services) AS Total_Services,
                        ROUND(SUM(Total_Services * Avg_Medicare_Allowed_Amt), 2) AS Total_Allowed_Spend
                    FROM read_parquet('{PROVIDER_SERVICE_PARQUET}')
                    WHERE Specialty_Type = '{drill_spec}'
                    GROUP BY HCPCS_Code, HCPCS_Desc, Is_Part_B_Drug
                    ORDER BY Total_Allowed_Spend DESC
                    LIMIT 10;
                """
                df_drill = run_query(query_drill)
                st.dataframe(df_drill[['HCPCS_Code', 'HCPCS_Desc', 'Category', 'Total_Allowed_Spend']], use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error in Drug vs Medical Module: {e}")


    # ================================================================================
    # TAB 5 — Provider Performance Quadrants (Risk vs Spend)
    # ================================================================================
    with tabs[4]:
        section_header(
            "Physician Risk-Adjusted Cost vs. Acuity Profiling",
            "Dynamic risk-adjusted tiering benchmarking clinicians against specialty medians with Inefficiency Exposure metrics."
        )
        
        try:
            specialties_quad = run_query(f"""
                SELECT DISTINCT Specialty_Type 
                FROM read_parquet('{PROVIDER_PARQUET}')
                WHERE Entity_Type = 'I' AND Specialty_Type IS NOT NULL
                ORDER BY Specialty_Type;
            """)['Specialty_Type'].tolist()
            
            target_specialty = st.selectbox("Select Specialty for Provider Profiling:", specialties_quad, index=0, key="quad_spec")
            
            query_quad = f"""
                SELECT 
                    NPI,
                    Provider_Last_or_Org_Name || ', ' || Provider_First_Name AS Provider_Name,
                    Specialty_Type, City, State,
                    Total_Beneficiaries, Total_Services, Bene_Avg_Risk_Score,
                    Spend_Per_Beneficiary, Risk_Adjusted_Spend_Per_Bene,
                    Drug_Standardized_Amt, Med_Standardized_Amt
                FROM read_parquet('{PROVIDER_PARQUET}')
                WHERE Entity_Type = 'I'
                  AND Specialty_Type = '{target_specialty}'
                  AND Total_Beneficiaries >= 30
                  AND Bene_Avg_Risk_Score IS NOT NULL
                  AND Spend_Per_Beneficiary > 0
                ORDER BY Total_Beneficiaries DESC;
            """
            df_quad = run_query(query_quad)
            
            if df_quad.empty:
                st.warning("No providers found meeting criteria for this specialty.")
            else:
                median_risk = df_quad['Bene_Avg_Risk_Score'].median()
                median_spend = df_quad['Spend_Per_Beneficiary'].median()
                
                def assign_quadrant(row):
                    if row['Bene_Avg_Risk_Score'] >= median_risk and row['Spend_Per_Beneficiary'] <= median_spend:
                        return '🟢 High Value (High Risk, Low Cost)'
                    elif row['Bene_Avg_Risk_Score'] < median_risk and row['Spend_Per_Beneficiary'] <= median_spend:
                        return '🔵 Efficient (Low Risk, Low Cost)'
                    elif row['Bene_Avg_Risk_Score'] >= median_risk and row['Spend_Per_Beneficiary'] > median_spend:
                        return '🟠 Expected High Resource (High Risk, High Cost)'
                    else:
                        return '🔴 Inefficient Outlier (Low Risk, High Cost)'
                        
                df_quad['Quadrant'] = df_quad.apply(assign_quadrant, axis=1)
                
                ineff_df = df_quad[df_quad['Quadrant'] == '🔴 Inefficient Outlier (Low Risk, High Cost)']
                exposure = ((ineff_df['Spend_Per_Beneficiary'] - median_spend) * ineff_df['Total_Beneficiaries']).sum()
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Providers Profiled (Dynamic)", f"{len(df_quad):,d}")
                c2.metric("Specialty Median Risk", f"{median_risk:.2f}")
                c3.metric("Specialty Median Spend", f"${median_spend:,.2f}")
                c4.metric("Inefficiency Exposure ($)", f"${exposure:,.2f}")
                
                color_map = {
                    '🟢 High Value (High Risk, Low Cost)': '#2ca02c',
                    '🔵 Efficient (Low Risk, Low Cost)': '#1f77b4',
                    '🟠 Expected High Resource (High Risk, High Cost)': '#ff7f0e',
                    '🔴 Inefficient Outlier (Low Risk, High Cost)': '#d62728',
                }
                
                fig_quad = go.Figure()
                for quad, color in color_map.items():
                    sub_df = df_quad[df_quad['Quadrant'] == quad]
                    fig_quad.add_trace(
                        go.Scatter(
                            x=sub_df['Bene_Avg_Risk_Score'], y=sub_df['Spend_Per_Beneficiary'], mode='markers', name=quad,
                            marker=dict(
                                size=sub_df['Total_Beneficiaries'], sizemode='area',
                                sizeref=2.0 * max(df_quad['Total_Beneficiaries']) / (25.0 ** 2), sizemin=3,
                                color=color, opacity=0.7,
                            ),
                            text=sub_df['Provider_Name'] + " (" + sub_df['City'] + ", " + sub_df['State'] + ")",
                            customdata=sub_df[['NPI', 'Total_Beneficiaries', 'Risk_Adjusted_Spend_Per_Bene', 'Drug_Standardized_Amt', 'Med_Standardized_Amt']],
                            hovertemplate=(
                                "<b>%{text}</b><br>NPI: %{customdata[0]}<br><br>"
                                "Avg Patient Risk Score: %{x:.2f}<br>"
                                "Spend Per Patient: $%{y:,.2f}<br>"
                                "Risk-Adjusted Spend: $%{customdata[2]:,.2f}<br>"
                                "Patient Panel Size: %{customdata[1]:,}<extra></extra>"
                            ),
                        )
                    )
                fig_quad.add_vline(x=median_risk, line_dash="dash", line_color="gray", annotation_text=f"Median Risk: {median_risk:.2f}", annotation_position="top left")
                fig_quad.add_hline(y=median_spend, line_dash="dash", line_color="gray", annotation_text=f"Median Spend: ${median_spend:.2f}", annotation_position="bottom right")
                
                fig_quad.update_layout(
                    xaxis_title="Patient Panel Acuity (Beneficiary Avg Risk Score)",
                    yaxis_title="Annual Spend Per Beneficiary ($)",
                    template="plotly_dark", height=650,
                    legend=dict(title="Payer Network Tier", yanchor="top", y=0.98, xanchor="left", x=0.02),
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(style_fig(fig_quad), use_container_width=True)
                
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    green_data = df_quad[df_quad['Quadrant'].str.contains("High Value")].to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Tier-1 Preferred Network Roster (CSV)", green_data, "Tier1_High_Value_Providers.csv", "text/csv")
                with col_exp2:
                    red_data = df_quad[df_quad['Quadrant'].str.contains("Inefficient")].to_csv(index=False).encode('utf-8')
                    st.download_button("🚨 Download Audit Target Roster (CSV)", red_data, "Audit_Target_Inefficient_Providers.csv", "text/csv")
        except Exception as e:
            st.error(f"Error in Provider Quadrant Module: {e}")


    # ================================================================================
    # TAB 6 — Rural & Regional Healthcare Equity Disparities
    # ================================================================================
    with tabs[5]:
        section_header(
            "Rural vs. Urban Health Disparities & Territory Analytics",
            "Audit the Rural Healthcare Equity Paradox across RUCA tiers with optional state-level filtering."
        )
        
        try:
            available_states = run_query(f"""
                SELECT DISTINCT State FROM read_parquet('{PROVIDER_PARQUET}') WHERE State IS NOT NULL ORDER BY State;
            """)['State'].tolist()
            
            sel_geo_state = st.selectbox("Filter State / Territory (or View National):", ["National (All States)"] + available_states, index=0)
            
            state_filter_sql = f"AND State = '{sel_geo_state}'" if sel_geo_state != "National (All States)" else ""
            
            query_ruca = f"""
                SELECT 
                    NPI, Specialty_Type, State, RUCA_Code,
                    CASE 
                        WHEN RUCA_Code BETWEEN 1.0 AND 3.9 THEN '1. Metropolitan / Urban'
                        WHEN RUCA_Code BETWEEN 4.0 AND 6.9 THEN '2. Micropolitan / Large Rural'
                        WHEN RUCA_Code BETWEEN 7.0 AND 9.9 THEN '3. Small Rural Town'
                        WHEN RUCA_Code >= 10.0 THEN '4. Isolated Rural / Frontier'
                        ELSE 'Other / Unclassified'
                    END AS Geographic_Tier,
                    Total_Beneficiaries, Bene_Avg_Risk_Score, Spend_Per_Beneficiary,
                    ROUND(COALESCE(Dual_Eligible_Benes, 0) * 100.0 / NULLIF(Total_Beneficiaries, 0), 1) AS Pct_Dual_Eligible
                FROM read_parquet('{PROVIDER_PARQUET}')
                WHERE Entity_Type = 'I'
                  AND RUCA_Code IS NOT NULL
                  AND Total_Beneficiaries >= 30
                  AND Spend_Per_Beneficiary BETWEEN 10 AND 5000
                  AND Bene_Avg_Risk_Score IS NOT NULL
                  {state_filter_sql}
                ORDER BY Total_Beneficiaries DESC
                LIMIT 25000;
            """
            df_ruca = run_query(query_ruca)
            df_ruca = df_ruca[df_ruca['Geographic_Tier'] != 'Other / Unclassified']
            
            if df_ruca.empty:
                st.warning("No rural taxonomy data found for the selected state filter.")
            else:
                tier_order = ['1. Metropolitan / Urban', '2. Micropolitan / Large Rural', '3. Small Rural Town', '4. Isolated Rural / Frontier']
                colors = {'1. Metropolitan / Urban': '#1f77b4', '2. Micropolitan / Large Rural': '#33a02c', '3. Small Rural Town': '#ff7f0e', '4. Isolated Rural / Frontier': '#e31a1c'}
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Providers Analyzed", f"{len(df_ruca):,d}")
                c2.metric("Rural Tiers Represented", f"{df_ruca['Geographic_Tier'].nunique()}")
                c3.metric("Avg Dual-Eligible Vulnerability", f"{df_ruca['Pct_Dual_Eligible'].mean():.1f}%")
                
                fig_ruca = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=("1. Standardized Spend ($)", "2. Patient Panel Acuity (Risk)", "3. Dual-Eligible Vulnerability (%)"),
                    horizontal_spacing=0.08
                )
                for tier in tier_order:
                    sub = df_ruca[df_ruca['Geographic_Tier'] == tier]
                    fig_ruca.add_trace(go.Box(x=sub['Geographic_Tier'], y=sub['Spend_Per_Beneficiary'], name=tier, marker_color=colors[tier], showlegend=False), row=1, col=1)
                    fig_ruca.add_trace(go.Violin(x=sub['Geographic_Tier'], y=sub['Bene_Avg_Risk_Score'], name=tier, line_color=colors[tier], box_visible=True, showlegend=False), row=1, col=2)
                    fig_ruca.add_trace(go.Box(x=sub['Geographic_Tier'], y=sub['Pct_Dual_Eligible'], name=tier, marker_color=colors[tier], showlegend=False), row=1, col=3)
                    
                fig_ruca.update_layout(template="plotly_dark", height=600, margin=dict(l=10, r=10, t=50, b=90))
                st.plotly_chart(style_fig(fig_ruca), use_container_width=True)
                with st.expander("View Underlying RUCA Tier Data"):
                    st.dataframe(df_ruca, use_container_width=True)
        except Exception as e:
            st.error(f"Error in Rural Disparities Module: {e}")


    # ================================================================================
    # TAB 7 — State Price Variation Choropleth & Inspection Drawer
    # ================================================================================
    with tabs[6]:
        section_header(
            "Regional Price Variation & State Inspection Drawer",
            "Analyze state-level price disparities and inspect dominant health systems and urban/rural price variances."
        )
        
        try:
            top_codes_df = run_query(f"""
                SELECT HCPCS_Code, HCPCS_Desc, SUM(Total_Services) AS National_Services
                FROM read_parquet('{GEO_PARQUET}')
                WHERE Geo_Level = 'State'
                GROUP BY HCPCS_Code, HCPCS_Desc
                ORDER BY National_Services DESC
                LIMIT 50;
            """)
            code_options = [f"{r.HCPCS_Code} - {str(r.HCPCS_Desc)[:40]}" for r in top_codes_df.itertuples()]
            selected_label = st.selectbox("Select Procedure Code for State Heatmap:", code_options, index=0, key="reg_code_select")
            selected_code = selected_label.split(" - ")[0]
            
            query_map = f"""
                SELECT Geo_Desc AS State_Name, HCPCS_Code, HCPCS_Desc,
                       Total_Beneficiaries, Total_Services, Avg_Submitted_Charge,
                       Avg_Medicare_Allowed_Amt, Avg_Medicare_Standardized_Amt, Billed_To_Allowed_Markup_Ratio
                FROM read_parquet('{GEO_PARQUET}')
                WHERE Geo_Level = 'State' AND HCPCS_Code = '{selected_code}'
            """
            df_map = run_query(query_map)
            df_map['State_Abbr'] = df_map['State_Name'].map(STATE_ABBR_MAP)
            df_map = df_map.dropna(subset=['State_Abbr'])
            
            if df_map.empty:
                st.warning("No geographic records found for this procedure.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("States Reporting", f"{df_map['State_Abbr'].nunique()}")
                c2.metric("Avg Standardized Spend", f"${df_map['Avg_Medicare_Standardized_Amt'].mean():,.2f}")
                c3.metric("State Pricing Disparity Spread", f"${(df_map['Avg_Medicare_Standardized_Amt'].max() - df_map['Avg_Medicare_Standardized_Amt'].min()):,.2f}")
                
                col_map, col_drawer = st.columns([3, 2])
                with col_map:
                    fig_map = go.Figure(
                        go.Choropleth(
                            locations=df_map['State_Abbr'], z=df_map['Avg_Medicare_Standardized_Amt'], locationmode='USA-states',
                            colorscale='Blues', colorbar_title="Standardized ($)",
                            hovertemplate="<b>%{hovertext}</b><br>Allowed: $%{z:,.2f}<extra></extra>",
                            hovertext=df_map['State_Name']
                        )
                    )
                    fig_map.update_layout(
                        geo=dict(scope='usa', showlakes=True, lakecolor='#131d30', bgcolor='#131d30', landcolor='#1a2740', subunitcolor='#2a3a58'),
                        margin=dict(l=0, r=0, t=20, b=0), height=520
                    )
                    st.plotly_chart(style_fig(fig_map), use_container_width=True)
                    
                with col_drawer:
                    st.subheader("📋 Click-to-Inspect State Drawer")
                    sel_inspect_state = st.selectbox("Select State to Inspect:", sorted(df_map['State_Abbr'].unique()), key="inspect_state_select")
                    st_row = df_map[df_map['State_Abbr'] == sel_inspect_state].iloc[0]
                    
                    tot_state_spend = st_row['Total_Services'] * st_row['Avg_Medicare_Standardized_Amt']
                    st.markdown(f"**State:** `{st_row['State_Name']} ({st_row['State_Abbr']})`")
                    st.markdown(f"**Standardized Allowed Rate:** `${st_row['Avg_Medicare_Standardized_Amt']:,.2f}`")
                    st.markdown(f"**Total State Procedural Spend:** `${tot_state_spend:,.2f}`")
                    st.markdown(f"**Billed Markup Multiple:** `{st_row['Billed_To_Allowed_Markup_Ratio']:.2f}x`")
                    
                    query_top_provs = f"""
                        SELECT 
                            Provider_Last_or_Org_Name || ', ' || Provider_First_Name AS Provider_Name,
                            City, Total_Services,
                            ROUND(Total_Services * Avg_Medicare_Allowed_Amt, 2) AS Spend
                        FROM read_parquet('{PROVIDER_SERVICE_PARQUET}')
                        WHERE State = '{sel_inspect_state}' AND HCPCS_Code = '{selected_code}' AND Entity_Type = 'I'
                        ORDER BY Spend DESC LIMIT 5;
                    """
                    df_top_provs = run_query(query_top_provs)
                    st.markdown("**Top 5 Clinicians by Spend in State:**")
                    st.dataframe(df_top_provs, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error in State Price Variation Module: {e}")


# ============================================================
# FRONTEND SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("## 🏥 VBC Command Center")
    st.markdown(
        '<div class="vbc-divider"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="vbc-section">NAVIGATION</div>',
        unsafe_allow_html=True,
    )

    dashboard_area = st.radio(
        "Dashboard",
        [
            "ACO Overview",
            "Hospital Overview",
            "GeoProvider Service",
        ],
        label_visibility="collapsed",
        key="dashboard_area",
    )

    if dashboard_area == "ACO Overview":
        st.markdown(
            '<div class="vbc-divider"></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="vbc-section">ACO OVERVIEW</div>',
            unsafe_allow_html=True,
        )

        aco_page = st.radio(
            "ACO Navigation",
            [
                "App",
                "Portfolio Overview",
                "Contract Scorecard",
                "Driver Analysis",
                "Recommendations",
                "Performance Year",
            ],
            label_visibility="collapsed",
            key="aco_page",
        )

    elif dashboard_area == "GeoProvider Service":
        st.markdown(
            '<div class="vbc-divider"></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="vbc-section">DATA SOURCE</div>',
            unsafe_allow_html=True,
        )
        st.caption("CMS Medicare Physician & Other Practitioners Public Use Files")
        st.caption("9 Part B claims-based analyses across provider cost, utilization, markup ratios, and regional disparities.")
        aco_page = None

    else:
        st.markdown(
            '<div class="vbc-divider"></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="vbc-section">DATA SOURCE</div>',
            unsafe_allow_html=True,
        )
        st.caption("CMS Hospital Value-Based Purchasing (HVBP) Safety Scores")
        aco_page = None


# ============================================================
# ROUTING
# ============================================================

if dashboard_area == "ACO Overview":

    if aco_page == "App":
        render_aco_app(get_current_year())

    elif aco_page == "Portfolio Overview":
        render_portfolio_overview(get_current_year())

    elif aco_page == "Contract Scorecard":
        render_contract_scorecard(get_current_year())

    elif aco_page == "Driver Analysis":
        render_driver_analysis(get_current_year())

    elif aco_page == "Recommendations":
        render_recommendations(get_current_year())

    elif aco_page == "Performance Year":
        render_performance_year()

elif dashboard_area == "Hospital Overview":
    render_hospital_overview()

elif dashboard_area == "GeoProvider Service":
    render_geoprovider_service()