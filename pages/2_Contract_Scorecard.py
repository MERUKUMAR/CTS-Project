import streamlit as st

import pandas as pd
import plotly.graph_objects as go
from data_utils import load_scorecard, render_year_selector, fmt_dollars, fmt_pct, fmt_variance, OUTCOME_COLORS, TIER_COLORS, load_aco_trend
st.set_page_config(page_title="Contract Scorecard", page_icon="📊", layout="wide")
st.title("📊 Contract Scorecard")
st.caption("Drill into a single ACO's cost, quality, and utilization performance")
year = render_year_selector()
df = load_scorecard(year)
if df.empty:
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
u1, u2, u3, u4 , u5= st.columns(5)

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

trend_df = load_aco_trend(row["ACO_ID"])

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
    
st.page_link("pages/3_Driver_Analysis.py", label="→ See full driver analysis for this ACO", icon="🔍")