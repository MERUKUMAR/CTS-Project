import streamlit as st

import pandas as pd
import plotly.graph_objects as go
from data_utils import load_scorecard, render_year_selector, fmt_variance

# ML helpers
try:
    from ml.predict import predict_for_row, get_shap_explanation, models_available
    ML_AVAILABLE = models_available()
except Exception:
    ML_AVAILABLE = False

st.set_page_config(page_title="Driver Analysis", page_icon="🔍", layout="wide")
st.title("🔍 Driver Analysis")
st.caption("What's actually driving this ACO's cost variance vs. peers + ML risk explanation")

year = render_year_selector()
df = load_scorecard(year)
if df.empty:
    st.stop()

aco_name = st.selectbox("Select an ACO", sorted(df["ACO_Name"].unique()))
row = df[df["ACO_Name"] == aco_name].iloc[0]

# ------------------------------------------------------------------
# Existing cost variance chart
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
# NEW: ML Risk Explanation with SHAP
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
        # Nice readable names
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

        # Horizontal bar chart (red = increases risk, green = decreases risk)
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

        # Simple text summary
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
# Population context (existing)
# ------------------------------------------------------------------
st.divider()
st.subheader("Population context")
c1, c2 = st.columns(2)

perc_dual = row.get("Perc_Dual") if "Perc_Dual" in row.index else None
perc_lti = row.get("Perc_LTI") if "Perc_LTI" in row.index else None

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

st.page_link("pages/4_Recommendations.py", label="→ Get recommended actions for this ACO", icon="✅")