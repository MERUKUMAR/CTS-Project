import streamlit as st

import pandas as pd
from datetime import date
from data_utils import load_scorecard, render_year_selector, fmt_dollars, fmt_pct, fmt_variance

try:
    from ml.predict import predict_for_row, models_available, get_model_metrics
    ML_AVAILABLE = models_available()
except Exception:
    ML_AVAILABLE = False

st.set_page_config(page_title="Recommendations", page_icon="✅", layout="wide")
st.title("✅ Recommendations & Meeting Brief")
st.caption("Rule-based + ML-powered recommended actions, and a one-page brief for the provider review meeting")

year = render_year_selector()
df = load_scorecard(year)
if df.empty:
    st.stop()

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