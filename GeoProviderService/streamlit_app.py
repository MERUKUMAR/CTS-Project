"""
CMS Medicare Provider Analytics Dashboard (Enhanced Payer Intelligence Edition)
================================================================================
A unified Streamlit dashboard integrating risk-adjusted profiling, chargemaster
markup defense, low-value care outlier detection, and site-of-service arbitrage.

Expected parquet files in working directory:
    - Medicare_By_Provider_Cleaned.parquet
    - Medicare_By_Provider_and_Service_Cleaned.parquet
    - Medicare_Geo_Service_Cleaned.parquet
"""

import duckdb
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------------
# PAGE CONFIG & GLOBAL STYLING
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="CMS Medicare Payer Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROVIDER_PARQUET = "Medicare_By_Provider_Cleaned.parquet"
PROVIDER_SERVICE_PARQUET = "Medicare_By_Provider_and_Service_Cleaned.parquet"
GEO_PARQUET = "Medicare_Geo_Service_Cleaned.parquet"

CUSTOM_CSS = """
<style>
    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", sans-serif;
    }
    .dashboard-header {
        padding: 1.5rem 2rem;
        border-radius: 14px;
        background: linear-gradient(120deg, #0f2545 0%, #1f4e8c 55%, #2b7bd1 100%);
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .dashboard-header h1 {
        font-size: 1.85rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #ffffff;
    }
    .dashboard-header p {
        font-size: 0.92rem;
        color: #cfe0f7;
        margin-bottom: 0;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(160deg, #1a2740 0%, #131d30 100%);
        border: 1px solid #2a3a58;
        border-radius: 12px;
        padding: 0.8rem 1rem;
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
    div[data-baseweb="tab-list"] {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 8px !important;
        background-color: #131d30;
        border: 1px solid #2a3a58;
        border-radius: 12px;
        padding: 8px;
    }
    button[data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.85rem;
        padding: 8px 14px !important;
        color: #b6c2d6;
        border-radius: 8px !important;
    }
    button[aria-selected="true"] {
        background-color: #1f2f4a !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(43, 123, 209, 0.25);
    }
    div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
        display: none !important;
    }
    .section-card {
        background: linear-gradient(160deg, #1a2740 0%, #131d30 100%);
        border-radius: 14px;
        padding: 1.2rem 1.5rem 0.5rem 1.5rem;
        border: 1px solid #2a3a58;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        margin-bottom: 1.1rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.15rem;
    }
    .section-sub {
        font-size: 0.82rem;
        color: #9fb0c9;
        margin-bottom: 0.7rem;
    }
    section[data-testid="stSidebar"] {
        background-color: #0f2545;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------------
# DATA CONNECTION & STYLING UTILS
# --------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------
# HEADER & NAVIGATION
# --------------------------------------------------------------------------------
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
