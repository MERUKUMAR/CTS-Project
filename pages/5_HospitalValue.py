import streamlit as st

import pandas as pd
import numpy as np
import plotly.express as px

# --- PAGE CONFIGURATION ---
#st.set_page_config(page_title="Hospital Safety KPI Dashboard", layout="wide")
st.title("🏥 Hospital Value-Based Purchasing (HVBP) Safety Dashboard")
st.markdown("Visualizing Comprehensive Safety KPIs for America's top reporting hospitals.")

# --- 1. DATA LOADING & PROCESSING CACHE ---
# This ensures the filtering and math only run once, making the dashboard lightning-fast.
@st.cache_data
def load_and_process_data():
    # Pandas will automatically handle the empty commas as NaN values
    from pathlib import Path
    DATA_PATH = Path(__file__).parent.parent / "data" / "calculated_hvbp_data.csv"
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

# Load the finalized data
df = load_and_process_data()

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
    st.subheader("1. Comprehensive Hospital Safety Score (CHSS) Distribution")
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

# Group by state for the choropleth map
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

# Create the Bar Chart (Histogram)
fig_improve = px.histogram(
    df, x='Improvement Rate', nbins=30, 
    color_discrete_sequence=['#27AE60'], # A nice green color for 'Improvement'
    labels={'Improvement Rate': 'Avg Improvement Score (Max 1.0)'}
)
# Rename the Y-axis so it's easy for non-technical people to read
fig_improve.update_layout(bargap=0.1, yaxis_title="Count of Hospitals")
st.plotly_chart(fig_improve, use_container_width=True)

# --- 5. INTERACTIVE DATA TABLE (COMMAND CENTER) ---
st.markdown("---")
st.subheader("🔎 Command Center: Contract Target Finder")
st.markdown("Filter the national database to identify specific high-effort hospitals for Value-Based Care contracts or safe hospitals for Network Tiering.")

# Create two columns for the filter tools
filter_col1, filter_col2 = st.columns(2)

with filter_col1:
    # 🌟 NEW: Dropdown to select the search mode
    search_mode = st.selectbox(
        "1. Select Search Goal:", 
        options=[
            "Search based on Improvement Score (Shared-Savings)", 
            "Search based on Safety Score (Network Tiering)"
        ]
    )

with filter_col2:
    # 🌟 NEW: Dynamic slider that changes based on the dropdown choice
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
            "2. Minimum Safety Score (CHSS):", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.50, 
            step=0.05
        )

# Apply the logic based on what the user selected
if search_mode == "Search based on Improvement Score (Shared-Savings)":
    filtered_df = df[df['Improvement Rate'] >= min_score]
    # Sort so the highest improvement scores are at the top
    sorted_df = filtered_df.sort_values(by='Improvement Rate', ascending=False)
else:
    filtered_df = df[df['CHSS'] >= min_score]
    # Sort so the highest safety scores are at the top
    sorted_df = filtered_df.sort_values(by='CHSS', ascending=False)

st.success(f"🎯 **Target List Generated:** Found {len(sorted_df)} individual hospitals nationwide matching your criteria.")

# Display the filtered data
display_cols = ['Facility Name', 'State', 'Improvement Rate', 'CHSS', 'SEP-1 Measure Score', 'Combined SSI Measure Score']
st.dataframe(sorted_df[display_cols].round(3), use_container_width=True)