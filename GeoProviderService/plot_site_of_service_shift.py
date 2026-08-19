import duckdb
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CLEANED_PARQUET = "Medicare_By_Provider_and_Service_Cleaned.parquet"
con = duckdb.connect()

# Query top procedure codes performed in BOTH Facility (F) and Non-Facility (O)
query = f"""
    WITH code_summary AS (
        SELECT 
            HCPCS_Code,
            HCPCS_Desc,
            Place_Of_Srvc_Code,
            SUM(Total_Services) AS Srvc_Count,
            SUM(Total_Services * Avg_Medicare_Allowed_Amt) AS Total_Spend,
            AVG(Avg_Medicare_Allowed_Amt) AS Avg_Allowed_Amt
        FROM read_parquet('{CLEANED_PARQUET}')
        WHERE Is_Part_B_Drug = 'N'
        GROUP BY HCPCS_Code, HCPCS_Desc, Place_Of_Srvc_Code
    ),
    pivoted AS (
        SELECT 
            f.HCPCS_Code,
            f.HCPCS_Desc,
            f.Srvc_Count AS Facility_Services,
            o.Srvc_Count AS Office_Services,
            f.Avg_Allowed_Amt AS Facility_Avg_Cost,
            o.Avg_Allowed_Amt AS Office_Avg_Cost,
            (f.Avg_Allowed_Amt - o.Avg_Allowed_Amt) AS Unit_Cost_Diff,
            -- Potential Savings if 30% of Facility volume shifts to Office/ASC
            (f.Srvc_Count * 0.30) * (f.Avg_Allowed_Amt - o.Avg_Allowed_Amt) AS Potential_30Pct_Shift_Savings
        FROM (SELECT * FROM code_summary WHERE Place_Of_Srvc_Code = 'F') f
        INNER JOIN (SELECT * FROM code_summary WHERE Place_Of_Srvc_Code = 'O') o
            ON f.HCPCS_Code = o.HCPCS_Code
        WHERE f.Avg_Allowed_Amt > o.Avg_Allowed_Amt
          AND f.Srvc_Count >= 25000
    )
    SELECT * 
    FROM pivoted 
    ORDER BY Potential_30Pct_Shift_Savings DESC
    LIMIT 12;
"""

df = con.execute(query).fetchdf()
df = df.sort_values(by="Potential_30Pct_Shift_Savings", ascending=True)
df['Label'] = df['HCPCS_Code'] + " - " + df['HCPCS_Desc'].str.slice(0, 30) + "..."

# Dual Subplot: (1) Cost Differential per Service, (2) Potential Realizable Savings
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Cost Differential per Case (Facility vs. Office)",
        "Potential Savings (Shifting 30% Facility Volume to Office)"
    ),
    horizontal_spacing=0.15
)

# Panel 1: Price Gap Bar Chart
fig.add_trace(
    go.Bar(
        y=df['Label'],
        x=df['Facility_Avg_Cost'],
        name='Facility (Hospital/HOPD)',
        orientation='h',
        marker=dict(color='#d9534f'),
        hovertemplate="Facility Cost: $%{x:,.2f}<extra></extra>"
    ),
    row=1, col=1
)

fig.add_trace(
    go.Bar(
        y=df['Label'],
        x=df['Office_Avg_Cost'],
        name='Non-Facility (Office/ASC)',
        orientation='h',
        marker=dict(color='#337ab7'),
        hovertemplate="Office Cost: $%{x:,.2f}<extra></extra>"
    ),
    row=1, col=1
)

# Panel 2: Potential Dollar Savings
fig.add_trace(
    go.Bar(
        y=df['Label'],
        x=df['Potential_30Pct_Shift_Savings'],
        name='Projected Savings ($)',
        orientation='h',
        marker=dict(color='#2ca02c'),
        hovertemplate="Projected Savings: $%{x:,.2f}<extra></extra>"
    ),
    row=1, col=2
)

fig.update_layout(
    barmode='group',
    title_text="Site-of-Service Optimization & Outpatient Arbitrage (Top 12 Opportunity Codes)",
    template="plotly_white",
    height=650,
    width=1450,
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    margin=dict(l=50, r=50, t=80, b=100)
)

fig.update_xaxes(title_text="Avg Medicare Allowed Amount ($)", row=1, col=1)
fig.update_xaxes(title_text="Potential Annual Dollar Savings ($)", row=1, col=2)

fig.show()
fig.write_html("site_of_service_shift_opportunity.html")
print("✅ Saved visualization to site_of_service_shift_opportunity.html")