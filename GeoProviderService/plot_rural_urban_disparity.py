import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROVIDER_PARQUET = "Medicare_By_Provider_Cleaned.parquet"
con = duckdb.connect()

# 1. Aggregate and Categorize Providers by Geographic RUCA Tier
query = f"""
    SELECT 
        NPI,
        Provider_Last_or_Org_Name || ', ' || Provider_First_Name AS Provider_Name,
        Specialty_Type,
        State,
        RUCA_Code,
        RUCA_Desc,
        
        -- Categorize into 4 Standard Demographic Tiers
        CASE 
            WHEN RUCA_Code BETWEEN 1.0 AND 3.9 THEN '1. Metropolitan / Urban'
            WHEN RUCA_Code BETWEEN 4.0 AND 6.9 THEN '2. Micropolitan / Large Rural'
            WHEN RUCA_Code BETWEEN 7.0 AND 9.9 THEN '3. Small Rural Town'
            WHEN RUCA_Code >= 10.0 THEN '4. Isolated Rural / Frontier'
            ELSE 'Other / Unclassified'
        END AS Geographic_Tier,

        Total_Beneficiaries,
        Bene_Avg_Risk_Score,
        Bene_Avg_Age,
        Spend_Per_Beneficiary,
        Risk_Adjusted_Spend_Per_Bene,
        
        -- Social Vulnerability Metric: % Dual Eligible (Medicare + Medicaid)
        ROUND(COALESCE(Dual_Eligible_Benes, 0) * 100.0 / NULLIF(Total_Beneficiaries, 0), 1) AS Pct_Dual_Eligible

    FROM read_parquet('{PROVIDER_PARQUET}')
    WHERE Entity_Type = 'I'
      AND RUCA_Code IS NOT NULL
      AND Total_Beneficiaries >= 30
      AND Spend_Per_Beneficiary BETWEEN 10 AND 5000  -- Filter extreme outliers for clear distribution
      AND Bene_Avg_Risk_Score IS NOT NULL
    ORDER BY Total_Beneficiaries DESC
    LIMIT 20000;
"""

df = con.execute(query).fetchdf()

# Filter out unclassified
df = df[df['Geographic_Tier'] != 'Other / Unclassified']

# 2. Build Multi-Panel Subplot (Violin Distributions)
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=(
        "Standardized Spend per Beneficiary ($)",
        "Patient Panel Risk Score (Clinical Acuity)",
        "Dual-Eligible Vulnerability (%)"
    ),
    horizontal_spacing=0.08
)

tier_order = [
    '1. Metropolitan / Urban', 
    '2. Micropolitan / Large Rural', 
    '3. Small Rural Town', 
    '4. Isolated Rural / Frontier'
]

colors = {
    '1. Metropolitan / Urban': '#1f77b4',
    '2. Micropolitan / Large Rural': '#33a02c',
    '3. Small Rural Town': '#ff7f0e',
    '4. Isolated Rural / Frontier': '#e31a1c'
}

# Panel 1: Spend Distribution
for tier in tier_order:
    sub = df[df['Geographic_Tier'] == tier]
    fig.add_trace(
        go.Violin(
            x=sub['Geographic_Tier'],
            y=sub['Spend_Per_Beneficiary'],
            name=tier,
            box_visible=True,
            meanline_visible=True,
            line_color=colors[tier],
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Spend: $%{y:,.2f}<extra></extra>"
        ),
        row=1, col=1
    )

# Panel 2: Clinical Risk Score Distribution
for tier in tier_order:
    sub = df[df['Geographic_Tier'] == tier]
    fig.add_trace(
        go.Violin(
            x=sub['Geographic_Tier'],
            y=sub['Bene_Avg_Risk_Score'],
            name=tier,
            box_visible=True,
            meanline_visible=True,
            line_color=colors[tier],
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Risk Score: %{y:.2f}<extra></extra>"
        ),
        row=1, col=2
    )

# Panel 3: Dual-Eligible Rate Distribution
for tier in tier_order:
    sub = df[df['Geographic_Tier'] == tier]
    fig.add_trace(
        go.Violin(
            x=sub['Geographic_Tier'],
            y=sub['Pct_Dual_Eligible'],
            name=tier,
            box_visible=True,
            meanline_visible=True,
            line_color=colors[tier],
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Dual Eligible: %{y:.1f}%<extra></extra>"
        ),
        row=1, col=3
    )

# Layout Formatting
fig.update_layout(
    title_text="Rural vs. Urban Health Disparities: Spending, Clinical Acuity & Social Vulnerability (CMS Data)",
    template="plotly_white",
    height=700,
    width=1500,
    margin=dict(l=50, r=50, t=90, b=120)
)

fig.update_xaxes(tickangle=30, row=1, col=1)
fig.update_xaxes(tickangle=30, row=1, col=2)
fig.update_xaxes(tickangle=30, row=1, col=3)

fig.update_yaxes(title_text="Spend per Beneficiary ($)", row=1, col=1)
fig.update_yaxes(title_text="Average Risk Score", row=1, col=2)
fig.update_yaxes(title_text="Dual-Eligible Beneficiaries (%)", row=1, col=3)

fig.show()
fig.write_html("rural_urban_disparity_violin.html")
print("✅ Saved interactive violin charts to rural_urban_disparity_violin.html")