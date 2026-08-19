import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CLEANED_PARQUET = "Medicare_Geo_Service_Cleaned.parquet"
con = duckdb.connect()

# 1. Aggregate National Level Metrics per HCPCS Code
query = f"""
    SELECT 
        HCPCS_Code,
        HCPCS_Desc,
        Is_Part_B_Drug,
        SUM(Total_Services) AS Total_National_Services,
        SUM(Total_Beneficiaries) AS Total_National_Benes,
        AVG(Avg_Submitted_Charge) AS Avg_National_Billed_Charge,
        AVG(Avg_Medicare_Allowed_Amt) AS Avg_National_Allowed_Amt,
        ROUND(AVG(Avg_Submitted_Charge) / NULLIF(AVG(Avg_Medicare_Allowed_Amt), 0), 2) AS National_Markup_Ratio
    FROM read_parquet('{CLEANED_PARQUET}')
    WHERE Geo_Level = 'National' 
      AND Avg_Medicare_Allowed_Amt > 5.0  -- Exclude trivial micro-reimbursements
    GROUP BY HCPCS_Code, HCPCS_Desc, Is_Part_B_Drug
    HAVING SUM(Total_Services) >= 50000  -- Focus on meaningful high-volume services
    ORDER BY National_Markup_Ratio DESC;
"""

df = con.execute(query).fetchdf()

# 2. Extract Top 20 Procedures by Markup Ratio for the Bar Chart
top_20_markup = df.head(20).sort_values(by="National_Markup_Ratio", ascending=True)

# Truncate descriptions for clean axis labels
top_20_markup['Label'] = top_20_markup['HCPCS_Code'] + " - " + top_20_markup['HCPCS_Desc'].str.slice(0, 35) + "..."

# 3. Create Subplot: (1) Top Bar Chart, (2) Scatter Distribution
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Top 20 Procedures by Billed Markup Ratio (≥50k Volume)",
        "Markup Ratio vs. Medicare Allowed Amount"
    ),
    horizontal_spacing=0.15,
    specs=[[{"type": "xy"}, {"type": "xy"}]]
)

# -------------------------------------------------------------
# Panel 1: Ranked Horizontal Bar Chart
# -------------------------------------------------------------
fig.add_trace(
    go.Bar(
        x=top_20_markup['National_Markup_Ratio'],
        y=top_20_markup['Label'],
        orientation='h',
        marker=dict(
            color=top_20_markup['National_Markup_Ratio'],
            colorscale='YlOrRd',
            showscale=False
        ),
        hovertemplate=(
            "<b>%{y}</b><br>" +
            "Markup Ratio: %{x:.2f}x<br>" +
            "<extra></extra>"
        )
    ),
    row=1, col=1
)

# -------------------------------------------------------------
# Panel 2: Scatter Plot (Markup vs Allowed Cost)
# -------------------------------------------------------------
fig.add_trace(
    go.Scatter(
        x=df['Avg_National_Allowed_Amt'],
        y=df['National_Markup_Ratio'],
        mode='markers',
        marker=dict(
            size=df['Total_National_Services'],
            sizemode='area',
            sizeref=2.0 * max(df['Total_National_Services']) / (40.0 ** 2),
            sizemin=4,
            color=df['National_Markup_Ratio'],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Markup (x)", x=1.02)
        ),
        text=df['HCPCS_Code'] + ": " + df['HCPCS_Desc'],
        hovertemplate=(
            "<b>%{text}</b><br><br>" +
            "Avg Allowed Amt: $%{x:,.2f}<br>" +
            "Markup Ratio: %{y:.2f}x<br>" +
            "Total Services: %{marker.size:,}<br>" +
            "<extra></extra>"
        )
    ),
    row=1, col=2
)

# Add Reference Line at 1.0x (No Markup / Parity Baseline)
fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=2,
              annotation_text="1.0x (Parity Baseline)", annotation_position="bottom right")

# 4. Layout Polish
fig.update_layout(
    title_text="Commercial Billed Charge Markup Analysis (CMS Geography & Service Data)",
    showlegend=False,
    template="plotly_white",
    height=700,
    width=1400,
    margin=dict(l=50, r=50, t=90, b=50)
)

fig.update_xaxes(title_text="Billed Markup Ratio (Submitted / Allowed)", row=1, col=1)
fig.update_xaxes(title_text="Avg Medicare Allowed Amount ($)", type="log", row=1, col=2)
fig.update_yaxes(title_text="Markup Ratio (x)", row=1, col=2)

# Display in browser & save static HTML
fig.show()
fig.write_html("commercial_markup_ratio_analysis.html")
print("✅ Saved interactive markup charts to commercial_markup_ratio_analysis.html")