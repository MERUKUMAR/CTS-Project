import duckdb
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CLEANED_PARQUET = "Medicare_By_Provider_Cleaned.parquet"
con = duckdb.connect()

# 1. Aggregate Total Medical vs Drug Spend across Top Specialties
query = f"""
    SELECT 
        Specialty_Type,
        COUNT(DISTINCT NPI) AS Total_Providers,
        SUM(Total_Beneficiaries) AS Total_Beneficiaries,
        SUM(Med_Standardized_Amt) AS Total_Medical_Spend,
        SUM(Drug_Standardized_Amt) AS Total_Drug_Spend,
        SUM(Total_Medicare_Standardized_Amt) AS Total_Spend,
        ROUND(SUM(Drug_Standardized_Amt) * 100.0 / NULLIF(SUM(Total_Medicare_Standardized_Amt), 0), 2) AS Pct_Drug_Spend
    FROM read_parquet('{CLEANED_PARQUET}')
    WHERE Entity_Type = 'I'
      AND Specialty_Type IS NOT NULL
      AND Specialty_Type NOT IN ('All Other Clinicians', 'Undefined')
      AND Total_Medicare_Standardized_Amt > 0
    GROUP BY Specialty_Type
    HAVING SUM(Total_Medicare_Standardized_Amt) >= 50000000 -- Focus on specialties with >= $50M spend
    ORDER BY Total_Spend DESC
    LIMIT 15;
"""

df = con.execute(query).fetchdf()

# Calculate Macro Totals for the Donut Chart
total_macro_med = df['Total_Medical_Spend'].sum()
total_macro_drug = df['Total_Drug_Spend'].sum()

# 2. Build Nested Data for Treemap
# Levels: Root -> Specialty -> Service Type (Medical vs Part B Drug)
treemap_labels = ["Total Part B Spend"]
treemap_parents = [""]
treemap_values = [df['Total_Spend'].sum()]
treemap_colors = ["#f0f0f0"]
treemap_hover = ["Macro Overview"]

for _, row in df.iterrows():
    spec = row['Specialty_Type']
    
    # Add Specialty Branch
    treemap_labels.append(spec)
    treemap_parents.append("Total Part B Spend")
    treemap_values.append(row['Total_Spend'])
    treemap_colors.append("#2b5c8f")
    treemap_hover.append(f"Total Spend: ${row['Total_Spend']:,.2f}<br>Drug Share: {row['Pct_Drug_Spend']:.1f}%")

    # Add Medical Leaf
    treemap_labels.append(f"{spec} - Medical")
    treemap_parents.append(spec)
    treemap_values.append(row['Total_Medical_Spend'])
    treemap_colors.append("#337ab7")
    treemap_hover.append(f"Medical Spend: ${row['Total_Medical_Spend']:,.2f}")

    # Add Drug Leaf
    treemap_labels.append(f"{spec} - Part B Drugs")
    treemap_parents.append(spec)
    treemap_values.append(row['Total_Drug_Spend'])
    treemap_colors.append("#d9534f")
    treemap_hover.append(f"Drug Spend: ${row['Total_Drug_Spend']:,.2f}")

# 3. Create Dual Subplot: (1) Donut, (2) Treemap
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.3, 0.7],
    specs=[[{"type": "domain"}, {"type": "treemap"}]],
    subplot_titles=("National Part B Spend Split", "Specialty Decomposition (Medical vs. Drugs)")
)

# Panel 1: Donut Chart
fig.add_trace(
    go.Pie(
        labels=["Medical Services", "Part B Drugs"],
        values=[total_macro_med, total_macro_drug],
        hole=0.55,
        marker=dict(colors=["#337ab7", "#d9534f"]),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Total Spend: $%{value:,.2f}<br>Share: %{percent}<extra></extra>"
    ),
    row=1, col=1
)

# Panel 2: Treemap
fig.add_trace(
    go.Treemap(
        labels=treemap_labels,
        parents=treemap_parents,
        values=treemap_values,
        branchvalues="total",
        hovertext=treemap_hover,
        hoverinfo="text+value",
        marker=dict(colorscale='Blues')
    ),
    row=1, col=2
)

# Layout Styling
fig.update_layout(
    title_text="Part B Drug vs. Medical Service Spend Decomposition (Top 15 Specialties)",
    template="plotly_white",
    height=650,
    width=1400,
    margin=dict(l=30, r=30, t=80, b=30)
)

fig.show()
fig.write_html("part_b_drug_vs_medical_decomposition.html")
print("✅ Saved visualization to part_b_drug_vs_medical_decomposition.html")