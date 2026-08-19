import duckdb
import pandas as pd
import plotly.graph_objects as go

CLEANED_PARQUET = "Medicare_By_Provider_Cleaned.parquet"
con = duckdb.connect()

# Query individual physicians in a target specialty (e.g., Internal Medicine)
# Filtering to Entity_Type = 'I' and Total_Beneficiaries >= 50 for statistical stability
TARGET_SPECIALTY = "Internal Medicine"

query = f"""
    SELECT 
        NPI,
        Provider_Last_or_Org_Name || ', ' || Provider_First_Name AS Provider_Name,
        Specialty_Type,
        City,
        State,
        Total_Beneficiaries,
        Total_Services,
        Bene_Avg_Risk_Score,
        Spend_Per_Beneficiary,
        Risk_Adjusted_Spend_Per_Bene,
        Drug_Standardized_Amt,
        Med_Standardized_Amt
    FROM read_parquet('{CLEANED_PARQUET}')
    WHERE Entity_Type = 'I'
      AND Specialty_Type = '{TARGET_SPECIALTY}'
      AND Total_Beneficiaries >= 50
      AND Bene_Avg_Risk_Score IS NOT NULL
      AND Spend_Per_Beneficiary > 0
    ORDER BY Total_Beneficiaries DESC
    LIMIT 2000;  -- High-density sample for fluid browser interactivity
"""

df = con.execute(query).fetchdf()

# Calculate specialty medians to define quadrant baselines
median_risk = df['Bene_Avg_Risk_Score'].median()
median_spend = df['Spend_Per_Beneficiary'].median()

# Assign Value-Based Quadrant Categories
def assign_quadrant(row):
    if row['Bene_Avg_Risk_Score'] >= median_risk and row['Spend_Per_Beneficiary'] <= median_spend:
        return 'High Value (High Risk, Low Cost)'
    elif row['Bene_Avg_Risk_Score'] < median_risk and row['Spend_Per_Beneficiary'] <= median_spend:
        return 'Efficient (Low Risk, Low Cost)'
    elif row['Bene_Avg_Risk_Score'] >= median_risk and row['Spend_Per_Beneficiary'] > median_spend:
        return 'Expected High Resource (High Risk, High Cost)'
    else:
        return 'Inefficient Outlier (Low Risk, High Cost)'

df['Quadrant'] = df.apply(assign_quadrant, axis=1)

# Color mapping for payer tiering
color_map = {
    'High Value (High Risk, Low Cost)': '#2ca02c',          # Green
    'Efficient (Low Risk, Low Cost)': '#1f77b4',             # Blue
    'Expected High Resource (High Risk, High Cost)': '#ff7f0e', # Orange
    'Inefficient Outlier (Low Risk, High Cost)': '#d62728'   # Red
}

fig = go.Figure()

for quad, color in color_map.items():
    sub_df = df[df['Quadrant'] == quad]
    fig.add_trace(
        go.Scatter(
            x=sub_df['Bene_Avg_Risk_Score'],
            y=sub_df['Spend_Per_Beneficiary'],
            mode='markers',
            name=quad,
            marker=dict(
                size=sub_df['Total_Beneficiaries'],
                sizemode='area',
                sizeref=2.0 * max(df['Total_Beneficiaries']) / (25.0 ** 2),
                sizemin=3,
                color=color,
                opacity=0.7
            ),
            text=sub_df['Provider_Name'] + " (" + sub_df['City'] + ", " + sub_df['State'] + ")",
            customdata=sub_df[['NPI', 'Total_Beneficiaries', 'Risk_Adjusted_Spend_Per_Bene', 'Drug_Standardized_Amt', 'Med_Standardized_Amt']],
            hovertemplate=(
                "<b>%{text}</b><br>" +
                "NPI: %{customdata[0]}<br><br>" +
                "Avg Patient Risk Score: %{x:.2f}<br>" +
                "Spend Per Patient: $%{y:,.2f}<br>" +
                "Risk-Adjusted Spend: $%{customdata[2]:,.2f}<br>" +
                "Patient Panel Size: %{customdata[1]:,}<br>" +
                "Part B Drug Spend: $%{customdata[3]:,.2f}<br>" +
                "Medical Spend: $%{customdata[4]:,.2f}<br>" +
                "<extra></extra>"
            )
        )
    )

# Add Quadrant Reference Lines
fig.add_vline(x=median_risk, line_dash="dash", line_color="gray",
              annotation_text=f"Median Risk: {median_risk:.2f}", annotation_position="top left")
fig.add_hline(y=median_spend, line_dash="dash", line_color="gray",
              annotation_text=f"Median Spend: ${median_spend:,.2f}", annotation_position="bottom right")

# Layout Polish
fig.update_layout(
    title_text=f"Physician Risk-Adjusted Cost vs. Acuity Profiling ({TARGET_SPECIALTY})",
    xaxis_title="Patient Panel Acuity (Beneficiary Average Risk Score)",
    yaxis_title="Annual Standardized Spend Per Beneficiary ($)",
    template="plotly_white",
    height=750,
    width=1350,
    legend=dict(title="Payer Network Tier", yanchor="top", y=0.98, xanchor="left", x=0.02)
)

fig.show()
fig.write_html("provider_risk_spend_quadrant.html")
print("✅ Saved interactive scatter plot to provider_risk_spend_quadrant.html")