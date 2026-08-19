import duckdb
import pandas as pd
import plotly.express as px

CLEANED_PARQUET = "Medicare_By_Provider_and_Service_Cleaned.parquet"
con = duckdb.connect()

# Target high-volume diagnostic & imaging codes monitored in low-value care measures
TARGET_CODES = {
    '72148': 'MRI Lumbar Spine w/o Dye',
    '93000': 'Electrocardiogram (EKG/ECG)',
    '71045': 'Chest X-Ray Single View',
    '70450': 'CT Head/Brain w/o Dye',
    '93306': 'Echocardiography Complete',
    '36415': 'Routine Venipuncture (Blood Draw)'
}

code_filter = ", ".join([f"'{c}'" for c in TARGET_CODES.keys()])

query = f"""
    SELECT 
        NPI,
        Provider_Last_or_Org_Name || ', ' || Provider_First_Name AS Provider_Name,
        Specialty_Type,
        State,
        HCPCS_Code,
        HCPCS_Code || ' - ' || HCPCS_Desc AS Full_Service_Label,
        Total_Beneficiaries,
        Total_Services,
        Services_Per_Beneficiary,
        Avg_Medicare_Allowed_Amt,
        ROUND(Total_Services * Avg_Medicare_Allowed_Amt, 2) AS Total_Reimbursement
    FROM read_parquet('{CLEANED_PARQUET}')
    WHERE Entity_Type = 'I'
      AND HCPCS_Code IN ({code_filter})
      AND Total_Beneficiaries >= 30  -- Minimum panel size for statistical relevance
      AND Services_Per_Beneficiary BETWEEN 1.0 AND 8.0  -- Filter extreme anomalies
    ORDER BY Total_Beneficiaries DESC
    LIMIT 25000;
"""

df = con.execute(query).fetchdf()

# Add short descriptive labels for clean X-axis display
df['Short_Label'] = df['HCPCS_Code'].map(lambda c: f"{c}: {TARGET_CODES.get(str(c), '')}" if pd.notna(c) else "")

# Generate Interactive Box-and-Whisker Plot
fig = px.box(
    df,
    x="Short_Label",
    y="Services_Per_Beneficiary",
    color="Short_Label",
    points="outliers",
    hover_name="Provider_Name",
    hover_data={
        "Short_Label": False,
        "Specialty_Type": True,
        "State": True,
        "Total_Beneficiaries": ":,",
        "Total_Services": ":,",
        "Services_Per_Beneficiary": ":.2f",
        "Total_Reimbursement": ":$,.2f"
    },
    title="Physician Diagnostic Utilization Intensity (Low-Value Care Screening)",
    labels={
        "Short_Label": "Diagnostic / Procedure Code",
        "Services_Per_Beneficiary": "Services per Distinct Beneficiary (Ordering Frequency)"
    },
    template="plotly_white",
    height=700,
    width=1350
)

# Reference baseline: 1 test per patient per year
fig.add_hline(
    y=1.0, 
    line_dash="dash", 
    line_color="gray", 
    annotation_text="1.0x Parity Baseline (1 Test / Patient)", 
    annotation_position="bottom right"
)

fig.update_layout(
    showlegend=False,
    xaxis_title="Monitored Procedure Code",
    yaxis_title="Services Rendered per Beneficiary",
    margin=dict(l=50, r=50, t=80, b=60)
)

fig.show()
fig.write_html("low_value_care_overutilization_boxplot.html")
print("✅ Saved interactive boxplot to low_value_care_overutilization_boxplot.html")