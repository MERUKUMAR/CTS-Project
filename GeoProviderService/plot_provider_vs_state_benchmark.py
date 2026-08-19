import duckdb
import pandas as pd
import plotly.express as px

GEO_PARQUET = "Medicare_Geo_Service_Cleaned.parquet"
PROVIDER_PARQUET = "Medicare_By_Provider_Cleaned.parquet"
PROVIDER_SERVICE_PARQUET = "Medicare_By_Provider_and_Service_Cleaned.parquet"

con = duckdb.connect()

# Comprehensive 50-State + Territory mapping table
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
    ('WA', 'Washington'), ('WV', 'West Virginia'), ('WI', 'Wisconsin'), ('WY', 'Wyoming'),
    ('PR', 'Puerto Rico'), ('VI', 'Virgin Islands'), ('GU', 'Guam'), ('AS', 'American Samoa')
]

# Register mapping in DuckDB
con.register('state_ref', pd.DataFrame(STATE_REFERENCE, columns=['State_Abbr', 'State_Full_Name']))

TARGET_HCPCS = '99214'  # Established Patient Visit (Moderate Complexity)
TARGET_POS = 'O'        # 'O' = Non-Facility (Office Setting)

query = f"""
    WITH state_benchmarks AS (
        -- 1. Extract State-Level Benchmark Spend using Full State Name
        SELECT 
            TRIM(Geo_Desc) AS State_Full_Name,
            HCPCS_Code,
            Place_Of_Srvc_Code,
            Avg_Medicare_Standardized_Amt AS State_Benchmark_Spend
        FROM read_parquet('{GEO_PARQUET}')
        WHERE Geo_Level = 'State'
          AND HCPCS_Code = '{TARGET_HCPCS}'
          AND Place_Of_Srvc_Code = '{TARGET_POS}'
    ),
    provider_proc AS (
        -- 2. Extract Procedure-Level Spend and map Postal Abbreviation to Full Name
        SELECT 
            ps.NPI,
            ref.State_Full_Name,
            ps.State AS State_Abbr,
            ps.HCPCS_Code,
            ps.Place_Of_Srvc_Code,
            ps.Total_Services AS Clinician_Proc_Services,
            ps.Avg_Medicare_Standardized_Amt AS Clinician_Avg_Spend
        FROM read_parquet('{PROVIDER_SERVICE_PARQUET}') ps
        INNER JOIN state_ref ref 
            ON ps.State = ref.State_Abbr
        WHERE ps.Entity_Type = 'I'
          AND ps.HCPCS_Code = '{TARGET_HCPCS}'
          AND ps.Place_Of_Srvc_Code = '{TARGET_POS}'
          AND ps.Total_Services >= 20  -- Meaningful procedure volume threshold
    ),
    provider_profile AS (
        -- 3. Extract Clinician Panel Risk Score and Beneficiary Volume
        SELECT 
            NPI,
            Provider_Last_or_Org_Name || ', ' || Provider_First_Name AS Provider_Name,
            Specialty_Type,
            City,
            Total_Beneficiaries AS Total_Patient_Panel,
            Bene_Avg_Risk_Score
        FROM read_parquet('{PROVIDER_PARQUET}')
        WHERE Entity_Type = 'I'
          AND Bene_Avg_Risk_Score IS NOT NULL
    )
    -- 4. Execute Multi-Table Join across all 3 CMS Datasets
    SELECT 
        p.NPI,
        p.Provider_Name,
        p.Specialty_Type,
        p.City,
        pp.State_Abbr AS State,
        p.Total_Patient_Panel,
        p.Bene_Avg_Risk_Score,
        pp.Clinician_Proc_Services,
        pp.Clinician_Avg_Spend,
        sb.State_Benchmark_Spend,
        
        -- Variance from State Peer Average ($)
        ROUND(pp.Clinician_Avg_Spend - sb.State_Benchmark_Spend, 2) AS Spend_Variance_From_State,
        ROUND((pp.Clinician_Avg_Spend - sb.State_Benchmark_Spend) * 100.0 / NULLIF(sb.State_Benchmark_Spend, 0), 1) AS Pct_Variance_From_State,
        
        -- Categorization into Payer Actionable Tiers
        CASE 
            WHEN pp.Clinician_Avg_Spend > sb.State_Benchmark_Spend AND p.Bene_Avg_Risk_Score < 1.0 THEN 'Higher Cost / Low Risk (Outlier)'
            WHEN pp.Clinician_Avg_Spend <= sb.State_Benchmark_Spend AND p.Bene_Avg_Risk_Score >= 1.0 THEN 'High Value (Low Cost / High Risk)'
            WHEN pp.Clinician_Avg_Spend <= sb.State_Benchmark_Spend THEN 'Cost Efficient'
            ELSE 'Expected High Spend (High Risk)'
        END AS Performance_Tier

    FROM provider_proc pp
    INNER JOIN provider_profile p 
        ON pp.NPI = p.NPI
    INNER JOIN state_benchmarks sb 
        ON pp.State_Full_Name = sb.State_Full_Name 
       AND pp.HCPCS_Code = sb.HCPCS_Code 
       AND pp.Place_Of_Srvc_Code = sb.Place_Of_Srvc_Code
    ORDER BY p.Total_Patient_Panel DESC
    LIMIT 3000;
"""

df = con.execute(query).fetchdf()
print(f"✅ Successfully matched and joined {len(df):,} provider records across all three datasets.")

# Build Interactive Plotly Bubble Chart
fig = px.scatter(
    df,
    x="Spend_Variance_From_State",
    y="Bene_Avg_Risk_Score",
    size="Total_Patient_Panel",
    color="Performance_Tier",
    hover_name="Provider_Name",
    hover_data={
        "NPI": True,
        "Specialty_Type": True,
        "City": True,
        "State": True,
        "Clinician_Avg_Spend": ":$,.2f",
        "State_Benchmark_Spend": ":$,.2f",
        "Spend_Variance_From_State": ":$,.2f",
        "Pct_Variance_From_State": ":+.1f%",
        "Bene_Avg_Risk_Score": ":.2f",
        "Total_Patient_Panel": ":,",
        "Performance_Tier": False
    },
    color_discrete_map={
        'High Value (Low Cost / High Risk)': '#2ca02c',       # Green
        'Cost Efficient': '#1f77b4',                          # Blue
        'Expected High Spend (High Risk)': '#ff7f0e',         # Orange
        'Higher Cost / Low Risk (Outlier)': '#d62728'         # Red
    },
    title=f"Provider Cost Variance vs. State Regional Benchmark (HCPCS: {TARGET_HCPCS} - Office Visits)",
    labels={
        "Spend_Variance_From_State": "Dollar Variance vs. State Peer Average ($)",
        "Bene_Avg_Risk_Score": "Beneficiary Average Risk Score (Illness Acuity)",
        "Performance_Tier": "Payer Network Tier"
    },
    template="plotly_white",
    height=750,
    width=1400
)

# Reference zero-line: State Benchmark Parity ($0)
fig.add_vline(
    x=0, 
    line_dash="dash", 
    line_color="black", 
    annotation_text="State Average Baseline ($0 Variance)", 
    annotation_position="top left"
)

# Reference risk-line: National Average Illness Acuity (1.0)
fig.add_hline(
    y=1.0, 
    line_dash="dash", 
    line_color="gray", 
    annotation_text="Medicare National Average Acuity (Risk = 1.0)", 
    annotation_position="bottom right"
)

fig.update_layout(
    legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
    margin=dict(l=50, r=50, t=80, b=50)
)

fig.show()
fig.write_html("provider_vs_state_benchmark_bubble.html")
print("✅ Saved populated bubble chart to provider_vs_state_benchmark_bubble.html")