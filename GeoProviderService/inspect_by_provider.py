import duckdb
import pandas as pd

# Update with your local By Provider file path (.csv, .parquet, etc.)
PROVIDER_FILE_PATH = "Medicare_Physician_by_Provider.csv"

con = duckdb.connect()

# Register raw view
con.execute(f"""
    CREATE OR REPLACE VIEW raw_provider AS 
    SELECT * FROM read_csv_auto('{PROVIDER_FILE_PATH}', all_varchar=true, ignore_errors=true);
""")

print("=" * 90)
print("1. DATASET VOLUME & COLUMN COUNT")
print("=" * 90)
row = con.execute("SELECT COUNT(*) FROM raw_provider;").fetchone()
total_rows = int(row[0]) if row and row[0] is not None else 0
schema_info = con.execute("DESCRIBE raw_provider;").fetchdf()
print(f"Total Provider Records: {total_rows:,}")
print(f"Total Columns: {len(schema_info)}")

print("\n" + "=" * 90)
print("2. MISSING / NULL / SUPPRESSED VALUE AUDIT (SAMPLE ACROSS KEY GROUPS)")
print("=" * 90)

cols = schema_info['column_name'].tolist()
audit_queries = []

for c in cols:
    audit_queries.append(f"""
        SELECT 
            '{c}' AS Column_Name,
            COUNT(CASE WHEN "{c}" IS NULL THEN 1 END) AS Null_Count,
            COUNT(CASE WHEN TRIM("{c}") = '' THEN 1 END) AS Empty_String_Count,
            COUNT(CASE WHEN TRIM("{c}") IN ('*', '#', '.') THEN 1 END) AS Suppressed_Symbol_Count,
            COUNT(CASE WHEN "{c}" IS NULL OR TRIM("{c}") = '' OR TRIM("{c}") IN ('*', '#', '.') THEN 1 END) AS Total_Missing,
            ROUND(COUNT(CASE WHEN "{c}" IS NULL OR TRIM("{c}") = '' OR TRIM("{c}") IN ('*', '#', '.') THEN 1 END) * 100.0 / {total_rows}, 2) AS Pct_Missing
        FROM raw_provider
    """)

audit_df = con.execute(" UNION ALL ".join(audit_queries)).fetchdf()

# Display high-priority subsets for review
print("\n--- Demographics & Risk Score Missingness ---")
demo_cols = ['Rndrng_NPI', 'Rndrng_Prvdr_Type', 'Bene_Avg_Risk_Scre', 'Bene_Avg_Age', 'Bene_Dual_Cnt', 'Bene_Feml_Cnt']
print(audit_df[audit_df['Column_Name'].isin(demo_cols)].to_string(index=False))

print("\n--- Chronic Condition Prevalence Missingness ---")
cc_cols = ['Bene_CC_PH_Diabetes_V2_Pct', 'Bene_CC_PH_CKD_V2_Pct', 'Bene_CC_PH_HF_NonIHD_V2_Pct', 'Bene_CC_BH_Depress_V1_Pct']
print(audit_df[audit_df['Column_Name'].isin(cc_cols)].to_string(index=False))

print("\n--- Financial & Spending Missingness ---")
fin_cols = ['Tot_Sbmtd_Chrg', 'Tot_Mdcr_Alowd_Amt', 'Tot_Mdcr_Stdzd_Amt', 'Drug_Tot_Mdcr_Stdzd_Amt', 'Med_Tot_Mdcr_Stdzd_Amt']
print(audit_df[audit_df['Column_Name'].isin(fin_cols)].to_string(index=False))

print("\n" + "=" * 90)
print("3. CLINICAL & FINANCIAL INTEGRITY CHECKS")
print("=" * 90)
integrity_df = con.execute("""
    SELECT
        COUNT(CASE WHEN TRY_CAST(Bene_Avg_Risk_Scre AS DOUBLE) <= 0 THEN 1 END) AS Invalid_Risk_Scores,
        COUNT(CASE WHEN TRY_CAST(REGEXP_REPLACE(Tot_Mdcr_Stdzd_Amt, '[$,]', '', 'g') AS DOUBLE) < 0 THEN 1 END) AS Negative_Standardized_Spend,
        COUNT(CASE WHEN TRY_CAST(TRIM(Tot_Benes) AS BIGINT) = 0 THEN 1 END) AS Zero_Bene_Count,
        COUNT(CASE WHEN Rndrng_Prvdr_Ent_Cd NOT IN ('I', 'O') THEN 1 END) AS Invalid_Entity_Code
    FROM raw_provider;
""").fetchdf()
print(integrity_df.to_string(index=False))