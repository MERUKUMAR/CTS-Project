import duckdb
import pandas as pd

# Path to your local CSV or Parquet file for Provider and Service
PROVIDER_SERVICE_FILE = "Medicare_Physician_by_Provider_and_Service.csv"

con = duckdb.connect()

print("Registering Provider and Service dataset in DuckDB...")
con.execute(f"""
    CREATE OR REPLACE VIEW raw_p_service AS 
    SELECT * FROM read_csv_auto('{PROVIDER_SERVICE_FILE}', all_varchar=true, ignore_errors=true);
""")

print("=" * 90)
print("1. DATASET VOLUME & GRAIN OVERVIEW")
print("=" * 90)
count_result = con.execute("SELECT COUNT(*) FROM raw_p_service;").fetchone()
total_rows = count_result[0] if count_result is not None else 0
schema_info = con.execute("DESCRIBE raw_p_service;").fetchdf()
print(f"Total Rows: {total_rows:,}")
print(f"Total Columns: {len(schema_info)}")

print("\n" + "=" * 90)
print("2. MISSING & SUPPRESSED VALUE AUDIT (KEY COLUMNS)")
print("=" * 90)

key_cols = [
    'Rndrng_NPI', 'Rndrng_Prvdr_Type', 'HCPCS_Cd', 'HCPCS_Desc', 
    'Place_Of_Srvc', 'Tot_Benes', 'Tot_Srvcs', 'Tot_Bene_Day_Srvcs',
    'Avg_Sbmtd_Chrg', 'Avg_Mdcr_Alowd_Amt', 'Avg_Mdcr_Pymt_Amt', 'Avg_Mdcr_Stdzd_Amt'
]

audit_queries = []
for c in key_cols:
    audit_queries.append(f"""
        SELECT 
            '{c}' AS Column_Name,
            COUNT(CASE WHEN "{c}" IS NULL THEN 1 END) AS Null_Count,
            COUNT(CASE WHEN TRIM("{c}") = '' THEN 1 END) AS Empty_Count,
            COUNT(CASE WHEN TRIM("{c}") = '*' THEN 1 END) AS Suppressed_Star_Count,
            ROUND(COUNT(CASE WHEN "{c}" IS NULL OR TRIM("{c}") = '' OR TRIM("{c}") = '*' THEN 1 END) * 100.0 / {total_rows}, 2) AS Pct_Missing
        FROM raw_p_service
    """)

audit_df = con.execute(" UNION ALL ".join(audit_queries)).fetchdf()
print(audit_df.to_string(index=False))

print("\n" + "=" * 90)
print("3. CLINICAL & PLACE OF SERVICE INTEGRITY CHECKS")
print("=" * 90)

integrity_df = con.execute("""
    SELECT
        COUNT(CASE WHEN Place_Of_Srvc NOT IN ('F', 'O') THEN 1 END) AS Invalid_Place_Of_Service,
        COUNT(CASE WHEN HCPCS_Drug_Ind NOT IN ('Y', 'N') AND HCPCS_Drug_Ind IS NOT NULL AND TRIM(HCPCS_Drug_Ind) != '' THEN 1 END) AS Invalid_Drug_Flag,
        COUNT(CASE WHEN TRY_CAST(REGEXP_REPLACE(Avg_Mdcr_Alowd_Amt, '[$,]', '', 'g') AS DOUBLE) < 0 THEN 1 END) AS Negative_Allowed_Amounts,
        COUNT(CASE WHEN TRY_CAST(REGEXP_REPLACE(Avg_Sbmtd_Chrg, '[$,]', '', 'g') AS DOUBLE) < 0 THEN 1 END) AS Negative_Submitted_Charges
    FROM raw_p_service;
""").fetchdf()
print(integrity_df.to_string(index=False))