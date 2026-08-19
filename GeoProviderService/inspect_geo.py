import duckdb
import pandas as pd

# Update this to your local file path (e.g., "Medicare_Geo_2023.csv" or ".parquet")
DATA_FILE_PATH = "Medicare_Geo_Service.csv"

# Connect to in-memory DuckDB
con = duckdb.connect()

# Register the table (DuckDB infers schema on the fly)
con.execute(f"""
    CREATE OR REPLACE VIEW raw_geo AS 
    SELECT * FROM read_csv_auto('{DATA_FILE_PATH}', all_varchar=true);
""")

print("=" * 80)
print("1. DATASET SHAPE & COLUMN TYPES")
print("=" * 80)
schema_info = con.execute("DESCRIBE raw_geo;").fetchdf()
count_result = con.execute("SELECT COUNT(*) FROM raw_geo;").fetchone()
total_rows = count_result[0] if count_result is not None and count_result[0] is not None else 0
print(f"Total Records: {total_rows:,}")
print(f"Total Columns: {len(schema_info)}")
print(schema_info[['column_name', 'column_type']])

print("\n" + "=" * 80)
print("2. DEEP MISSING / NULL / SUPPRESSED / EMPTY VALUE AUDIT")
print("=" * 80)

# Build dynamic SQL to count NULL, Blank (''), and CMS Asterisk ('*') per column
cols = schema_info['column_name'].tolist()
audit_queries = []

for c in cols:
    audit_queries.append(f"""
        SELECT 
            '{c}' AS Column_Name,
            COUNT(CASE WHEN "{c}" IS NULL THEN 1 END) AS Null_Count,
            COUNT(CASE WHEN TRIM("{c}") = '' THEN 1 END) AS Empty_String_Count,
            COUNT(CASE WHEN TRIM("{c}") = '*' THEN 1 END) AS Suppressed_Star_Count,
            COUNT(CASE WHEN "{c}" IS NULL OR TRIM("{c}") = '' OR TRIM("{c}") = '*' THEN 1 END) AS Total_Missing_Count,
            ROUND(COUNT(CASE WHEN "{c}" IS NULL OR TRIM("{c}") = '' OR TRIM("{c}") = '*' THEN 1 END) * 100.0 / {total_rows}, 2) AS Pct_Missing
        FROM raw_geo
    """)

full_audit_sql = " UNION ALL ".join(audit_queries)
audit_df = con.execute(full_audit_sql).fetchdf()
print(audit_df.to_string(index=False))

print("\n" + "=" * 80)
print("3. INAPPROPRIATE / ANOMALOUS VALUE SCAN")
print("=" * 80)

# Value validation queries
anomalies = con.execute("""
    SELECT
        -- Check Geographic levels
        COUNT(CASE WHEN Rndrng_Prvdr_Geo_Lvl NOT IN ('National', 'State') THEN 1 END) AS Invalid_Geo_Lvl_Count,
        
        -- Check Place of Service codes
        COUNT(CASE WHEN Place_Of_Srvc NOT IN ('F', 'O') THEN 1 END) AS Invalid_Place_Of_Srvc_Count,
        
        -- Check HCPCS Drug Indicator
        COUNT(CASE WHEN HCPCS_Drug_Ind NOT IN ('Y', 'N') AND HCPCS_Drug_Ind IS NOT NULL AND TRIM(HCPCS_Drug_Ind) != '' THEN 1 END) AS Invalid_Drug_Ind_Count,
        
        -- Check for negative financial numbers
        COUNT(CASE WHEN TRY_CAST(REGEXP_REPLACE(Avg_Mdcr_Alowd_Amt, '[$,]', '', 'g') AS DOUBLE) < 0 THEN 1 END) AS Negative_Allowed_Amt_Count,
        COUNT(CASE WHEN TRY_CAST(REGEXP_REPLACE(Avg_Sbmtd_Chrg, '[$,]', '', 'g') AS DOUBLE) < 0 THEN 1 END) AS Negative_Submitted_Chrg_Count
    FROM raw_geo;
""").fetchdf()

print(anomalies.to_string(index=False))