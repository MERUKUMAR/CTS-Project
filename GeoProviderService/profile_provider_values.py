import duckdb
import pandas as pd

PROVIDER_FILE_PATH = "Medicare_Physician_by_Provider.csv"
con = duckdb.connect()

con.execute(f"""
    CREATE OR REPLACE VIEW raw_provider AS 
    SELECT * FROM read_csv_auto('{PROVIDER_FILE_PATH}', all_varchar=true, ignore_errors=true);
""")

print("=" * 90)
print("1. SUMMARY OF NUMERIC RANGES & OUTLIERS")
print("=" * 90)

numeric_profile = con.execute("""
    SELECT 
        'Bene_Avg_Risk_Scre' AS Column_Name,
        MIN(TRY_CAST(Bene_Avg_Risk_Scre AS DOUBLE)) AS Min_Val,
        ROUND(AVG(TRY_CAST(Bene_Avg_Risk_Scre AS DOUBLE)), 2) AS Mean_Val,
        MAX(TRY_CAST(Bene_Avg_Risk_Scre AS DOUBLE)) AS Max_Val,
        COUNT(CASE WHEN TRY_CAST(Bene_Avg_Risk_Scre AS DOUBLE) IS NULL THEN 1 END) AS Cast_Failures
    FROM raw_provider
    UNION ALL
    SELECT 
        'Bene_Avg_Age',
        MIN(TRY_CAST(Bene_Avg_Age AS DOUBLE)),
        ROUND(AVG(TRY_CAST(Bene_Avg_Age AS DOUBLE)), 2),
        MAX(TRY_CAST(Bene_Avg_Age AS DOUBLE)),
        COUNT(CASE WHEN TRY_CAST(Bene_Avg_Age AS DOUBLE) IS NULL THEN 1 END)
    FROM raw_provider
    UNION ALL
    SELECT 
        'Tot_Benes',
        MIN(TRY_CAST(NULLIF(REPLACE(Tot_Benes, '*', ''), '') AS BIGINT)),
        ROUND(AVG(TRY_CAST(NULLIF(REPLACE(Tot_Benes, '*', ''), '') AS BIGINT)), 2),
        MAX(TRY_CAST(NULLIF(REPLACE(Tot_Benes, '*', ''), '') AS BIGINT)),
        COUNT(CASE WHEN TRY_CAST(NULLIF(REPLACE(Tot_Benes, '*', ''), '') AS BIGINT) IS NULL THEN 1 END)
    FROM raw_provider
    UNION ALL
    SELECT 
        'Tot_Mdcr_Stdzd_Amt',
        MIN(TRY_CAST(REGEXP_REPLACE(Tot_Mdcr_Stdzd_Amt, '[$, ]', '', 'g') AS DOUBLE)),
        ROUND(AVG(TRY_CAST(REGEXP_REPLACE(Tot_Mdcr_Stdzd_Amt, '[$, ]', '', 'g') AS DOUBLE)), 2),
        MAX(TRY_CAST(REGEXP_REPLACE(Tot_Mdcr_Stdzd_Amt, '[$, ]', '', 'g') AS DOUBLE)),
        COUNT(CASE WHEN TRY_CAST(REGEXP_REPLACE(Tot_Mdcr_Stdzd_Amt, '[$, ]', '', 'g') AS DOUBLE) IS NULL THEN 1 END)
    FROM raw_provider;
""").fetchdf()

print(numeric_profile.to_string(index=False))

print("\n" + "=" * 90)
print("2. CHRONIC CONDITION PERCENTAGE RANGE CHECK (Are they 0-100 or 0-1.0?)")
print("=" * 90)

cc_profile = con.execute("""
    SELECT 
        MIN(TRY_CAST(Bene_CC_PH_Diabetes_V2_Pct AS DOUBLE)) AS Min_Diabetes_Pct,
        MAX(TRY_CAST(Bene_CC_PH_Diabetes_V2_Pct AS DOUBLE)) AS Max_Diabetes_Pct,
        MIN(TRY_CAST(Bene_CC_PH_CKD_V2_Pct AS DOUBLE)) AS Min_CKD_Pct,
        MAX(TRY_CAST(Bene_CC_PH_CKD_V2_Pct AS DOUBLE)) AS Max_CKD_Pct
    FROM raw_provider
    WHERE Bene_CC_PH_Diabetes_V2_Pct IS NOT NULL 
      AND TRIM(Bene_CC_PH_Diabetes_V2_Pct) != '*';
""").fetchdf()

print(cc_profile.to_string(index=False))

print("\n" + "=" * 90)
print("3. CATEGORICAL DISTINCT VALUE AUDIT")
print("=" * 90)

distinct_check = con.execute("""
    SELECT 
        'Rndrng_Prvdr_Ent_Cd' AS Category, 
        Rndrng_Prvdr_Ent_Cd AS Value, 
        COUNT(*) AS Row_Count 
    FROM raw_provider 
    GROUP BY Rndrng_Prvdr_Ent_Cd
    UNION ALL
    SELECT 
        'Rndrng_Prvdr_Mdcr_Prtcptg_Ind', 
        Rndrng_Prvdr_Mdcr_Prtcptg_Ind, 
        COUNT(*) 
    FROM raw_provider 
    GROUP BY Rndrng_Prvdr_Mdcr_Prtcptg_Ind;
""").fetchdf()

print(distinct_check.to_string(index=False))