import duckdb

DATA_FILE_PATH = "Medicare_Geo_Service.csv"
OUTPUT_PARQUET = "Medicare_Geo_Service_Cleaned.parquet"

con = duckdb.connect()

print("Cleaning and transforming Medicare Geography & Service dataset...")

con.execute(f"""
    CREATE OR REPLACE TABLE geo_service_cleaned AS
    SELECT
        -- 1. Standardize Geographic Hierarchy & Impute Nulls
        TRIM(Rndrng_Prvdr_Geo_Lvl) AS Geo_Level,
        
        -- Handle 13,468 National nulls by setting them to 'US'
        CASE 
            WHEN TRIM(Rndrng_Prvdr_Geo_Lvl) = 'National' THEN 'US'
            ELSE COALESCE(TRIM(Rndrng_Prvdr_Geo_Cd), 'UNKNOWN')
        END AS Geo_Code,
        
        -- Handle 5 missing descriptions
        COALESCE(NULLIF(TRIM(Rndrng_Prvdr_Geo_Desc), ''), 'Unknown / Other') AS Geo_Desc,

        -- 2. Service & Procedure Identifiers
        TRIM(HCPCS_Cd) AS HCPCS_Code,
        TRIM(HCPCS_Desc) AS HCPCS_Desc,
        COALESCE(NULLIF(TRIM(HCPCS_Drug_Ind), ''), 'N') AS Is_Part_B_Drug,

        -- 3. Place of Service Categorization
        TRIM(Place_Of_Srvc) AS Place_Of_Srvc_Code,
        CASE 
            WHEN TRIM(Place_Of_Srvc) = 'F' THEN 'Facility (Hospital/HOPD)'
            WHEN TRIM(Place_Of_Srvc) = 'O' THEN 'Non-Facility (Office/ASC)'
            ELSE 'Other'
        END AS Place_Of_Srvc_Desc,

        -- 4. Cast Volume Metrics to Integers / Floats
        TRY_CAST(TRIM(Tot_Rndrng_Prvdrs) AS BIGINT) AS Total_Rendering_Providers,
        TRY_CAST(TRIM(Tot_Benes) AS BIGINT) AS Total_Beneficiaries,
        TRY_CAST(TRIM(Tot_Srvcs) AS DOUBLE) AS Total_Services,
        TRY_CAST(TRIM(Tot_Bene_Day_Srvcs) AS DOUBLE) AS Total_Bene_Day_Services,

        -- 5. Clean & Cast Financial Metrics (Stripping $, commas)
        TRY_CAST(REGEXP_REPLACE(Avg_Sbmtd_Chrg, '[$, ]', '', 'g') AS DOUBLE) AS Avg_Submitted_Charge,
        TRY_CAST(REGEXP_REPLACE(Avg_Mdcr_Alowd_Amt, '[$, ]', '', 'g') AS DOUBLE) AS Avg_Medicare_Allowed_Amt,
        TRY_CAST(REGEXP_REPLACE(Avg_Mdcr_Pymt_Amt, '[$, ]', '', 'g') AS DOUBLE) AS Avg_Medicare_Payment_Amt,
        TRY_CAST(REGEXP_REPLACE(Avg_Mdcr_Stdzd_Amt, '[$, ]', '', 'g') AS DOUBLE) AS Avg_Medicare_Standardized_Amt,

        -- 6. Essential Derived Payer Metrics for VBC Visualizations
        -- Commercial Charge Markup Ratio (Gross Billed / Medicare Allowed)
        ROUND(
            TRY_CAST(REGEXP_REPLACE(Avg_Sbmtd_Chrg, '[$, ]', '', 'g') AS DOUBLE) / 
            NULLIF(TRY_CAST(REGEXP_REPLACE(Avg_Mdcr_Alowd_Amt, '[$, ]', '', 'g') AS DOUBLE), 0), 
            2
        ) AS Billed_To_Allowed_Markup_Ratio,

        -- Utilization Intensity per Patient
        ROUND(
            TRY_CAST(TRIM(Tot_Srvcs) AS DOUBLE) / 
            NULLIF(TRY_CAST(TRIM(Tot_Benes) AS BIGINT), 0), 
            2
        ) AS Services_Per_Beneficiary

    FROM read_csv_auto('{DATA_FILE_PATH}', all_varchar=true)
    WHERE HCPCS_Cd IS NOT NULL AND TRIM(HCPCS_Cd) != '';
""")

# Verify zero nulls remain in the final table
post_check = con.execute("""
    SELECT 
        COUNT(*) AS Cleaned_Rows,
        COUNT(CASE WHEN Geo_Code IS NULL THEN 1 END) AS Null_Geo_Codes,
        COUNT(CASE WHEN Geo_Desc IS NULL THEN 1 END) AS Null_Geo_Descs,
        COUNT(CASE WHEN Avg_Medicare_Standardized_Amt IS NULL THEN 1 END) AS Null_Standardized_Amt
    FROM geo_service_cleaned;
""").fetchdf()

print("\n--- Post-Clean Verification ---")
print(post_check.to_string(index=False))

# Export clean Parquet
con.execute(f"COPY geo_service_cleaned TO '{OUTPUT_PARQUET}' (FORMAT PARQUET);")
print(f"\n Cleaned dataset exported: {OUTPUT_PARQUET}")