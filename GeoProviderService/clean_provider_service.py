import duckdb

DATA_FILE_PATH = "Medicare_Physician_by_Provider_and_Service.csv"
OUTPUT_PARQUET = "Medicare_By_Provider_and_Service_Cleaned.parquet"

con = duckdb.connect()
print("Cleaning and transforming 'By Provider and Service' dataset...")

con.execute(f"""
    CREATE OR REPLACE TABLE provider_service_cleaned AS
    SELECT
        -- 1. Provider & Specialty Identifiers
        TRIM(Rndrng_NPI) AS NPI,
        TRIM(Rndrng_Prvdr_Last_Org_Name) AS Provider_Last_or_Org_Name,
        COALESCE(NULLIF(TRIM(Rndrng_Prvdr_First_Name), ''), '') AS Provider_First_Name,
        COALESCE(NULLIF(TRIM(Rndrng_Prvdr_Crdntls), ''), '') AS Provider_Credentials,
        TRIM(Rndrng_Prvdr_Ent_Cd) AS Entity_Type, -- 'I' = Individual, 'O' = Organization
        TRIM(Rndrng_Prvdr_Type) AS Specialty_Type,
        COALESCE(NULLIF(TRIM(Rndrng_Prvdr_Mdcr_Prtcptg_Ind), ''), 'N') AS Medicare_Participating,

        -- 2. Geographic Attributes
        TRIM(Rndrng_Prvdr_City) AS City,
        TRIM(Rndrng_Prvdr_State_Abrvtn) AS State,
        TRIM(Rndrng_Prvdr_Zip5) AS Zip5,
        TRY_CAST(TRIM(Rndrng_Prvdr_RUCA) AS DOUBLE) AS RUCA_Code,
        COALESCE(NULLIF(TRIM(Rndrng_Prvdr_RUCA_Desc), ''), 'Unknown') AS RUCA_Desc,

        -- 3. Procedure / HCPCS Identifiers
        TRIM(HCPCS_Cd) AS HCPCS_Code,
        TRIM(HCPCS_Desc) AS HCPCS_Desc,
        COALESCE(NULLIF(TRIM(HCPCS_Drug_Ind), ''), 'N') AS Is_Part_B_Drug,

        -- 4. Setting (Place of Service)
        TRIM(Place_Of_Srvc) AS Place_Of_Srvc_Code,
        CASE 
            WHEN TRIM(Place_Of_Srvc) = 'F' THEN 'Facility (Hospital/HOPD)'
            WHEN TRIM(Place_Of_Srvc) = 'O' THEN 'Non-Facility (Office/ASC)'
            ELSE 'Other'
        END AS Place_Of_Srvc_Desc,

        -- 5. Utilization & Service Volumes
        TRY_CAST(NULLIF(REPLACE(TRIM(Tot_Benes), '*', ''), '') AS BIGINT) AS Total_Beneficiaries,
        TRY_CAST(NULLIF(REPLACE(TRIM(Tot_Srvcs), '*', ''), '') AS DOUBLE) AS Total_Services,
        TRY_CAST(NULLIF(REPLACE(TRIM(Tot_Bene_Day_Srvcs), '*', ''), '') AS DOUBLE) AS Total_Bene_Day_Services,

        -- 6. Financial Amounts (Cleaned & Cast to DOUBLE)
        TRY_CAST(REGEXP_REPLACE(Avg_Sbmtd_Chrg, '[$, ]', '', 'g') AS DOUBLE) AS Avg_Submitted_Charge,
        TRY_CAST(REGEXP_REPLACE(Avg_Mdcr_Alowd_Amt, '[$, ]', '', 'g') AS DOUBLE) AS Avg_Medicare_Allowed_Amt,
        TRY_CAST(REGEXP_REPLACE(Avg_Mdcr_Pymt_Amt, '[$, ]', '', 'g') AS DOUBLE) AS Avg_Medicare_Payment_Amt,
        TRY_CAST(REGEXP_REPLACE(Avg_Mdcr_Stdzd_Amt, '[$, ]', '', 'g') AS DOUBLE) AS Avg_Medicare_Standardized_Amt,

        -- 7. Derived VBC Utilization Metrics
        -- Commercial Billed Markup Ratio
        ROUND(
            TRY_CAST(REGEXP_REPLACE(Avg_Sbmtd_Chrg, '[$, ]', '', 'g') AS DOUBLE) / 
            NULLIF(TRY_CAST(REGEXP_REPLACE(Avg_Mdcr_Alowd_Amt, '[$, ]', '', 'g') AS DOUBLE), 0), 
            2
        ) AS Billed_To_Allowed_Markup_Ratio,

        -- Procedure Frequency per Patient
        ROUND(
            TRY_CAST(NULLIF(REPLACE(TRIM(Tot_Srvcs), '*', ''), '') AS DOUBLE) / 
            NULLIF(TRY_CAST(NULLIF(REPLACE(TRIM(Tot_Benes), '*', ''), '') AS BIGINT), 0), 
            2
        ) AS Services_Per_Beneficiary

    FROM read_csv_auto('{DATA_FILE_PATH}', all_varchar=true, ignore_errors=true)
    WHERE TRIM(Rndrng_NPI) IS NOT NULL AND TRIM(HCPCS_Cd) IS NOT NULL;
""")

print("Exporting cleaned data to Parquet...")
con.execute(f"COPY provider_service_cleaned TO '{OUTPUT_PARQUET}' (FORMAT PARQUET);")
print(f" Cleaned parquet file saved: {OUTPUT_PARQUET}")