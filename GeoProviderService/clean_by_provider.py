import duckdb

DATA_FILE_PATH = "Medicare_Physician_by_Provider.csv"
OUTPUT_PARQUET = "Medicare_By_Provider_Cleaned.parquet"

con = duckdb.connect()
print("Executing pre-validated preprocessing on 'By Provider' dataset...")

con.execute(f"""
    CREATE OR REPLACE TABLE provider_cleaned AS
    SELECT
        -- 1. Clinician & Entity Details
        TRIM(Rndrng_NPI) AS NPI,
        TRIM(Rndrng_Prvdr_Last_Org_Name) AS Provider_Last_or_Org_Name,
        COALESCE(NULLIF(TRIM(Rndrng_Prvdr_First_Name), ''), '') AS Provider_First_Name,
        COALESCE(NULLIF(TRIM(Rndrng_Prvdr_Crdntls), ''), '') AS Provider_Credentials,
        TRIM(Rndrng_Prvdr_Ent_Cd) AS Entity_Type, -- 'I' = Individual Clinician, 'O' = Organization
        TRIM(Rndrng_Prvdr_Type) AS Specialty_Type,
        COALESCE(NULLIF(TRIM(Rndrng_Prvdr_Mdcr_Prtcptg_Ind), ''), 'N') AS Medicare_Participating,

        -- 2. Geography
        TRIM(Rndrng_Prvdr_City) AS City,
        TRIM(Rndrng_Prvdr_State_Abrvtn) AS State,
        TRIM(Rndrng_Prvdr_Zip5) AS Zip5,
        TRY_CAST(TRIM(Rndrng_Prvdr_RUCA) AS DOUBLE) AS RUCA_Code,
        COALESCE(NULLIF(TRIM(Rndrng_Prvdr_RUCA_Desc), ''), 'Unknown') AS RUCA_Desc,

        -- 3. Panel Metrics & Acuity
        TRY_CAST(TRIM(Tot_Benes) AS BIGINT) AS Total_Beneficiaries,
        TRY_CAST(TRIM(Tot_Srvcs) AS DOUBLE) AS Total_Services,
        TRY_CAST(TRIM(Tot_HCPCS_Cds) AS INTEGER) AS Total_Distinct_HCPCS,
        TRY_CAST(TRIM(Bene_Avg_Risk_Scre) AS DOUBLE) AS Bene_Avg_Risk_Score,
        TRY_CAST(TRIM(Bene_Avg_Age) AS DOUBLE) AS Bene_Avg_Age,

        -- 4. Demographics (CMS Suppressed <11 preserved as NULL)
        TRY_CAST(NULLIF(TRIM(Bene_Dual_Cnt), '') AS BIGINT) AS Dual_Eligible_Benes,
        TRY_CAST(NULLIF(TRIM(Bene_Feml_Cnt), '') AS BIGINT) AS Female_Benes,
        TRY_CAST(NULLIF(TRIM(Bene_Male_Cnt), '') AS BIGINT) AS Male_Benes,

        -- 5. Chronic Disease Prevalence (%)
        TRY_CAST(NULLIF(TRIM(Bene_CC_PH_Diabetes_V2_Pct), '') AS DOUBLE) AS Pct_Diabetes,
        TRY_CAST(NULLIF(TRIM(Bene_CC_PH_CKD_V2_Pct), '') AS DOUBLE) AS Pct_CKD,
        TRY_CAST(NULLIF(TRIM(Bene_CC_PH_HF_NonIHD_V2_Pct), '') AS DOUBLE) AS Pct_Heart_Failure,
        TRY_CAST(NULLIF(TRIM(Bene_CC_PH_COPD_V2_Pct), '') AS DOUBLE) AS Pct_COPD,
        TRY_CAST(NULLIF(TRIM(Bene_CC_PH_Hypertension_V2_Pct), '') AS DOUBLE) AS Pct_Hypertension,
        TRY_CAST(NULLIF(TRIM(Bene_CC_BH_Depress_V1_Pct), '') AS DOUBLE) AS Pct_Depression,
        TRY_CAST(NULLIF(TRIM(Bene_CC_BH_Anxiety_V1_Pct), '') AS DOUBLE) AS Pct_Anxiety,

        -- 6. Clean Financials
        TRY_CAST(REGEXP_REPLACE(Tot_Sbmtd_Chrg, '[$, ]', '', 'g') AS DOUBLE) AS Total_Submitted_Charges,
        TRY_CAST(REGEXP_REPLACE(Tot_Mdcr_Alowd_Amt, '[$, ]', '', 'g') AS DOUBLE) AS Total_Medicare_Allowed_Amt,
        TRY_CAST(REGEXP_REPLACE(Tot_Mdcr_Pymt_Amt, '[$, ]', '', 'g') AS DOUBLE) AS Total_Medicare_Payment_Amt,
        TRY_CAST(REGEXP_REPLACE(Tot_Mdcr_Stdzd_Amt, '[$, ]', '', 'g') AS DOUBLE) AS Total_Medicare_Standardized_Amt,

        -- 7. Part B Drug & Medical Spend (Corrected CMS column names)
        COALESCE(TRY_CAST(REGEXP_REPLACE(Drug_Mdcr_Stdzd_Amt, '[$, ]', '', 'g') AS DOUBLE), 0.0) AS Drug_Standardized_Amt,
        COALESCE(TRY_CAST(REGEXP_REPLACE(Med_Mdcr_Stdzd_Amt, '[$, ]', '', 'g') AS DOUBLE), 0.0) AS Med_Standardized_Amt,

        -- 8. Derived Value-Based Care Spend Metrics
        ROUND(
            TRY_CAST(REGEXP_REPLACE(Tot_Mdcr_Stdzd_Amt, '[$, ]', '', 'g') AS DOUBLE) / 
            NULLIF(TRY_CAST(TRIM(Tot_Benes) AS BIGINT), 0),
            2
        ) AS Spend_Per_Beneficiary,

        ROUND(
            (TRY_CAST(REGEXP_REPLACE(Tot_Mdcr_Stdzd_Amt, '[$, ]', '', 'g') AS DOUBLE) / 
             NULLIF(TRY_CAST(TRIM(Tot_Benes) AS BIGINT), 0)) / 
            NULLIF(TRY_CAST(TRIM(Bene_Avg_Risk_Scre) AS DOUBLE), 0),
            2
        ) AS Risk_Adjusted_Spend_Per_Bene

    FROM read_csv_auto('{DATA_FILE_PATH}', all_varchar=true, ignore_errors=true)
    WHERE TRIM(Rndrng_NPI) IS NOT NULL AND TRIM(Rndrng_NPI) != '';
""")

con.execute(f"COPY provider_cleaned TO '{OUTPUT_PARQUET}' (FORMAT PARQUET);")
print(f"✅ Cleaned parquet file saved successfully: {OUTPUT_PARQUET}")