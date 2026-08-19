import duckdb

con = duckdb.connect()

print("--- Geography Dataset State Identifiers ---")
print(con.execute("""
    SELECT DISTINCT Geo_Code, Geo_Desc 
    FROM read_parquet('Medicare_Geo_Service_Cleaned.parquet') 
    WHERE Geo_Level = 'State' 
    LIMIT 5;
""").fetchdf())

print("\n--- Provider Dataset State Identifiers ---")
print(con.execute("""
    SELECT DISTINCT State 
    FROM read_parquet('Medicare_By_Provider_Cleaned.parquet') 
    LIMIT 5;
""").fetchdf())