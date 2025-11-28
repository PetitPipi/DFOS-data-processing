import duckdb
import glob
import os
import json

files = sorted(glob.glob("*.parquet"))

print(f"{'FILENAME':<55} | {'ROWS':<8} | {'SENSORS':<8} | {'FIRST VAL':<10} | {'STATUS'}")
print("-" * 115)

for f in files:
    filename = os.path.basename(f)
    try:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        if size_mb < 0.01:
            print(f"{filename:<55} | {'-':<8} | {'-':<8} | {'-':<10} | CORRUPT (0 MB)")
            continue

        # --- FIX IS HERE ---
        # We assume the array column is named "Data_Array"
        # We grab the length of the array from the first row
        query = f"""
            SELECT 
                COUNT(*) as rows, 
                list_extract(Data_Array, 1) as first_val, 
                len(Data_Array) as sensors
            FROM '{f}'
            LIMIT 1
        """
        
        # We run two queries: one for count (fast), one for structure (fast)
        # Actually, let's do it in two steps to be safe against empty files
        
        row_count = duckdb.sql(f"SELECT COUNT(*) FROM '{f}'").fetchone()[0]
        
        if row_count > 0:
            stats = duckdb.sql(f"SELECT len(Data_Array), Data_Array[1] FROM '{f}' LIMIT 1").fetchone()
            sensors = stats[0]
            first_val = stats[1]
        else:
            sensors = 0
            first_val = None

        # Determine Status
        status = "OK"
        if first_val is None: status = "⚠️ NULL DATA"
        elif str(first_val) == "nan": status = "⚠️ ALL NaN"
        elif sensors < 10: status = "⚠️ LOW SENSORS"

        print(f"{filename:<55} | {row_count:<8} | {sensors:<8} | {str(first_val)[:8]:<10} | {status}")

    except Exception as e:
        print(f"{filename:<55} | ERROR: {str(e)}")
        