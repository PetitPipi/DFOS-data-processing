import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import glob
import os
import json
import time
import numpy as np

# --- SETTINGS ---
CHUNK_SIZE = 2000

files = sorted(glob.glob("*.tsv"))
print(f"Found {len(files)} files to process.\n")

for input_file in files:
    output_file = input_file.replace(".tsv", ".parquet")
    meta_file = input_file.replace(".tsv", "_metadata.json")
    
    if os.path.exists(output_file) and os.path.getsize(output_file) > 1024 * 1024:
        # print(f"Skipping {input_file} (Already done)")
        continue

    print(f"Processing: {input_file}")
    start_time = time.time()

    # 1. SCAN HEADER FOR METADATA & GAGE COUNT
    # We need to find the specific line that lists the gages
    num_sensors = 0
    header_found_method = "None"
    data_start_row = 0
    
    # Store physics vars just in case
    phys_len_m = 0
    phys_pitch_mm = 0
    
    with open(input_file, 'r') as f:
        # Read header block
        lines = f.readlines(8192) 
        
        # First pass: Look for Metadata (Length/Pitch) and Separator
        sep_index = -1
        for i, line in enumerate(lines):
            l_lower = line.lower()
            
            # Physics capture
            if "length (m)" in l_lower:
                try: phys_len_m = float(line.split(":")[1].strip().split()[0])
                except: pass
            if "gage pitch (mm)" in l_lower:
                try: phys_pitch_mm = float(line.split(":")[1].strip().split()[0])
                except: pass
                
            if "-------" in line:
                sep_index = i
        
        # Second pass: Look for Header Lines (Gage or X-Axis)
        # We start searching around the separator
        search_start = sep_index if sep_index != -1 else 0
        
        for i in range(search_start, min(len(lines), search_start + 20)):
            line = lines[i]
            parts = line.split('\t') # Raw split to preserve position
            
            # PRIORITY 1: "Gage" Line with "All Gages"
            if line.lower().startswith("gage"):
                # Filter: Count ONLY columns containing "All Gages"
                # This ignores "Vertical 1", "Segment A", etc.
                valid_gages = [p for p in parts if "All Gages" in p]
                
                if len(valid_gages) > 100: # Sanity check: must be a real data file
                    num_sensors = len(valid_gages)
                    header_found_method = "Gage Name Filter"
                    data_start_row = i + 1
                    break
            
            # PRIORITY 2: "x-axis" Line (Standard)
            elif line.lower().startswith("x-axis"):
                # Count numbers
                valid_nums = []
                for p in parts[1:]:
                    try: valid_nums.append(float(p))
                    except: continue
                
                if len(valid_nums) > 100:
                    num_sensors = len(valid_nums)
                    header_found_method = "x-axis Row"
                    data_start_row = i + 1
                    break

    # 3. FALLBACK: PHYSICS CALCULATION
    if num_sensors == 0 and phys_len_m > 0 and phys_pitch_mm > 0:
        print("   [INFO] Header scan failed. Falling back to Physics Calculation.")
        # Calc sensors: (Length * 1000) / Pitch
        # We round to nearest integer
        num_sensors = int(round((phys_len_m * 1000) / phys_pitch_mm))
        header_found_method = f"Physics ({phys_len_m}m / {phys_pitch_mm}mm)"
        
        # We assume data starts after separator + 2 lines (Tare, Gage)
        if sep_index != -1:
            data_start_row = sep_index + 3
        else:
            data_start_row = 30 # absolute guess if no separator

    print(f"   -> Method: {header_found_method}")
    print(f"   -> Sensor Count: {num_sensors}")

    if num_sensors == 0:
        print("   [ERROR] Could not determine sensor count. Skipping.")
        continue

    # 4. SAVE METADATA (Generative)
    # Since we might not have grabbed the names, we generate meters
    # Default pitch 1.3mm if not found? Or 0.65mm?
    pitch_to_use = phys_pitch_mm if phys_pitch_mm > 0 else 1.3 # standard guess
    
    generated_locs = list(np.arange(num_sensors) * (pitch_to_use / 1000.0))
    
    metadata = {
        "locations": generated_locs,
        "gages": [f"{x:.4f}m" for x in generated_locs]
    }
    with open(meta_file, 'w') as f:
        json.dump(metadata, f)

    # 5. PROCESS DATA
    try:
        my_schema = pa.schema([
            ('Time', pa.string()),
            ('Data_Array', pa.list_(pa.float64())) 
        ])

        csv_stream = pd.read_csv(
            input_file, sep='\t', chunksize=CHUNK_SIZE, header=None,
            skiprows=data_start_row, engine='c', low_memory=False
        )
        
        writer = None
        rows_processed = 0

        for chunk in csv_stream:
            # Drop empty rows (sometimes happen at EOF)
            chunk = chunk.dropna(subset=[0])
            if len(chunk) == 0: continue

            # Time is Col 0
            time_col = chunk.iloc[:, 0].astype(str)
            
            # Data is RIGHT-ALIGNED
            # We skip 'Vertical 1', 'Segment A', etc by taking the last N cols
            raw_data = chunk.iloc[:, -num_sensors:]
            
            clean_matrix = raw_data.apply(pd.to_numeric, errors='coerce').to_numpy(dtype='float64')

            compact_df = pd.DataFrame({
                'Time': time_col,
                'Data_Array': list(clean_matrix)
            })
            
            table = pa.Table.from_pandas(compact_df, schema=my_schema)
            
            if writer is None:
                writer = pq.ParquetWriter(output_file, table.schema, compression='snappy')
            
            writer.write_table(table)
            rows_processed += len(compact_df)
            print(f"   -> Processed {rows_processed} rows...", end='\r')

        if writer: writer.close()
        
        duration = time.time() - start_time
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\n   -> DONE! Time: {duration:.1f}s | Size: {size_mb:.1f}MB\n")

    except Exception as e:
        print(f"\n   -> ERROR: {e}\n")
        if os.path.exists(output_file): os.remove(output_file)