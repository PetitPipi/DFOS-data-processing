import glob
import json
import os

files = sorted(glob.glob("*.tsv"))
print(f"Scanning {len(files)} files for Sensor Type & Units...\n")

for tsv_file in files:
    json_file = tsv_file.replace(".tsv", "_metadata.json")
    
    # Skip if we haven't processed this file yet
    if not os.path.exists(json_file):
        print(f"[SKIP] No metadata found for {tsv_file} (Run transform first)")
        continue
        
    # defaults
    s_type = "Unknown"
    s_unit = "-"
    
    # 1. READ HEADER (Scan first 50 lines)
    with open(tsv_file, 'r') as f:
        for _ in range(50):
            line = f.readline()
            if not line: break
            
            clean_line = line.strip()
            
            # Extract "Sensor Type: Strain"
            if clean_line.startswith("Sensor Type:"):
                s_type = clean_line.split(":", 1)[1].strip()
                
            # Extract "Units: microstrain"
            # (Make sure we don't accidentally grab "X-Axis Units")
            if clean_line.startswith("Units:"):
                s_unit = clean_line.split(":", 1)[1].strip()

    print(f"Update: {tsv_file[:40]}... -> Type: {s_type} | Unit: {s_unit}")

    # 2. UPDATE JSON
    with open(json_file, 'r') as f:
        meta_data = json.load(f)
    
    meta_data["sensor_type"] = s_type
    meta_data["units"] = s_unit
    
    with open(json_file, 'w') as f:
        json.dump(meta_data, f)

print("\nDone! Metadata updated.")