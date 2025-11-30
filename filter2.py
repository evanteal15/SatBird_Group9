#!/usr/bin/env python3
import os
import re
import sys
import csv
from collections import defaultdict

# -------------------------------
# Configuration and constants
# -------------------------------
FIELD_LOCALITY_ID = 26   # Field 27 in 1-based AWK indexing
FIELD_OBS_DATE    = 30   # Field 31 in 1-based AWK indexing
FIELD_ALL_SPECIES = 41   # Field 42 in 1-based AWK indexing

BUFFER_LIMIT_BYTES = 128 * 1024 * 1024  # 128 MB flush threshold
MONTH_PATTERN = re.compile(r"202[3-5]-(06|07|08)-")

# -------------------------------
# Helper functions
# -------------------------------
def lid(locality_id: str) -> int:
    """Convert 'L123456' -> 123456 (numeric part)."""
    return int(locality_id[1:]) if locality_id.startswith("L") else int(locality_id)

def row_exists_in_csv(row, file_path):
    # row should be a dict
    with open(file_path, "r", newline='', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for existing_row in reader:
            # Direct comparison may be strict; adjust if needed
            if all(str(existing_row[k]) == str(v) for k, v in row.items() if k in existing_row):
                return True
    return False

def flush_buffer(buffer, localities, bufferdirs, header):
    """Write out buffered data to CSV files and clear buffer."""
    for l_id, lines in buffer.items():
        i = lid(l_id)
        print(f"This is our specific file id {l_id}")
        subdir = f"output2/split-{i % 1000:03d}/"
        file_path = os.path.join(subdir, f"{l_id}.csv")

        # Create parent directory if needed
        if subdir not in bufferdirs:
            print("Creating Parent Directory")
            os.makedirs(subdir, exist_ok=True)
            bufferdirs.add(subdir)

        # Write header once per locality
        write_header = not os.path.exists(file_path)
        with open(file_path, "a", encoding="utf-8") as f:
            print("Writing out specific file")
            if write_header:
                f.write(header + "\n")
                localities.add(l_id)
            for row in lines:
                if not row_exists_in_csv(row, file_path):
                    f.write("\t".join(row.values()) + "\n")

    buffer.clear()

# -------------------------------
# Main processing
# -------------------------------
def main(input_path):
    buffer = defaultdict(list)
    localities = set()
    bufferdirs = set()
    buffer_size = 0

    with open(input_path, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile, delimiter='\t')
        header = "\t".join(reader.fieldnames)
        print("File opened!")
        for line_num, line in enumerate(reader, start=2):
            # parts = line.rstrip("\n").split("\t")

            # # Skip malformed lines
            # if len(parts) <= FIELD_ALL_SPECIES:
            #     continue

            all_species = line["ALL SPECIES REPORTED"]
            obs_date = line["OBSERVATION DATE"]
            locality_id = line["LOCALITY ID"]
            
            # if(2049900 < line_num and 2050000 > line_num):
            #     print("Has all Species: " + all_species + " Date Observed: " + obs_date + "\n")

            # Apply filters: All Species == 1 and month is Jan, Jun, Jul, Dec
            if all_species == "1" and MONTH_PATTERN.match(obs_date):
                # print("There is a good checklist here on line " + line_num)
                buffer[locality_id].append(line)
                buffer_size += len(line)

                if buffer_size >= BUFFER_LIMIT_BYTES:
                    flush_buffer(buffer, localities, bufferdirs, header)
                    buffer_size = 0

    # Flush any remaining data
    flush_buffer(buffer, localities, bufferdirs, header)
    print("✅ Done. Filtered output written to ./output/")

# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 filter.py <input_file>")
        sys.exit(1)
    main(sys.argv[1])
