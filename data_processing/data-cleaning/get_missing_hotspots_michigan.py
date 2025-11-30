import pandas as pd
import os
from os import listdir
from os.path import isfile, join

# --- Paths ---
hotspot_csv_files_path = "data_processing/data-cleaning/hotspot_csv_data_cleaned_michigan/"
output_file = "data_processing/data-cleaning/missing_hotspot_michigan.csv"
hotspot_list_file = "data_processing/data-cleaning/hotspotlist_with_50_complete_checklists_michigan.csv"

def find_missing_hotspots(hotspot_csv_files_path):
    """Find Michigan hotspots with >=50 checklists that are missing individual CSVs"""
    hotspot_files = [f for f in listdir(hotspot_csv_files_path) if isfile(join(hotspot_csv_files_path, f))]
    hotspot_names_from_files = [os.path.splitext(each)[0] for each in hotspot_files]

    hotspots_more_than_50 = pd.read_csv(hotspot_list_file)['LOCALITY_ID'].to_list()
    diff_hotspot = list(set(hotspots_more_than_50) - set(hotspot_names_from_files))
    return diff_hotspot

def main():
    diff_hotspot = find_missing_hotspots(hotspot_csv_files_path)
    print("Number of missing Michigan hotspots:", len(diff_hotspot))

    missing_hotspot_df = pd.DataFrame(diff_hotspot, columns=["missing_hotspot"])
    missing_hotspot_df.to_csv(output_file, index=False)
    print(f"Missing hotspot IDs saved in {output_file}")

if __name__ == "__main__":
    main()
