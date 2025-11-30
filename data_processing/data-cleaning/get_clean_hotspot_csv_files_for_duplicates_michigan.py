import pandas as pd
import os
from os import listdir
from os.path import isfile, join

# --- Paths ---
src_csv_folder = "data_processing/data-cleaning/hotspot_csv_data/"
dest_csv_folder = "data_processing/data-cleaning/hotspot_csv_data_cleaned_michigan/"
os.makedirs(dest_csv_folder, exist_ok=True)

# Optional: Michigan hotspot list
michigan_hotspots_file = "data_processing/ebird/michigan_hotspots.csv"
michigan_hotspots = pd.read_csv(michigan_hotspots_file)['hotspot_id'].to_list()

def remove_row_duplicates(mypath, output_csv_folder, onlyfiles):
    """Removes duplicate rows and filters Michigan hotspots"""
    for idx, file in enumerate(onlyfiles):
        csv_path = mypath + file
        output_path = output_csv_folder + file

        df = pd.read_csv(csv_path, delimiter=",")
        df = df.drop_duplicates()
        df = df.loc[df['GLOBAL UNIQUE IDENTIFIER'] != 'GLOBAL UNIQUE IDENTIFIER']

        # Filter for Michigan hotspots only
        if 'LOCALITY ID' in df.columns:
            df = df[df['LOCALITY ID'].isin(michigan_hotspots)]

        df.to_csv(output_path, index=False)
        if idx % 100 == 0:
            print("#", end='')

def main():
    onlyfiles = [f for f in listdir(src_csv_folder) if isfile(join(src_csv_folder, f))]
    remove_row_duplicates(src_csv_folder, dest_csv_folder, onlyfiles)
    print("Done cleaning Michigan hotspot CSVs")

if __name__ == "__main__":
    main()
