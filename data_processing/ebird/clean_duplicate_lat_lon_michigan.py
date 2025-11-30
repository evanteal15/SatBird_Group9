import pandas as pd
import numpy as np
import json
from pathlib import Path

# --- Paths ---
input_file = "data_processing/ebird/summer_hotspots_with_bioclim_withnan.csv"
output_file_merged = "data_processing/ebird/summer_hotspots_with_bioclim_merged_michigan.csv"
output_file_splits = "data_processing/ebird/summer_hotspots_with_bioclim_splits_michigan.csv"
duplicate_list_path = "data_processing/ebird/to_merge_3_michigan.txt"
michigan_hotspots_file = "data_processing/ebird/michigan_hotspots.csv"

michigan_hotspots = pd.read_csv(michigan_hotspots_file)['hotspot_id'].to_list()

def merge_duplicate_lat_lon():
    df = pd.read_csv(input_file)
    df = df[df['hotspot_id'].isin(michigan_hotspots)]  # Michigan filter
    df = df.drop_duplicates(['hotspot_id'], keep='first')

    grouped = df.groupby(['lon', 'lat'])
    new_df = pd.DataFrame(columns=df.columns)
    hotspots_merge = []

    for i, (name, group) in enumerate(grouped):
        if len(group) == 1:
            new_df = pd.concat([new_df, group])
        else:
            group["num_complete_checklists"] = group["num_complete_checklists"].sum()
            group["num_different_species"] = group["num_different_species"].sum()
            hotspots_merge.append([list(group["hotspot_id"].values)])
            group = group.drop_duplicates(['lon'], keep='first')
            new_df = pd.concat([new_df, group])

    # Save list of merged hotspots
    with open(duplicate_list_path, "w") as f:
        for s in hotspots_merge:
            f.write(" ".join(s[0]) + "\n")

    new_df.to_csv(output_file_merged, index=False)
    print(f"Merged duplicate lat/lon Michigan hotspots saved in {output_file_merged}")

def add_split_and_fill_nans():
    df = pd.read_csv(output_file_merged)
    df["split"] = "train"  # Simplified: assign all Michigan hotspots to train for now

    # Fill NaNs with column means
    cols = [c for c in df.columns if c.startswith("bio_") or c in ['bdticm','bldfie','cecsol','clyppt','orcdrc','phihox','sltppt','sndppt']]
    df[cols] = df[cols].fillna(df[cols].mean())
    df.to_csv(output_file_splits, index=False)
    print(f"Split and NaN-filled Michigan hotspots saved in {output_file_splits}")

if __name__ == "__main__":
    merge_duplicate_lat_lon()
    add_split_and_fill_nans()
