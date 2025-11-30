import os
import pandas as pd
from tqdm import tqdm

# --- Michigan-specific paths ---
checklist_folder = "data_processing/ebird/checklists_USA/summer/"
michigan_hotspots_file = "data_processing/ebird/michigan_hotspots.csv"
output_file = "data_processing/ebird/checklists_USA/summer_list_michigan.txt"

# Load Michigan hotspot IDs
michigan_ids = pd.read_csv(michigan_hotspots_file)["hotspot_id"].to_list()

with open(output_file, "w") as f:
    paths = os.listdir(checklist_folder)
    for file in tqdm(paths):
        df = pd.read_csv(os.path.join(checklist_folder, file))
        hotspot_id = df["LOCALITY ID"][0]
        if hotspot_id in michigan_ids and len(df) > 5:
            f.write(f"{hotspot_id},{df['LATITUDE'][0]},{df['LONGITUDE'][0]}\n")

print(f"Done. Michigan summer checklist saved to {output_file}")
