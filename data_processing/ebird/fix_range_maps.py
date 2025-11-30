#!/usr/bin/env python
import pickle
import pandas as pd

# Path to your current range_maps.pkl
INPUT_PATH = "USA_summer_michigan/range_maps.pkl"
OUTPUT_PATH = INPUT_PATH  # overwrite original

# Load existing array
with open(INPUT_PATH, "rb") as f:
    range_maps_array = pickle.load(f)

# Convert to DataFrame
range_maps_df = pd.DataFrame(range_maps_array)

# Optional: if you have hotspot IDs, you can set them as index:
# hotspot_ids = pd.read_csv("train_split.csv")['hotspot_id']
# range_maps_df.index = hotspot_ids

# Save back to pickle
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(range_maps_df, f)

print(f"Saved fixed range_maps.pkl to {OUTPUT_PATH}")
