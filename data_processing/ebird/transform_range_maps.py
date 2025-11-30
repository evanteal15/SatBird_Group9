#!/usr/bin/env python
"""
Transform range_maps.pkl dictionary into a 2D NumPy array for training.
"""

import os
import pickle
import numpy as np

# --- Config ---
RANGE_MAP_PATH = "USA_summer_michigan/range_maps.pkl"  # path to original pickle
OUTPUT_PATH = RANGE_MAP_PATH  # overwrite
TOTAL_SPECIES = 670  # must match config

# --- Load original pickle ---
with open(RANGE_MAP_PATH, "rb") as f:
    range_maps_dict = pickle.load(f)

# --- Ensure it's a dict ---
if not isinstance(range_maps_dict, dict):
    raise ValueError(f"Expected a dictionary in {RANGE_MAP_PATH}, got {type(range_maps_dict)}")

# --- Convert dict values to 2D NumPy array ---
range_maps_list = []
for key in sorted(range_maps_dict.keys()):  # sort keys for consistency
    arr = np.array(range_maps_dict[key], dtype=np.float32).flatten()
    range_maps_list.append(arr)

range_maps = np.stack(range_maps_list, axis=0)  # shape: [num_samples, num_species]
print(f"Original shape from dict: {range_maps.shape}")

# --- Adjust number of species ---
num_samples, current_species = range_maps.shape

if current_species < TOTAL_SPECIES:
    padding = np.zeros((num_samples, TOTAL_SPECIES - current_species), dtype=np.float32)
    range_maps = np.concatenate([range_maps, padding], axis=1)
    print(f"Padded from {current_species} to {TOTAL_SPECIES} species")
elif current_species > TOTAL_SPECIES:
    range_maps = range_maps[:, :TOTAL_SPECIES]
    print(f"Truncated from {current_species} to {TOTAL_SPECIES} species")

# --- Clip to [0,1] ---
range_maps = np.clip(range_maps, 0.0, 1.0)

# --- Save transformed pickle ---
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(range_maps, f)

print(f"Transformed range_maps.pkl saved at {OUTPUT_PATH}")
print(f"Final shape: {range_maps.shape}, dtype: {range_maps.dtype}")
