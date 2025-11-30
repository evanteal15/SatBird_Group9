#!/usr/bin/env python

from __future__ import annotations
import pickle
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
import argparse

# --- Constants ---
INPUT_DIR = Path("USA_summer_michigan")  # Folder containing train_split.csv etc.
OUTPUT_FILE = INPUT_DIR / "range_maps.pkl"  # Final output for SatBird
TOTAL_SPECIES = 670  # Must match training YAML config

# --- Utility: Create 2D occupancy grid ---
def generate_range_map(lat, lon, lat_bins, lon_bins):
    grid = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1), dtype=np.float32)
    lat_idx = np.searchsorted(lat_bins, lat) - 1
    lon_idx = np.searchsorted(lon_bins, lon) - 1
    lat_idx = np.clip(lat_idx, 0, grid.shape[0] - 1)
    lon_idx = np.clip(lon_idx, 0, grid.shape[1] - 1)
    grid[lat_idx, lon_idx] = 1.0
    return grid.flatten()  # flatten to 1D per hotspot

# --- Main generation ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=float, default=0.05, help="Grid resolution in degrees")
    args = parser.parse_args()

    # --- Load all splits ---
    splits = ["train_split.csv", "valid_split.csv", "test_split.csv"]
    dfs = [pd.read_csv(INPUT_DIR / s) for s in splits]
    all_data = pd.concat(dfs).drop_duplicates(subset=["hotspot_id"]).reset_index(drop=True)
    print(f"Total Michigan hotspots to process: {len(all_data)}")

    # --- Define lat/lon bins ---
    lat_min, lat_max = all_data["lat"].min(), all_data["lat"].max()
    lon_min, lon_max = all_data["lon"].min(), all_data["lon"].max()
    lat_bins = np.arange(lat_min, lat_max + args.resolution, args.resolution)
    lon_bins = np.arange(lon_min, lon_max + args.resolution, args.resolution)

    # --- Generate flattened range maps ---
    range_maps = []
    hotspot_ids = []

    for _, row in tqdm(all_data.iterrows(), total=len(all_data), desc="Generating range maps"):
        hotspot_id = row["hotspot_id"]
        lat, lon = row["lat"], row["lon"]
        grid_flat = generate_range_map(lat, lon, lat_bins, lon_bins)
        range_maps.append(grid_flat)
        hotspot_ids.append(hotspot_id)

    range_maps = np.stack(range_maps, axis=0)
    num_samples, current_species = range_maps.shape
    print(f"Initial shape: {range_maps.shape}")

    # --- Pad or truncate to match TOTAL_SPECIES ---
    if current_species < TOTAL_SPECIES:
        padding = np.zeros((num_samples, TOTAL_SPECIES - current_species), dtype=np.float32)
        range_maps = np.concatenate([range_maps, padding], axis=1)
        print(f"Padded from {current_species} to {TOTAL_SPECIES} species")
    elif current_species > TOTAL_SPECIES:
        range_maps = range_maps[:, :TOTAL_SPECIES]
        print(f"Truncated from {current_species} to {TOTAL_SPECIES} species")

    # --- Normalize and clip ---
    range_maps = np.clip(range_maps, 0.0, 1.0)

    # --- Convert to DataFrame with hotspot_id index ---
    range_maps_df = pd.DataFrame(range_maps)
    range_maps_df.insert(0, "hotspot_id", hotspot_ids)
    range_maps_df.set_index("hotspot_id", inplace=True)

    # --- Save final pickle ---
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(range_maps_df, f)

    print(f"\nSaved final training-ready range_maps.pkl to {OUTPUT_FILE}")
    print(f"Final shape: {range_maps_df.shape}, dtype: {range_maps_df.dtypes[0]}")

if __name__ == "__main__":
    main()
