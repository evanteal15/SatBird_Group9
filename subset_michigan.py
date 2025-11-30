import os
import shutil
import pandas as pd
from tqdm import tqdm

# === CONFIGURATION ===
base_dir = "USA_summer"              # where your extracted dataset lives
out_dir = "USA_summer_michigan"      # output folder
folders_to_subset = ["environmental", "images", "images_visual", "targets"]

# === STEP 1: Read train/test splits ===
train = pd.read_csv(os.path.join(base_dir, "train_split.csv"))
test = pd.read_csv(os.path.join(base_dir, "test_split.csv"))
valid = pd.read_csv(os.path.join(base_dir, "valid_split.csv"))

# === STEP 2: Filter to Michigan ===
train_mi = train[train["state"].str.lower() == "michigan"]
test_mi = test[test["state"].str.lower() == "michigan"]
valid_mi = valid[valid["state"].str.lower() == "michigan"]

# Combine IDs from both splits
ids_mi = pd.concat([train_mi, test_mi, valid_mi])["hotspot_id"].unique().tolist()

print(f"Found {len(ids_mi)} Michigan entries across train and test.")

# === STEP 3: Prepare output structure ===
os.makedirs(out_dir, exist_ok=True)
for folder in folders_to_subset:
    os.makedirs(os.path.join(out_dir, folder), exist_ok=True)

# === STEP 4: Copy files for each Michigan hotspot ===
for folder in folders_to_subset:
    print(f"\nCopying {folder} files...")
    src_folder = os.path.join(base_dir, folder)
    dst_folder = os.path.join(out_dir, folder)

    for hid in tqdm(ids_mi):
        # Match any file starting with the ID (handles different extensions)
        matches = [f for f in os.listdir(src_folder) if f.startswith(hid)]
        for m in matches:
            shutil.copy(os.path.join(src_folder, m), os.path.join(dst_folder, m))

# === STEP 5: Save new CSV splits ===
train_mi.to_csv(os.path.join(out_dir, "train_split_michigan.csv"), index=False)
test_mi.to_csv(os.path.join(out_dir, "test_split_michigan.csv"), index=False)
valid_mi.to_csv(os.path.join(out_dir, "valid_split_michigan.csv"), index=False)

print("\n Subsetting complete! New dataset saved to:", out_dir)

