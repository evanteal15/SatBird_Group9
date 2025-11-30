import pandas as pd

import os

from pathlib import Path



def sync_hotspot_data(

    csv_a_path: str,

    csv_b_path: str,

    dir_a_root: str,

    dir_b_root: str,

    id_column: str = 'hotspot_id'

) -> None:

    """

    Synchronizes two main CSV files based on a shared ID column and cleans

    out corresponding CSV files in two associated directory structures.



    Only Hotspot IDs present in *both* main CSV files will be kept.

    All IDs not in the intersection will be removed from their respective CSVs and directories.



    Args:

        csv_a_path: Path to the first main CSV file (e.g., Summer data).

        csv_b_path: Path to the second main CSV file (e.g., Winter data).

        dir_a_root: Root directory containing split CSV files for CSV A.

        dir_b_root: Root directory containing split CSV files for CSV B.

        id_column: The name of the column containing the unique Hotspot ID.

    """

    print(f"--- Starting Data Synchronization ---")



    # --- 1. Load DataFrames ---

    try:

        df_a = pd.read_csv(csv_a_path)

        df_b = pd.read_csv(csv_b_path)

    except FileNotFoundError as e:

        print(f"ERROR: Could not find a file. Check path: {e}")

        return

    except Exception as e:

        print(f"ERROR reading CSVs: {e}")

        return



    # Ensure column names are stripped of whitespace for safe access

    df_a.columns = df_a.columns.str.strip()

    df_b.columns = df_b.columns.str.strip()



    if id_column not in df_a.columns or id_column not in df_b.columns:

        print(f"ERROR: Required ID column '{id_column}' not found in one or both CSVs.")

        return



    # --- 2. Find Intersection of Hotspot IDs ---

    ids_a = set(df_a[id_column].unique())

    ids_b = set(df_b[id_column].unique())



    print(f"Total IDs in CSV A: {len(ids_a)}")

    print(f"Total IDs in CSV B: {len(ids_b)}")



    # Calculate the set of IDs common to both (the desired intersection)

    common_ids = ids_a.intersection(ids_b)

    print(f"Total Common IDs (Intersection): {len(common_ids)}")



    # Calculate the set of IDs to be removed from A and B

    ids_to_remove_from_a = ids_a - common_ids

    ids_to_remove_from_b = ids_b - common_ids



    print(f"IDs unique to A (to be removed): {len(ids_to_remove_from_a)}")

    # OUTPUT: Display the list of IDs to be removed from A

    print(f"Hotspot IDs removed from A: {ids_to_remove_from_a}") 

    

    print(f"IDs unique to B (to be removed): {len(ids_to_remove_from_b)}")

    # OUTPUT: Display the list of IDs to be removed from B

    print(f"Hotspot IDs removed from B: {ids_to_remove_from_b}")



    # --- 3. Clean Main CSV Files ---



    # Filter CSV A

    df_a_cleaned = df_a[df_a[id_column].isin(common_ids)].sort_values(by="hotspot_id")

    df_a_cleaned.to_csv(csv_a_path + "_syncd.csv", index=False)

    print(f"Updated CSV A: Kept {len(df_a_cleaned)} IDs. Saved to {csv_a_path}")



    # Filter CSV B

    df_b_cleaned = df_b[df_b[id_column].isin(common_ids)].sort_values(by="hotspot_id")

    df_b_cleaned.to_csv(csv_b_path + "_syncd.csv", index=False)

    print(f"Updated CSV B: Kept {len(df_b_cleaned)} IDs. Saved to {csv_b_path}")



    # --- 4. Clean Corresponding Directories ---



    def cleanup_directory(root_dir, ids_to_remove):

        """Helper function to find and delete specific CSV files."""

        if not ids_to_remove:

            return 0

        

        removed_count = 0

        root_path = Path(root_dir)

        

        # Iterate over all files ending with .csv in the root and all subdirectories

        for filepath in root_path.rglob('*.csv'):

            filename = filepath.stem # Get the filename without extension (e.g., L1234567)

            

            if filename in ids_to_remove:

                try:

                    os.remove(filepath)

                    removed_count += 1

                except OSError as e:

                    print(f"Warning: Could not delete file {filepath}. Error: {e}")

        

        print(f"Cleaned directory {root_dir}. Removed {removed_count} files.")

        return removed_count



    # Clean Directory A

    # cleanup_directory(dir_a_root, ids_to_remove_from_a)



    # Clean Directory B

    # cleanup_directory(dir_b_root, ids_to_remove_from_b)



    print("\n--- Synchronization Complete ---")





if __name__ == "__main__":

    # --- Configuration ---

    # NOTE: You must replace these placeholder paths with your actual paths!

    

    # Example paths based on the context of your previous code:

    

    # Main CSV A (e.g., Summer Hotspots Metadata)

    CSV_A = "./SatBird_data_10-22/all_summer_hotspots.csv"

    

    # Main CSV B (e.g., Winter Hotspots Metadata)

    CSV_B = "./SatBird_data_23-25/all_summer_hotspots.csv" 

    

    # Root Directory A (e.g., where split Summer Checklist files are stored)

    DIR_A_ROOT = "/network/scratch/t/tengmeli/newebd/output/summer_targets/" 

    

    # Root Directory B (e.g., where split Winter Checklist files are stored)

    DIR_B_ROOT = "/network/scratch/t/tengmeli/newebd/output/winter_targets/" 

    

    HOTSPOT_ID_COLUMN = 'hotspot_id'



    # --- Execution ---

    sync_hotspot_data(

        csv_a_path=CSV_A,

        csv_b_path=CSV_B,

        dir_a_root=DIR_A_ROOT,

        dir_b_root=DIR_B_ROOT,

        id_column=HOTSPOT_ID_COLUMN

    )
