import pandas as pd

import numpy as np

from pathlib import Path



def count_duplicate_hotspots(

    file_path: str = "/network/projects/ecosystem-embeddings/ebird_new/summer_hotspots_with_bioclim_withnan.csv",

) -> None:

    """

    Analyzes a CSV containing hotspot data (including latitude, longitude, and ID)

    to identify and count instances where multiple unique hotspot IDs share the

    exact same latitude and longitude coordinates.



    Args:

        file_path: The full path to the input CSV file.

    """

    print(f"--- Analyzing file: {file_path} ---")



    # --- 1. Load Data ---

    try:

        df = pd.read_csv(file_path)

    except FileNotFoundError:

        print(f"ERROR: File not found at {file_path}. Please check the path.")

        return

    except Exception as e:

        print(f"ERROR: Could not read file. Check CSV format. Details: {e}")

        return



    # Ensure column names are clean and present

    df.columns = df.columns.str.strip()

    required_cols = ['hotspot_id', 'lat', 'lon']

    if not all(col in df.columns for col in required_cols):

        missing = [col for col in required_cols if col not in df.columns]

        print(f"ERROR: Input CSV is missing required columns: {missing}")

        print(f"Available columns: {df.columns.to_list()}")

        return



    # --- 2. Initial Cleanup ---

    # Drop rows that are duplicates on hotspot_id (in case the input file has duplicates before merge)

    initial_rows = len(df)

    df = df.drop_duplicates(subset=["hotspot_id"], keep="first")

    print(f"Total unique hotspot IDs found in file: {len(df)}")

    print(f"Removed {initial_rows - len(df)} rows that were duplicates of the same hotspot ID.")



    # --- 3. Group by Coordinates ---

    # Group the DataFrame by latitude and longitude.

    grouped = df.groupby(['lat', 'lon'])



    # --- 4. Identify Duplicates (Groups with size > 1) ---

    # Filter the groups to find those that contain more than one row (i.e., more than one Hotspot ID)

    duplicate_groups = grouped.filter(lambda x: len(x) > 1)



    # --- 5. Final Counting and Summary ---

    total_duplicate_hotspots = len(duplicate_groups)

    print("\n--- Summary ---")



    if total_duplicate_hotspots == 0:

        print("RESULT: No instances found where multiple Hotspot IDs share the same latitude/longitude.")

        return



    # Count how many unique *locations* have duplicates

    num_duplicate_locations = duplicate_groups[['lat', 'lon']].drop_duplicates().shape[0]



    # Count the total number of duplicate IDs involved

    total_ids_involved = duplicate_groups['hotspot_id'].nunique()



    print(f"Total Unique Hotspot IDs found with duplicate coordinates: {total_ids_involved}")

    print(f"Total number of unique (lat, lon) locations involved: {num_duplicate_locations}")



    # --- 6. Detailed Output (Optional) ---

    print("\n--- Details of Duplicated Locations (First 10) ---")

    duplicate_hotspot_ids = duplicate_groups.groupby(['lat', 'lon'])['hotspot_id'].apply(list)

    

    # Create a nice summary DataFrame for display

    summary_df = pd.DataFrame({

        'Duplicate Hotspot IDs': duplicate_hotspot_ids,

        'Count': duplicate_hotspot_ids.apply(len)

    })

    

    # Reset index to make lat/lon columns

    summary_df = summary_df.reset_index()

    

    print(summary_df.sort_values(by='Count', ascending=False).head(10).to_markdown(index=False))

    

    # Save the list of duplicate IDs to a file, similar to the original script

    output_path = Path("./duplicate_hotspot_list.txt")

    print(f"\nSaving list of all duplicate ID groups to {output_path.name}")

    with output_path.open("w") as f:

        for ids in duplicate_hotspot_ids.values:

            f.write(" ".join(ids) + "\n")





if __name__ == "__main__":

    # NOTE: Set the correct file path for your environment.

    # The default path used in the original script is provided here:

    DEFAULT_PATH = "./SatBird_data_23-25/all_summer_hotspots.csv"

    count_duplicate_hotspots(file_path=DEFAULT_PATH)
