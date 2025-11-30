import os

import rasterio

import numpy as np

import xarray as xr

import pandas as pd



def process_all_hotspots(csv_path, output_dir):

    # 1. LOAD HOTSPOTS

    df = pd.read_csv(csv_path)

    

    # 2. INITIALIZE DATA REGISTRY

    # Dictionary structure: { 'hotspot_id': numpy_array of shape (3, 15, 12) }

    # 3 params, 15 years (2010-2024), 12 months

    hotspot_data = {}

    

    # Pre-fill the dictionary with empty arrays for every hotspot

    # This avoids creating new arrays constantly

    for index, row in df.iterrows():

        hotspot_data[row["hotspot_id"]] = np.zeros((3, 15, 12), dtype=np.float32)



    params = ['tmax', 'tmin', 'prec']

    

    # 3. OUTER LOOP: PARAMETERS

    for p_idx, p in enumerate(params):

        print(f"Processing parameter: {p}...")

        

        # Define your directory sequence here

        directories = [

            f"wc2.1_cruts4.09_2.5m_{p}_2010-2019",

            f"wc2.1_cruts4.09_2.5m_{p}_2020-2024"

        ]

        

        # Track the total month count for this parameter (0 to 179)

        total_month_idx = 0

        

        # 4. MIDDLE LOOP: DIRECTORIES & FILES

        for directory in directories:

            if not os.path.exists(directory):

                print(f"Warning: Directory {directory} not found. Skipping.")

                continue

                

            files = sorted([f for f in os.listdir(directory) if f.endswith('.tif')])

            

            for filename in files:

                file_path = os.path.join(directory, filename)

                

                # --- OPTIMIZATION START ---

                # We open the file ONCE per month, not per hotspot.

                with rasterio.open(file_path) as src:

                    # Read the entire band into memory once

                    # output shape: (height, width)

                    raster_data = src.read(1) 

                    

                    # 5. INNER LOOP: EXTRACT DATA FOR ALL HOTSPOTS

                    # We iterate through the dataframe to get coords for every hotspot

                    for index, row in df.iterrows():

                        hid = row["hotspot_id"]

                        lat = row["lat"]

                        lon = row["lon"]

                        

                        try:

                            # Convert lat/lon to row/col

                            # index() is fast, but if performance is still tight, 

                            # we can cache these row/col pairs since they likely won't change between files.

                            r, c = src.index(lon, lat)

                            

                            # Extract value

                            val = raster_data[r, c]

                            

                            # Map linear month index to (year, month)

                            # Year index: 0-14, Month index: 0-11

                            y_idx = total_month_idx // 12

                            m_idx = total_month_idx % 12

                            

                            # Update the specific hotspot's array in our registry

                            if y_idx < 15: # Safety check for array bounds

                                hotspot_data[hid][p_idx, y_idx, m_idx] = val

                                

                        except IndexError:

                            # Handle cases where lat/lon might be out of bounds

                            pass

                # --- OPTIMIZATION END ---

                

                total_month_idx += 1



    # 6. SAVE TO DISK

    print("Processing complete. Saving NetCDF files...")

    

    # Define coordinates for xarray

    coords = {

        'param': params,

        'year': np.arange(2010, 2025),

        'month': np.arange(1, 13)

    }



    for hid, array in hotspot_data.items():

        da = xr.DataArray(

            data=array,

            dims=['param', 'year', 'month'],

            coords=coords,

            name='env_data'

        )

        

        save_path = os.path.join(output_dir, f"{hid}.nc")

        da.to_netcdf(save_path, mode='w')



if __name__ == "__main__":

    input_csv = "./SatBird_data_complete/all_hotspots_cleaned.csv"

    output_folder = "./SatBird_data_complete/env_timeseries/"

    

    # Safety check for directory

    if os.path.exists(output_folder):

        # Optional: remove this raise if you want to allow overwriting

        # raise ValueError('Directory already exists.')

        pass

    else:

        os.makedirs(output_folder)

        

    process_all_hotspots(input_csv, output_folder)
