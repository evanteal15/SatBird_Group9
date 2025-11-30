import csv

import pandas as pd

from data_processing.satellite.utils import bounding_box_from_centre

def main():

    # Path of all individual CSV files for each hotspot

    src_csv_folder = 'SatBird_data_complete/all_hotspots_cleaned.csv'

    dest_csv_folder = 'SatBird_data_complete/polygons.csv'

    # read in src as a DataFrame

    data = pd.read_csv(src_csv_folder)

    print(data.head())

    new_df = pd.DataFrame(columns=['hotspot_id', 'geometry'])


    for _, row in data.iterrows():

        # lets extract the lat and long

        lat = row['lat']

        lon = row['lon']

        id = row['hotspot_id']

        polygo = bounding_box_from_centre(lat, lon, 5000)

        new_df = new_df.append({'hotspot_id':id, 'geometry':polygo}, ignore_index=True)


    new_df.to_csv(dest_csv_folder, index=False, mode='w')

if __name__ == "__main__":

    main()
