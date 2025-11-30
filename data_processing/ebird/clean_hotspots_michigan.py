import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import tifffile
import os

# --- Paths ---
input_file = "data_processing/ebird/summer_hotspots_with_bioclim_splits_michigan.csv"
output_file_final = "data_processing/ebird/summer_hotspots_clean_michigan.csv"
shapefile_path = "data_processing/ebird/cb_2018_us_nation_5m.shp"
raster_folder = "data_processing/ebird/rasters_new/summer_rasters/"

def filter_by_rasters(df):
    available_rasters = [r.strip(".tif") for r in os.listdir(raster_folder)]
    df = df[df['hotspot_id'].isin(available_rasters)]
    return df

def filter_by_geography(df):
    gdf = gpd.read_file(shapefile_path)
    indices_to_drop = [i for i, row in enumerate(df[['lon', 'lat']].values) if not gdf.geometry[0].contains(Point(row[0], row[1]))]
    df = df.drop(indices_to_drop)
    return df

def filter_by_size(df):
    indices_to_drop = []
    for i, hotspot in enumerate(df['hotspot_id'].values):
        w,h,b = tifffile.imread(os.path.join(raster_folder, f"{hotspot}.tif")).shape
        if w<128 or h<128:
            indices_to_drop.append(i)
    df = df.drop(indices_to_drop)
    return df

if __name__ == "__main__":
    df = pd.read_csv(input_file)
    df = filter_by_rasters(df)
    df = filter_by_geography(df)
    df = filter_by_size(df)
    df.to_csv(output_file_final, index=False)
    print(f"Clean Michigan hotspots saved in {output_file_final}")
