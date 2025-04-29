# %%
import os
from goes2go import GOES
import pandas as pd
from datetime import datetime
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import subprocess
import geopandas as gpd
import json

# Path to GeoJSON file
aoi_path = '../AOI/AOI_conus.geojson'

# Radiance to Brightness Temperature
def radiance_to_temperature(ds):
    # Radiance values
    radiance = ds['Rad'].values
    # Get calibration coefficients
    fk1 = ds['planck_fk1'].values
    fk2 = ds['planck_fk2'].values
    bc1 = ds['planck_bc1'].values
    bc2 = ds['planck_bc2'].values
    # Convert radiance to brightness temperature
    temp = (fk2 / np.log((fk1 / radiance) + 1) - bc1) / bc2
    return temp


def crop_aoi(aoi_path, input_tif, output_tif):
    # Load the GeoJSON file defining the Area of Interest (AOI)
    with open(aoi_path) as f:
        aoi = json.load(f)
    # Convert GeoJSON to a Shapefile for GDAL
    # gdf = gpd.GeoDataFrame.from_features(aoi['features'], crs="EPSG:4326")
    # aoi_shapefile = "../AOI/aoi.shp"
    # gdf.to_file(aoi_shapefile)

    aoi_shapefile = "../AOI/aoi.shp"
    # Crop the image to the area of interest
    script = f"gdalwarp -cutline {aoi_shapefile} -crop_to_cutline {input_tif} {output_tif}"
    subprocess.run(script, shell=True)


def convert_netcdf2geotif(path):
    """
    Convert the netCDF file to GeoTIFF file
    Convert Radiance to Brightness Temperature

    Args:
        path (_type_): path to the netCDF file (GEOS-16)
    """
    # Load nedcdf data
    ds = xr.open_dataset(path)
    # Radiance values
    temperature = radiance_to_temperature(ds)
    # Add the brightness temperature to the dataset
    ds['Temp'] = (('y', 'x'), temperature)
    # Add Cordinate information to Temp
    ds['Temp'].attrs = ds['Rad'].attrs
    # Save as netCDF file (named "temp.nc")
    ds.to_netcdf('temp.nc')
    # Convert to TIF
    out_fname = path.replace(".nc", "")
    # Convert GeoTIFF file (convert "Temp" from temp.nc to TIF)
    script = f"gdal_translate NETCDF:temp.nc:Temp {out_fname}.tif"
    subprocess.run(script, shell=True)
    # Change CRS to EPSG:4326
    script = f"gdalwarp -t_srs EPSG:4326 -r near {out_fname}.tif {out_fname}_epsg.tif"
    subprocess.run(script, shell=True)
    # Crop the image to the area of interest
    crop_aoi(aoi_path, f"{out_fname}_epsg.tif", f"{out_fname}_aoi.tif")
    # Remove the intermediate output file
    subprocess.run("rm temp.nc", shell=True)
    subprocess.run(f"rm {out_fname}.tif", shell=True)
    subprocess.run(f"rm {out_fname}_epsg.tif", shell=True)


# Convert all netCDF files to GeoTIFF files
def convert_all_nc2tif():
    path_nc = "./data/noaa-goes16/ABI-L1b-RadC/*/*/*/*.nc"
    nc_files = sorted(glob.glob(path_nc))

    for path in nc_files:
        convert_netcdf2geotif(path)

    # ncファイルを削除
    subprocess.run(f"rm {path_nc}", shell=True)


# %%
# Parameters
start = datetime(2018, 1, 1, 0, 0, 0)
end = datetime(2018, 12, 31, 23, 59, 59)
target_bands = [8, 9, 10, 11, 12, 13, 14, 15, 16]
dir = 'data/'
print("Downloading data from: ", start, " to ", end)

# %%
G = GOES(satellite=16, product="ABI-L1b-Rad", domain='C',
         bands=target_bands)
# Get data in 30-minute intervals
times = pd.date_range(start, end, freq='30min')

# %%
# Downloading and converting to GeoTIFF
for t in times:
    # Run Downloading (get data closest to timestamp)
    try:
        bands = G.nearesttime(t, save_dir=dir, download=True)
    except Exception as e:
        with open("failed_files.txt", "a") as f:
            f.write(f"{t}: {e}\n")
    convert_all_nc2tif()
    print(f"Completed: {t}")


