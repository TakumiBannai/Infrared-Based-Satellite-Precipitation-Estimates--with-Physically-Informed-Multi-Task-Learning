# %%
# Download the data
import os
from datetime import datetime, timedelta
import subprocess
import gzip
import pygrib
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS


def download_data(year, month, day, hour, save_dir):
    # Generate Target URL
    path = f"https://mtarchive.geol.iastate.edu/{year}/{month}/{day}/mrms/ncep/PrecipRate/PrecipRate_00.00_{year}{month}{day}-{hour}0000.grib2.gz"
    print("Start: ", f"PrecipRate_00.00_{year}{month}{day}-{hour}0000.grib2.gz")
    # Create the directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Download the data
    script = f"wget {path} -P {save_dir}"
    subprocess.run(script, shell=True)


def read_mrms_grib2_bz(file_path):
    # Decompress bz2 file
    with gzip.open(file_path, 'rb') as bz2_file:
        grib_data = bz2_file.read()
    # Create temporary grib2 file
    temp_grib_file = 'temp_mrms.grib2'
    with open(temp_grib_file, 'wb') as grib_file:
        grib_file.write(grib_data)
    # Read grib2 file
    grbs = pygrib.open(temp_grib_file)
    # Get the first message (usually MRMS data has only one message)
    grb = grbs[1]
    # Get data and latitude/longitude
    data, lats, lons = grb.data()
    # Remove temporary file
    os.remove(temp_grib_file)
    return data, lats, lons


# Convert and save as GeoTIF
def save_as_geotiff(data, lats, lons, output_file):
    # Get latitude and longitude ranges
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    # Adjust longitude to -180 ~ 180 range
    lons = (lons + 180) % 360 - 180
    lon_min, lon_max = lons.min(), lons.max()
    # Calculate latitude/longitude resolution
    lat_res = (lat_max - lat_min) / data.shape[0]
    lon_res = (lon_max - lon_min) / data.shape[1]
    # Create affine transformation matrix
    transform = from_origin(lon_min, lat_max, lon_res, lat_res)
    # Save as GeoTIFF file
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=CRS.from_epsg(4326),  # WGS84 coordinate system
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def download_and_process_data(start_date, end_date, time_interval):
    current_time = start_date
    while current_time <= end_date:
        year = current_time.strftime("%Y")
        month = current_time.strftime("%m")
        day = current_time.strftime("%d")
        hour = current_time.strftime("%H")
        # Download data
        save_dir = f"./data/{year}/{month}/{day}/"
        os.makedirs(save_dir, exist_ok=True)
        download_data(year, month, day, hour, save_dir)
        # Read the downloaded data
        try:
            file_path = f"{save_dir}PrecipRate_00.00_{year}{month}{day}-{hour}0000.grib2.gz"
            precipitation_data, latitudes, longitudes = read_mrms_grib2_bz(file_path)
            # Save as GeoTIFF
            output_file = f"{save_dir}PrecipRate_00.00_{year}{month}{day}-{hour}0000.tif"
            save_as_geotiff(precipitation_data, latitudes, longitudes, output_file)
            print(f"Done: {output_file}.tif")

        except Exception as e:
            # If an error occurs, records the file path in a text file
            with open("failed_files.txt", "a") as f:
                f.write(file_path + "\n")
            print(f"Failed: {file_path}")
        
        # Increment the current time by the specified interval
        current_time += time_interval


# %%
# Define the start and end dates
start_date = datetime(2019, 2, 10, 0, 0)
end_date = datetime(2021, 12, 31, 23, 0)
time_interval = timedelta(hours=1)
# Execute the function
download_and_process_data(start_date, end_date, time_interval)


# %%
# Check download status
import glob
import os
import sys
import shutil
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling


# Function to create date list
def make_date_list(start, end, freq, day_index):
    if day_index is None:
        date_list = pd.date_range(start=start, end=end, freq=freq)
        # YY-M-DD-HH
        date_list = date_list.strftime("%Y-%m-%d-%H")
        return date_list
    else:
        date_list = pd.date_range(start=start, end=end, freq=freq)
        # YY-M-DD-HH
        date_list_f = date_list.strftime("%Y-%m-%d-%H")
        yyyy, mm, dd, hh = date_list_f[day_index].split("-")
        # print(f"Targeting date: {date_list_f[day_index]}")
        # YY-DDD-HH (3-digits day of year)
        date_list_f = date_list.strftime("%Y-%j-%H")
        yyyy, ddd, hh = date_list_f[day_index].split("-")
        return yyyy, mm, dd, hh, ddd


def get_path(yyyy, mm, dd, hh, ddd):
    mrms_target = glob.glob(f"../mrms/data/{yyyy}/{mm}/{dd}/*{yyyy}{mm}{dd}-{hh}0000_aoi.tif")
    # Assert error if number of files is not 1
    assert len(mrms_target) == 1, f"MRMS: Error in {[yyyy, mm, dd, hh, ddd]}"
    return mrms_target[0]


start_date = "2019-1-1"
end_date = "2019-12-31"
print(f"Checking...{start_date} to {end_date}")

# Data set completion (Missing data)
valid_date, invalid_date = [], []
n_day_index = len(make_date_list(start_date, end_date, freq="h", day_index=None))
for idx in range(n_day_index):
    try:
        yyyy, mm, dd, hh, ddd = make_date_list(start_date, end_date, freq="h", day_index=idx)
        paths = get_path(yyyy, mm, dd, hh, ddd)
        valid_date.append([yyyy, mm, dd, hh, ddd])
    # Confim Missing data
    except AssertionError as e:
        error_message = str(e)
        print(error_message)
        invalid_date.append([yyyy, mm, dd, hh, ddd])
print("Total date: ", n_day_index)
print("Valid date (Missing data): ", len(valid_date))
print("Invalid date (Missing data): ", len(invalid_date))


# %%
