import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
import rasterio
import rasterio.mask
from rasterio.transform import Affine
from fiona.crs import from_epsg
import geopandas as gpd
import subprocess
from tqdm import tqdm

def sh_command(script):
    output = subprocess.run(script.split(" "), stdout=subprocess.PIPE, encoding='UTF-8')
    print(output.stdout)
    
def get_geotif_params(lon, lat, confirmation=False):
    lat_start, lat_end,lat_size = lat[0], lat[-1], lat.size
    lon_start, lon_end, lon_size = lon[0],lon[-1], lon.size
    if confirmation is True:
        print("lat start from: ", lat_start, "end width: ", lat_end)
        print("lon start from: ", lon_start, "end width: ", lon_end)
        print("Shape-size for lat ", lat_size, "for lon ", lon_size)
    height_of_pixel = (lat_start + -1*lat_end)/lat_size
    with_of_pixel = (-1*lon_start + lon_end)/lon_size
    lon_upper_left = lon_start
    lat_upper_left = lat_start
    if confirmation is True:
        print("Tif params -------")
        print("height_of_pixel", height_of_pixel)
        print("with_of_pixel", with_of_pixel)
        print("lon_upper_left", lon_upper_left)
        print("lat_upper_left", lat_upper_left)
    return height_of_pixel, with_of_pixel, lon_upper_left, lat_upper_left


def convert_np2tif(data, out_file,
                   with_of_pixel, height_of_pixel, lon_upper_left, lat_upper_left,
                   crs):
    affine = Affine(with_of_pixel, 0, lon_upper_left, 0, -1*height_of_pixel, lat_upper_left)
    with rasterio.open(out_file, 'w',
                        driver='GTiff',
                        height=data.shape[0],
                        width=data.shape[1],
                        count=1,
                        dtype=data.dtype,
                        crs=crs,
                        transform=affine
                        ) as dst:
        dst.write(data.reshape(1,data.shape[0],data.shape[1]))  

# Data loading
def dataload(year="2013", date="001", hour="00"):
    url = f"https://persiann.eng.uci.edu/CHRSdata/PERSIANN-CCS/hrly/{year}/rgccs1h{year[2:]}{date}{hour}.bin.gz"
    print(url)
    save_name = f"{year}{date}{hour}.tif"
    # Downloading
    sh_command(f"wget {url}") 
    file_name = url.split("/")[-1]
    sh_command(f"gunzip {file_name}")# Unzip
    # Read binary as Big-endian(>) and short-integer(h)
    arr = np.fromfile(file_name[:-3], dtype='>h',sep='').reshape(3000, 9000)
    # Change Unit (mm/h)
    arr = arr/100
    arr = np.hstack((arr[:, 4500:], arr[:, :4500]))
    # Save as TIF
    convert_np2tif(arr, save_name,
                   with_of_pixel, height_of_pixel, lon_upper_left, lat_upper_left,
                   set_crs)
    # Move files
    os.makedirs("raw", exist_ok=True)
    os.makedirs("tif", exist_ok=True)
    sh_command(f"mv {file_name[:-3]} raw") 
    sh_command(f"mv {save_name} tif") 


lon = np.arange(-179.98, 180, 0.04)
lat = np.arange(-59.98, 60, 0.04)[::-1]
set_crs = from_epsg(4326)
with_of_pixel, height_of_pixel, lon_upper_left, lat_upper_left = get_geotif_params(lon, lat)


year = "2019"
dates = np.arange(1, 366, 1)  # for 1 year
hours = np.arange(0, 24, 1)  # for 1 day

# %%
for date in tqdm(dates):
    for hour in hours:
        print(f"Start. {year}-{date}-{hour}")
        try:
            dataload(year=year, date=f"{date:0=3}", hour=f"{hour:0=2}")
        except Exception as e:
            # If an error occurs, records the file path in a text file
            with open("failed_files.txt", "a") as f:
                f.write(f"Error in. {year}-{date}-{hour}" + "\n")
            print(f"Error in. {year}-{date}-{hour}")

# %%
# TEMP) 2020, 2021を終えたら削除
year = "2020"
for date in tqdm(dates):
    for hour in hours:
        print(f"Start. {year}-{date}-{hour}")
        try:
            dataload(year=year, date=f"{date:0=3}", hour=f"{hour:0=2}")
        except Exception as e:
            # If an error occurs, records the file path in a text file
            with open("failed_files.txt", "a") as f:
                f.write(f"Error in. {year}-{date}-{hour}" + "\n")
            print(f"Error in. {year}-{date}-{hour}")

year = "2021"
for date in tqdm(dates):
    for hour in hours:
        print(f"Start. {year}-{date}-{hour}")
        try:
            dataload(year=year, date=f"{date:0=3}", hour=f"{hour:0=2}")
        except Exception as e:
            # If an error occurs, records the file path in a text file
            with open("failed_files.txt", "a") as f:
                f.write(f"Error in. {year}-{date}-{hour}" + "\n")
            print(f"Error in. {year}-{date}-{hour}")


