# %%
""" Preprocess Pipeline
- Check Complete dataset. 1 (Exclude dataset including missing data)
- Check Complete dataset. 2 (Exclude dataset including non-sequential data)
- Resample to 2km
- Split to 112 by 112 pixels patch
- Save train/val and test dataset by time step

Usage:
python preprosess.py 2018-1-1 2018-12-31
"""

# %%
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
    cw_target = glob.glob(f"../era5/tif/total_column_cloud_liquid_water_*_{yyyy}{mm}{dd}{hh}00_aoi.tif")
    assert len(cw_target) == 1, f"ERA5 Cloud Water: Error in {[yyyy, mm, dd, hh, ddd]}"
    ci_target = glob.glob(f"../era5/tif/total_column_cloud_ice_water_*_{yyyy}{mm}{dd}{hh}00_aoi.tif")
    assert len(ci_target) == 1, f"ERA5 Ice Water: Error in {[yyyy, mm, dd, hh, ddd]}"
    goes_ch08_target = glob.glob(f"../goes_conus/data/noaa-goes16/ABI-L1b-RadC/{yyyy}/{ddd}/{hh}/OR_ABI-L1b-RadC-M*C08_G16_s{yyyy}{ddd}{hh}0*_aoi.tif")
    assert len(goes_ch08_target) == 1, f"GOES Ch08: Error in {[yyyy, mm, dd, hh, ddd]}"
    goes_ch10_target = glob.glob(f"../goes_conus/data/noaa-goes16/ABI-L1b-RadC/{yyyy}/{ddd}/{hh}/OR_ABI-L1b-RadC-M*C10_G16_s{yyyy}{ddd}{hh}0*_aoi.tif")
    assert len(goes_ch10_target) == 1, f"GOES Ch10: Error in {[yyyy, mm, dd, hh, ddd]}"
    goes_c11_target = glob.glob(f"../goes_conus/data/noaa-goes16/ABI-L1b-RadC/{yyyy}/{ddd}/{hh}/OR_ABI-L1b-RadC-M*C11_G16_s{yyyy}{ddd}{hh}0*_aoi.tif")
    assert len(goes_c11_target) == 1, f"GOES Ch11: Error in {[yyyy, mm, dd, hh, ddd]}"
    goes_c14_target = glob.glob(f"../goes_conus/data/noaa-goes16/ABI-L1b-RadC/{yyyy}/{ddd}/{hh}/OR_ABI-L1b-RadC-M*C14_G16_s{yyyy}{ddd}{hh}0*_aoi.tif")
    assert len(goes_c14_target) == 1, f"GOES Ch14: Error in {[yyyy, mm, dd, hh, ddd]}"
    goes_c15_target = glob.glob(f"../goes_conus/data/noaa-goes16/ABI-L1b-RadC/{yyyy}/{ddd}/{hh}/OR_ABI-L1b-RadC-M*C15_G16_s{yyyy}{ddd}{hh}0*_aoi.tif")
    assert len(goes_c15_target) == 1, f"GOES Ch15: Error in {[yyyy, mm, dd, hh, ddd]}"
    return mrms_target[0], cw_target[0], ci_target[0], goes_ch08_target[0], goes_ch10_target[0], goes_c11_target[0], goes_c14_target[0], goes_c15_target[0]


# read tif
def read_tif(path, mask_value=None):
    with rasterio.open(path) as src:
        arr = src.read()
        if mask_value is not None:
            arr = np.where(arr == mask_value, np.nan, arr)
        print("fname: ", os.path.basename(path))
        print("Shape size: ", arr.shape)
        # Stats
        print("min: ", np.nanmin(arr))
        print("max: ", np.nanmax(arr))
        print("mean: ", np.nanmean(arr))
        return arr


def resample_tif(input_tif, h_pixel=677, w_pixel=1761, mask_value=None):
    with rasterio.open(input_tif) as dataset:
        arr = dataset.read(
            out_shape=(
                dataset.count,
                int(h_pixel),
                int(w_pixel)),
            resampling=Resampling.nearest)
    if mask_value is not None:
        arr = np.where(arr == mask_value, np.nan, arr)
    # print("Shape size: ", arr.shape)
    # Stats
    # print("min: ", np.nanmin(arr))
    # print("max: ", np.nanmax(arr))
    # rint("mean: ", np.nanmean(arr))
    return arr


def extract_patches(features, patch_size):
    channels, height, width = features.shape
    patches = []
    for i in range(0, height - patch_size + 1, patch_size):
        for j in range(0, width - patch_size + 1, patch_size):
            patch = features[:, i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return np.array(patches)


def move_npy_files(src_dir):
    # Create train and test directories if they don't exist
    train_dir = os.path.join(src_dir, "train")
    test_dir = os.path.join(src_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    # Get all .npy files in the source directory
    npy_files = glob.glob(os.path.join(src_dir, "*.npy"))
    for file in npy_files:
        # Extract the year from the filename
        filename = os.path.basename(file)
        year = int(filename.split('_')[1][:4])
        if 2018 <= year <= 2020:
            shutil.move(file, train_dir)
        elif year == 2021:
            shutil.move(file, test_dir)


# %%
# Parameter setting
args = sys.argv
start_date = args[1]
end_date = args[2]

# %%
print(f"Processing...{start_date} to {end_date}")

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

# %%
# Data set completion (Sequential data)
judge = []
for i in range(len(valid_date)):
    # Time t
    yyyy, mm, dd, hh, ddd = valid_date[i]
    date_t = pd.to_datetime(f"{yyyy}-{mm}-{dd} {hh}:00:00")
    # Time t-1
    date_t1 = date_t - pd.Timedelta(hours=1)
    # Check if date_t1 exists in valid_date
    date_t1_str = date_t1.strftime("%Y-%m-%d-%H")
    j = date_t1_str in [f"{yyyy}-{mm}-{dd}-{hh}" for yyyy, mm, dd, hh, ddd in valid_date]
    judge.append(j)

valid_date = np.array(valid_date)[judge].tolist()
print("Valid date (Sequential data): ", len(valid_date))

# %%
# Generate feature dataset
for i in range(len(valid_date)):
    # Dataset at time t
    yyyy, mm, dd, hh, ddd = valid_date[i]
    paths_t = get_path(yyyy, mm, dd, hh, ddd)
    # Dataset at time t-1
    yyyy, mm, dd, hh, ddd = valid_date[i-1]
    paths_t1 = get_path(yyyy, mm, dd, hh, ddd)

    # Read tif files
    mrms_t = resample_tif(paths_t[0], h_pixel=677, w_pixel=1761, mask_value=-3)
    ch08_t = resample_tif(paths_t[3], h_pixel=677, w_pixel=1761)
    ch08_t1 = resample_tif(paths_t1[3], h_pixel=677, w_pixel=1761)
    ch10_t = resample_tif(paths_t[4], h_pixel=677, w_pixel=1761)
    ch10_t1 = resample_tif(paths_t1[4], h_pixel=677, w_pixel=1761)
    ch11_t = resample_tif(paths_t[5], h_pixel=677, w_pixel=1761)
    ch11_t1 = resample_tif(paths_t1[5], h_pixel=677, w_pixel=1761)
    ch14_t = resample_tif(paths_t[6], h_pixel=677, w_pixel=1761)
    ch14_t1 = resample_tif(paths_t1[6], h_pixel=677, w_pixel=1761)
    ch15_t = resample_tif(paths_t[7], h_pixel=677, w_pixel=1761)
    ch15_t1 = resample_tif(paths_t1[7], h_pixel=677, w_pixel=1761)
    cw_t = resample_tif(paths_t[1], h_pixel=677, w_pixel=1761)
    cw_t1 = resample_tif(paths_t1[1], h_pixel=677, w_pixel=1761)
    ci_t = resample_tif(paths_t[2], h_pixel=677, w_pixel=1761)
    ci_t1 = resample_tif(paths_t1[2], h_pixel=677, w_pixel=1761)
    # Concatinate
    features = np.concatenate([mrms_t, ch08_t, ch08_t1, ch10_t, ch10_t1, ch11_t, ch11_t1, ch14_t, ch14_t1, ch15_t, ch15_t1, cw_t, cw_t1, ci_t, ci_t1], axis=0)
    # Split to 112 by 112 pixels patch
    patches = extract_patches(features, patch_size=112)
    # Save train/val and test dataset by time step
    np.save(f"./patches_{yyyy}{mm}{dd}{hh}.npy", patches)

# Organize files
move_npy_files("./")
print(f"Done...{start_date} to {end_date}")



