# %% Setting
import time
import os
import glob
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.dates as mdates
import rasterio
from rasterio.enums import Resampling
import cartopy.crs as ccrs
from matplotlib.gridspec import GridSpec

# Constants
DATA_PATHS = {
    "mrms": "../data/mrms/data/{yyyy}/{mm}/{dd}/*{yyyy}{mm}{dd}-{hh}0000_aoi.tif",
    "cw": "../data/era5/tif/total_column_cloud_liquid_water_*_{yyyy}{mm}{dd}{hh}00_aoi.tif",
    "ci": "../data/era5/tif/total_column_cloud_ice_water_*_{yyyy}{mm}{dd}{hh}00_aoi.tif",
    "goes_ch08": "../data/goes_conus/data/noaa-goes16/ABI-L1b-RadC/{yyyy}/{ddd}/{hh}/OR_ABI-L1b-RadC-M*C08_G16_s{yyyy}{ddd}{hh}0*_aoi.tif",
    "ccs": "../data/persiann_ccs/tif/{yyyy}{ddd}{hh}_aoi.tif",
    "scampr": "../data/scampr/remapped/{yyyy}/{ddd}/RRR_CONUS{yyyy}{ddd}{hh}.001h_aoi.tif",
    "rqi": "../data/mrms_quality/rqi_data/{yyyy}/{mm}/{dd}/RadarQualityIndex_00.00_{yyyy}{mm}{dd}-{hh}0000_cropped.tif"
}

# Prediction data shape
PREDICTION_SHAPE = (1, 672, 1680)

def make_date_list(start: str, end: str, freq: str = "h", day_index: Optional[int] = None) -> Union[List[str], Tuple[str, str, str, str, str]]:
    """
    Generate a list of dates for the specified period.

    Parameters:
    ----------
    start : str
        Start date (YYYY-MM-DD format)
    end : str
        End date (YYYY-MM-DD format)
    freq : str, optional
        Frequency (default: "h" = hourly)
    day_index : int, optional
        Index of a specific date. If specified, returns detailed information for that day.

    Returns:
    -------
    Union[List[str], Tuple[str, str, str, str, str]]
        If day_index is None: List of date strings
        If day_index is specified: (yyyy, mm, dd, hh, ddd) format information for that day
    """
    date_list = pd.date_range(start=start, end=end, freq=freq)
    
    if day_index is None:
        # Return date list in YY-M-DD-HH format
        return date_list.strftime("%Y-%m-%d-%H")
    else:
        # Return detailed information for the specified date
        date_str = date_list[day_index].strftime("%Y-%m-%d-%H")
        yyyy, mm, dd, hh = date_str.split("-")
        
        # Get day of year
        ddd = date_list[day_index].strftime("%j")
        
        return yyyy, mm, dd, hh, ddd

def get_path(yyyy: str, mm: str, dd: str, hh: str, ddd: str, 
           required_vars: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Get file paths for each data source.
    
    Parameters:
    ----------
    yyyy, mm, dd, hh, ddd : str
        Date and time components
    required_vars : List[str], optional
        List of required data sources. If None, all data sources are retrieved
        
    Returns:
    -------
    Dict[str, str]
        Dictionary with data source names as keys and file paths as values
    
    Raises:
    ------
    AssertionError
        If required files are not found
    """
    # Get all data sources by default
    if required_vars is None:
        required_vars = list(DATA_PATHS.keys())
    
    # Search for files using formatted paths
    paths = {}
    for name, path_pattern in DATA_PATHS.items():
        # Process only required data sources
        if name in required_vars:
            path = glob.glob(path_pattern.format(yyyy=yyyy, mm=mm, dd=dd, hh=hh, ddd=ddd))
            # Assert error if number of files is not 1
            assert len(path) == 1, f"{name.upper()}: Error in {[yyyy, mm, dd, hh, ddd]}"
            paths[name] = path[0]
    
    return paths

def read_tif(path: str, mask_value: Optional[float] = None, verbose: bool = False) -> np.ndarray:
    """
    Read a GeoTIFF file.
    
    Parameters:
    ----------
    path : str
        File path
    mask_value : float, optional
        Mask value (this value will be replaced with NaN)
    verbose : bool, optional
        Whether to output detailed information
        
    Returns:
    -------
    np.ndarray
        Read array
    """
    with rasterio.open(path) as src:
        arr = src.read()
        if mask_value is not None:
            arr = np.where(arr == mask_value, np.nan, arr)
        
        if verbose:
            print("Filename: ", os.path.basename(path))
            print("Shape: ", arr.shape)
            print("Min value: ", np.nanmin(arr))
            print("Max value: ", np.nanmax(arr))
            print("Mean value: ", np.nanmean(arr))
            
        return arr

def read_data(data_type: str, model: str, start_date: str, end_date: str, freq: str = "h", 
              n_day_index: int = 1, overlap_data: bool = False) -> List[np.ndarray]:
    """
    Read prediction or label data.
    
    Parameters:
    ----------
    data_type : str
        Specify "pred" or "label"
    model : str
        Model name
    start_date, end_date : str
        Start and end dates of the period
    freq : str
        Frequency
    n_day_index : int
        Number of days to read
    overlap_data : bool
        Whether to use overlap data
        
    Returns:
    -------
    List[np.ndarray]
        List of data
    """
    result = []
    
    for day_index in range(n_day_index):
        yyyy, mm, dd, hh, ddd = make_date_list(start_date, end_date, freq, day_index)
        
        # Build path
        base_dir = f"saved_pred/{data_type}_overlap" if overlap_data else f"saved_pred/{data_type}"
        path_data = f"./{base_dir}/{model}/{yyyy}{mm}{dd}{hh}.npy"
        
        try:
            arr = np.load(path_data)
            # Inverse log transformation
            arr = np.expm1(arr)
            result.append(arr)
        except Exception as e:
            nan_arr = np.full(PREDICTION_SHAPE, np.nan)
            result.append(nan_arr)
            
    return result

def resample_tif(input_tif: str, h_pixel: int = 677, w_pixel: int = 1761, 
                 mask_value: Optional[float] = None) -> np.ndarray:
    """
    Resample a GeoTIFF file.
    
    Parameters:
    ----------
    input_tif : str
        Input file path
    h_pixel, w_pixel : int
        Height and width of output image
    mask_value : float, optional
        Mask value
        
    Returns:
    -------
    np.ndarray
        Resampled array
    """
    with rasterio.open(input_tif) as dataset:
        arr = dataset.read(
            out_shape=(
                dataset.count,
                int(h_pixel),
                int(w_pixel)),
            resampling=Resampling.nearest)
            
    if mask_value is not None:
        arr = np.where(arr == mask_value, np.nan, arr)
        
    return arr

def plot_map(data: np.ndarray, title: str, cmap: str = "RdBu", vmin: float = -2, 
             vmax: float = 2, save_fig: Optional[str] = None) -> None:
    """
    Plot data on a map.
    
    Parameters:
    ----------
    data : np.ndarray
        Data array to plot
    title : str
        Plot title
    cmap : str
        Colormap
    vmin, vmax : float
        Value range
    save_fig : str, optional
        File path to save
    """
    lat_min, lat_max = 25, 50
    lon_min, lon_max = -130, -65

    fig, ax = plt.subplots(figsize=(18, 6), dpi=300,
                           subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    im = ax.imshow(data, origin="upper", extent=[lon_min, lon_max, lat_min, lat_max],
                   transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.coastlines()
    fig.suptitle(title, fontsize=16, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.5, pad=0.05)
    fig.tight_layout()
    
    if save_fig is not None:
        plt.savefig(save_fig)
        
    plt.show()

def prepare_datasets(start_date: str, end_date: str, 
                    load_vars: Optional[List[str]] = None,
                    overlap_data: bool = False) -> Dict[str, np.ndarray]:
    """
    Prepare datasets for analysis.
    
    Parameters:
    ----------
    start_date, end_date : str
        Data retrieval period
    load_vars : List[str], optional
        List of variables to load. If None, all variables are loaded.
        Example: ["mrms", "cw", "ci", "goes_ch08", "ccs", "scampr", "pred_mtl_cw", "pred_stl_ir", "label", "rqi"]
    overlap_data : bool
        Whether to use overlap data
        
    Returns:
    -------
    Dict[str, np.ndarray]
        Dictionary with loaded variables as keys and corresponding data as values
    """
    # Default settings for loading all variables
    all_vars = ["mrms", "cw", "ci", "goes_ch08", "ccs", "scampr", "rqi", "pred_mtl_cw", "pred_stl_ir", "label"]
    if load_vars is None:
        load_vars = all_vars

    # Set flags (whether to load each variable)
    load_flags = {var: (var in load_vars) for var in all_vars}

    freq = "h"
    n_day_index = len(make_date_list(start_date, end_date, freq="h", day_index=None))
    print("Data period:", start_date, "-", end_date)

    results = {}

    # ① Load base data
    base_vars = ["mrms", "cw", "ci", "goes_ch08", "ccs", "scampr", "rqi"]
    
    if any(load_flags[var] for var in base_vars):
        data_vars = {var: [] if load_flags[var] else None for var in base_vars}
        
        # Specify which variables to load (based on load_flags)
        required_base_vars = [var for var in base_vars if load_flags[var]]

        # Define mask values
        var_mask_values = {
            "mrms": -3,
            "scampr": -9990
        }
        
        for day_index in range(n_day_index):
            yyyy, mm, dd, hh, ddd = make_date_list(start_date, end_date, freq, day_index)
            try:
                target_files = get_path(yyyy, mm, dd, hh, ddd, required_vars=required_base_vars)
                
                for var in required_base_vars:
                    if load_flags[var]:
                        try:
                            if var in target_files:
                                mask_value = var_mask_values.get(var, None)
                                data_vars[var].append(resample_tif(target_files[var], mask_value=mask_value))
                            else:
                                PREDICTION_SHAPE_FULL_IMAGE = (1, 677, 1761)
                                data_vars[var].append(np.full(PREDICTION_SHAPE_FULL_IMAGE, np.nan))
                        except Exception as e:
                            print(f"Error ({var}, Day {day_index}): {e}")
                            PREDICTION_SHAPE_FULL_IMAGE = (1, 677, 1761)
                            data_vars[var].append(np.full(PREDICTION_SHAPE_FULL_IMAGE, np.nan))
                            
            except Exception as e:
                print(f"Path retrieval error (Day {day_index}): {e}")
                PREDICTION_SHAPE_FULL_IMAGE = (1, 677, 1761)
                for var in required_base_vars:
                    if load_flags[var]:
                        data_vars[var].append(np.full(PREDICTION_SHAPE_FULL_IMAGE, np.nan))
        # Convert to numpy array and exclude bottom-right pixels that fall out of patches
        for var in base_vars:
            if load_flags[var] and data_vars[var]:
                data_vars[var] = np.array(data_vars[var])[:, :, :672, :1680]

        results.update({k: v for k, v in data_vars.items() if v is not None})

    # ② Load predictions and labels
    if load_flags["pred_mtl_cw"]:
        exp_name = "val2"
        model_type = "MTL_CW"
        pred_mtl_cw = read_data("pred", f"{exp_name}_{model_type}", start_date=start_date,
                                end_date=end_date, freq=freq, n_day_index=n_day_index, 
                                overlap_data=overlap_data)
        results["pred_mtl_cw"] = np.array(pred_mtl_cw)
        
    if load_flags["pred_stl_ir"]:
        pred_stl_ir = read_data("pred", "val2_STL_IR", start_date=start_date,
                               end_date=end_date, freq=freq, n_day_index=n_day_index, 
                               overlap_data=overlap_data)
        results["pred_stl_ir"] = np.array(pred_stl_ir)
        
    if load_flags["label"]:
        label = read_data("label", "val2_STL_IR", start_date=start_date,
                          end_date=end_date, freq=freq, n_day_index=n_day_index, 
                          overlap_data=overlap_data)
        results["label"] = np.array(label)

    return results

def apply_masks_to_arrays(arrays_dict, rqi_array, rqi_th=0.8):
    """
    Apply masks to arrays in original shape and return data before/after masking and 1d arrays.
    
    Generated array types:
    - orig_before: Original shape array with only NaN mask applied
    - orig_after: Original shape array with NaN mask + RQI mask applied
    - 1d_before: 1d array converted from orig_before (including NaN)
    - 1d_after: 1d array converted from orig_after (including NaN)
    - 1d_valid_before: 1d_before with NaN values removed (valid data only)
    - 1d_valid_after: 1d_after with NaN values removed (high-quality data)
    
    Parameters:
    -----------
    arrays_dict : dict
        Dictionary with variable names as keys and numpy arrays as values
    rqi_array : numpy.ndarray
        Array containing RQI values
    rqi_th : float, default=0.8
        RQI threshold
        
    Returns:
    --------
    dict
        Dictionary containing processing results
        - 'orig_before': Dictionary of original shape arrays before masking
        - 'orig_after': Dictionary of original shape arrays after masking
        - '1d_before': Dictionary of 1d arrays after conversion
        - '1d_after': Dictionary of 1d arrays after masking
        - '1d_valid_before': Dictionary of 1d arrays with valid data only (before)
        - '1d_valid_after': Dictionary of 1d arrays with valid data only (after)
        - 'stats': Statistical information
        - 'nan_mask': NaN mask
        - 'rqi_mask': RQI mask
    """
    # Check array shapes
    shape = rqi_array.shape
    for var_name, arr in arrays_dict.items():
        assert arr.shape == shape, f"{var_name} shape({arr.shape}) does not match RQI shape({shape})"
    
    print("Creating masks for original shape arrays...")
    # Exclude NaN (OR condition) - No RQI filtering
    nan_mask = np.isnan(rqi_array)
    for arr in arrays_dict.values():
        nan_mask |= np.isnan(arr)
    
    # RQI mask (NaN excluded + RQI < rqi_th)
    rqi_mask = (nan_mask | (rqi_array < rqi_th))
    
    # "Before" data in original shape (only NaN excluded)
    print("\nApplying NaN mask to original arrays...")
    orig_before = {}
    
    # Include RQI
    arrays_with_rqi = {**arrays_dict, 'rqi': rqi_array}
    
    for var_name, arr in arrays_with_rqi.items():
        # Before (NaN mask only)
        arr_before = np.copy(arr)
        arr_before[nan_mask] = np.nan
        orig_before[var_name] = arr_before
    
    # "After" data in original shape (NaN excluded + RQI >= rqi_th)
    print("Applying NaN and RQI masks to original arrays...")
    orig_after = {}
    
    for var_name, arr in arrays_with_rqi.items():
        # After (NaN + RQI mask)
        arr_after = np.copy(arr)
        arr_after[rqi_mask] = np.nan
        orig_after[var_name] = arr_after
    
    # Convert to 1-d array
    print("\nConverting original arrays to 1-d arrays...")
    d1_before = {}
    d1_after = {}
    
    for var_name, arr in orig_before.items():
        d1_before[var_name] = arr.reshape(-1)
    
    for var_name, arr in orig_after.items():
        d1_after[var_name] = arr.reshape(-1)
    
    # Extract valid data only by removing NaN
    print("\nExtracting valid data from 1-d arrays...")
    # Create mask (extract data that is not NaN for any variable)
    valid_mask_before = ~np.isnan(d1_before['rqi'])
    valid_mask_after = ~np.isnan(d1_after['rqi'])
    
    d1_valid_before = {}
    d1_valid_after = {}
    
    for var_name, arr in d1_before.items():
        d1_valid_before[var_name] = arr[valid_mask_before]
    
    for var_name, arr in d1_after.items():
        d1_valid_after[var_name] = arr[valid_mask_after]
    
    # Calculate statistics
    total_pixels = rqi_array.size
    nan_pixels = np.sum(nan_mask)
    rqi_filtered_pixels = np.sum(~nan_mask & (rqi_array < rqi_th))
    
    stats = {
        'total_pixels': total_pixels,
        'nan_pixels': nan_pixels,
        'nan_ratio': nan_pixels / total_pixels,
        'rqi_filtered_pixels': rqi_filtered_pixels,
        'rqi_filtered_ratio': rqi_filtered_pixels / total_pixels,
        'total_filtered_pixels': np.sum(rqi_mask),
        'total_filtered_ratio': np.sum(rqi_mask) / total_pixels,
        'valid_points_before': np.sum(valid_mask_before),
        'valid_points_after': np.sum(valid_mask_after),
        'reduction_percentage': (1 - np.sum(valid_mask_after) / np.sum(valid_mask_before)) * 100 if np.sum(valid_mask_before) > 0 else 0
    }
    
    return {
        'orig_before': orig_before,
        'orig_after': orig_after,
        '1d_before': d1_before,
        '1d_after': d1_after,
        '1d_valid_before': d1_valid_before,
        '1d_valid_after': d1_valid_after,
        'stats': stats,
        'nan_mask': nan_mask,
        'rqi_mask': rqi_mask
    }

def calculate_daily_mean(data, hours_per_day=24):
    """
    Calculate daily mean from hourly data by averaging over channel, latitude, and longitude dimensions.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input array with shape (time in hours, channel, latitude, longitude)
    hours_per_day : int, optional
        Number of hours per day, default is 24
    
    Returns:
    --------
    numpy.ndarray
        1D array of daily means
    """
    # Average over channel, latitude, and longitude dimensions (axes 1,2,3)
    hourly_means = np.nanmean(data, axis=(1, 2, 3))  # Shape: (time in hours,)
    
    # Calculate number of complete days
    num_complete_days = hourly_means.shape[0] // hours_per_day
    
    # Reshape time data into days and calculate mean for each day
    daily_means = hourly_means[:num_complete_days * hours_per_day].reshape(num_complete_days, hours_per_day).mean(axis=1)
    
    return daily_means

def calculate_daily_mean_keep_spatial(data, hours_per_day=24):
    """
    Calculate daily mean from hourly data while maintaining spatial information.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input array with shape (time in hours, channel, latitude, longitude)
    hours_per_day : int, optional
        Number of hours per day, default is 24
    
    Returns:
    --------
    numpy.ndarray
        4D array of daily means with shape (time in days, channel, latitude, longitude)
    """
    # Get original data shape
    total_hours, num_channels, num_latitudes, num_longitudes = data.shape
    
    # Calculate number of complete days
    num_complete_days = total_hours // hours_per_day
    
    # Use only complete days' data
    truncated_data = data[:num_complete_days * hours_per_day]
    
    # Reshape data to separate days and hours
    # New shape: (num_complete_days, hours_per_day, channel, latitude, longitude)
    reshaped_data = truncated_data.reshape(num_complete_days, hours_per_day, num_channels, num_latitudes, num_longitudes)
    
    # Calculate mean over time axis (axis 1)
    # Result shape: (num_complete_days, channel, latitude, longitude)
    daily_means = np.nanmean(reshaped_data, axis=1)
    
    return daily_means

def calculate_daily_sum(data, hours_per_day=24):
    """
    Calculate daily sum from hourly data by summing over channel, latitude, and longitude dimensions.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input array with shape (time in hours, channel, latitude, longitude)
    hours_per_day : int, optional
        Number of hours per day, default is 24
    
    Returns:
    --------
    numpy.ndarray
        1D array of daily sums
    """
    # Sum over channel, latitude, and longitude dimensions (axes 1,2,3)
    hourly_sums = np.nansum(data, axis=(1, 2, 3))  # Shape: (time in hours,)
    
    # Calculate number of complete days
    num_complete_days = hourly_sums.shape[0] // hours_per_day
    
    # Reshape time data into days and calculate sum for each day
    daily_sums = hourly_sums[:num_complete_days * hours_per_day].reshape(num_complete_days, hours_per_day).sum(axis=1)
    
    return daily_sums

def calculate_daily_sum_keep_spatial(data, hours_per_day=24):
    """
    Calculate daily sum from hourly data while maintaining spatial information.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input array with shape (time in hours, channel, latitude, longitude)
    hours_per_day : int, optional
        Number of hours per day, default is 24
    
    Returns:
    --------
    numpy.ndarray
        4D array of daily sums with shape (time in days, channel, latitude, longitude)
    """
    # Get original data shape
    total_hours, num_channels, num_latitudes, num_longitudes = data.shape
    
    # Calculate number of complete days
    num_complete_days = total_hours // hours_per_day
    
    # Use only complete days' data
    truncated_data = data[:num_complete_days * hours_per_day]
    
    # Reshape data to separate days and hours
    # New shape: (num_complete_days, hours_per_day, channel, latitude, longitude)
    reshaped_data = truncated_data.reshape(num_complete_days, hours_per_day, num_channels, num_latitudes, num_longitudes)
    
    # Calculate sum over time axis (axis 1)
    # Result shape: (num_complete_days, channel, latitude, longitude)
    daily_sums = np.nansum(reshaped_data, axis=1)
    
    return daily_sums

# %%
# Define overall period - adjust these dates as needed
overall_start = "2021-01-01"  # Start from January 2021
overall_end = "2021-12-31"    # End at December 2021

# Ensure output directory exists
output_dir = "./saved_daily_agg_sum"
os.makedirs(output_dir, exist_ok=True)

# Convert to datetime objects
start_dt = pd.to_datetime(overall_start)
end_dt = pd.to_datetime(overall_end)

# Generate month periods and process each
current_month = start_dt.replace(day=1)
while current_month <= end_dt:
    # Get start and end date for current month
    month_start = current_month
    if current_month.month == 12:
        month_end = current_month.replace(year=current_month.year+1, month=1, day=1) - pd.Timedelta(days=1)
    else:
        month_end = current_month.replace(month=current_month.month+1, day=1) - pd.Timedelta(days=1)
    
    # Format dates in the original format (without zero-padding)
    start_date = f"{month_start.year}-{month_start.month}-{month_start.day}"
    end_date = f"{month_end.year}-{month_end.month}-{month_end.day}"
    
    print(f"\nProcessing period: {start_date} to {end_date}")
    
    # Start timing for this month
    start = time.time()
    rqi_th = 0.8

    try:
        # Prepare dataset (add RQI)
        results = prepare_datasets(
            start_date,
            end_date,
            load_vars=["mrms", "pred_mtl_cw", "pred_stl_ir",
                       "ccs", "scampr",
                       "rqi"],
            overlap_data=False
        )

        # Get results
        mrms = results["mrms"]
        pred_mtl_cw = results["pred_mtl_cw"]
        pred_stl_ir = results["pred_stl_ir"]
        ccs = results["ccs"]
        scampr = results["scampr"]
        rqi = results["rqi"]  # Get RQI data

        # Check shapes of various data
        print("MRMS: ", mrms.shape)
        print("Pred MTL_CW: ", pred_mtl_cw.shape)
        print("Pred STL_IR: ", pred_stl_ir.shape)
        print("CCS: ", ccs.shape)
        print("SCAMPR: ", scampr.shape)
        print("RQI: ", rqi.shape)

        # Daily Aggregation
        time_index = make_date_list(start_date, end_date, freq="h", day_index=None)

        mrms_daily = calculate_daily_sum(mrms)
        pred_mtl_cw_daily = calculate_daily_sum(pred_mtl_cw)
        pred_stl_ir_daily = calculate_daily_sum(pred_stl_ir)
        ccs_daily = calculate_daily_sum(ccs)
        scampr_daily = calculate_daily_sum(scampr)

        df = pd.DataFrame({"MRMS": mrms_daily,
                           "Pred MTL_CW": pred_mtl_cw_daily,
                           "Pred STL_IR": pred_stl_ir_daily,
                           "CCS": ccs_daily,
                           "SCaMPR": scampr_daily},
                           index=time_index[:len(mrms_daily)])

        # Save df
        df.to_csv(f"{output_dir}/daily_sum_df_{start_date}.csv", index=True)

        # Daily Aggregation with Spatial Information
        mrms_2d_daily = calculate_daily_sum_keep_spatial(mrms)
        pred_mtl_cw_2d_daily = calculate_daily_sum_keep_spatial(pred_mtl_cw)
        pred_stl_ir_2d_daily = calculate_daily_sum_keep_spatial(pred_stl_ir)
        ccs_2d_daily = calculate_daily_sum_keep_spatial(ccs)
        scampr_2d_daily = calculate_daily_sum_keep_spatial(scampr)

        # Save 2D daily data
        np.save(f"{output_dir}/mrms_2d_daily_{start_date}.npy", mrms_2d_daily)
        np.save(f"{output_dir}/pred_mtl_cw_2d_daily_{start_date}.npy", pred_mtl_cw_2d_daily)
        np.save(f"{output_dir}/pred_stl_ir_2d_daily_{start_date}.npy", pred_stl_ir_2d_daily)
        np.save(f"{output_dir}/ccs_2d_daily_{start_date}.npy", ccs_2d_daily)
        np.save(f"{output_dir}/scampr_2d_daily_{start_date}.npy", scampr_2d_daily)
        
        end = time.time()
        print(f"Processing completed in {end - start:.2f} seconds")
    
    except Exception as e:
        print(f"Error processing {start_date} to {end_date}: {e}")
    
    # Move to next month
    current_month = current_month + relativedelta(months=1)

print("\nAll months processed.")

