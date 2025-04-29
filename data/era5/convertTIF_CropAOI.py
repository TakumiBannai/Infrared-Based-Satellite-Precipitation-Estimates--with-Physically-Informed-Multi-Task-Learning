import os
import sys
import xarray as xr
import geopandas as gpd
import rioxarray
import pandas as pd
from rioxarray.exceptions import NoDataInBounds, MissingCRS

data_type = sys.argv[1]  # total_column_cloud_ice_water
year = sys.argv[2]  # 2018

# Set file paths
netcdf_path = f'./raw/{data_type}_{year}.nc'
aoi_shapefile = '../AOI/aoi.shp'
output_dir = './tif'

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load netCDF data
ds = xr.open_dataset(netcdf_path)

# Set variable name (replace with actual variable name)
variable_name = 'tciw'  # Change if adjustment is needed
data = ds[variable_name]

# Time dimension name
time_dim_name = 'valid_time'

# Check and set CRS
if not data.rio.crs:
    # Set data CRS (change if needed)
    data = data.rio.write_crs("EPSG:4326")

# Load shapefile
aoi = gpd.read_file(aoi_shapefile)

# Match AOI CRS to data CRS
if aoi.crs != data.rio.crs:
    aoi = aoi.to_crs(data.rio.crs)

# Get AOI bounds
aoi_bounds = aoi.total_bounds

# Loop through each time step
for timestep in data[time_dim_name]:
    # Select data for current time step
    data_time = data.sel({time_dim_name: timestep})

    # Convert longitude (from 0-360 degrees to -180-180 degrees)
    if data_time.longitude.max() > 180:
        data_time = data_time.assign_coords(
            longitude = xr.where(data_time.longitude >= 180, data_time.longitude - 360, data_time.longitude)
        ).sortby('longitude').rio.write_crs("EPSG:4326")

    # Check CRS
    if not data_time.rio.crs:
        data_time.rio.write_crs("EPSG:4326", inplace=True)

    # Get and check data bounds
    data_bounds = data_time.rio.bounds()


    # Attempt clipping
    try:
        clipped = data_time.rio.clip(aoi.geometry, aoi.crs)
    except (NoDataInBounds, MissingCRS) as e:
        print(f"Skipping time {timestep.values} due to error: {e}")
        continue

    # Get time information
    time_obj = pd.to_datetime(timestep.values)

    # Extract year and time string
    year = time_obj.year
    timestr = time_obj.strftime('%Y%m%d%H%M')

    # Build output filename
    output_filename = f"total_column_cloud_ice_water_{year}_{timestr}_aoi.tif"
    output_path = os.path.join(output_dir, output_filename)

    # Save clipped data
    clipped.rio.to_raster(output_path)

    print(f"Saved {output_path}")
