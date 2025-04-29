import xarray as xr
import os
import time
import glob

dir = "./raw/"
data_type = "total_column_cloud_ice_water"
year = "2018"

print(f"Merge {data_type} data for {year}...")
start = time.time()
file_paths = glob.glob(os.path.join(dir, f"{data_type}_{year}_*.nc"))
file_paths = sorted(file_paths)
file_paths
# %%
# Read all files and combine into xarray dataset
datasets = [xr.open_dataset(fp) for fp in file_paths]
combined_dataset = xr.concat(datasets, dim='time')

# Save combined dataset as a single NetCDF file
out_fname = os.path.join(dir, f"{data_type}_{year}.nc")
combined_dataset.to_netcdf(out_fname)
# %%
elapsed_time = time.time() - start
print(f"elapsed_time:{elapsed_time}")
