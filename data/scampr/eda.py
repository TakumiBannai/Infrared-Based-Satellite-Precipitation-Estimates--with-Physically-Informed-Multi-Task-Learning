import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

# ファイルの読み込み
path_file = "RRR_CONUS20210010001_remapped.nc"
ds = nc.Dataset(path_file, mode='r')

# メタデータの確認
print("Variables:", list(ds.variables.keys()))
print("Dimensions:", {dim: ds.dimensions[dim].size for dim in ds.dimensions})

# 緯度・経度の取得
cols = ds.dimensions['Columns'].size
rows = ds.dimensions['Rows'].size
lon = np.linspace(-135, -65, cols)  # 仮の経度範囲
lat = np.linspace(15, 60, rows)[::-1]  # 反転して正しく表示

data_var = "RRQPE"  # 既知の変数

data = ds.variables[data_var][:] if data_var in ds.variables else None

ds.close()

# 可視化
if data is not None:
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_title(f"Visualization of {data_var}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    img = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(), cmap='viridis')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    plt.colorbar(img, ax=ax, label=data_var)
    plt.show()
else:
    print("No valid data found in the NetCDF file.")

# %%
