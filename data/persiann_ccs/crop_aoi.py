import glob
import subprocess
import geopandas as gpd
import json

# GeoJSONファイルのパス
aoi_path = '../AOI/AOI_conus.geojson'


def crop_aoi(aoi_path, input_tif, output_tif):
    # Load the GeoJSON file defining the Area of Interest (AOI)
    with open(aoi_path) as f:
        aoi = json.load(f)
    # Convert GeoJSON to a Shapefile for GDAL
    # gdf = gpd.GeoDataFrame.from_features(aoi['features'], crs="EPSG:4326")
    # aoi_shapefile = "../AOI/aoi.shp"
    # gdf.to_file(aoi_shapefile)

    # Avoid conflict Error
    aoi_shapefile = "./aoi.shp"
    
    # Crop the image to the area of interest
    script = f"gdalwarp -cutline {aoi_shapefile} -crop_to_cutline {input_tif} {output_tif}"
    subprocess.run(script, shell=True)


# %%
# Run the crop_aoi function on all MRMS files
files = sorted(glob.glob('./tif/*.tif'))

# %%
for input_tif in files:
    out_fname = input_tif.replace('.tif', '_aoi.tif')
    crop_aoi(aoi_path, f"{input_tif}", f"{out_fname}")
