"""
Script to extract and save patches for use
Create and save patch directories in train and test directories
Extraction conditions:
Train = Patches without NaN in all channels and precipitation pixel rate of 10% or more
Test = Patches without NaN in all channels
test_precp = Patches with precipitation pixel rate of 10% or more

Result: Note that patch definitions differ between train and test
- train/patch/*.npy: 10% filter + Nan
- test/patch/*.npy: Nan
- test_precp/patch/*.npy: 10% filter + Nan

"""
# %%
import os
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt


# %%
def filter_save_patch(path_files, prcp_th=0.1, prcp_rate_th=0.1):
    """_summary_

    Args:
        path_files (_type_): List of npy files in train/test directories
        prcp_th (float, optional): Precipitation threshold. Defaults to 0.1.
        prcp_rate_th (float, optional): Threshold for precipitation pixel rate per patch. Defaults to 0.1.
    """
    for i in tqdm(range(len(path_files))):
        arr = np.load(path_files[i])
        for patch_id in range(arr.shape[0]):
            # Precipitation filter
            prcp = arr[patch_id, 0, :, :]
            prcp_rate = np.count_nonzero(prcp) / prcp.size
            prcp_rate = (prcp >= prcp_th).sum() / prcp.size
            if prcp_rate >= prcp_rate_th:
                # Nan filter
                if not np.isnan(arr[patch_id, :, :, :]).any():
                    # Save patch
                    np.save(f"{os.path.dirname(path_files[i])}/patch/{os.path.basename(path_files[i])}_{patch_id}.npy", arr[patch_id, :, :, :])


# Create patch directories in train/test directories
os.makedirs("train/patch", exist_ok=True)
os.makedirs("test/patch", exist_ok=True)

# %%
print("train data processing...")
path_files = sorted(glob.glob("train/*.npy"))
filter_save_patch(path_files, prcp_th=0.1, prcp_rate_th=0.1)

# print("test data processing...")
# path_files = sorted(glob.glob("test/*.npy"))
# filter_save_patch(path_files, prcp_th=0.1, prcp_rate_th=0.0)  # Test data includes all except Nan
print("Done.")

# %%
# Check the saved patches
# path = "train/patch/*.npy"
# path_files = sorted(glob.glob(path))
# np.random.seed(0)
# np.random.shuffle(path_files)
# print(len(path_files))

# # Display 10 patches
# for i in range(3):
#     arr = np.load(path_files[10])
#     print(arr.shape)
#     plt.imshow(arr[0, :, :])
#     plt.show()
