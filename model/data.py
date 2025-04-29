import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyDataset(Dataset):
    def __init__(self, data_paths, mean_tensor, std_tensor, normalize=True, log_prcp=False):
        self.data_paths = data_paths
        self.normalize = normalize
        self.log_prcp = log_prcp
        self.mean_tensor = mean_tensor
        self.std_tensor = std_tensor

    def _normalize(self, tensor):
        normalized_tensor = (tensor - self.mean_tensor) / self.std_tensor
        return normalized_tensor

    def _log_transform(self, tensor):
        return torch.log1p(tensor)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data = np.load(self.data_paths[idx])
        data_tensor = torch.from_numpy(data).float()
        if self.normalize:
            data_tensor = self._normalize(data_tensor)
        y = data_tensor[0, :, :]
        if self.log_prcp:
            y = self._log_transform(y)
        x_ir_ch08 = data_tensor[1:3, :, :]
        x_ir_ch10 = data_tensor[3:5, :, :]
        x_ir_ch11 = data_tensor[5:7, :, :]
        x_ir_ch14 = data_tensor[7:9, :, :]
        x_ir_ch15 = data_tensor[9:11, :, :]
        x_cw = data_tensor[11:13, :, :]
        x_ci = data_tensor[13:, :, :]
        return y, x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci


class PatchDataset(Dataset):
    """_summary_
    単一のnumpyファイルからパッチを取得するためのデータセットクラス
    Args:
        Dataset (path): path to numpy file
    """
    def __init__(self, numpy_file_path, mean_tensor, std_tensor, normalize=True, log_prcp=False):
        self.data = np.load(numpy_file_path)
        self.num_patches = self.data.shape[0]
        self.normalize = normalize
        self.log_prcp = log_prcp
        self.mean_tensor = mean_tensor
        self.std_tensor = std_tensor

    def __len__(self):
        return self.num_patches

    def _normalize(self, tensor):
        normalized_tensor = (tensor - self.mean_tensor) / self.std_tensor
        return normalized_tensor

    def _log_transform(self, tensor):
        return torch.log1p(tensor)

    def __getitem__(self, idx):
        patch = self.data[idx]
        data_tensor = torch.from_numpy(patch).float()
        if self.normalize:
            data_tensor = self._normalize(data_tensor)
        y = data_tensor[0, :, :]
        if self.log_prcp:
            y = self._log_transform(y)
        x_ir_ch08 = data_tensor[1:3, :, :]
        x_ir_ch10 = data_tensor[3:5, :, :]
        x_ir_ch11 = data_tensor[5:7, :, :]
        x_ir_ch14 = data_tensor[7:9, :, :]
        x_ir_ch15 = data_tensor[9:11, :, :]
        x_cw = data_tensor[11:13, :, :]
        x_ci = data_tensor[13:, :, :]
        return y, x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci


# Normalizationを戻す関数
def denormalize(tensor, std_tensor, mean_tensor):
    denormalized_tensor = tensor * std_tensor + mean_tensor
    return denormalized_tensor


# log1p変換を戻す関数
def delog(tensor):
    return torch.expm1(tensor)


def extract_patches(features, patch_size):
    channels, height, width = features.shape
    patches = []
    for i in range(0, height - patch_size + 1, patch_size):
        for j in range(0, width - patch_size + 1, patch_size):
            patch = features[:, i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return np.array(patches)


def reconstruct_image(patches, num_patches_height, num_patches_width, patch_size):
    channels = patches.shape[1]
    height = num_patches_height * patch_size
    width = num_patches_width * patch_size
    # 元の画像サイズに合わせた空の配列を用意
    reconstructed = np.zeros((channels, height, width))
    # パッチを順番に元の画像に配置
    patch_idx = 0
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            reconstructed[:, i:i+patch_size, j:j+patch_size] = patches[patch_idx]
            patch_idx += 1
    return reconstructed
