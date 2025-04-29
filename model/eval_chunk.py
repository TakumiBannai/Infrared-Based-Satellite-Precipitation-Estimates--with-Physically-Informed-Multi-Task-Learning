import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix
from matplotlib import cm
from tqdm import tqdm


import numpy as np
from sklearn.metrics import confusion_matrix


# EvaluationIndices
class EvaluationIndices():
    def __init__(self, pred_reg, label_reg, rain_th=0.1, filter_regression=False, chunk_size=100000):
        self.rain_th = rain_th
        self.filter_regression = filter_regression
        self.chunk_size = chunk_size

        # 回帰評価用の累積変数
        self.sum_abs_error = 0
        self.sum_squared_error = 0
        self.sum_pred = 0
        self.sum_label = 0
        self.sum_pred_label = 0
        self.sum_pred2 = 0
        self.sum_label2 = 0
        self.n_regression = 0

        # 分類評価用の累積変数
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

        # チャンクごとにデータを処理
        if len(pred_reg) > 0 and len(label_reg) > 0:
            self.process_chunks(pred_reg, label_reg)

    def process_chunks(self, pred_reg, label_reg):
        pred_reg = np.array(pred_reg)
        label_reg = np.array(label_reg)
        total_length = pred_reg.size
        for i in range(0, total_length, self.chunk_size):
            pred_chunk = pred_reg.ravel()[i:i+self.chunk_size]
            label_chunk = label_reg.ravel()[i:i+self.chunk_size]

            # 回帰評価指標の計算
            if self.filter_regression:
                mask = label_chunk >= self.rain_th
                pred_reg_chunk = pred_chunk[mask]
                label_reg_chunk = label_chunk[mask]
            else:
                pred_reg_chunk = pred_chunk
                label_reg_chunk = label_chunk

            n_chunk = len(pred_reg_chunk)
            if n_chunk > 0:
                abs_error = np.abs(pred_reg_chunk - label_reg_chunk)
                squared_error = (pred_reg_chunk - label_reg_chunk) ** 2
                self.sum_abs_error += np.sum(abs_error)
                self.sum_squared_error += np.sum(squared_error)
                self.sum_pred += np.sum(pred_reg_chunk)
                self.sum_label += np.sum(label_reg_chunk)
                self.sum_pred_label += np.sum(pred_reg_chunk * label_reg_chunk)
                self.sum_pred2 += np.sum(pred_reg_chunk ** 2)
                self.sum_label2 += np.sum(label_reg_chunk ** 2)
                self.n_regression += n_chunk

            # 分類評価指標の計算
            pred_cls_chunk = (pred_chunk >= self.rain_th).astype(int)
            label_cls_chunk = (label_chunk >= self.rain_th).astype(int)

            # 混同行列の要素を累積
            tp_chunk = np.sum((pred_cls_chunk == 1) & (label_cls_chunk == 1))
            tn_chunk = np.sum((pred_cls_chunk == 0) & (label_cls_chunk == 0))
            fp_chunk = np.sum((pred_cls_chunk == 1) & (label_cls_chunk == 0))
            fn_chunk = np.sum((pred_cls_chunk == 0) & (label_cls_chunk == 1))

            self.tp += tp_chunk
            self.tn += tn_chunk
            self.fp += fp_chunk
            self.fn += fn_chunk

    def evaluate(self):
        # 回帰評価指標の計算
        MAE = self.sum_abs_error / self.n_regression if self.n_regression > 0 else np.nan
        RMSE = np.sqrt(self.sum_squared_error / self.n_regression) if self.n_regression > 0 else np.nan
        n = self.n_regression
        if n > 0:
            numerator = n * self.sum_pred_label - self.sum_pred * self.sum_label
            denominator = np.sqrt((n * self.sum_pred2 - self.sum_pred ** 2) * (n * self.sum_label2 - self.sum_label ** 2))
            CC = numerator / denominator if denominator != 0 else np.nan
        else:
            CC = np.nan

        # 分類評価指標の計算
        h = self.tp
        c = self.tn
        f = self.fp
        m = self.fn

        POD = self.pod(h, m)
        FAR = self.far(f, h)
        CSI = self.csi(h, f, m)

        return {
            "MAE": MAE,
            "RMSE": RMSE,
            "CC": CC,
            "POD": POD,
            "FAR": FAR,
            "CSI": CSI
        }

    # 分類評価指標の計算関数
    def pod(self, h, m):
        return h / (h + m) if (h + m) > 0 else np.nan

    def far(self, f, h):
        return f / (h + f) if (h + f) > 0 else np.nan

    def csi(self, h, f, m):
        denom = h + f + m
        return h / denom if denom > 0 else np.nan


class CategoryEvaluation():
    def __init__(self, pred_reg, label_reg, rain_th=0.1, chunk_size=100000):
        self.rain_th = rain_th
        self.chunk_size = chunk_size
        self.categories = ["no_rain", "weak", "moderate", "strong"]
        self.evaluations = {category: EvaluationIndices(np.array([]), np.array([])) for category in self.categories}
        self.process_chunks(pred_reg, label_reg)

    def process_chunks(self, pred_reg, label_reg):
        pred_reg = np.array(pred_reg)
        label_reg = np.array(label_reg)
        total_length = pred_reg.size
        for i in range(0, total_length, self.chunk_size):
            pred_chunk = pred_reg.ravel()[i:i+self.chunk_size]
            label_chunk = label_reg.ravel()[i:i+self.chunk_size]

            for category in self.categories:
                pred_bin_chunk, label_bin_chunk = binning_chunk(pred_chunk, label_chunk, bin=category)
                if len(pred_bin_chunk) > 0:
                    self.evaluations[category].process_chunks(pred_bin_chunk, label_bin_chunk)

    def evaluate(self):
        results = {}
        for category in self.categories:
            results[category] = self.evaluations[category].evaluate()
        return pd.DataFrame(results)


def binning_chunk(arr_pred_chunk, arr_label_chunk, bin="weak"):
    no_rain = (0 <= arr_label_chunk) & (arr_label_chunk < 0.1)
    weak = (0.1 <= arr_label_chunk) & (arr_label_chunk < 1.0)
    mod = (1.0 <= arr_label_chunk) & (arr_label_chunk < 10.0)
    strong = (10 <= arr_label_chunk)
    if bin == "weak":
        return arr_pred_chunk[weak], arr_label_chunk[weak]
    elif bin == "moderate":
        return arr_pred_chunk[mod], arr_label_chunk[mod]
    elif bin == "strong":
        return arr_pred_chunk[strong], arr_label_chunk[strong]
    elif bin == "no_rain":
        return arr_pred_chunk[no_rain], arr_label_chunk[no_rain]
    else:
        raise ValueError(f"Invalid bin value: {bin}")


def compute_index_chunk(A_chunk, B_chunk):
    A_chunk = A_chunk.reshape(-1)
    B_chunk = B_chunk.reshape(-1)
    # Number of Pixel
    num_pix = len(A_chunk)
    if num_pix == 0:
        return num_pix, np.nan, np.nan, np.nan, np.nan
    else:
        # MAE
        mae = mean_absolute_error(A_chunk, B_chunk)
        # MSE
        mse = mean_squared_error(A_chunk, B_chunk)
        # RMSE
        rmse = np.sqrt(mse)
        # Corr
        corr_coef = np.corrcoef(A_chunk, B_chunk)[0][1]
        return num_pix, mae, mse, rmse, corr_coef

def process_chunks(arr_pred, arr_label, chunk_size):
    num_chunks = int(np.ceil(len(arr_pred) / chunk_size))
    all_results = {'All': [], 'No_rain': [], 'Weak': [], 'Moderate': [], 'Strong': []}
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        pred_chunk = arr_pred[start:end]
        label_chunk = arr_label[start:end]
        
        # 全体の評価指標
        all_results['All'].append(compute_index_chunk(pred_chunk, label_chunk))
        
        # 各降水強度での評価指標をチャンクごとに計算
        for bin_type in ['no_rain', 'weak', 'moderate', 'strong']:
            A_chunk, B_chunk = binning_chunk(pred_chunk, label_chunk, bin=bin_type)
            all_results[bin_type.capitalize()].append(compute_index_chunk(A_chunk, B_chunk))
    
    # チャンクごとの結果を集計
    final_results = {key: np.nanmean(np.array(all_results[key]), axis=0) for key in all_results.keys()}
    return final_results

def compute_index_bin(pred, label, chunk_size=10000, out_type='df'):
    # チャンク処理を使用して計算を実行
    final_results = process_chunks(pred, label, chunk_size)
    
    if out_type == 'df':
        out = pd.DataFrame(final_results, index=["n_pixel", "MAE", "MSE", "RMSE", "CC"])
    elif out_type == 'arr':
        out = np.array(list(final_results.values()))
    else:
        raise ValueError(f"Invalid out_type: {out_type}")
    
    return out



def show_histgram(label_1d, pred_1d, chunk_size=1000000):
    bins = np.arange(0.001, 10, 0.1)
    label_hist = np.zeros(len(bins) - 1, dtype=np.uint64)
    pred_hist = np.zeros(len(bins) - 1, dtype=np.uint64)

    length = len(label_1d)
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        label_chunk = label_1d[start:end]
        pred_chunk = pred_1d[start:end]
        # ヒストグラムのカウントをuint64型にキャスト
        label_hist += np.histogram(label_chunk, bins=bins)[0].astype(np.uint64)
        pred_hist += np.histogram(pred_chunk, bins=bins)[0].astype(np.uint64)

    fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
    ax[0].bar(bins[:-1], label_hist, width=0.1, alpha=0.7, color='blue', label='Label Data')
    ax[1].bar(bins[:-1], pred_hist, width=0.1, alpha=0.7, color='green', label='Prediction Data')
    ax[0].set_title('Distribution of Label Data', fontsize=14)
    ax[1].set_title('Distribution of Prediction Data', fontsize=14)
    ax[0].set_ylim(0, 3.5*1e7)
    ax[1].set_ylim(0, 3.5*1e7)

    ax[0].set_xlabel('Value', fontsize=12)
    ax[0].set_ylabel('Frequency', fontsize=12)
    ax[1].set_xlabel('Value', fontsize=12)
    ax[1].set_ylabel('Frequency', fontsize=12)

    ax[0].grid(True)
    ax[1].grid(True)

    ax[0].legend(loc='upper right', fontsize=10)
    ax[1].legend(loc='upper right', fontsize=10)

    # 統計情報（平均と標準偏差）の追加
    mean_label = label_1d.mean()
    std_label = label_1d.std()
    mean_pred = pred_1d.mean()
    std_pred = pred_1d.std()

    ax[0].axvline(mean_label, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_label:.2f}')
    ax[0].axvline(mean_label + std_label, color='red', linestyle='dotted', linewidth=2, label=f'Std: {std_label:.2f}')
    # ax[0].axvline(mean_label - std_label, color='red', linestyle='dotted', linewidth=2)
    ax[0].legend()

    ax[1].axvline(mean_pred, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_pred:.2f}')
    ax[1].axvline(mean_pred + std_pred, color='red', linestyle='dotted', linewidth=2, label=f'Std: {std_pred:.2f}')
    # ax[1].axvline(mean_pred - std_pred, color='red', linestyle='dotted', linewidth=2)
    ax[1].legend()

    plt.tight_layout()
    plt.show()


def show_density_scatterplot(label, pred, title_name, vmax, chunk_size=10000):
    # log10変換（値が0以下の場合はマスク）
    with np.errstate(divide='ignore', invalid='ignore'):
        label_log = np.log10(label)
        pred_log = np.log10(pred)
    # マスク処理
    valid_mask = np.isfinite(label_log) & np.isfinite(pred_log)
    label_log = label_log[valid_mask]
    pred_log = pred_log[valid_mask]

    # ヒストグラムのビン設定
    xedges = np.arange(0.01, 2.0, 0.075)
    yedges = np.arange(0.01, 2.0, 0.075)
    # xedges = np.linspace(0.0001, 2.0, 37)
    # yedges = np.linspace(0.0001, 2.0, 37)
    H = np.zeros((len(xedges)-1, len(yedges)-1), dtype=np.float64)

    # チャンクごとに2Dヒストグラムを計算して累積
    length = len(label_log)
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        label_chunk = label_log[start:end]
        pred_chunk = pred_log[start:end]
        # 2Dヒストグラムを計算
        H_chunk, _, _ = np.histogram2d(label_chunk, pred_chunk, bins=[xedges, yedges])
        # 累積
        H += H_chunk

    # プロット
    plt.figure(figsize=(7, 5.5))
    plt.imshow(H.T, origin='lower', aspect='auto',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               vmin=0,
               vmax=vmax, cmap=cm.jet)
    plt.colorbar()
    plt.xlabel('Label: Precipitation ($log_{10}$[mm/h])')
    plt.ylabel('Pred: Precipitation ($log_{10}$[mm/h])')
    plt.title(title_name)
    plt.xticks(np.arange(0., 2, 0.25))
    plt.yticks(np.arange(0., 2, 0.25))
    # 対角線
    ident = [0.01, 1.95]
    plt.plot(ident, ident, ls="--", lw=1.2, c="gray")
    plt.show()

