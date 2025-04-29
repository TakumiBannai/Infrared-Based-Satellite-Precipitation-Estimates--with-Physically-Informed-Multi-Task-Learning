import glob
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from data import NumpyDataset, denormalize
from model import STL_IR, STL_IR_CW, STL_IR_CI, STL_IR_CWCI
from eval_chunk import EvaluationIndices, CategoryEvaluation, show_histgram, show_density_scatterplot

reference_values = np.load("/lustre/home/bannai/IR_PrecpEstimate/data/feature/reference_values.npy", mmap_mode='r')

def get_args():
    parser = argparse.ArgumentParser(description="Training parameters")
    # Define command-line arguments
    parser.add_argument('--exp_name', type=str, default="devTest", help="Experiment name")
    parser.add_argument('--model_type', type=str, default="STL_IR", help="Type of the model")
    parser.add_argument('--sample', type=str, default=None, help="Sample data (can be an int or 'None')")
    args = parser.parse_args()
    # Convert sample to None if it's the string 'None', otherwise to int
    if args.sample == 'None':
        args.sample = None
    elif args.sample is not None:
        try:
            args.sample = int(args.sample)
        except ValueError:
            raise ValueError("The --sample argument must be an integer or 'None'.")
    return args


start = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Parameters
args = get_args()
exp_name = args.exp_name
model_type = args.model_type
sample = args.sample
# exp_name = "val2"
# model_type = "STL_IR"
# sample = None

print(f"Experiment name: {exp_name}")
print(f"Model type: {model_type}")
print(f"Sampling: {sample}")

# Evaluation ---------------------------------
print("Model evaluation start...")
# Load dataset (1-d Preds value)
pred_1d = np.load(f"./saved_output/{exp_name}_{model_type}_pred_1d.npy", mmap_mode='r')
label_1d = np.load(f"./saved_output/{exp_name}_{model_type}_label_1d.npy", mmap_mode='r')

# Radam sample
if sample is not None:
    np.random.seed(42)
    print(f"Sampling: {sample:,} from {len(pred_1d):,}")
    idx = np.random.choice(len(pred_1d), sample, replace=False)
    pred_1d = pred_1d[idx]
    label_1d = label_1d[idx]

# Denormalize (対数変換の逆変換)
pred_1d = np.expm1(pred_1d)
label_1d = np.expm1(label_1d)


# Evaluation
print("Metrics calculation start.")
evaluater = EvaluationIndices(pred_1d, label_1d, rain_th=0.1, filter_regression=True)
eval_test = pd.DataFrame([evaluater.evaluate()])
eval_test.to_csv(f"./saved_output/{exp_name}_{model_type}_indexEval.csv")

# Rain-intensity interval
print("Rain-intensity interval calculation start.")
evaluator = CategoryEvaluation(pred_1d, label_1d)
eval_rain_intensity = evaluator.evaluate()
eval_rain_intensity.to_csv(f"./saved_output/{exp_name}_{model_type}_indexEvalInterval.csv")

# Histgram
print("Histgram calculation start.")
show_histgram(label_1d, pred_1d)
plt.savefig(f"./saved_output/{exp_name}_{model_type}_histgram.png")

# Density scatter plot
print("Density scatter plot calculation start.")
show_density_scatterplot(label_1d, pred_1d, f"{exp_name}_{model_type}", vmax=900000)
plt.savefig(f"./saved_output/{exp_name}_{model_type}_densityScatterPlot.png")

# Monthly Metrics calculation
print("Monthly Metrics calculation start.")

def get_test_index(target_month, return_path=False):
    path_feature = "../data/feature/test/patch/*.npy"
    file_paths = sorted(glob.glob(path_feature))
    index = [i for i, file_path in enumerate(file_paths) if f"{target_month}" in file_path]
    if return_path:
        return index, np.array(file_paths)[index]
    elif return_path == False:
        return index

# Load dataset (3-d Preds value)
pred_3d = np.load(f"./saved_output/{exp_name}_{model_type}_pred_3d.npy", mmap_mode='r')
label_3d = np.load(f"./saved_output/{exp_name}_{model_type}_label_3d.npy", mmap_mode='r')

eval_test_df = pd.DataFrame()
eval_test_rain_intensity_df = pd.DataFrame()
eval_test_qm_df = pd.DataFrame()
eval_test_rain_intensity_qm_df = pd.DataFrame()
for target_month in ["202101", "202102", "202103", "202104", "202105", "202106", "202107", "202108", "202109", "202110", "202111", "202112"]:
    print(f"Metrics calculation start for {target_month}.")
    index = get_test_index(target_month)
    pred_3d_target = pred_3d[index]
    label_3d_target = label_3d[index]
    pred_1d_target = pred_3d_target.flatten()
    label_1d_target = label_3d_target.flatten()
    # Denormalize
    pred_1d_target = np.expm1(pred_1d_target)
    label_1d_target = np.expm1(label_1d_target)
    # Evaluation by Month
    evaluater = EvaluationIndices(pred_1d_target, label_1d_target, rain_th=0.1, filter_regression=True)
    eval_test = pd.DataFrame([evaluater.evaluate()])
    eval_test_df = pd.concat([eval_test_df, eval_test])

    # Evaluation By Month and Rain-intensity
    evaluator = CategoryEvaluation(pred_1d_target, label_1d_target)
    eval_rain_intensity = evaluator.evaluate()
    eval_rain_intensity["Month"] = target_month
    eval_test_rain_intensity_df = pd.concat([eval_test_rain_intensity_df, eval_rain_intensity])

eval_test_df.to_csv(f"./saved_output/{exp_name}_{model_type}_indexEval_byMonth.csv")
eval_test_rain_intensity_df.to_csv(f"./saved_output/{exp_name}_{model_type}_indexEvalInterval_byMonth.csv")


# %%
print("Model evaluation finished.")
elapsed_time = time.time() - start
print(f"Elapsed time: {elapsed_time:.2f} [sec]")

