import os
import glob
import time
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from data import NumpyDataset, denormalize
from model import MTL_CW, MTL_CI, MTL_CWCI, MTL_CW_multi
from loss import MTLLoss_CW, MTLLoss_CI, MTLLoss_CWCI, MTLLoss_CW_Weighting
from util import EarlyStopping
import torch.cuda.amp

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

half_prec = False

torch.backends.cuda.matmul.allow_tf32 = half_prec
torch.backends.cudnn.allow_tf32 = half_prec


def get_args():
    parser = argparse.ArgumentParser(description="Training parameters")
    # Define command-line arguments
    parser.add_argument('--exp_name', type=str, default="devTest", help="Experiment name")
    parser.add_argument('--model_type', type=str, default="STL_IR", help="Type of the model")
    parser.add_argument('--train_mode', action='store_true', help="Enable train mode")
    parser.add_argument('--test_mode', action='store_true', help="Enable test mode")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--epoch', type=int, default=120, help="Number of epochs")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers")
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
train_mode = args.train_mode
test_mode = args.test_mode
batch_size = args.batch_size
lr = args.lr
epoch = args.epoch
num_workers = args.num_workers
sample = args.sample
save_train = True
save_test = False

# Multi-task learning parameters
strategy = {
    "rainmask_start": 0,
    "rainmask_end": 20,
    "mix_start": 20,
    "mix_end": 50,
    "rainrate_start": 50,
    "rainrate_end": 150,

    "weighted_mix_start": 300,
    "weighted_mix_end": 350,

    "cloudwater_start": 9999,
    "cloudwater_end": 10000,
    "cloudice_start": 9999,
    "cloudice_end": 10000
}


# mean and std for each channel (precipitating data)
mean = [0.0, 229.4375165954692, 229.29896317314396,
        239.5470371473738, 239.4295285823659, 250.06714141347626,
        250.09161894143443, 250.53426028530805, 250.536461936128,
        249.185426698166, 249.1588386738013, 0.14642600553042465,
        0.14477387601887984, 0.0990693896937506, 0.09959578019675766]
std = [1.0, 7.118123604027681, 7.0398058403316925,
       11.554776291989917, 11.517932550826567, 19.042804229936664,
       19.248360638593393, 20.025186165814315, 20.236383972925015,
       19.49095823831401, 19.656779246266485, 0.21483776668687404,
       0.21368900271619656, 0.13138535457406586, 0.13191988206331662]

mean_tensor = torch.tensor(mean).view(-1, 1, 1)
std_tensor = torch.tensor(std).view(-1, 1, 1)

print(f"Experiment name: {exp_name}")
print(f"Model type: {model_type}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {lr}")
print(f"Number of workers: {num_workers}")
print(f"Sampling: {sample}")

# %%
# Train
if train_mode:
    # データセットの作成
    path_feature = "../data/feature/train/patch/*.npy"
    file_paths = sorted(glob.glob(path_feature))
    print(f"Total Data: {len(file_paths)} files")

    # Randam Sampling for Train/Val (optional)
    if sample is not None:
        np.random.seed(42)
        file_paths = np.random.choice(file_paths, sample, replace=False)

    # Dataset/Loaderの作成
    dataset = NumpyDataset(file_paths, mean_tensor, std_tensor, normalize=True, log_prcp=True)
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                            generator=torch.Generator().manual_seed(42))
    # DataLoaderの作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Train tiles: {len(train_dataset)}")
    print(f"Val tiles: {len(val_dataset)}")

    # %%
    # Multi-task learning
    if model_type == "MTL_CW":
        model = MTL_CW()
        loss_weight = [1.0, 1.0, 1.0]
        loss_reg = nn.MSELoss()
        loss_clsf = nn.BCEWithLogitsLoss()
        loss_fn = MTLLoss_CW(loss_reg, loss_clsf, strategy, loss_weight)

    elif model_type == "MTL_CW_multi":
        model = MTL_CW_multi()
        loss_weight = [1.0, 1.0, 1.0]
        loss_reg = nn.MSELoss()
        loss_clsf = nn.CrossEntropyLoss()
        loss_fn = MTLLoss_CW(loss_reg, loss_clsf, strategy, loss_weight)

    elif model_type == "MTL_CI":
        model = MTL_CI()
        loss_weight = [1.0, 1.0, 1.0]
        loss_reg = nn.MSELoss()
        loss_clsf = nn.BCEWithLogitsLoss()
        loss_fn = MTLLoss_CI(loss_reg, loss_clsf, strategy, loss_weight)

    elif model_type == "MTL_CWCI":
        model = MTL_CWCI()
        loss_weight = [1.0, 1.0, 1.0, 1.0]
        loss_reg = nn.MSELoss()
        loss_clsf = nn.BCEWithLogitsLoss()
        loss_fn = MTLLoss_CWCI(loss_reg, loss_clsf, strategy, loss_weight)

    elif model_type == "MTL_CW_Weight":
        model = MTL_CW()
        loss_weight = [1.0, 1.0, 1.0]
        alpha = 1.0
        beta = 60.0
        loss_reg = nn.MSELoss()
        loss_clsf = nn.BCEWithLogitsLoss()
        loss_fn = MTLLoss_CW_Weighting(loss_reg, loss_clsf, strategy, loss_weight, alpha, beta)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # %%
    # Model training --------------
    print("Model training start...")

    scaler = torch.amp.GradScaler(init_scale=2.**16, growth_interval=2000)

    train_loss, val_loss = [], []
    earlystopping = EarlyStopping(patience=8, verbose=True,
                                path=f"./saved_model/{exp_name}_{model_type}.pth")

    for i in range(epoch):
        # Train loop
        model.train()
        train_batch_loss = []
        for y, x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci in train_loader:
            y, x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci = y.to(device).float(), x_ir_ch08.to(device).float(), x_ir_ch10.to(device).float(), x_ir_ch11.to(device).float(), x_ir_ch14.to(device).float(), x_ir_ch15.to(device).float(), x_cw.to(device).float(), x_ci.to(device).float()
            y_mask = (y >= 0.1).float().to(device)  # Rain mask for MTL
            # Multi-class Label
            y_multimask = torch.zeros_like(y, dtype=torch.long).to(device) 
            y_multimask[(0 <= y) & (y < 0.1)] = 0  # no_rain
            y_multimask[(0.1 <= y) & (y < 1.0)] = 1  # weak
            y_multimask[(1.0 <= y) & (y < 10.0)] = 2  # moderate
            y_multimask[(10 <= y)] = 3  # strong
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=True):
                if model_type == "MTL_CW":
                    output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw)
                    p_rainrate = output[0]
                    p_rainmask = output[1]
                    p_cw = output[2]
                    # Compute loss
                    loss = loss_fn(
                        p_rainrate,
                        p_rainmask,
                        p_cw,
                        y.unsqueeze(1),
                        y_mask.unsqueeze(1),
                        x_cw[:, 0, :, :].unsqueeze(1),
                        i
                        )
                elif model_type == "MTL_CW_multi":
                    output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw)
                    p_rainrate = output[0]
                    p_rainmask = output[1]  # Mukti-classに変更
                    p_cw = output[2]
                    # Compute loss
                    loss = loss_fn(
                        p_rainrate,
                        p_rainmask,
                        p_cw,
                        y.unsqueeze(1),
                        y_multimask,
                        x_cw[:, 0, :, :].unsqueeze(1),
                        i
                        )
                elif model_type == "MTL_CI":
                    output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_ci)
                    p_rainrate = output[0]
                    p_rainmask = output[1]
                    p_ci = output[2]
                    # Compute loss
                    loss = loss_fn(
                        p_rainrate,
                        p_rainmask,
                        p_ci,
                        y.unsqueeze(1),
                        y_mask.unsqueeze(1),
                        x_ci[:, 0, :, :].unsqueeze(1),
                        i
                        )
                elif model_type == "MTL_CWCI":
                    output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci)
                    p_rainrate = output[0]
                    p_rainmask = output[1]
                    p_cw = output[2]
                    p_ci = output[3]
                    # Compute loss
                    loss = loss_fn(
                        p_rainrate,
                        p_rainmask,
                        p_cw,
                        p_ci,
                        y.unsqueeze(1),
                        y_mask.unsqueeze(1),
                        x_cw[:, 0, :, :].unsqueeze(1),
                        x_ci[:, 0, :, :].unsqueeze(1),
                        i
                    )
                elif model_type == "MTL_CW_Weight":
                    output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw)
                    p_rainrate = output[0]
                    p_rainmask = output[1]
                    p_cw = output[2]
                    # Compute loss
                    loss = loss_fn(
                        p_rainrate,
                        p_rainmask,
                        p_cw,
                        y.unsqueeze(1),
                        y_mask.unsqueeze(1),
                        x_cw[:, 0, :, :].unsqueeze(1),
                        i
                    )

            scaler.scale(loss).backward()

            # 勾配クリッピング
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            train_batch_loss.append(loss.item())

        # GPUメモリキャッシュを解放
        torch.cuda.empty_cache()
        
        # val loop
        model.eval()
        val_batch_loss = []
        for y, x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci in val_loader:
            y, x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci = y.to(device).float(), x_ir_ch08.to(device).float(), x_ir_ch10.to(device).float(), x_ir_ch11.to(device).float(), x_ir_ch14.to(device).float(), x_ir_ch15.to(device).float(), x_cw.to(device).float(), x_ci.to(device).float()
            y_mask = (y >= 0.1).float().to(device)  # Rain mask for MTL
            # Multi-class Label
            y_multimask = torch.zeros_like(y, dtype=torch.long).to(device) 
            y_multimask[(0 <= y) & (y < 0.1)] = 0  # no_rain
            y_multimask[(0.1 <= y) & (y < 1.0)] = 1  # weak
            y_multimask[(1.0 <= y) & (y < 10.0)] = 2  # moderate
            y_multimask[(10 <= y)] = 3  # strong
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', enabled=True):
                    if model_type == "MTL_CW":
                        output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw)
                        p_rainrate = output[0]
                        p_rainmask = output[1]
                        p_cw = output[2]
                        # Compute loss
                        loss = loss_fn(
                            p_rainrate,
                            p_rainmask,
                            p_cw,
                            y.unsqueeze(1),
                            y_mask.unsqueeze(1),
                            x_cw[:, 0, :, :].unsqueeze(1),
                            i
                        )
                    elif model_type == "MTL_CW_multi":
                        output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw)
                        p_rainrate = output[0]
                        p_rainmask = output[1]
                        p_cw = output[2]
                        # Compute loss
                        loss = loss_fn(
                            p_rainrate,
                            p_rainmask,
                            p_cw,
                            y.unsqueeze(1),
                            y_multimask,
                            x_cw[:, 0, :, :].unsqueeze(1),
                            i
                        )
                    elif model_type == "MTL_CI":
                        output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_ci)
                        p_rainrate = output[0]
                        p_rainmask = output[1]
                        p_ci = output[2]
                        # Compute loss
                        loss = loss_fn(
                            p_rainrate,
                            p_rainmask,
                            p_ci,
                            y.unsqueeze(1),
                            y_mask.unsqueeze(1),
                            x_ci[:, 0, :, :].unsqueeze(1),
                            i
                        )
                    
                    elif model_type == "MTL_CWCI":
                        output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci)
                        p_rainrate = output[0]
                        p_rainmask = output[1]
                        p_cw = output[2]
                        p_ci = output[3]
                        # Compute loss
                        loss = loss_fn(
                            p_rainrate,
                            p_rainmask,
                            p_cw,
                            p_ci,
                            y.unsqueeze(1),
                            y_mask.unsqueeze(1),
                            x_cw[:, 0, :, :].unsqueeze(1),
                            x_ci[:, 0, :, :].unsqueeze(1),
                            i
                        )

                    elif model_type == "MTL_CW_Weight":
                        output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw)
                        p_rainrate = output[0]
                        p_rainmask = output[1]
                        p_cw = output[2]
                        # Compute loss
                        loss = loss_fn(
                            p_rainrate,
                            p_rainmask,
                            p_cw,
                            y.unsqueeze(1),
                            y_mask.unsqueeze(1),
                            x_cw[:, 0, :, :].unsqueeze(1),
                            i
                        )

                val_batch_loss.append(loss.item())

        # GPUメモリキャッシュを解放
        torch.cuda.empty_cache()

        train_loss.append(np.mean(train_batch_loss))
        val_loss.append(np.mean(val_batch_loss))
        if i % 10 == 0:
            print(f"Epoch: {i}, Train Loss: {np.mean(train_batch_loss):.4f}, Val Loss: {np.mean(val_batch_loss):.4f}", "Elapsed time: {:.2f} [hour]".format((time.time() - start)/3600))
        if i > 50:
            earlystopping(np.mean(val_batch_loss), model)
            if earlystopping.early_stop:
                print(f"Early stopping at epoch {i}")
                break

    print("Model training finished.")
    # %%
    # Plot loss graph
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Train Loss", color='blue', linestyle='-', linewidth=2, marker='o', markersize=5)
    plt.plot(val_loss, label="Val Loss", color='orange', linestyle='--', linewidth=2, marker='s', markersize=5)
    plt.title("Training and Validation Loss Over Epochs", fontsize=16, fontweight='bold')
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"./saved_output/{exp_name}_{model_type}_lossGraph.png")
    plt.show()

    # %%
    elapsed_time = time.time() - start
    print(f"Elapsed time: {elapsed_time/3600:.2f} [hour]")

# %%
# Test
if test_mode:
    print("Model Test start.")
    # Load the model
    if model_type == "MTL_CW":
        model = MTL_CW()
    elif model_type == "MTL_CW_multi":
        model = MTL_CW_multi()
    elif model_type == "MTL_CI":
        model = MTL_CI()
    elif model_type == "MTL_CWCI":
        model = MTL_CWCI()
    elif model_type == "MTL_CW_Weight":
        model = MTL_CW()
    model.load_state_dict(torch.load(f"./saved_model/{exp_name}_{model_type}.pth", weights_only=True))
    model = model.to(device)

    # Train/Val dataloader
    path_feature = "../data/feature/train/patch/*.npy"
    file_paths = sorted(glob.glob(path_feature))

    # Randam Sampling for Train/Val (optional)
    if sample is not None:
        np.random.seed(42)
        file_paths = np.random.choice(file_paths, sample, replace=False)

    # Dataset/Loaderの作成
    train_dataset = NumpyDataset(file_paths, mean_tensor, std_tensor, normalize=True, log_prcp=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"Train tiles: {len(train_dataset)}")

    # Test dataloader
    path_feature = "../data/feature/test/patch/*.npy"
    file_paths = sorted(glob.glob(path_feature))

    # Randam Sampling (optional)
    if sample is not None:
        file_paths = np.random.choice(file_paths, sample, replace=False)

    test_dataset = NumpyDataset(file_paths, mean_tensor, std_tensor, normalize=True, log_prcp=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"Test tiles: {len(test_dataset)}")

    # %%
    model.eval()
    pred_train, label_train = [], []
    pred_test, label_test = [], []

    # Train/Val loop
    if save_train:
        for y, x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci in train_loader:
            y, x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci = y.to(device).float(), x_ir_ch08.to(device).float(), x_ir_ch10.to(device).float(), x_ir_ch11.to(device).float(), x_ir_ch14.to(device).float(), x_ir_ch15.to(device).float(), x_cw.to(device).float(), x_ci.to(device).float()
            with torch.no_grad():
                if model_type == "MTL_CW":
                    output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw)
                elif model_type == "MTL_CW_multi":
                    output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw)
                elif model_type == "MTL_CI":
                    output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_ci)
                elif model_type == "MTL_CWCI":
                    output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci)
                elif model_type == "MTL_CW_Weight":
                    output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw)

                # Concat&Denormalize label
                data = torch.cat([y.unsqueeze(1), x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci], dim=1)
                # data = denormalize(data.cpu(), mean_tensor, std_tensor)
                y = data[:, 0, :, :]
                # Denormalize output  * Pred-output:[rainrate, rainmask, cloudwater, ci]
                data = torch.cat([output[0], x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci], dim=1)
                # data = denormalize(data.cpu(), mean_tensor, std_tensor)

                output = data[:, 0, :, :]
                # Prediction on val data
                pred_train.append(output.cpu().numpy().astype(np.float32))
                label_train.append(y.cpu().numpy().astype(np.float32))

        pred_train = np.concatenate(pred_train)
        label_train = np.concatenate(label_train)
        # Save Val values
        np.save(f"./saved_output/{exp_name}_{model_type}_pred_train_3d.npy", pred_train)
        np.save(f"./saved_output/{exp_name}_{model_type}_label_train_3d.npy", label_train)
        np.save(f"./saved_output/{exp_name}_{model_type}_pred_train_1d.npy", pred_train.reshape(-1))
        np.save(f"./saved_output/{exp_name}_{model_type}_label_train_1d.npy", label_train.reshape(-1))

    # Test loop
    if save_test:
        for y, x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci in test_loader:
            y, x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci = y.to(device).float(), x_ir_ch08.to(device).float(), x_ir_ch10.to(device).float(), x_ir_ch11.to(device).float(), x_ir_ch14.to(device).float(), x_ir_ch15.to(device).float(), x_cw.to(device).float(), x_ci.to(device).float()
            with torch.no_grad():
                if model_type == "MTL_CW":
                    output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw)
                elif model_type == "MTL_CW_multi":
                    output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw)
                elif model_type == "MTL_CI":
                    output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_ci)
                elif model_type == "MTL_CWCI":
                    output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci)
                elif model_type == "MTL_CW_Weight":
                    output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw)

                # Concat&Denormalize label
                data = torch.cat([y.unsqueeze(1), x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci], dim=1)
                # data = denormalize(data.cpu(), mean_tensor, std_tensor)
                y = data[:, 0, :, :]
                # Denormalize output  * Pred-output:[rainrate, rainmask, cloudwater, ci]
                data = torch.cat([output[0], x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci], dim=1)
                # data = denormalize(data.cpu(), mean_tensor, std_tensor)
                output = data[:, 0, :, :]

                # Prediction on test data
                pred_test.append(output.cpu().numpy().astype(np.float32))
                label_test.append(y.cpu().numpy().astype(np.float32))

        pred_test = np.concatenate(pred_test)
        label_test = np.concatenate(label_test)
        # Save Preds value
        np.save(f"./saved_output/{exp_name}_{model_type}_pred_3d.npy", pred_test)
        np.save(f"./saved_output/{exp_name}_{model_type}_label_3d.npy", label_test)
        np.save(f"./saved_output/{exp_name}_{model_type}_pred_1d.npy", pred_test.reshape(-1))
        np.save(f"./saved_output/{exp_name}_{model_type}_label_1d.npy", label_test.reshape(-1))

# %%
    print("Model evaluation finished.")
    elapsed_time = time.time() - start
    print(f"Epaplse time: {elapsed_time/3600:.2f} [hour]")
