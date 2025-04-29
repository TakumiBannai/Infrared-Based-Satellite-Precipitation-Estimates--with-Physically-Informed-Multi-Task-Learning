import glob
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import PatchDataset, reconstruct_image
from model import STL_IR, STL_IR_CW, STL_IR_CI, STL_IR_CWCI, MTL_CW, MTL_CI, MTL_CWCI

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def get_args():
    parser = argparse.ArgumentParser(description="Training parameters")
    # Define command-line arguments
    parser.add_argument('--exp_name', type=str, default="devTest", help="Experiment name")
    parser.add_argument('--model_type', type=str, default="STL_IR", help="Type of the model")
    args = parser.parse_args()
    return args


# Parameters
args = get_args()
exp_name = args.exp_name
model_type = args.model_type
num_workers = 15
# exp_name = "val2"
# model_type = "STL_IR_CW"
# num_workers = 15

print(f"Experiment name: {exp_name}")
print(f"Model type: {model_type}")

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

# Month-to-days dictionary to handle variable number of days
month_days = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
}


# Function to handle inference
def run_inference(year, month, day, hour):
    path_feature = f"../data/feature/test/*{year}{month:02d}{day:02d}{hour:02d}*.npy"
    file_paths = sorted(glob.glob(path_feature))
    
    if len(file_paths) != 1:
        print(f"Skipping {year}/{month:02d}/{day:02d} {hour:02d}:00 due to missing or multiple files.")
        return

    patch_dataset = PatchDataset(file_paths[0], mean_tensor, std_tensor, normalize=True, log_prcp=True)
    patch_loader = DataLoader(patch_dataset, batch_size=1, shuffle=False, num_workers=15, pin_memory=True)

    if model_type == "STL_IR":
        model = STL_IR()
    elif model_type == "STL_IR_CW":
        model = STL_IR_CW()
    elif model_type == "STL_IR_CI":
        model = STL_IR_CI()
    elif model_type == "STL_IR_CWCI":
        model = STL_IR_CWCI()

    elif model_type == "MTL_CW":
        model = MTL_CW()
    elif model_type == "MTL_CI":
        model = MTL_CI()
    elif model_type == "MTL_CWCI":
        model = MTL_CWCI()


    model.load_state_dict(torch.load(f"./saved_model/{exp_name}_{model_type}.pth", weights_only=True))
    model = model.to(device)

    model.eval()
    output_patch = []
    label_patch = []
    for y, x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci in patch_loader:
        y, x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci = y.to(device).float(), x_ir_ch08.to(device).float(), x_ir_ch10.to(device).float(), x_ir_ch11.to(device).float(), x_ir_ch14.to(device).float(), x_ir_ch15.to(device).float(), x_cw.to(device).float(), x_ci.to(device).float()
        with torch.no_grad():
            if model_type == "STL_IR":
                output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15)
            elif model_type == "STL_IR_CW":
                output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw)
            elif model_type == "STL_IR_CI":
                output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_ci)
            elif model_type == "STL_IR_CWCI":
                output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci)
            
            elif model_type == "MTL_CW":
                output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw)
                output = output[0]  # Only use the first output (precipitation)
            elif model_type == "MTL_CI":
                output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_ci)
                output = output[0]  # Only use the first output (precipitation)
            elif model_type == "MTL_CWCI":
                output = model(x_ir_ch08, x_ir_ch10, x_ir_ch11, x_ir_ch14, x_ir_ch15, x_cw, x_ci)
                output = output[0]  # Only use the first output (precipitation)

            # Collect Precipitation data
            label_patch.append(y.unsqueeze(1).cpu().numpy().astype(np.float32))
            output_patch.append(output.cpu().numpy().astype(np.float32))

    output_patch = np.array(output_patch)
    label_patch = np.array(label_patch)

    output_patch = np.squeeze(output_patch, axis=1)
    label_patch = np.squeeze(label_patch, axis=1)

    # Reconstruct Image from Patch
    output_image = reconstruct_image(output_patch, num_patches_height=6, num_patches_width=15, patch_size=112)
    label_image = reconstruct_image(label_patch, num_patches_height=6, num_patches_width=15, patch_size=112)

    # Save Daily Preds and Label value
    # make dir if not exist
    os.makedirs(f"./saved_pred/pred/{exp_name}_{model_type}", exist_ok=True)
    os.makedirs(f"./saved_pred/label/{exp_name}_{model_type}", exist_ok=True)
    np.save(f"./saved_pred/pred/{exp_name}_{model_type}/{year}{month:02d}{day:02d}{hour:02d}.npy", output_image)
    np.save(f"./saved_pred/label/{exp_name}_{model_type}/{year}{month:02d}{day:02d}{hour:02d}.npy", label_image)
    print(f"Finished inference for {year}/{month:02d}/{day:02d} {hour:02d}:00")


# %%
# Main loop to process each month, day, and hour
for month in range(1, 13):  # Loop over each month (1~13)
    days_in_month = month_days[month]  # Get number of days in the current month
    for day in range(1, days_in_month + 1):  # Loop over each day in the month
        for hour in range(24):  # Loop over each hour in the day (0-23)
            run_inference(year=2021, month=month, day=day, hour=hour)
