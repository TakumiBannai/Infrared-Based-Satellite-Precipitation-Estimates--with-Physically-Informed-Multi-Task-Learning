# MTL_IR-retrieval

This repository contains a Multi-Task Learning (MTL) model for Infrared (IR) retrieval of precipitation data. The project processes various meteorological data sources and uses them to train a deep learning model for precipitation estimation.

## Project Structure

### Data Processing Pipeline (`data/`)

The data processing pipeline consists of several components:

1. **Data Sources**:
   - `persiann_ccs/`: PERSIANN-CCS precipitation data processing
   - `mrms/`: Multi-Radar Multi-Sensor (MRMS) data processing
   - `goes_conus/`: GOES-16 CONUS data processing
   - `era5/`: ERA5 reanalysis data processing
   - `AOI/`: Area of Interest (AOI) definition files
   - `feature/`: Feature extraction and preprocessing
   - `scampr/`: Additional data processing utilities

2. **Data Processing Steps**:
   - Data downloading and format conversion
   - Area of Interest (AOI) cropping
   - Feature extraction and patch creation
   - Data preprocessing and normalization

### Model Architecture (`model/`)

The model implementation includes:

1. **Core Components**:
   - `model.py`: Main model architecture
   - `loss.py`: Custom loss functions
   - `data.py`: Data loading and preprocessing
   - `util.py`: Utility functions

2. **Training and Evaluation**:
   - `train_STL.py`: Single-Task Learning training script
   - `train_MTL.py`: Multi-Task Learning training script
   - `eval_result.py`: Model evaluation and result analysis
   - `eval_chunk.py`: Evaluation functions
   - `generate_dailyPred.py`: Inference Daily prediction result

3. **Data Processing and Visualization**:
   - `make_data_for_fig.py`: Data processing for figure generation
     - Functions for data loading and preprocessing
     - Daily aggregation (mean/sum) calculations
   - `vis_data_for_fig.py`: Visualization tools
     - Scatter plots with regression lines
     - Residual boxplots
     - RMSE analysis by precipitation intensity
     - Kernel density estimation plots

