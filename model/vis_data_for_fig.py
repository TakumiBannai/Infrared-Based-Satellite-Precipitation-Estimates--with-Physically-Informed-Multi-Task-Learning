import pandas as pd
import numpy as np
import os
import glob
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.dates as mdates
import rasterio
from rasterio.enums import Resampling
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import torch
from matplotlib.gridspec import GridSpec


def load_monthly_df(year, month, agg="mean"):
    """
    Load monthly data file for a specific year and month.
    Each file contains data for all days in the month.
    
    Parameters:
    -----------
    year : int
        Year (e.g., 2021)
    month : int
        Month (e.g., 1-12)
    agg : str, optional
        Aggregation type ("mean" or "sum")
        
    Returns:
    --------
    pandas.DataFrame
        Loaded monthly dataframe (contains daily records)
    """
    if agg == "mean":
        base_dir = "./saved_daily_agg_mean"
        filepath = f"{base_dir}/daily_mean_df_{year}-{month}-1.csv"
    elif agg == "sum":
        base_dir = "./saved_daily_agg_sum"
        filepath = f"{base_dir}/daily_sum_df_{year}-{month}-1.csv"
    
    if not os.path.exists(filepath):
        print(f"Warning: Monthly file not found: {filepath}")
        return None
    
    # Load index as datetime type
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    print(f"Loaded monthly file: {filepath}")
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    print(f"Number of records: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df


def load_monthly_2d_data(year, month, data_type, agg="mean"):
    """
    Load monthly 2D spatial data for a specific year, month, and data type.
    
    Parameters:
    -----------
    year : int
        Year (e.g., 2021)
    month : int
        Month (e.g., 1-12)
    data_type : str
        Type of data ('mrms', 'pred_mtl_cw', 'pred_stl_ir', 'ccs', 'scampr')
    agg : str, optional
        Aggregation type ("mean" or "sum")
        
    Returns:
    --------
    numpy.ndarray
        Loaded monthly 2D spatial data
    """
    if agg == "mean":
        base_dir = "./saved_daily_agg_mean"
    elif agg == "sum":
        base_dir = "./saved_daily_agg_sum"
    filepath = f"{base_dir}/{data_type}_2d_daily_{year}-{month}-1.npy"
    
    if not os.path.exists(filepath):
        print(f"Warning: Monthly 2D file not found: {filepath}")
        return None
    
    data = np.load(filepath)
    
    print(f"Loaded monthly 2D data: {filepath}")
    print(f"Data shape: {data.shape}")
    
    return data


def load_year_data(year=2021, agg="mean"):
    """
    Load one year of data for the specified year.
    
    Parameters:
    -----------
    year : int
        Year to load (e.g., 2021)
    agg : str, optional
        Aggregation type ("mean" or "sum")
        
    Returns:
    --------
    pandas.DataFrame
        Combined dataframe containing one year of data
    """
    # Initialize empty dataframe
    yearly_df = pd.DataFrame()
    
    # Count successful monthly loads
    success_count = 0
    
    # Iterate from January to December
    for month in range(1, 13):
        try:
            # Load monthly data
            monthly_df = load_monthly_df(year, month, agg=agg)
            
            # If data loaded successfully, concatenate
            if monthly_df is not None:
                yearly_df = pd.concat([yearly_df, monthly_df], axis=0)
                success_count += 1
        except Exception as e:
            print(f"Error: Failed to load data for {year} year {month} month: {e}")
    
    # Remove duplicate indices (dates) if any
    if not yearly_df.empty:
        yearly_df = yearly_df.loc[~yearly_df.index.duplicated(keep='first')]
        
        # Sort
        yearly_df = yearly_df.sort_index()
    
    print(f"Loading complete: Loaded {success_count} months of data.")
    
    if not yearly_df.empty:
        print(f"Period: {yearly_df.index[0]} to {yearly_df.index[-1]}")
        print(f"Total rows: {len(yearly_df)}")
    else:
        print("Failed to load data.")
    
    return yearly_df


def load_year_2d_data(year=2021, data_types=None, agg="mean"):
    """
    Load one year of 2D data for each data type and return as stacked arrays.
    
    Parameters:
    -----------
    year : int
        Year to load (e.g., 2021)
    data_types : list or None
        List of data types to load (e.g., ['mrms', 'pred_mtl_cw', 'pred_stl_ir', 'ccs', 'scampr'])
        If None, loads all data types
    agg : str, optional
        Aggregation type ("mean" or "sum")
        
    Returns:
    --------
    dict
        Dictionary with data types as keys and stacked arrays as values
        Each array has shape (date, channel, height, width)
    """
    # Default data types
    if data_types is None:
        data_types = ['mrms', 'pred_mtl_cw', 'pred_stl_ir', 'ccs', 'scampr']
    
    # Initialize dictionary for intermediate results
    temp_data = {data_type: {} for data_type in data_types}
    
    # Dictionary for final results
    yearly_2d_arrays = {}
    
    # Count successful monthly loads for each data type
    success_counts = {data_type: 0 for data_type in data_types}
    
    # Iterate from January to December
    for month in range(1, 13):
        # Process each data type
        for data_type in data_types:
            try:
                # Load 2D data
                data = load_monthly_2d_data(year, month, data_type, agg=agg)
                
                # If data loaded successfully, add to dictionary
                if data is not None:
                    temp_data[data_type][month] = data
                    success_counts[data_type] += 1
            except Exception as e:
                print(f"Error: Failed to load {data_type} data for {year} year {month} month: {e}")
    
    # Stack monthly data for each data type
    for data_type in data_types:
        if success_counts[data_type] > 0:
            # Get list of months with data
            months = sorted(temp_data[data_type].keys())
            print(f"{data_type}: Loaded {success_counts[data_type]} months of data.")
            print(f"  Loaded months: {months}")
            
            # Check shape of first month's data
            first_shape = temp_data[data_type][months[0]].shape
            print(f"  Data shape: {first_shape}")
            
            # Stack all months' data into one array
            # Shape will be (date, channel, height, width)
            stacked_data = []
            for month in months:
                monthly_data = temp_data[data_type][month]
                stacked_data.append(monthly_data)
            
            # Use numpy.concatenate to concatenate month-wise data
            yearly_array = np.concatenate(stacked_data, axis=0)
            print(f"  Stacked data shape: {yearly_array.shape}")
            
            # Store result in dictionary
            yearly_2d_arrays[data_type] = yearly_array
        else:
            print(f"{data_type}: Failed to load data.")
            yearly_2d_arrays[data_type] = None
    
    return yearly_2d_arrays


def load_year_all_data(year=2021, data_types=None, agg="mean"):
    """
    Load one year of time series data and 2D data for all specified data types.
    
    Parameters:
    -----------
    year : int
        Year to load (e.g., 2021)
    data_types : list or None
        List of 2D data types to load
    agg : str, optional
        Aggregation type ("mean" or "sum")
        
    Returns:
    --------
    tuple
        (Time series dataframe, 2D data dictionary)
    """
    # Load time series data
    print("=== Start loading time series data ===")
    yearly_df = load_year_data(year=year, agg=agg)
    
    # Load 2D data
    print("\n=== Start loading 2D data ===")
    yearly_2d_data = load_year_2d_data(year=year, data_types=data_types, agg=agg)
    
    return yearly_df, yearly_2d_data


def plot_scatter_with_regression_lines(df, x_col, y_cols, colors=None, labels=None, 
                                     title="Scatter Plot with Regression Lines", 
                                     xlabel="X", ylabel="Y", figsize=(10, 8), 
                                     alpha=0.5, show_identity_line=True,
                                     show_regression_lines=None, 
                                     fontsize=14, legend_fontsize=12):
    """
    Draw scatter plot with regression lines (optimized for paper)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing data to plot
    x_col : str
        Column name to use as x-axis
    y_cols : list of str
        List of column names to use as y-axis
    colors : list of str, optional
        List of colors corresponding to each y column. None uses default colors
    labels : list of str, optional
        List of labels corresponding to each y column. None uses column names
    title : str, optional
        Plot title
    xlabel, ylabel : str, optional
        Axis labels
    figsize : tuple, optional
        Figure size
    alpha : float, optional
        Transparency of scatter plot
    show_identity_line : bool, optional
        1:1 line to be displayed
    show_regression_lines : list of str, bool, or None, optional
        List of column names to display regression lines.
        True: Display regression lines for all columns
        False: Do not display regression lines for any column
        List: Display regression lines only for specified columns
        None: Display regression lines for all columns (default)
    fontsize : int, optional
        Font size for main text (title, axis labels)
    legend_fontsize : int, optional
        Font size for legend
    
    Returns:
    --------
    matplotlib.figure.Figure
        Generated figure object
    """
    # Times New Roman font setting
    # plt.rcParams['font.family'] = 'Times New Roman'
    
    # Default value setting
    if colors is None:
        colors = ['blue', 'red', 'green', 'purple', 'orange']  # Default color list
    
    if labels is None:
        labels = y_cols  # Default to use column names as labels
    
    # Set columns to display regression lines
    if show_regression_lines is None or show_regression_lines is True:
        cols_with_regression = y_cols  # Display regression lines for all columns
    elif show_regression_lines is False:
        cols_with_regression = []  # Do not display regression lines
    else:
        # If list is provided, display regression lines only for specified columns
        cols_with_regression = [col for col in show_regression_lines if col in y_cols]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot scatter plot and regression lines for each y column
    for i, y_col in enumerate(y_cols):
        color = colors[i % len(colors)]  # Cycle through color list
        label = labels[i % len(labels)]  # Cycle through label list
        
        # Scatter plot
        ax.scatter(df[x_col], df[y_col], c=color, label=label, alpha=alpha, s=70)  # Increase marker size
        
        # Check if this column should display regression line
        if y_col in cols_with_regression:
            # Regression line (NaN filtering)
            mask = ~np.isnan(df[x_col]) & ~np.isnan(df[y_col])
            
            if np.sum(mask) > 1:  # Ensure there are at least two data points
                slope, intercept = np.polyfit(df.loc[mask, x_col], df.loc[mask, y_col], 1)
                x_range = np.linspace(df.loc[mask, x_col].min(), df.loc[mask, x_col].max(), 100)
                ax.plot(x_range, slope * x_range + intercept, color=color, linestyle='-', linewidth=2,
                       label=f'{label} Fit (y = {slope:.2f}x + {intercept:.2f})')
    
    # 1:1 line (optional)
    if show_identity_line:
        valid_x = df[x_col].dropna()
        if len(valid_x) > 0:
            x_min = valid_x.min()
            x_max = valid_x.max()
            
            # Consider y-axis range
            y_values = []
            for y_col in y_cols:
                valid_y = df[y_col].dropna()
                if len(valid_y) > 0:
                    y_values.extend([valid_y.min(), valid_y.max()])
            
            if y_values:
                plot_min = min(x_min, min(y_values))
                plot_max = max(x_max, max(y_values))
                
                identity_range = np.linspace(plot_min, plot_max, 100)
                ax.plot(identity_range, identity_range, c='black', linestyle='--', linewidth=2, label='1:1 Line')
    
    # Graph decoration - Increase font size
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize+2)  # Title is even larger
    ax.legend(fontsize=legend_fontsize, facecolor='white', edgecolor='black', framealpha=1.0)
    ax.grid(True)
    
    # Set tick font size
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    
    # Increase axis line width
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        
    plt.tight_layout()
    
    return fig


def plot_residual_boxplot(df, truth_col, pred_cols, colors=None, labels=None,
                          title="Residual Boxplot", xlabel="Models", ylabel="Residuals",
                          figsize=(10, 6), residual_type="difference", showfliers=True,
                          grid=True, horizontal_line=0, fontsize=14, legend_fontsize=12):
    """
    Draw boxplot of residuals between predictions and truth (optimized for paper)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing data
    truth_col : str
        Column name to use as truth
    pred_cols : list of str
        List of column names to use as predictions
    colors : list of str, optional
        List of colors corresponding to each prediction column. None uses default colors
    labels : list of str, optional
        List of labels corresponding to each prediction column. None uses column names
    title : str, optional
        Plot title
    xlabel, ylabel : str, optional
        Axis labels
    figsize : tuple, optional
        Figure size
    residual_type : str, optional
        Method to calculate residuals. Choose from 'difference' (prediction - truth), 'absolute' (|prediction - truth|),
        'relative' ((prediction - truth)/truth * 100)
    showfliers : bool, optional
        Whether to display outliers
    grid : bool, optional
        Whether to display grid
    horizontal_line : float or None, optional
        Position to draw horizontal line. None to not draw horizontal line
    fontsize : int, optional
        Font size for main text (title, axis labels)
    legend_fontsize : int, optional
        Font size for legend
        
    Returns:
    --------
    matplotlib.figure.Figure
        Generated figure object
    """
    # Times New Roman font setting
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Default value setting
    if colors is None:
        colors = ['blue', 'red', 'green', 'purple', 'orange']  # Default color list
    
    if labels is None:
        labels = pred_cols  # Default to use column names as labels
    
    # Calculate residuals
    residuals = []
    valid_labels = []
    valid_colors = []
    
    for i, pred_col in enumerate(pred_cols):
        # NaN filtering
        mask = ~np.isnan(df[truth_col]) & ~np.isnan(df[pred_col])
        
        if np.sum(mask) > 0:  # If valid data exists
            if residual_type == "difference":
                # Simple difference
                res = df.loc[mask, pred_col] - df.loc[mask, truth_col]
                ylabel_text = "Residuals (Prediction - Observation)"
            elif residual_type == "absolute":
                # Absolute difference
                res = np.abs(df.loc[mask, pred_col] - df.loc[mask, truth_col])
                ylabel_text = "Absolute Residuals |Prediction - Observation|"
            elif residual_type == "relative":
                # Relative error (%)
                # Prevent division by zero
                truth_vals = df.loc[mask, truth_col]
                non_zero_mask = truth_vals != 0
                
                if np.sum(non_zero_mask) > 0:
                    res = (df.loc[mask, pred_col].loc[non_zero_mask] - truth_vals.loc[non_zero_mask]) / truth_vals.loc[non_zero_mask] * 100
                    ylabel_text = "Relative Residuals (%) (Prediction - Observation)/Truth × 100"
                else:
                    continue
            else:
                raise ValueError(f"Unknown residual_type: {residual_type}. Choose from 'difference', 'absolute', or 'relative'.")
            
            residuals.append(res)
            valid_labels.append(labels[i % len(labels)])
            valid_colors.append(colors[i % len(colors)])
    
    # If no valid data
    if not residuals:
        raise ValueError("No valid data for boxplot.")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create boxplot - Increase line width
    boxprops = dict(linestyle='-', linewidth=2)
    whiskerprops = dict(linestyle='-', linewidth=2)
    medianprops = dict(linestyle='-', linewidth=2.5, color='black')
    capprops = dict(linestyle='-', linewidth=2)
    flierprops = dict(marker='o', markerfacecolor='black', markersize=6, 
                     markeredgecolor='black', alpha=0.7)
    
    bp = ax.boxplot(residuals, patch_artist=True, labels=valid_labels, 
                   showfliers=showfliers, whis=1.5,
                   boxprops=boxprops, whiskerprops=whiskerprops, 
                   medianprops=medianprops, capprops=capprops,
                   flierprops=flierprops)
    
    # Set colors for boxes
    for box, color in zip(bp['boxes'], valid_colors):
        box.set(facecolor=color, alpha=0.6)
    
    # Add horizontal line (optional)
    if horizontal_line is not None:
        ax.axhline(y=horizontal_line, color='black', linestyle='--', linewidth=2, alpha=0.7)
    
    # Graph decoration - Increase font size
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel_text if 'ylabel_text' in locals() else ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize+2)  # Title is even larger
    
    # Set tick font size
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    
    # Increase axis line width
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    if grid:
        ax.grid(True, linestyle='--', linewidth=1.5, alpha=0.7)
    
    plt.tight_layout()
    
    return fig


def plot_combined_histograms(df, columns, colors=None, labels=None, title="Histogram",
                           xlabel="Value", ylabel="Frequency", figsize=(10, 6),
                           bins=30, alpha=0.5, range=None, density=False,
                           show_stats=False, grid=True, y_log=False,
                           show_hist=True, show_kde=False, kde_bandwidth=None):
    """
    Draw histograms and kernel density estimation for multiple columns on one plot
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing data
    columns : list of str
        List of column names to display as histograms
    colors : list of str, optional
        List of colors corresponding to each column. None uses default colors
    labels : list of str, optional
        List of labels corresponding to each column. None uses column names
    title : str, optional
        Plot title
    xlabel, ylabel : str, optional
        Axis labels
    figsize : tuple, optional
        Figure size
    bins : int or sequence or str, optional
        Number of bins or sequence of bin edges
    alpha : float, optional
        Transparency of histograms
    range : tuple of (float, float), optional
        Range of histogram. None for automatic determination
    density : bool, optional
        Whether to normalize as density function
    show_stats : bool, optional
        Whether to display statistical information on plot
    grid : bool, optional
        Whether to display grid
    y_log : bool, optional
        Whether to display y-axis on logarithmic scale
    show_hist : bool, optional
        Whether to display histogram (default is True)
    show_kde : bool, optional
        Whether to display kernel density estimation (default is False)
    kde_bandwidth : float or str, optional
        Bandwidth for kernel density estimation. 'scott', 'silverman' or number. None for automatic determination
        
    Returns:
    --------
    matplotlib.figure.Figure
        Generated figure object
    """
    # Necessary imports
    import matplotlib.pyplot as plt
    from scipy import stats
    import numpy as np
    
    # Default value setting
    if colors is None:
        colors = ['blue', 'red', 'green', 'purple', 'orange']  # Default color list
    
    if labels is None:
        labels = columns  # Default to use column names as labels
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate common range (if range argument is not specified)
    if range is None and len(columns) > 0:
        min_vals = []
        max_vals = []
        for col in columns:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                min_vals.append(valid_data.min())
                max_vals.append(valid_data.max())
        
        if min_vals and max_vals:
            range = (min(min_vals), max(max_vals))
    
    # Draw histograms and KDE for each column
    for i, col in enumerate(columns):
        color = colors[i % len(colors)]  # Cycle through color list
        label = labels[i % len(labels)]  # Cycle through label list
        
        # NaN filtering
        valid_data = df[col].dropna()
        
        if len(valid_data) > 0:
            # Draw histogram (optional)
            if show_hist:
                hist_label = label if not show_kde else f"{label} (Hist)"
                n, bins_edges, patches = ax.hist(valid_data, bins=bins, alpha=alpha, color=color, 
                                                label=hist_label, range=range, density=density)
            
            # Draw KDE (optional)
            if show_kde:
                kde_label = label if not show_hist else f"{label} (KDE)"
                
                # Calculate KDE
                if range is not None:
                    x_range = np.linspace(range[0], range[1], 1000)
                else:
                    x_min, x_max = valid_data.min(), valid_data.max()
                    # Add a little margin
                    margin = (x_max - x_min) * 0.1
                    x_range = np.linspace(x_min - margin, x_max + margin, 1000)
                
                # Calculate KDE
                kde = stats.gaussian_kde(valid_data, bw_method=kde_bandwidth)
                y_kde = kde(x_range)
                
                # Scale to match histogram (scale of histogram and KDE should match)
                if density and show_hist:
                    # No need to adjust as it's already scaled as density
                    pass
                elif show_hist:
                    # Scale to histogram scale
                    hist_area = n.sum() * (bins_edges[1] - bins_edges[0])
                    kde_area = np.trapz(y_kde, x_range)
                    y_kde = y_kde * (hist_area / kde_area)
                
                # Draw KDE
                ax.plot(x_range, y_kde, color=color, label=kde_label, linewidth=2)
            
            # Display statistical information (optional)
            if show_stats:
                mean_val = valid_data.mean()
                median_val = valid_data.median()
                std_val = valid_data.std()
                
                # Vertical line for mean value
                ax.axvline(x=mean_val, color=color, linestyle='--', alpha=0.8)
                
                # Display statistical information in text
                text_x = 0.7  # Text x position (relative)
                text_y = 0.9 - i * 0.1  # Text y position (relative), shift down for each column
                ax.text(text_x, text_y, f'{label}: Mean={mean_val:.2f}, Median={median_val:.2f}, Std={std_val:.2f}',
                       transform=ax.transAxes, fontsize=9, color=color)
    
    # Set y-axis to logarithmic scale (optional)
    if y_log:
        ax.set_yscale('log')
        # y-axis handling when 0
        bottom, top = ax.get_ylim()
        if bottom <= 0:
            ax.set_ylim(bottom=0.1)  # Set to value greater than 0
    
    # Graph decoration
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig


def plot_rmse_by_precipitation_intensity(df, 
                                        reference_col="MRMS", 
                                        model_cols=["Pred MTL_CW", "Pred STL_IR", "CCS", "SCaMPR"],
                                        colors=['black', 'blue', 'red', 'green', 'purple'],
                                        intensity_ranges=None,
                                        figsize=(10, 8),
                                        save_path=None,
                                        dpi=300):
    """
    Draw bar graph of RMSE by precipitation intensity and display distribution of reference data using KDE
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame for analysis
    reference_col : str
        Column name for reference data (truth)
    model_cols : list
        List of column names for model prediction values
    colors : list
        Colors for each model
    intensity_ranges : list of tuples
        List of tuples defining intensity ranges. Example: [(0, 1), (1, 5), (5, 10), (10, np.inf)]
        None uses default value
    figsize : tuple
        Figure size
    save_path : str
        Save path for figure. None to not save
    dpi : int
        Resolution of figure to save
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error
    from scipy import stats
    import matplotlib.ticker as ticker
    
    # Default intensity range setting
    if intensity_ranges is None:
        intensity_ranges = [(0, 1), (1, 5), (5, 10), (10, np.inf)]
    
    # Create intensity range labels (corrected to show up to third decimal place)
    intensity_labels = [f"{low:.3f}-{high:.3f}" if high != np.inf else f">{low:.3f}" for low, high in intensity_ranges]
    
    # Calculate RMSE for each model and intensity range
    rmse_results = {}
    
    for model_col in model_cols:
        rmse_values = []
        
        for low, high in intensity_ranges:
            # Filter data by intensity range
            mask = (df[reference_col] >= low) & (df[reference_col] < high)
            subset = df[mask]
            
            # Skip if no data in this range
            if len(subset) == 0:
                rmse_values.append(np.nan)
                continue
            
            # Exclude rows with NaN
            valid_data = subset.dropna(subset=[reference_col, model_col])
            
            # Skip if no valid data
            if len(valid_data) == 0:
                rmse_values.append(np.nan)
                continue
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(valid_data[reference_col], valid_data[model_col]))
            rmse_values.append(rmse)
        
        rmse_results[model_col] = rmse_values
    
    # Percentile calculation using only valid data
    valid_reference_data = df[df[reference_col] > 0][reference_col].dropna()
    percentiles = []
    
    # Calculate percentile value for each intensity range
    for low, _ in intensity_ranges:
        # Calculate what percentile 0 or above values are
        if low > 0:
            percentile = 100 * (valid_reference_data <= low).mean()
            percentiles.append(f"{percentile:.1f}%")
        else:
            percentiles.append("0%")
    # Add percentile for maximum value (100%)
    percentiles.append("100%")
    
    # Create plot
    fig = plt.figure(figsize=figsize)
    
    # Set grid spec (set main and KDE height ratio) - Order changed
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
    
    # Bar chart subplot (top)
    ax_bar = fig.add_subplot(gs[0])
    
    # KDE plot subplot (bottom)
    ax_kde = fig.add_subplot(gs[1])
    
    # Create bar chart
    # Set bar width
    bar_width = 0.15
    index = np.arange(len(intensity_labels))
    
    # Plot bars for each model
    for i, (model_col, color) in enumerate(zip(model_cols, colors[1:])):
        ax_bar.bar(index + i * bar_width, rmse_results[model_col], bar_width, 
               label=model_col, color=color, edgecolor='black', linewidth=0.5, alpha=0.7)
    
    # Add bar chart labels
    ax_bar.set_ylabel('RMSE (mm/h)', fontsize=12)
    ax_bar.set_title('RMSE by Precipitation Intensity', fontsize=14)
    ax_bar.set_xticks(index + bar_width * (len(model_cols) - 1) / 2)
    
    # Set x-axis label (include percentile information)
    x_labels = []
    for i, label in enumerate(intensity_labels):
        x_labels.append(f"{label}\n({percentiles[i]}-{percentiles[i+1]})")
    ax_bar.set_xticklabels(x_labels, fontsize=12)
    
    # Set x-axis label for bar chart (to be shared with KDE plot)
    ax_bar.set_xlabel('')
    
    # Legend to upper right outside
    # ax_bar.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    
    # Scientific paper style setting
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create KDE plot
    # Set x-axis range for KDE plot
    # Plot up to 99th percentile (exclude extreme outliers)
    x_max = min(valid_reference_data.quantile(0.99), intensity_ranges[-1][0] * 2)
    if np.isinf(x_max):  # If last range is infinite
        x_max = valid_reference_data.quantile(0.99)
    x_kde = np.linspace(0, x_max, 1000)
    
    # Calculate KDE
    kde = stats.gaussian_kde(valid_reference_data)
    ax_kde.plot(x_kde, kde(x_kde), color='black')
    
    # Add vertical lines at intensity range boundaries
    for i, (low, high) in enumerate(intensity_ranges):
        if i > 0:  # First boundary (0) is not displayed
            ax_kde.axvline(low, color='gray', linestyle='--', alpha=0.7)
            # Add label (value and percentile)
            ax_kde.text(low, ax_kde.get_ylim()[1] * 0.9, f"{low:.3f}\n({percentiles[i]})", 
                    rotation=90, ha='right', va='top', fontsize=12)
    
    # KDE plot decoration
    ax_kde.set_ylabel('Density', fontsize=10)
    ax_kde.set_xlabel('Precipitation Intensity (mm/h)', fontsize=12)
    ax_kde.spines['top'].set_visible(False)
    ax_kde.spines['right'].set_visible(False)
    ax_kde.set_xlim(0, x_max)  # Set x-axis range
    
    # Remove y-axis ticks (KDE value itself is not important)
    ax_kde.yaxis.set_major_formatter(ticker.NullFormatter())
    
    # Overall layout adjustment
    plt.tight_layout()
    fig.subplots_adjust(right=0.8)  # Create white space on right side
    
    # Save path is specified, save figure
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def determine_intensity_ranges(df, reference_col="MRMS", n_bins=4, method="percentile"):
    """
    Determine intensity ranges based on data distribution
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame for analysis
    reference_col : str
        Column name for reference data (truth)
    n_bins : int
        Number of bins to split
    method : str
        'percentile': Split based on percentile
        'log': Split on equal intervals on logarithmic scale
        'kmeans': Split by K-means clustering
        
    Returns:
    --------
    intensity_ranges : list of tuples
        List of tuples defining intensity ranges
    """
    import numpy as np
    from sklearn.cluster import KMeans
    
    # Exclude missing values and negative values
    valid_data = df[df[reference_col] > 0][reference_col].dropna()
    
    if method == "percentile":
        # Split based on percentile
        percentiles = np.linspace(0, 100, n_bins+1)
        thresholds = np.percentile(valid_data, percentiles)
        
        # Remove duplicates and sort
        thresholds = sorted(set(thresholds))
        
    elif method == "log":
        # Split on equal intervals on logarithmic scale
        min_val = valid_data.min()
        max_val = valid_data.max()
        
        # Add a small value if 0 is included
        if min_val <= 0:
            min_val = 0.01
            
        # Split on equal intervals on logarithmic scale
        log_thresholds = np.linspace(np.log10(min_val), np.log10(max_val), n_bins+1)
        thresholds = 10 ** log_thresholds
        
    elif method == "kmeans":
        # Split by K-means clustering
        kmeans = KMeans(n_clusters=n_bins, random_state=42)
        kmeans.fit(valid_data.values.reshape(-1, 1))
        
        # Sort cluster centers
        centers = sorted(kmeans.cluster_centers_.flatten())
        
        # Calculate boundary values from centers
        thresholds = [0]  # Start from 0
        for i in range(len(centers)-1):
            thresholds.append((centers[i] + centers[i+1]) / 2)
        thresholds.append(np.inf)  # Last is infinite
        
    else:
        raise ValueError("method must be one of 'percentile', 'log', or 'kmeans'")
    
    # Create range tuples from thresholds
    intensity_ranges = []
    for i in range(len(thresholds)-1):
        intensity_ranges.append((thresholds[i], thresholds[i+1]))
    
    # Last bin is infinite
    if intensity_ranges[-1][1] != np.inf:
        intensity_ranges[-1] = (intensity_ranges[-1][0], np.inf)
    
    return intensity_ranges
