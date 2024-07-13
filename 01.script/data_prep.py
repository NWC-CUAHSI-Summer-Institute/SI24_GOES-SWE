import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import pairwise_distances_argmin
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# %%

class RadDataset(Dataset):
    def __init__(self, xarray_data, swe_data):
        self.data = xarray_data
        self.swe = swe_data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.swe[idx]
        return sample, label
    

def prep(station_name, batch_size, shuffle_value):
    zarr_file = f'/home/shared/GOES_DA/stacked_resampled/{station_name}_b1_b3_b5_swe_2023.zarr'

    xr_dataset = xr.open_zarr(zarr_file) #.isel( time=slice(0, 30)) # x=slice(x_min, x_max+1), y=slice(y_min, y_max+1),

    # Extract the bands data from the xarray dataset
    rad1_data = xr_dataset['Rad_1'].values  # Extract the first band
    rad3_data = xr_dataset['Rad_3'].values  # Extract the second band

    # Stack the bands along the channel dimension
    rad_data = np.stack([rad1_data, rad3_data], axis=1)  # Shape: (time, channels, y, x)

    # Check the dimensions of rad_data
    print(f"Original shape: {rad_data.shape}")

    # Reshape the data to 2D for StandardScaler
    num_samples, num_channels, height, width = rad_data.shape
    rad_data_reshaped = rad_data.reshape(num_samples, -1)

    # Scale the input data
    scaler_in = StandardScaler()
    rad_data_scaled_reshaped = scaler_in.fit_transform(rad_data_reshaped)

    # Reshape back to 4D
    rad_data_scaled = rad_data_scaled_reshaped.reshape(num_samples, num_channels, height, width)

    # Ensure that the input dimensions are large enough for U-Net
    print(f"After adding channel dimension: {rad_data_scaled.shape}")

    # Ensure that the height and width are sufficiently large
    min_size = 64  # Example minimum size, adjust based on your model
    if height < min_size or width < min_size:
        raise ValueError(f"Input height and width must be at least {min_size} pixels.")

    # Process actual SWE values
    actual_swe_values = torch.tensor(xr_dataset['swe'].isel(x=0, y=0).values, dtype=torch.float32)

    # Scale the SWE values
    scaler_out = StandardScaler()
    swe_scaled = scaler_out.fit_transform(actual_swe_values.reshape(-1, 1))

    # Check the shapes
    print(f"Scaled rad_data shape: {rad_data_scaled.shape}")
    print(f"Scaled SWE shape: {swe_scaled.shape}")



    # Convert to PyTorch tensor
    rad_data_tensor = torch.tensor(rad_data_scaled, dtype=torch.float32)

    # Create the PyTorch dataset
    dataset = RadDataset(rad_data_tensor, torch.tensor(swe_scaled, dtype=torch.float32))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_value)

    return train_loader, scaler_out, actual_swe_values