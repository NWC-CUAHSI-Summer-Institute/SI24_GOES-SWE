# %% import the packages

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
from data_prep import prep as pr
from unet import UNet

# %% Functions

def plot_tensor(tensor, title="Tensor Plot"):
    tensor = tensor.detach().cpu().numpy()
    # Assuming you want to plot the first channel of the first batch
    tensor = tensor[0, 0, :, :]
    plt.imshow(tensor, cmap='viridis')
    plt.title(title)
    plt.colorbar()
    plt.show()



def train_model(epochs, train_loader):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to device
            
            outputs, x = model(inputs)
            targets = targets.view(-1, 1)  # Reshape targets to match outputs
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        plot_tensor(x)
        # Step the learning rate scheduler
        scheduler.step()
        # Print the learning rate for each epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Learning Rate: {current_lr}")


def eval_model(train_loader, scaler_out, actual_swe_values):
    # Set model to evaluation mode
    model.eval()
    predicted_swe = []

    # Run the model on the test data
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to device
            outputs, x = model(inputs)
            # outputs = outputs.reshape(-1, 1)
            
            outputs = scaler_out.inverse_transform(outputs)
            predicted_swe.extend(outputs)  # Collect the predicted values

    # Convert list to numpy array
    predicted_swe = np.array(predicted_swe).flatten()



    # Convert actual SWE values to a numpy array for easier handling
    actual_swe_values_np = actual_swe_values.cpu().numpy()

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(predicted_swe, label='Predicted SWE', color='red')
    plt.plot(actual_swe_values_np, label='Actual SWE', color='blue')
    plt.xlabel('Time')
    plt.ylabel('SWE')
    plt.title('Predicted vs. Actual SWE')
    plt.legend()
    plt.show()



# %% Hyperparameters

batch_size = 2
shuffle_value = True
# Define the model and move it to the device
in_channels = 2  # Two channels for Rad data
out_channels = 1  # Output is the SWE time series length
epochs = 30
station_name = 'uty'




# if __name__ == 'main':

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

print(f'Device: {device}')

train_loader, scaler_out, actual_swe_values = pr(station_name, batch_size, shuffle_value)

model = UNet(in_channel=in_channels, out_channel=out_channels).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
criterion = nn.SmoothL1Loss()

# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.98)


# %%

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to device
        
        outputs, x = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    plot_tensor(x)
    # Step the learning rate scheduler
    scheduler.step()
    # Print the learning rate for each epoch
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Learning Rate: {current_lr}")


# %%

model.eval()
predicted_swe = []

# Run the model on the test data
with torch.no_grad():
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to device
        outputs, x = model(inputs)
        # outputs = outputs.reshape(-1, 1)
        
        outputs = scaler_out.inverse_transform(outputs)
        predicted_swe.extend(outputs)  # Collect the predicted values

# Convert list to numpy array
predicted_swe = np.array(predicted_swe).flatten()



# Convert actual SWE values to a numpy array for easier handling
actual_swe_values_np = actual_swe_values.cpu().numpy()

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(predicted_swe, label='Predicted SWE', color='red')
plt.plot(actual_swe_values_np, label='Actual SWE', color='blue')
plt.xlabel('Time')
plt.ylabel('SWE')
plt.title('Predicted vs. Actual SWE')
plt.legend()
plt.show()