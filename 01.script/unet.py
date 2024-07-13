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


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, activation=nn.SELU, dropout_rate=0.5):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            activation(),
            nn.Dropout(dropout_rate),  # Add dropout here
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            activation(),
            nn.Dropout(dropout_rate),  # And here
        )
        
    def forward(self, x):
        return self.double_conv(x)

# Downsampling Layer
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.SELU):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, activation=activation)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# Upsampling Layer
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, activation=nn.SELU):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, activation)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, activation=activation)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Output Convolutional Layer
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# U-Net Architecture
class UNet(nn.Module):
    def __init__(self, in_channel, out_channel, activation=nn.Mish, bilinear=True, base_channels=16):
        super().__init__()
        self.in_channels = in_channel
        self.out_channels = out_channel
        self.bilinear = bilinear
        c = base_channels
        self.inc = DoubleConv(self.in_channels, c, activation=activation)
        self.down1 = Down(c, 2 * c, activation=activation)
        self.down2 = Down(2 * c, 4 * c, activation=activation)
        self.down3 = Down(4 * c, 8 * c, activation=activation)
        factor = 2 if bilinear else 1
        self.down4 = Down(8 * c, 16 * c // factor, activation=activation)
        self.up1 = Up(16 * c, 8 * c // factor, bilinear, activation=activation)
        self.up2 = Up(8 * c, 4 * c // factor, bilinear, activation=activation)
        self.up3 = Up(4 * c, 2 * c // factor, bilinear, activation=activation)
        self.up4 = Up(2 * c, c, bilinear, activation=activation)
        self.outc = OutConv(c, self.out_channels)
        # self.fc = nn.Linear(c * 8 * 8, 1)  # This will be updated dynamically
                # Additional layers for temporal prediction
        self.global_pool = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x_out = self.outc(x)
        # x = x_out.view(x.size(0), -1)
        # z = self.fc(x)

        # Global average pooling and fully connected layer for time dimension output
        pooled_output = self.global_pool(x_out)
        pooled_output = pooled_output.view(pooled_output.size(0), -1)
        # time_output = self.fc(pooled_output)
        

        # x = x_out.view(x.size(0), -1)
        self.fc = nn.Linear(pooled_output.size(1), 1)
        z = self.fc(pooled_output)
        return z , x_out

    def get_feature_map_size(self, input_size):
        dummy_input = torch.zeros(1, self.in_channels, *input_size)
        with torch.no_grad():
            x1 = self.inc(dummy_input)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            x = self.outc(x)
        return x.view(x.size(0), -1).size(1)
    
