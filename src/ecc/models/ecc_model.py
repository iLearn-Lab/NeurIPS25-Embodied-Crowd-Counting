"""
Embodied Crowd Counting Model

This module implements the core architecture for embodied crowd counting,
which combines visual perception with embodied navigation for accurate
crowd density estimation in complex environments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbodiedEncoder(nn.Module):
    """
    Encoder module that processes visual input with embodied perspective.
    Captures both spatial features and viewpoint-dependent information.
    """
    
    def __init__(self, in_channels=3, base_channels=64):
        super(EmbodiedEncoder, self).__init__()
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels * 2)
        
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_channels * 4)
        
        self.conv4 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(base_channels * 8)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Encoder pathway
        x1 = self.relu(self.bn1(self.conv1(x)))
        x1_pool = self.pool(x1)
        
        x2 = self.relu(self.bn2(self.conv2(x1_pool)))
        x2_pool = self.pool(x2)
        
        x3 = self.relu(self.bn3(self.conv3(x2_pool)))
        x3_pool = self.pool(x3)
        
        x4 = self.relu(self.bn4(self.conv4(x3_pool)))
        
        return x4, [x1, x2, x3, x4]


class DensityDecoder(nn.Module):
    """
    Decoder module that generates density maps from encoded features.
    Uses skip connections for multi-scale feature integration.
    """
    
    def __init__(self, base_channels=64):
        super(DensityDecoder, self).__init__()
        
        # Decoder layers with upsampling
        self.up1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.conv_up1 = nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1)
        self.bn_up1 = nn.BatchNorm2d(base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv_up2 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1)
        self.bn_up2 = nn.BatchNorm2d(base_channels * 2)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv_up3 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.bn_up3 = nn.BatchNorm2d(base_channels)
        
        # Final density prediction layer
        self.density_pred = nn.Conv2d(base_channels, 1, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, skip_connections):
        x1, x2, x3, x4 = skip_connections
        
        # Decoder pathway with skip connections
        up1 = self.up1(x)
        concat1 = torch.cat([up1, x3], dim=1)
        dec1 = self.relu(self.bn_up1(self.conv_up1(concat1)))
        
        up2 = self.up2(dec1)
        concat2 = torch.cat([up2, x2], dim=1)
        dec2 = self.relu(self.bn_up2(self.conv_up2(concat2)))
        
        up3 = self.up3(dec2)
        concat3 = torch.cat([up3, x1], dim=1)
        dec3 = self.relu(self.bn_up3(self.conv_up3(concat3)))
        
        # Generate density map
        density_map = self.density_pred(dec3)
        
        return density_map


class ECCModel(nn.Module):
    """
    Complete Embodied Crowd Counting Model
    
    This model integrates embodied perception with crowd density estimation,
    considering the agent's viewpoint and navigation context.
    
    Args:
        in_channels (int): Number of input channels (default: 3 for RGB)
        base_channels (int): Base number of channels for the network (default: 64)
    """
    
    def __init__(self, in_channels=3, base_channels=64):
        super(ECCModel, self).__init__()
        
        self.encoder = EmbodiedEncoder(in_channels, base_channels)
        self.decoder = DensityDecoder(base_channels)
        
    def forward(self, x):
        """
        Forward pass of the ECC model
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        
        Returns:
            torch.Tensor: Predicted density map of shape (B, 1, H, W)
        """
        encoded, skip_connections = self.encoder(x)
        density_map = self.decoder(encoded, skip_connections)
        
        return density_map
    
    def predict_count(self, x):
        """
        Predict the total crowd count from input image
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        
        Returns:
            torch.Tensor: Predicted count for each image in batch
        """
        density_map = self.forward(x)
        count = torch.sum(density_map.view(density_map.size(0), -1), dim=1)
        return count
