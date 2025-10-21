"""
Data transformations for embodied crowd counting
"""

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import numpy as np


def get_train_transforms():
    """
    Get training data transformations
    
    Returns:
        torchvision.transforms.Compose: Composed transforms for training
    """
    return T.Compose([
        T.Resize((512, 512)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms():
    """
    Get validation/test data transformations
    
    Returns:
        torchvision.transforms.Compose: Composed transforms for validation/test
    """
    return T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class DensityMapTransform:
    """
    Transform for density maps to match image transformations
    """
    
    def __init__(self, size=(512, 512)):
        self.size = size
    
    def __call__(self, density_map):
        """
        Apply transformation to density map
        
        Args:
            density_map (np.ndarray): Input density map
        
        Returns:
            torch.Tensor: Transformed density map
        """
        # Convert to tensor
        if isinstance(density_map, np.ndarray):
            density_map = torch.from_numpy(density_map)
        
        # Add channel dimension if needed
        if density_map.ndim == 2:
            density_map = density_map.unsqueeze(0)
        
        # Resize density map
        if self.size:
            density_map = TF.resize(density_map.unsqueeze(0), self.size).squeeze(0)
        
        return density_map
