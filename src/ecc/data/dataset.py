"""
Dataset classes for embodied crowd counting
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Callable


class CrowdCountingDataset(Dataset):
    """
    Dataset for crowd counting with embodied perspective
    
    This dataset handles images and their corresponding density maps or point annotations.
    Supports various crowd counting datasets with embodied viewpoint information.
    
    Args:
        root_dir (str): Root directory containing the dataset
        split (str): Dataset split ('train', 'val', or 'test')
        transform (Optional[Callable]): Optional transform to be applied on images
        target_transform (Optional[Callable]): Optional transform for density maps
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.density_dir = os.path.join(root_dir, split, 'density_maps')
        
        # Load image list
        self.image_files = []
        if os.path.exists(self.image_dir):
            self.image_files = sorted([
                f for f in os.listdir(self.image_dir)
                if f.endswith(('.jpg', '.png', '.jpeg'))
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Args:
            idx (int): Index of the sample
        
        Returns:
            tuple: (image, density_map, count) where density_map is the ground truth
                   density map and count is the total number of people
        """
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load or create density map
        density_name = img_name.replace('.jpg', '.npy').replace('.png', '.npy').replace('.jpeg', '.npy')
        density_path = os.path.join(self.density_dir, density_name)
        
        if os.path.exists(density_path):
            density_map = np.load(density_path).astype(np.float32)
        else:
            # If no density map exists, create a zero map
            density_map = np.zeros((image.height, image.width), dtype=np.float32)
        
        # Calculate count from density map
        count = np.sum(density_map)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            density_map = self.target_transform(density_map)
        else:
            # Convert to tensor if no transform provided
            density_map = torch.from_numpy(density_map).unsqueeze(0)
        
        return image, density_map, torch.tensor(count, dtype=torch.float32)
    
    def get_image_path(self, idx):
        """Get the file path for an image at given index"""
        img_name = self.image_files[idx]
        return os.path.join(self.image_dir, img_name)
