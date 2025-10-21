"""
Visualization utilities for crowd counting
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional


def plot_density_map(density_map: np.ndarray,
                     title: str = "Density Map",
                     save_path: Optional[str] = None):
    """
    Plot a density map
    
    Args:
        density_map: Density map to visualize
        title: Title for the plot
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(density_map, cmap='jet')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def visualize_predictions(image: np.ndarray,
                         pred_density: np.ndarray,
                         gt_density: Optional[np.ndarray] = None,
                         pred_count: Optional[float] = None,
                         gt_count: Optional[float] = None,
                         save_path: Optional[str] = None):
    """
    Visualize image with predicted and ground truth density maps
    
    Args:
        image: Input image (H, W, 3)
        pred_density: Predicted density map (H, W)
        gt_density: Ground truth density map (H, W), optional
        pred_count: Predicted count, optional
        gt_count: Ground truth count, optional
        save_path: Optional path to save the figure
    """
    n_plots = 2 if gt_density is None else 3
    
    fig, axes = plt.subplots(1, n_plots, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Predicted density map
    im1 = axes[1].imshow(pred_density, cmap='jet')
    title = 'Predicted Density Map'
    if pred_count is not None:
        title += f'\nCount: {pred_count:.1f}'
    axes[1].set_title(title)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # Ground truth density map
    if gt_density is not None:
        im2 = axes[2].imshow(gt_density, cmap='jet')
        title = 'Ground Truth Density Map'
        if gt_count is not None:
            title += f'\nCount: {gt_count:.1f}'
        axes[2].set_title(title)
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def denormalize_image(tensor: torch.Tensor,
                     mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]) -> np.ndarray:
    """
    Denormalize an image tensor for visualization
    
    Args:
        tensor: Normalized image tensor (C, H, W)
        mean: Mean used for normalization
        std: Std used for normalization
    
    Returns:
        np.ndarray: Denormalized image (H, W, C)
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    
    # Denormalize
    for i in range(3):
        tensor[i] = tensor[i] * std[i] + mean[i]
    
    # Clip to valid range and convert to HWC format
    tensor = np.clip(tensor, 0, 1)
    tensor = np.transpose(tensor, (1, 2, 0))
    
    return tensor
