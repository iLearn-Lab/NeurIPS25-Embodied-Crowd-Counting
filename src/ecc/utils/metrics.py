"""
Evaluation metrics for crowd counting
"""

import torch
import numpy as np
from typing import Union


def compute_mae(pred: Union[torch.Tensor, np.ndarray], 
                target: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute Mean Absolute Error (MAE)
    
    Args:
        pred: Predicted counts
        target: Ground truth counts
    
    Returns:
        float: MAE value
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    return float(np.mean(np.abs(pred - target)))


def compute_mse(pred: Union[torch.Tensor, np.ndarray],
                target: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute Mean Squared Error (MSE)
    
    Args:
        pred: Predicted counts
        target: Ground truth counts
    
    Returns:
        float: MSE value
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    return float(np.mean((pred - target) ** 2))


def compute_rmse(pred: Union[torch.Tensor, np.ndarray],
                 target: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute Root Mean Squared Error (RMSE)
    
    Args:
        pred: Predicted counts
        target: Ground truth counts
    
    Returns:
        float: RMSE value
    """
    mse = compute_mse(pred, target)
    return float(np.sqrt(mse))


def compute_density_map_loss(pred_density: torch.Tensor,
                             target_density: torch.Tensor) -> torch.Tensor:
    """
    Compute loss between predicted and target density maps
    
    Args:
        pred_density: Predicted density map
        target_density: Target density map
    
    Returns:
        torch.Tensor: Loss value
    """
    return torch.nn.functional.mse_loss(pred_density, target_density)
