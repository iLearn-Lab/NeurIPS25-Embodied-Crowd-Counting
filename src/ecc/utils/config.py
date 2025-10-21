"""
Configuration management utilities
"""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to the configuration file
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        save_path (str): Path to save the configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration
    
    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
        'model': {
            'in_channels': 3,
            'base_channels': 64,
        },
        'training': {
            'batch_size': 8,
            'num_epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'num_workers': 4,
        },
        'data': {
            'root_dir': './data',
            'image_size': [512, 512],
        },
        'logging': {
            'log_dir': './logs',
            'checkpoint_dir': './checkpoints',
            'save_frequency': 10,
        }
    }
