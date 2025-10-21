"""
Inference script for Embodied Crowd Counting model
"""

import os
import argparse
import torch
from PIL import Image
import numpy as np

from ecc.models import ECCModel
from ecc.data import get_val_transforms
from ecc.utils import load_config, visualize_predictions, denormalize_image


def predict_single_image(model, image_path, transform, device):
    """
    Predict crowd count for a single image
    
    Args:
        model: The ECC model
        image_path: Path to the input image
        transform: Image transformation pipeline
        device: Device to run inference on
    
    Returns:
        tuple: (predicted_count, density_map, original_image)
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Transform image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        density_map = model(image_tensor)
        predicted_count = density_map.sum().item()
    
    # Get density map as numpy array
    density_map_np = density_map[0, 0].cpu().numpy()
    
    # Get denormalized image
    image_np = denormalize_image(image_tensor[0])
    
    return predicted_count, density_map_np, image_np


def main():
    parser = argparse.ArgumentParser(description='Run inference with Embodied Crowd Counting model')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output visualization')
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Config file {args.config} not found. Using default configuration.")
        from ecc.utils.config import get_default_config
        config = get_default_config()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model
    model = ECCModel(
        in_channels=config['model']['in_channels'],
        base_channels=config['model']['base_channels']
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from {args.checkpoint}')
    
    # Get transforms
    transform = get_val_transforms()
    
    # Run inference
    print(f'\nProcessing image: {args.image}')
    predicted_count, density_map, image = predict_single_image(
        model, args.image, transform, device
    )
    
    print(f'Predicted count: {predicted_count:.1f}')
    
    # Save visualization
    if args.output:
        visualize_predictions(
            image, density_map,
            pred_count=predicted_count,
            save_path=args.output
        )
        print(f'Visualization saved to {args.output}')
    else:
        visualize_predictions(
            image, density_map,
            pred_count=predicted_count
        )


if __name__ == '__main__':
    main()
