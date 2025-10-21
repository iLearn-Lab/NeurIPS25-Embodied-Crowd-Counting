"""
Evaluation script for Embodied Crowd Counting model
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from ecc.models import ECCModel
from ecc.data import CrowdCountingDataset, get_val_transforms
from ecc.utils import load_config, compute_mae, compute_mse, compute_rmse
from ecc.utils import visualize_predictions, denormalize_image


def evaluate(model, dataloader, device, save_visualizations=False, output_dir='outputs'):
    """Evaluate the model"""
    model.eval()
    
    all_pred_counts = []
    all_gt_counts = []
    
    if save_visualizations:
        os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        for idx, (images, density_maps, counts) in enumerate(pbar):
            images = images.to(device)
            density_maps = density_maps.to(device)
            counts = counts.to(device)
            
            # Forward pass
            pred_density = model(images)
            
            # Calculate counts
            pred_counts = pred_density.sum(dim=(1, 2, 3)).cpu().numpy()
            gt_counts = counts.cpu().numpy()
            
            all_pred_counts.extend(pred_counts)
            all_gt_counts.extend(gt_counts)
            
            # Save visualizations for first few samples
            if save_visualizations and idx < 10:
                for i in range(min(len(images), 3)):
                    img = denormalize_image(images[i])
                    pred_dm = pred_density[i, 0].cpu().numpy()
                    gt_dm = density_maps[i, 0].cpu().numpy()
                    
                    save_path = os.path.join(output_dir, f'sample_{idx * len(images) + i}.png')
                    visualize_predictions(
                        img, pred_dm, gt_dm,
                        pred_counts[i], gt_counts[i],
                        save_path
                    )
    
    # Calculate metrics
    all_pred_counts = np.array(all_pred_counts)
    all_gt_counts = np.array(all_gt_counts)
    
    mae = compute_mae(all_pred_counts, all_gt_counts)
    mse = compute_mse(all_pred_counts, all_gt_counts)
    rmse = compute_rmse(all_pred_counts, all_gt_counts)
    
    return mae, mse, rmse


def main():
    parser = argparse.ArgumentParser(description='Evaluate Embodied Crowd Counting model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test',
                       choices=['val', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualization of predictions')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Directory to save outputs')
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
    
    # Create dataset
    dataset = CrowdCountingDataset(
        root_dir=config['data']['root_dir'],
        split=args.split,
        transform=get_val_transforms()
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Evaluate
    print(f'\nEvaluating on {args.split} set...')
    mae, mse, rmse = evaluate(
        model, dataloader, device,
        save_visualizations=args.visualize,
        output_dir=args.output_dir
    )
    
    print(f'\nResults:')
    print(f'  MAE: {mae:.2f}')
    print(f'  MSE: {mse:.2f}')
    print(f'  RMSE: {rmse:.2f}')
    
    if args.visualize:
        print(f'\nVisualizations saved to {args.output_dir}')


if __name__ == '__main__':
    main()
