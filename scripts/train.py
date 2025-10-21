"""
Training script for Embodied Crowd Counting model
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ecc.models import ECCModel
from ecc.data import CrowdCountingDataset, get_train_transforms, get_val_transforms
from ecc.utils import load_config, compute_mae, compute_mse


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for images, density_maps, counts in pbar:
        images = images.to(device)
        density_maps = density_maps.to(device)
        counts = counts.to(device)
        
        # Forward pass
        pred_density = model(images)
        
        # Calculate loss
        loss = criterion(pred_density, density_maps)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        pred_counts = pred_density.sum(dim=(1, 2, 3))
        mae = compute_mae(pred_counts, counts)
        mse = compute_mse(pred_counts, counts)
        
        total_loss += loss.item()
        total_mae += mae
        total_mse += mse
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'MAE': f'{mae:.2f}',
            'MSE': f'{mse:.2f}'
        })
    
    n_batches = len(dataloader)
    return total_loss / n_batches, total_mae / n_batches, total_mse / n_batches


def validate(model, dataloader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
        for images, density_maps, counts in pbar:
            images = images.to(device)
            density_maps = density_maps.to(device)
            counts = counts.to(device)
            
            # Forward pass
            pred_density = model(images)
            
            # Calculate loss
            loss = criterion(pred_density, density_maps)
            
            # Calculate metrics
            pred_counts = pred_density.sum(dim=(1, 2, 3))
            mae = compute_mae(pred_counts, counts)
            mse = compute_mse(pred_counts, counts)
            
            total_loss += loss.item()
            total_mae += mae
            total_mse += mse
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'MAE': f'{mae:.2f}',
                'MSE': f'{mse:.2f}'
            })
    
    n_batches = len(dataloader)
    return total_loss / n_batches, total_mae / n_batches, total_mse / n_batches


def main():
    parser = argparse.ArgumentParser(description='Train Embodied Crowd Counting model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
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
    
    # Create datasets
    train_dataset = CrowdCountingDataset(
        root_dir=config['data']['root_dir'],
        split='train',
        transform=get_train_transforms()
    )
    
    val_dataset = CrowdCountingDataset(
        root_dir=config['data']['root_dir'],
        split='val',
        transform=get_val_transforms()
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Setup tensorboard
    writer = SummaryWriter(config['logging']['log_dir'])
    
    # Create checkpoint directory
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    
    # Training loop
    start_epoch = 0
    best_mae = float('inf')
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_mae = checkpoint.get('best_mae', float('inf'))
        print(f'Resumed from epoch {start_epoch}')
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        # Train
        train_loss, train_mae, train_mse = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_mae, val_mse = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MAE/train', train_mae, epoch)
        writer.add_scalar('MAE/val', val_mae, epoch)
        writer.add_scalar('MSE/train', train_mse, epoch)
        writer.add_scalar('MSE/val', val_mse, epoch)
        
        print(f'\nEpoch {epoch}:')
        print(f'  Train Loss: {train_loss:.4f}, MAE: {train_mae:.2f}, MSE: {train_mse:.2f}')
        print(f'  Val Loss: {val_loss:.4f}, MAE: {val_mae:.2f}, MSE: {val_mse:.2f}')
        
        # Save checkpoint
        if (epoch + 1) % config['logging']['save_frequency'] == 0:
            checkpoint_path = os.path.join(
                config['logging']['checkpoint_dir'],
                f'checkpoint_epoch_{epoch}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_mae': best_mae,
            }, checkpoint_path)
            print(f'Saved checkpoint to {checkpoint_path}')
        
        # Save best model
        if val_mae < best_mae:
            best_mae = val_mae
            best_model_path = os.path.join(
                config['logging']['checkpoint_dir'],
                'best_model.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mae': best_mae,
            }, best_model_path)
            print(f'Saved best model with MAE: {best_mae:.2f}')
    
    writer.close()
    print('Training completed!')


if __name__ == '__main__':
    main()
