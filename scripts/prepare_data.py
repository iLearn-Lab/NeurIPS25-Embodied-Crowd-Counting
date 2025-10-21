"""
Utility script to prepare dataset for crowd counting

This script helps convert various crowd counting dataset formats
into the format expected by the ECC framework.
"""

import os
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import json


def create_density_map_from_points(points, image_shape, sigma=15):
    """
    Create density map from point annotations using Gaussian kernels
    
    Args:
        points: List of (x, y) coordinates of people
        image_shape: (height, width) tuple
        sigma: Standard deviation for Gaussian kernel
    
    Returns:
        np.ndarray: Density map
    """
    density = np.zeros(image_shape, dtype=np.float32)
    
    for x, y in points:
        # Check bounds
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            density[int(y), int(x)] = 1.0
    
    # Apply Gaussian filter
    density = gaussian_filter(density, sigma)
    
    return density


def load_annotations(annotation_file, format='json'):
    """
    Load annotations from file
    
    Args:
        annotation_file: Path to annotation file
        format: Annotation format ('json', 'txt', 'mat')
    
    Returns:
        List of (x, y) coordinates
    """
    if format == 'json':
        with open(annotation_file, 'r') as f:
            data = json.load(f)
            # Assume format: {"points": [[x1, y1], [x2, y2], ...]}
            return data.get('points', [])
    
    elif format == 'txt':
        points = []
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    points.append([x, y])
        return points
    
    elif format == 'mat':
        from scipy.io import loadmat
        mat_data = loadmat(annotation_file)
        # Adjust key based on your .mat file structure
        points = mat_data.get('image_info', [[]])[0][0][0][0][0]
        return points.tolist()
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def prepare_dataset(
    images_dir,
    annotations_dir,
    output_dir,
    annotation_format='json',
    sigma=15,
    splits={'train': 0.7, 'val': 0.15, 'test': 0.15}
):
    """
    Prepare dataset by creating density maps and organizing files
    
    Args:
        images_dir: Directory containing images
        annotations_dir: Directory containing annotations
        output_dir: Output directory for organized dataset
        annotation_format: Format of annotation files
        sigma: Gaussian kernel sigma for density maps
        splits: Dictionary with train/val/test split ratios
    """
    # Create output directories
    for split in splits.keys():
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'density_maps'), exist_ok=True)
    
    # Get list of images
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])
    
    print(f"Found {len(image_files)} images")
    
    # Determine split indices
    n_train = int(len(image_files) * splits['train'])
    n_val = int(len(image_files) * splits['val'])
    
    split_ranges = {
        'train': (0, n_train),
        'val': (n_train, n_train + n_val),
        'test': (n_train + n_val, len(image_files))
    }
    
    # Process each image
    for idx, img_file in enumerate(tqdm(image_files, desc="Processing images")):
        # Determine split
        split = None
        for s, (start, end) in split_ranges.items():
            if start <= idx < end:
                split = s
                break
        
        if split is None:
            continue
        
        # Load image
        img_path = os.path.join(images_dir, img_file)
        img = Image.open(img_path)
        img_array = np.array(img)
        
        # Get base filename
        base_name = os.path.splitext(img_file)[0]
        
        # Try to load annotations
        annotation_file = None
        for ext in ['.json', '.txt', '.mat']:
            potential_path = os.path.join(annotations_dir, base_name + ext)
            if os.path.exists(potential_path):
                annotation_file = potential_path
                break
        
        if annotation_file:
            # Load points
            points = load_annotations(annotation_file, annotation_format)
            
            # Create density map
            if len(img_array.shape) == 3:
                h, w = img_array.shape[:2]
            else:
                h, w = img_array.shape
            
            density_map = create_density_map_from_points(points, (h, w), sigma)
        else:
            print(f"Warning: No annotations found for {img_file}, using empty density map")
            if len(img_array.shape) == 3:
                h, w = img_array.shape[:2]
            else:
                h, w = img_array.shape
            density_map = np.zeros((h, w), dtype=np.float32)
        
        # Save image (copy to new location)
        output_img_path = os.path.join(output_dir, split, 'images', img_file)
        img.save(output_img_path)
        
        # Save density map
        output_density_path = os.path.join(output_dir, split, 'density_maps', base_name + '.npy')
        np.save(output_density_path, density_map)
    
    print("\nDataset preparation complete!")
    for split in splits.keys():
        start, end = split_ranges[split]
        n_samples = end - start
        print(f"{split}: {n_samples} samples")


def main():
    parser = argparse.ArgumentParser(description='Prepare crowd counting dataset')
    parser.add_argument('--images_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--annotations_dir', type=str, required=True,
                       help='Directory containing annotations')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Output directory for organized dataset')
    parser.add_argument('--annotation_format', type=str, default='json',
                       choices=['json', 'txt', 'mat'],
                       help='Format of annotation files')
    parser.add_argument('--sigma', type=float, default=15,
                       help='Gaussian kernel sigma for density maps')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Ratio of validation data')
    
    args = parser.parse_args()
    
    # Calculate test ratio
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    
    if test_ratio < 0:
        raise ValueError("Train and val ratios sum to more than 1.0")
    
    splits = {
        'train': args.train_ratio,
        'val': args.val_ratio,
        'test': test_ratio
    }
    
    prepare_dataset(
        args.images_dir,
        args.annotations_dir,
        args.output_dir,
        args.annotation_format,
        args.sigma,
        splits
    )


if __name__ == '__main__':
    main()
