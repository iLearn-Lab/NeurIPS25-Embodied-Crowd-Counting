# Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

## Installation

1. Clone the repository:
```bash
git clone https://github.com/longrunling/ECC.git
cd ECC
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Preparing Your Data

Your dataset should follow this structure:

```
data/
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── density_maps/
│       ├── img_001.npy
│       ├── img_002.npy
│       └── ...
├── val/
│   ├── images/
│   └── density_maps/
└── test/
    ├── images/
    └── density_maps/
```

### Density Map Format

Density maps should be saved as NumPy arrays (`.npy` files) with:
- Same spatial dimensions as the corresponding image
- Float32 dtype
- Each pixel represents the probability density of a person at that location

Example code to create a density map from point annotations:

```python
import numpy as np
from scipy.ndimage import gaussian_filter

def create_density_map(points, image_shape, sigma=15):
    """
    Create density map from point annotations
    
    Args:
        points: List of (x, y) coordinates
        image_shape: (height, width) of the image
        sigma: Gaussian kernel size
    """
    density = np.zeros(image_shape, dtype=np.float32)
    
    for x, y in points:
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            density[int(y), int(x)] = 1.0
    
    # Apply Gaussian filter
    density = gaussian_filter(density, sigma)
    
    return density
```

## Running the Model

### 1. Train a Model

```bash
python scripts/train.py --config configs/default.yaml
```

Training will:
- Save checkpoints every 10 epochs to `./checkpoints/`
- Save the best model based on validation MAE
- Log metrics to TensorBoard in `./logs/`

Monitor training with TensorBoard:
```bash
tensorboard --logdir logs
```

### 2. Evaluate the Model

```bash
python scripts/eval.py \
    --checkpoint checkpoints/best_model.pth \
    --split test \
    --visualize \
    --output_dir results
```

### 3. Run Inference on a Single Image

```bash
python scripts/infer.py \
    --image path/to/your/image.jpg \
    --checkpoint checkpoints/best_model.pth \
    --output prediction.png
```

## Tips

### For Small Datasets

Use the small model configuration for faster training:
```bash
python scripts/train.py --config configs/small_model.yaml
```

### For Limited GPU Memory

Reduce batch size in your config file:
```yaml
training:
  batch_size: 4  # Reduce from default 8
```

### Resume Training

If training is interrupted:
```bash
python scripts/train.py \
    --config configs/default.yaml \
    --resume checkpoints/checkpoint_epoch_30.pth
```

## Troubleshooting

### Out of Memory Error

1. Reduce batch size in config
2. Use the small model configuration
3. Reduce image size in config

### Dataset Not Found

Make sure your data directory matches the path in your config file:
```yaml
data:
  root_dir: ./data  # Update this path
```

### Import Errors

Ensure you've installed the package:
```bash
pip install -e .
```

## Next Steps

- Experiment with different model configurations
- Try transfer learning from a pretrained model
- Adjust hyperparameters for your specific dataset
- Implement data augmentation strategies

For more details, see the main [README.md](README.md).
