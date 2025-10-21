# Embodied Crowd Counting (ECC)

Official implementation for "Embodied Crowd Counting" - NeurIPS 2025

## Overview

Embodied Crowd Counting (ECC) is a novel approach to crowd counting that incorporates embodied perspective into the density estimation process. Unlike traditional crowd counting methods that treat images in isolation, ECC considers the agent's viewpoint and navigation context to improve counting accuracy in complex, real-world environments.

## Features

- **Embodied Perspective Integration**: Incorporates agent viewpoint and navigation context
- **Multi-Scale Feature Extraction**: Captures crowd patterns at different scales
- **Density Map Prediction**: Generates detailed density maps for spatial crowd distribution
- **Flexible Architecture**: Configurable model size for different computational budgets
- **Comprehensive Evaluation**: Standard crowd counting metrics (MAE, MSE, RMSE)

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/longrunling/ECC.git
cd ECC
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

## Project Structure

```
ECC/
├── src/ecc/
│   ├── models/          # Model architectures
│   ├── data/            # Dataset and data loading
│   └── utils/           # Utility functions
├── scripts/
│   ├── train.py         # Training script
│   ├── eval.py          # Evaluation script
│   └── infer.py         # Inference script
├── configs/             # Configuration files
├── requirements.txt     # Python dependencies
└── README.md
```

## Usage

### Data Preparation

Organize your dataset in the following structure:

```
data/
├── train/
│   ├── images/
│   └── density_maps/
├── val/
│   ├── images/
│   └── density_maps/
└── test/
    ├── images/
    └── density_maps/
```

Density maps should be saved as `.npy` files with the same name as their corresponding images.

### Training

Train the model with default configuration:

```bash
python scripts/train.py --config configs/default.yaml
```

Resume training from a checkpoint:

```bash
python scripts/train.py --config configs/default.yaml --resume checkpoints/checkpoint_epoch_50.pth
```

### Evaluation

Evaluate a trained model:

```bash
python scripts/eval.py --checkpoint checkpoints/best_model.pth --split test
```

Evaluate with visualization:

```bash
python scripts/eval.py --checkpoint checkpoints/best_model.pth --split test --visualize --output_dir results
```

### Inference

Run inference on a single image:

```bash
python scripts/infer.py --image path/to/image.jpg --checkpoint checkpoints/best_model.pth --output result.png
```

## Model Architecture

The ECC model consists of two main components:

1. **Embodied Encoder**: Multi-scale convolutional encoder that captures spatial features with embodied perspective awareness
2. **Density Decoder**: Upsampling decoder with skip connections that generates high-resolution density maps

### Configuration

Model and training parameters can be configured via YAML files. Key parameters:

- `model.base_channels`: Base channel size (32 for small model, 64 for default)
- `training.batch_size`: Batch size for training
- `training.learning_rate`: Learning rate for optimizer
- `data.image_size`: Input image size

## Evaluation Metrics

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and ground truth counts
- **MSE (Mean Squared Error)**: Average squared difference between predicted and ground truth counts
- **RMSE (Root Mean Squared Error)**: Square root of MSE

## Results

Results will be updated after experiments on standard crowd counting benchmarks.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{ecc2025,
  title={Embodied Crowd Counting},
  author={Research Team},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## License

This project is released under the MIT License.

## Acknowledgements

This work was developed for NeurIPS 2025. We thank the crowd counting community for their valuable contributions to the field.
