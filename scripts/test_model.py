"""
Simple test script to validate the ECC model
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from ecc.models import ECCModel


def test_model_creation():
    """Test if model can be created"""
    print("Testing model creation...")
    model = ECCModel(in_channels=3, base_channels=64)
    print(f"✓ Model created successfully")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def test_forward_pass():
    """Test if forward pass works"""
    print("\nTesting forward pass...")
    model = ECCModel(in_channels=3, base_channels=64)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 512, 512)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Output shape: {output.shape}")
    
    expected_shape = (batch_size, 1, 512, 512)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"
    print(f"✓ Output shape is correct")


def test_predict_count():
    """Test if predict_count method works"""
    print("\nTesting predict_count method...")
    model = ECCModel(in_channels=3, base_channels=64)
    model.eval()
    
    input_tensor = torch.randn(2, 3, 512, 512)
    
    with torch.no_grad():
        counts = model.predict_count(input_tensor)
    
    print(f"✓ Predict count successful")
    print(f"  Predicted counts shape: {counts.shape}")
    print(f"  Sample counts: {counts.numpy()}")
    
    assert counts.shape == (2,), f"Expected count shape (2,), got {counts.shape}"
    print(f"✓ Count shape is correct")


def test_small_model():
    """Test if small model variant works"""
    print("\nTesting small model variant...")
    model = ECCModel(in_channels=3, base_channels=32)
    print(f"✓ Small model created successfully")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    input_tensor = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"✓ Small model forward pass successful")


def main():
    print("=" * 60)
    print("ECC Model Tests")
    print("=" * 60)
    
    try:
        test_model_creation()
        test_forward_pass()
        test_predict_count()
        test_small_model()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
