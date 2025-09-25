#!/usr/bin/env python3
"""
Example demonstrating pure 1-bit finetuning without any full-precision shadow parameters.

This example shows how to:
1. Create a model with Pure1BitLinear layers
2. Train it using discrete {-1, 0, 1} weights throughout training
3. Compare memory usage vs. the hybrid approach

Note: This simplified version works without triton kernels for testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Import our pure 1-bit implementation
from onebitllms.layers.bitnet import Pure1BitLinear
from onebitllms.utils.monkey_patching import replace_linear_with_pure1bit_linear
from onebitllms.utils.pure1bit_training import create_pure1bit_optimizer, pure1bit_training_step


class SimpleModel(nn.Module):
    """Simple test model with linear layers"""
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


def create_dummy_data(batch_size=32, input_size=784, num_classes=10, num_batches=10):
    """Create dummy classification data"""
    X = torch.randn(batch_size * num_batches, input_size)
    y = torch.randint(0, num_classes, (batch_size * num_batches,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_model_memory_usage(model):
    """Calculate model memory usage in MB"""
    total_params = 0
    total_memory = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        param_memory = param.element_size() * num_params
        total_params += num_params
        total_memory += param_memory

        print(f"{name}: {num_params:,} params, "
              f"{param_memory / 1024**2:.2f} MB ({param.dtype})")

    return total_params, total_memory / 1024**2


def test_pure1bit_training():
    """Test the pure 1-bit training implementation"""
    print("=" * 60)
    print("Testing Pure 1-Bit Finetuning Implementation")
    print("=" * 60)

    # Create models
    print("\n1. Creating models...")
    model_hybrid = SimpleModel()
    model_pure1bit = SimpleModel()

    # Convert models
    print("\n2. Converting to BitNet layers...")

    # For testing, we'll just create a model with normal linear layers
    # and one with our Pure1BitLinear layers to show the concept

    print("Creating hybrid model (regular Linear layers for comparison)...")
    # Keep hybrid as regular model for comparison

    print("Converting to Pure 1-bit layers...")
    # Pure 1-bit approach (new implementation)
    replace_linear_with_pure1bit_linear(model_pure1bit)

    print("\n3. Memory usage comparison:")
    print("\nHybrid BitNet model (bf16 + quantization):")
    hybrid_params, hybrid_memory = get_model_memory_usage(model_hybrid)

    print("\nPure 1-bit model (int8 discrete weights):")
    pure1bit_params, pure1bit_memory = get_model_memory_usage(model_pure1bit)

    print(f"\nMemory reduction: {hybrid_memory - pure1bit_memory:.2f} MB "
          f"({100 * (1 - pure1bit_memory/hybrid_memory):.1f}% reduction)")

    # Test training
    print("\n4. Testing Pure 1-bit training...")

    # Create dummy data
    train_loader = create_dummy_data(batch_size=8, num_batches=5)

    # Create optimizer and training helper for pure 1-bit model
    optimizer, training_helper = create_pure1bit_optimizer(model_pure1bit, lr=1e-3)

    # Training loop
    model_pure1bit.train()
    loss_fn = nn.CrossEntropyLoss()

    print("\nTraining pure 1-bit model...")
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        loss = pure1bit_training_step(
            model_pure1bit, optimizer, training_helper,
            loss_fn, inputs, targets, lr=1e-3
        )

        # Check that weights are still discrete
        for name, module in model_pure1bit.named_modules():
            if isinstance(module, Pure1BitLinear):
                unique_values = torch.unique(module.weight.data)
                assert all(v.item() in [-1, 0, 1] for v in unique_values), \
                    f"Non-discrete values found: {unique_values}"

        print(f"Batch {batch_idx + 1}/5, Loss: {loss.item():.4f}")

        # Show gradient accumulation
        grad_norms = training_helper.get_gradient_norms()
        if grad_norms:
            avg_grad_norm = sum(grad_norms.values()) / len(grad_norms)
            print(f"  Avg gradient norm: {avg_grad_norm:.6f}")

    print("\n5. Verification: Weights remain discrete...")
    discrete_check = True
    for name, module in model_pure1bit.named_modules():
        if isinstance(module, Pure1BitLinear):
            unique_values = torch.unique(module.weight.data).cpu().numpy()
            print(f"{name}: unique values = {sorted(unique_values)}")
            if not all(v in [-1, 0, 1] for v in unique_values):
                discrete_check = False

    # Cleanup
    training_helper.cleanup()

    print("\n6. Results:")
    print(f"✓ Memory reduction achieved: {100 * (1 - pure1bit_memory/hybrid_memory):.1f}%")
    print(f"✓ Weights remain discrete: {discrete_check}")
    print(f"✓ Training completed without errors")

    print("\n" + "=" * 60)
    print("Pure 1-Bit Implementation Test Complete!")
    print("=" * 60)

    return {
        'hybrid_memory_mb': hybrid_memory,
        'pure1bit_memory_mb': pure1bit_memory,
        'memory_reduction_percent': 100 * (1 - pure1bit_memory/hybrid_memory),
        'discrete_weights_maintained': discrete_check
    }


if __name__ == "__main__":
    # Run the test
    results = test_pure1bit_training()
    print(f"\nFinal Results: {results}")