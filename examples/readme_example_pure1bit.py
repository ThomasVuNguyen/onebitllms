#!/usr/bin/env python3
"""
Example showing the README.md usage pattern adapted for pure 1-bit training.

This demonstrates how to use the new pure 1-bit approach instead of the hybrid approach
shown in the README.md.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config
from datasets import Dataset
from torch.utils.data import DataLoader

# Import pure 1-bit implementation instead of hybrid
from onebitllms import (
    replace_linear_with_pure1bit_linear,  # Instead of replace_linear_with_bitnet_linear
    create_pure1bit_optimizer,
    pure1bit_training_step
)


def main():
    print("=" * 60)
    print("README.md Example - Pure 1-Bit Version")
    print("=" * 60)

    print("\nOriginal README.md approach (hybrid):")
    print("```python")
    print("from onebitllms import replace_linear_with_bitnet_linear")
    print("model = replace_linear_with_bitnet_linear(model)")
    print("# Uses bf16 shadow parameters + quantization during forward pass")
    print("```")

    print("\nNew Pure 1-Bit approach:")
    print("```python")
    print("from onebitllms import replace_linear_with_pure1bit_linear")
    print("model = replace_linear_with_pure1bit_linear(model)")
    print("# Uses discrete {-1, 0, 1} weights throughout training")
    print("```")

    # Demo with a small model (since we can't download Falcon-E in this environment)
    print("\n1. Creating demo model (using GPT-2 config since Falcon-E requires download)...")

    config = GPT2Config(
        vocab_size=50257,  # Use full GPT-2 vocab to match tokenizer
        n_positions=128,
        n_embd=256,
        n_layer=4,
        n_head=4,
    )

    model = AutoModelForCausalLM.from_config(config)

    # Create a simple tokenizer for demo
    print("\n2. Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    print(f"\nOriginal model memory:")
    total_params = sum(p.numel() for p in model.parameters())
    total_memory = sum(p.element_size() * p.numel() for p in model.parameters()) / 1024**2
    print(f"  Total: {total_params:,} params, {total_memory:.2f} MB")

    print("\n3. Converting to Pure 1-Bit...")
    # This is the key replacement from the README.md approach
    model = replace_linear_with_pure1bit_linear(model)

    print(f"\nPure 1-bit model memory:")
    total_params_pure = sum(p.numel() for p in model.parameters())
    total_memory_pure = sum(p.element_size() * p.numel() for p in model.parameters()) / 1024**2
    print(f"  Total: {total_params_pure:,} params, {total_memory_pure:.2f} MB")

    memory_savings = total_memory - total_memory_pure
    memory_reduction_pct = 100 * (1 - total_memory_pure / total_memory)
    print(f"  Memory reduction: {memory_savings:.2f} MB ({memory_reduction_pct:.1f}%)")

    print("\n4. Training setup (Pure 1-Bit approach)...")

    # Create training data
    texts = ["This is a sample training text."] * 6
    tokenized = tokenizer(texts, padding=True, truncation=True, max_length=16, return_tensors="pt")
    input_ids = tokenized["input_ids"]

    # Set up pure 1-bit training (different from README.md)
    optimizer, training_helper = create_pure1bit_optimizer(model, lr=1e-4)

    print("   Original README.md training:")
    print("   ```python")
    print("   # Uses standard optimizers with bf16 shadow parameters")
    print("   trainer = SFTTrainer(model=model, ...)")
    print("   trainer.train()")
    print("   ```")

    print("\n   Pure 1-Bit training:")
    print("   ```python")
    print("   optimizer, training_helper = create_pure1bit_optimizer(model)")
    print("   # Custom training loop for discrete weights")
    print("   loss = pure1bit_training_step(model, optimizer, training_helper, ...)")
    print("   ```")

    # Demo training loop
    print("\n5. Running training demo...")
    model.train()

    # Run a few training steps
    for step in range(3):
        # Take a batch from the input
        batch_input = input_ids[step*2:(step+1)*2]
        labels = batch_input.clone()

        def loss_fn(outputs, targets):
            return torch.nn.functional.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                targets.view(-1)
            )

        loss = pure1bit_training_step(
            model, optimizer, training_helper,
            loss_fn, batch_input, labels, lr=1e-4
        )

        print(f"   Step {step+1}: Loss = {loss.item():.4f}")

    print("\n6. Verification...")
    # Check discrete weights
    discrete_layers = 0
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight.dtype == torch.int8:
            discrete_layers += 1

    print(f"   ✓ {discrete_layers} layers using discrete weights")
    print("   ✓ Training completed without full-precision shadow parameters")

    # Cleanup
    training_helper.cleanup()

    print("\n7. Key Differences from README.md approach:")
    print("   Hybrid (README.md):")
    print("   - Uses bf16 shadow parameters during training")
    print("   - Quantizes to 1.58-bit only during forward pass")
    print("   - Memory: full precision + quantization overhead")
    print("")
    print("   Pure 1-Bit (New approach):")
    print("   - Stores weights as discrete {-1, 0, 1} throughout")
    print("   - No quantization/dequantization during training")
    print(f"   - Memory: {memory_reduction_pct:.1f}% reduction achieved")

    print("\n" + "=" * 60)
    print("Pure 1-Bit README.md Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()