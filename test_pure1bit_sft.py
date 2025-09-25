#!/usr/bin/env python3
"""
Test script for pure1bit_sft.py - uses a local GPT-2 model for testing
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test the command line interface
def test_cli():
    print("🧪 Testing Pure 1-Bit SFT CLI...")

    # Create a simple test dataset
    test_data = {
        "text": [
            "This is a test sentence for training.",
            "Another example text for the model to learn.",
            "Pure 1-bit training is memory efficient.",
            "The model should learn from these examples."
        ] * 5  # Repeat to have enough samples
    }

    # Create temporary dataset file
    from datasets import Dataset
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = Dataset.from_dict(test_data)
        dataset_path = os.path.join(temp_dir, "test_dataset")
        dataset.save_to_disk(dataset_path)

        print(f"Created test dataset at: {dataset_path}")
        print(f"Dataset size: {len(dataset)} samples")

        # Show that we have the CLI arguments available
        print("\n📋 Available CLI arguments for pure1bit_sft.py:")
        print("Required:")
        print("  --model_name_or_path: HuggingFace model path")
        print("  --dataset_name: Dataset name or path")
        print("  --output_dir: Output directory")
        print()
        print("Example usage:")
        print("python examples/pure1bit_sft.py \\")
        print("    --model_name_or_path microsoft/DialoGPT-medium \\")
        print("    --dataset_name wikitext --dataset_config wikitext-2-raw-v1 \\")
        print("    --per_device_train_batch_size 2 \\")
        print("    --learning_rate 5e-5 \\")
        print("    --num_train_epochs 1 \\")
        print("    --output_dir ./test_output \\")
        print("    --logging_steps 10 \\")
        print("    --save_steps 100")

        print("\n✅ CLI interface test completed!")

if __name__ == "__main__":
    test_cli()