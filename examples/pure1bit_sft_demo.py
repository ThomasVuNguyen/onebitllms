#!/usr/bin/env python3
"""
Demo showing how to use the new pure 1-bit finetuning approach
with a real transformer model (using a small GPT-2 for demonstration).

This demonstrates the pure 1-bit approach that eliminates all bf16/fp32
shadow parameters during training.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from datasets import Dataset
import argparse

# Import our pure 1-bit implementation
from onebitllms import (
    replace_linear_with_pure1bit_linear,
    create_pure1bit_optimizer,
    pure1bit_training_step,
    Pure1BitTrainingHelper
)


def create_simple_dataset(tokenizer, num_samples=100):
    """Create a simple text dataset for demonstration"""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a powerful programming language.",
        "Machine learning is transforming the world.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models can learn complex patterns from data.",
        "Transformers have revolutionized natural language processing.",
        "Attention mechanisms help models focus on relevant information.",
        "Large language models can generate coherent text.",
        "Fine-tuning adapts pre-trained models to specific tasks.",
        "Pure 1-bit training reduces memory requirements significantly."
    ] * (num_samples // 10 + 1)

    # Tokenize the texts
    tokenized = tokenizer(
        texts[:num_samples],
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt"
    )

    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    })

    return dataset


def get_model_memory_usage(model, name="Model"):
    """Calculate and display model memory usage"""
    total_params = 0
    total_memory = 0

    print(f"\n{name} memory usage:")
    for n, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            param_memory = param.element_size() * num_params
            total_params += num_params
            total_memory += param_memory

            # Only show key layers to avoid clutter
            if any(x in n for x in ['weight', 'bias']) and len(n.split('.')) <= 4:
                print(f"  {n}: {num_params:,} params, {param_memory / 1024**2:.2f} MB ({param.dtype})")

    print(f"  Total: {total_params:,} params, {total_memory / 1024**2:.2f} MB")
    return total_params, total_memory / 1024**2


def main():
    parser = argparse.ArgumentParser(description="Pure 1-bit SFT Demo")
    parser.add_argument("--model_size", type=str, default="tiny", choices=["tiny", "small"],
                       help="Model size (tiny=~10M params, small=~50M params)")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of training samples")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--use_pure1bit", action="store_true", help="Use pure 1-bit training")

    args = parser.parse_args()

    print("=" * 60)
    print("Pure 1-Bit SFT Demo")
    print("=" * 60)

    # Create a small GPT-2 config for demonstration
    if args.model_size == "tiny":
        config = GPT2Config(
            vocab_size=50257,
            n_positions=512,
            n_embd=256,
            n_layer=4,
            n_head=4,
        )
    else:  # small
        config = GPT2Config(
            vocab_size=50257,
            n_positions=512,
            n_embd=512,
            n_layer=6,
            n_head=8,
        )

    print(f"\n1. Creating {args.model_size} GPT-2 model...")

    # Create two models for comparison
    model_hybrid = GPT2LMHeadModel(config)
    model_pure1bit = GPT2LMHeadModel(config) if args.use_pure1bit else None

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    print("\n2. Model conversion...")

    # Show original memory usage
    hybrid_params, hybrid_memory = get_model_memory_usage(model_hybrid, "Original Model")

    if args.use_pure1bit:
        print("\nConverting to pure 1-bit...")
        # Convert to pure 1-bit (exclude lm_head as it's typically kept in full precision)
        replace_linear_with_pure1bit_linear(model_pure1bit)

        pure1bit_params, pure1bit_memory = get_model_memory_usage(model_pure1bit, "Pure 1-bit Model")

        print(f"\nMemory reduction: {hybrid_memory - pure1bit_memory:.2f} MB "
              f"({100 * (1 - pure1bit_memory/hybrid_memory):.1f}% reduction)")

    print(f"\n3. Creating dataset with {args.num_samples} samples...")
    dataset = create_simple_dataset(tokenizer, args.num_samples)

    # Create data collator and loader
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        pad_to_multiple_of=8
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator
    )

    if args.use_pure1bit and model_pure1bit is not None:
        print("\n4. Training pure 1-bit model...")

        # Set up pure 1-bit training
        model_pure1bit.train()
        optimizer, training_helper = create_pure1bit_optimizer(
            model_pure1bit, lr=args.learning_rate
        )

        # Training loop
        total_loss = 0
        num_batches = 0

        for epoch in range(args.num_epochs):
            epoch_loss = 0
            epoch_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                # Prepare inputs and targets
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']

                # For causal LM, labels are the same as input_ids
                labels = input_ids.clone()

                # Create loss function
                def loss_fn(outputs, targets):
                    logits = outputs.logits
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = targets[..., 1:].contiguous()
                    # Flatten the tokens
                    loss_fct = nn.CrossEntropyLoss()
                    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                    shift_labels = shift_labels.view(-1)
                    loss = loss_fct(shift_logits, shift_labels)
                    return loss

                # Training step
                loss = pure1bit_training_step(
                    model_pure1bit, optimizer, training_helper,
                    loss_fn, input_ids, labels, lr=args.learning_rate
                )

                epoch_loss += loss.item()
                epoch_batches += 1

                if batch_idx % 5 == 0:  # Print every 5 batches
                    print(f"  Epoch {epoch+1}/{args.num_epochs}, "
                          f"Batch {batch_idx+1}/{len(dataloader)}, "
                          f"Loss: {loss.item():.4f}")

            avg_epoch_loss = epoch_loss / epoch_batches
            total_loss += epoch_loss
            num_batches += epoch_batches

            print(f"  Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")

        avg_loss = total_loss / num_batches
        print(f"\nTraining completed! Average loss: {avg_loss:.4f}")

        # Verify weights are still discrete
        print("\n5. Verifying discrete weights...")
        discrete_check = True
        discrete_layers = 0

        for name, module in model_pure1bit.named_modules():
            if hasattr(module, 'weight') and module.weight.dtype == torch.int8:
                unique_values = torch.unique(module.weight.data)
                if not all(v.item() in [-1, 0, 1] for v in unique_values):
                    discrete_check = False
                    print(f"  WARNING: {name} has non-discrete values: {unique_values}")
                else:
                    discrete_layers += 1

        print(f"  ✓ {discrete_layers} layers maintain discrete weights")
        print(f"  ✓ All weights remain in {{-1, 0, 1}}: {discrete_check}")

        # Cleanup
        training_helper.cleanup()

        print("\n6. Final Results:")
        print(f"  ✓ Training completed successfully")
        print(f"  ✓ Memory reduction: {100 * (1 - pure1bit_memory/hybrid_memory):.1f}%")
        print(f"  ✓ Discrete weights maintained: {discrete_check}")
        print(f"  ✓ Final training loss: {avg_loss:.4f}")

    else:
        print("\n4. Skipping pure 1-bit training (use --use_pure1bit flag to enable)")
        print("   This demo shows the memory comparison only.")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()