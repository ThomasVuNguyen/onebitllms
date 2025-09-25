#!/usr/bin/env python3
"""
Simplified Pure 1-Bit SFT Training Script

Usage:
python examples/pure1bit_sft_simple.py \
    --model_name_or_path tiiuae/Falcon-E-1B-Base \
    --model_revision prequantized \
    --dataset_name trl-lib/Capybara \
    --output_dir Falcon-E-Capybara-Pure1Bit \
    --learning_rate 0.0001 \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --save_steps 100 \
    --gradient_accumulation_steps 16
"""

import argparse
import os
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler
)
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
from tqdm.auto import tqdm

# Import pure 1-bit implementation
from onebitllms import (
    replace_linear_with_pure1bit_linear,
    create_pure1bit_optimizer,
    pure1bit_training_step,
    Pure1BitTrainingHelper
)


def main():
    parser = argparse.ArgumentParser(description="Pure 1-Bit SFT Training")

    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Model name or path")
    parser.add_argument("--model_revision", type=str, default=None, help="Model revision")
    parser.add_argument("--torch_dtype", type=str, default="float32", help="Model dtype")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")

    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset config")
    parser.add_argument("--dataset_train_split", type=str, default="train", help="Train split")
    parser.add_argument("--dataset_test_split", type=str, default="test", help="Test split")

    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "gpu"], help="Device to use (auto/cpu/gpu)")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Eval batch size")
    parser.add_argument("--num_train_epochs", type=float, default=1.0, help="Number of epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["steps", "epoch"], help="Save strategy")
    parser.add_argument("--eval_strategy", type=str, default="no", choices=["no", "steps", "epoch"], help="Eval strategy")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    if args.torch_dtype == "float32":
        torch_dtype = torch.float32
    elif args.torch_dtype == "float16":
        torch_dtype = torch.float16
    elif args.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    # Determine device based on args and availability
    if args.device == "auto":
        use_gpu = torch.cuda.is_available()
        device_name = "CUDA (auto)" if use_gpu else "CPU (auto)"
    elif args.device == "gpu":
        if not torch.cuda.is_available():
            print("❌ GPU requested but CUDA not available, falling back to CPU")
            use_gpu = False
            device_name = "CPU (GPU unavailable)"
        else:
            use_gpu = True
            device_name = "CUDA (forced)"
    else:  # args.device == "cpu"
        use_gpu = False
        device_name = "CPU (forced)"

    # Adjust dtype based on device
    if not use_gpu and torch_dtype == torch.bfloat16:
        print("⚠️  bf16 not supported on CPU, falling back to float32")
        torch_dtype = torch.float32

    print("🚀 Starting Pure 1-Bit SFT Training")
    print(f"Model: {args.model_name_or_path}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {device_name}")
    print(f"Dtype: {torch_dtype}")

    ################
    # Model Loading
    ################
    model_kwargs = dict(
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map="auto" if use_gpu else None,
    )

    # Use the instruct tokenizer for fine-tuning if available
    model_id = args.model_name_or_path
    if "Falcon-E" in model_id and "Base" in model_id:
        tokenizer_id = model_id.replace("Base", "Instruct")
    else:
        tokenizer_id = model_id

    # Create model
    print(f"📦 Loading model: {model_id}")
    try:
        config = AutoConfig.from_pretrained(args.model_name_or_path, revision=args.model_revision)
        valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()

        if config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures):
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 This might be due to network access or authentication issues.")
        print("💡 For testing, you can use a local model like 'gpt2' or 'microsoft/DialoGPT-medium'")
        return

    # Show original memory usage
    print(f"\n📊 Original model memory usage:")
    original_memory = get_model_memory_usage(model)

    # Convert to pure 1-bit
    print(f"\n🔧 Converting to Pure 1-Bit...")
    model = replace_linear_with_pure1bit_linear(model)

    # Show pure 1-bit memory usage
    print(f"\n📊 Pure 1-Bit model memory usage:")
    pure1bit_memory = get_model_memory_usage(model)

    memory_reduction = original_memory - pure1bit_memory
    memory_reduction_pct = 100 * (1 - pure1bit_memory / original_memory) if original_memory > 0 else 0
    print(f"\n💾 Memory reduction: {memory_reduction:.2f} MB ({memory_reduction_pct:.1f}%)")

    # Create tokenizer
    print(f"🔤 Loading tokenizer: {tokenizer_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=args.trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return

    ################
    # Dataset Loading
    ################
    print(f"📚 Loading dataset: {args.dataset_name}")
    try:
        if args.dataset_config:
            dataset = load_dataset(args.dataset_name, args.dataset_config)
        else:
            dataset = load_dataset(args.dataset_name)

        train_dataset = dataset[args.dataset_train_split]
        eval_dataset = dataset[args.dataset_test_split] if args.eval_strategy != "no" and args.dataset_test_split in dataset else None

        print(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            print(f"Evaluation samples: {len(eval_dataset)}")

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("💡 For testing, you can use datasets like 'wikitext' or create a simple text dataset")
        return

    # Preprocess dataset
    def preprocess_function(examples):
        # Handle different dataset formats
        if "conversations" in examples:
            texts = []
            for conversation in examples["conversations"]:
                formatted_text = ""
                for turn in conversation:
                    if turn.get("from") == "human":
                        formatted_text += f"Human: {turn['value']}\n"
                    elif turn.get("from") == "gpt":
                        formatted_text += f"Assistant: {turn['value']}\n"
                texts.append(formatted_text)
        elif "text" in examples:
            texts = examples["text"]
        elif "input" in examples:
            texts = examples["input"]
        else:
            # Fallback: use the first string field found
            for key, values in examples.items():
                if isinstance(values, list) and len(values) > 0 and isinstance(values[0], str):
                    texts = values
                    break
            else:
                raise ValueError("Could not find text data in dataset")

        # Tokenize - padding=False is correct, we'll pad later in the collator
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,  # Don't pad here, let the collator handle it
            max_length=args.max_length,
            return_special_tokens_mask=False,
        )

        # For causal LM, labels are the same as input_ids
        # Make sure labels is a copy, not a reference
        tokenized["labels"] = [ids.copy() if isinstance(ids, list) else ids for ids in tokenized["input_ids"]]
        return tokenized

    print("🔄 Preprocessing dataset...")
    try:
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing training data"
        )

        if eval_dataset:
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=eval_dataset.column_names,
                desc="Tokenizing evaluation data"
            )
    except Exception as e:
        print(f"❌ Error preprocessing dataset: {e}")
        return

    # Simple custom data collator that handles variable lengths properly
    def collate_fn(batch):
        # Find max length in this batch
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids_batch = []
        labels_batch = []

        for item in batch:
            input_ids = item["input_ids"]
            labels = item["labels"]

            # Pad to max length
            pad_length = max_len - len(input_ids)
            if pad_length > 0:
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_length
                labels = labels + [-100] * pad_length  # -100 is ignored in loss computation

            input_ids_batch.append(input_ids)
            labels_batch.append(labels)

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long)
        }

    data_collator = collate_fn

    ################
    # Training Setup
    ################
    print("🏋️ Setting up Pure 1-Bit training...")

    # Set up pure 1-bit optimizer and training helper
    optimizer, training_helper = create_pure1bit_optimizer(
        model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        drop_last=True,
        pin_memory=False  # Disable pin_memory to avoid potential issues
    )

    if eval_dataset:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
            drop_last=False
        )

    # Calculate training steps
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    max_train_steps = int(args.num_train_epochs * num_update_steps_per_epoch)

    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * max_train_steps),
        num_training_steps=max_train_steps
    )

    print(f"Total training steps: {max_train_steps}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")

    ################
    # Training Loop
    ################
    print("🚂 Starting training...")

    model.train()
    global_step = 0
    total_loss = 0

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

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

    for epoch in range(int(args.num_train_epochs)):
        print(f"\n📖 Epoch {epoch + 1}/{int(args.num_train_epochs)}")
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

        for step, batch in enumerate(progress_bar):
            try:
                # Move batch to same device as model
                device = next(model.parameters()).device
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                # Training step with pure 1-bit
                loss = pure1bit_training_step(
                    model, optimizer, training_helper,
                    loss_fn, input_ids, labels,
                    lr=args.learning_rate
                )

                epoch_loss += loss.item()
                total_loss += loss.item()

                # Update learning rate
                lr_scheduler.step()

                # Logging
                if global_step % args.logging_steps == 0:
                    avg_loss = total_loss / (global_step + 1)
                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "avg_loss": f"{avg_loss:.4f}",
                        "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"
                    })

                    print(f"Step {global_step}: loss={loss.item():.4f}, avg_loss={avg_loss:.4f}")

                # Save checkpoint
                if (args.save_strategy == "steps" and
                    args.save_steps > 0 and
                    global_step % args.save_steps == 0 and
                    global_step > 0):

                    save_checkpoint(model, tokenizer, args.output_dir, global_step)
                    print(f"💾 Saved checkpoint at step {global_step}")

                global_step += 1

                if global_step >= max_train_steps:
                    break

            except Exception as e:
                print(f"❌ Error during training step {global_step}: {e}")
                continue

        avg_epoch_loss = epoch_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

        # Save at end of epoch if strategy is epoch
        if args.save_strategy == "epoch":
            save_checkpoint(model, tokenizer, args.output_dir, f"epoch_{epoch + 1}")
            print(f"💾 Saved checkpoint at epoch {epoch + 1}")

    # Final save
    save_checkpoint(model, tokenizer, args.output_dir, "final")
    print(f"💾 Saved final checkpoint")

    # Verification
    print("\n🔍 Verifying discrete weights...")
    discrete_layers = 0
    total_discrete_params = 0

    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight.dtype == torch.int8:
            discrete_layers += 1
            total_discrete_params += module.weight.numel()

    print(f"✓ {discrete_layers} layers using discrete weights")
    print(f"✓ {total_discrete_params:,} discrete parameters")
    print(f"✓ Memory reduction achieved: {memory_reduction_pct:.1f}%")

    # Cleanup
    training_helper.cleanup()

    print(f"\n🎉 Pure 1-Bit training completed successfully!")
    print(f"Final model saved to: {args.output_dir}")


def get_model_memory_usage(model):
    """Calculate model memory usage in MB"""
    total_memory = 0
    for param in model.parameters():
        if param.requires_grad:
            param_memory = param.element_size() * param.numel()
            total_memory += param_memory
    return total_memory / 1024**2


def save_checkpoint(model, tokenizer, output_dir, step):
    """Save model checkpoint"""
    checkpoint_dir = Path(output_dir) / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save tokenizer
    tokenizer.save_pretrained(checkpoint_dir)

    # Save model with safetensors format (safer and more compatible)
    try:
        model.save_pretrained(checkpoint_dir, safe_serialization=True)
        print(f"✅ Saved model with safetensors format")
    except Exception as e:
        print(f"⚠️  Safetensors save failed: {e}")
        # Fallback to pytorch format
        model_state = {}
        for name, param in model.named_parameters():
            model_state[name] = param.data.clone()
        torch.save(model_state, checkpoint_dir / "pytorch_model.bin")
        print(f"✅ Saved model with pytorch format (fallback)")

    # Save config
    model.config.save_pretrained(checkpoint_dir)


if __name__ == "__main__":
    main()