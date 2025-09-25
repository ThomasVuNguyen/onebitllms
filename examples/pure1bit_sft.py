# Copyright 2025 The HuggingFace Team. and Falcon-LLM team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pure 1-Bit SFT Training Script

Usage:
python examples/pure1bit_sft.py \
    --model_name_or_path tiiuae/Falcon-E-1B-Base \
    --model_revision prequantized \
    --torch_dtype bfloat16 \
    --learning_rate 0.0001 \
    --dataset_name trl-lib/Capybara \
    --per_device_train_batch_size 1 \
    --output_dir Falcon-E-Capybara \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 100 \
    --packing \
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
    TrainingArguments,
    get_scheduler
)
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
from tqdm.auto import tqdm

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    TrlParser,
    get_kbit_device_map,
    get_quantization_config,
)

# Import pure 1-bit implementation
from onebitllms import (
    replace_linear_with_pure1bit_linear,
    create_pure1bit_optimizer,
    pure1bit_training_step,
    Pure1BitTrainingHelper
)


def main(script_args, training_args, model_args):
    ################
    # Model init kwargs & Tokenizer
    ################
    print("🚀 Starting Pure 1-Bit SFT Training")
    print(f"Model: {model_args.model_name_or_path}")
    print(f"Dataset: {script_args.dataset_name}")
    print(f"Output: {training_args.output_dir}")

    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model_id = model_args.model_name_or_path

    # Use the instruct tokenizer for fine-tuning
    if "Falcon-E" in model_id and "Base" in model_id:
        tokenizer_id = model_id.replace("Base", "Instruct")
    else:
        tokenizer_id = model_id

    # Create model
    print(f"📦 Loading model: {model_id}")
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()

    if config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures):
        from transformers import AutoModelForImageTextToText
        model_kwargs.pop("use_cache", None)  # Image models do not support cache
        model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

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
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    # Add PAD token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    print(f"📚 Loading dataset: {script_args.dataset_name}")
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None

    print(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Evaluation samples: {len(eval_dataset)}")

    # Preprocess dataset
    def preprocess_function(examples):
        # For conversational datasets, format the conversations
        if "conversations" in examples:
            texts = []
            for conversation in examples["conversations"]:
                formatted_text = ""
                for turn in conversation:
                    if turn["from"] == "human":
                        formatted_text += f"Human: {turn['value']}\n"
                    elif turn["from"] == "gpt":
                        formatted_text += f"Assistant: {turn['value']}\n"
                texts.append(formatted_text)
        else:
            # Fallback to text field if available
            texts = examples.get("text", examples.get("input", [""] * len(examples)))

        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,  # We'll pad in the collator
            max_length=training_args.max_length if hasattr(training_args, 'max_length') else 512,
            return_special_tokens_mask=False,
        )

        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("🔄 Preprocessing dataset...")
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

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    ################
    # Training Setup
    ################
    print("🏋️ Setting up Pure 1-Bit training...")

    # Set up pure 1-bit optimizer and training helper
    optimizer, training_helper = create_pure1bit_optimizer(
        model,
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay
    )

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        drop_last=True
    )

    if eval_dataset:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
            drop_last=False
        )

    # Calculate training steps
    num_update_steps_per_epoch = len(train_dataloader) // training_args.gradient_accumulation_steps
    max_train_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)

    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(training_args.warmup_ratio * max_train_steps),
        num_training_steps=max_train_steps
    )

    print(f"Total training steps: {max_train_steps}")
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")

    ################
    # Training Loop
    ################
    print("🚂 Starting training...")

    model.train()
    global_step = 0
    total_loss = 0

    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)

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

    for epoch in range(int(training_args.num_train_epochs)):
        print(f"\n📖 Epoch {epoch + 1}/{int(training_args.num_train_epochs)}")
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

        for step, batch in enumerate(progress_bar):
            # Training step with pure 1-bit
            loss = pure1bit_training_step(
                model, optimizer, training_helper,
                loss_fn, batch["input_ids"], batch["labels"],
                lr=training_args.learning_rate
            )

            epoch_loss += loss.item()
            total_loss += loss.item()

            # Update learning rate
            lr_scheduler.step()

            # Logging
            if global_step % training_args.logging_steps == 0:
                avg_loss = total_loss / (global_step + 1)
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{avg_loss:.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"
                })

                print(f"Step {global_step}: loss={loss.item():.4f}, avg_loss={avg_loss:.4f}")

            # Save checkpoint
            if (training_args.save_strategy == "steps" and
                global_step % training_args.save_steps == 0 and
                global_step > 0):

                save_checkpoint(model, tokenizer, training_args.output_dir, global_step)
                print(f"💾 Saved checkpoint at step {global_step}")

            global_step += 1

            if global_step >= max_train_steps:
                break

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

        # Save at end of epoch if strategy is epoch
        if training_args.save_strategy == "epoch":
            save_checkpoint(model, tokenizer, training_args.output_dir, f"epoch_{epoch + 1}")
            print(f"💾 Saved checkpoint at epoch {epoch + 1}")

    # Final save
    save_checkpoint(model, tokenizer, training_args.output_dir, "final")
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
    print(f"Final model saved to: {training_args.output_dir}")


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

    # Save model state dict (custom saving for pure 1-bit weights)
    model_state = {}
    for name, param in model.named_parameters():
        model_state[name] = param.data.clone()

    torch.save(model_state, checkpoint_dir / "pytorch_model.bin")

    # Save config
    model.config.save_pretrained(checkpoint_dir)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("pure1bit_sft", help="Run Pure 1-Bit SFT training", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Fix CPU/GPU compatibility issues
    if not torch.cuda.is_available():
        training_args.bf16 = False
        training_args.fp16 = False
        training_args.dataloader_pin_memory = False
        if model_args.torch_dtype == torch.bfloat16:
            model_args.torch_dtype = torch.float32

    main(script_args, training_args, model_args)