#!/usr/bin/env python3
"""
Test script for the newly trained Pure 1-Bit model

This script loads your trained pure 1-bit model and tests it with various prompts
to verify it's working correctly and producing good outputs.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def load_pure1bit_model(checkpoint_path):
    """Load the trained pure 1-bit model"""
    print(f"🔄 Loading model from: {checkpoint_path}")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model - try different loading methods
        try:
            # First try loading base model and converting to Pure1BitLinear
            print("🔧 Loading base model and converting to Pure1BitLinear...")
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)

            # Create base model
            model = AutoModelForCausalLM.from_config(
                config,
                dtype=torch.float32
            )

            # Convert to Pure1BitLinear
            from onebitllms import replace_linear_with_pure1bit_linear
            model = replace_linear_with_pure1bit_linear(model)

            # Now try to load the trained weights from sharded safetensors
            import json
            index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")

            if os.path.exists(index_path):
                print(f"Loading sharded safetensors weights...")
                from safetensors.torch import load_file

                # Load the index to get shard information
                with open(index_path, 'r') as f:
                    index = json.load(f)

                # Load all shards and combine state dict
                state_dict = {}
                for shard_file in set(index['weight_map'].values()):
                    shard_path = os.path.join(checkpoint_path, shard_file)
                    print(f"Loading shard: {shard_file}")
                    shard_state = load_file(shard_path)
                    state_dict.update(shard_state)

                model.load_state_dict(state_dict, strict=False)
                print(f"✅ Loaded {len(state_dict)} parameters from sharded safetensors")
            else:
                raise FileNotFoundError("No sharded safetensors found")

        except Exception as e1:
            print(f"❌ Failed to load model: {str(e1)[:200]}...")
            print("💡 Make sure the checkpoint was trained with pure 1-bit format")
            return None, None

        # Verify it's still pure 1-bit
        discrete_layers = 0
        total_discrete_params = 0

        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight.dtype == torch.int8:
                discrete_layers += 1
                total_discrete_params += module.weight.numel()

        print(f"✅ Model loaded successfully!")
        print(f"✅ {discrete_layers} layers using discrete weights")
        print(f"✅ {total_discrete_params:,} discrete parameters")

        return model, tokenizer

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None


def test_generation(model, tokenizer, prompt, max_length=100, temperature=0.7, stream=True):
    """Test text generation with the model"""
    print(f"\n🤖 Testing prompt: '{prompt}'")

    # Tokenize input (exclude token_type_ids which Falcon doesn't use)
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)

    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    model.eval()
    with torch.no_grad():
        if stream:
            print(f"📝 Response: ", end="", flush=True)

            # Initialize generation
            generated = inputs["input_ids"]
            prompt_length = generated.size(1)

            for _ in range(max_length - prompt_length):
                # Get next token logits
                outputs = model(generated)
                next_token_logits = outputs.logits[0, -1, :]

                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature

                    # Sample from the distribution
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Stop if EOS token
                if next_token.item() == tokenizer.eos_token_id:
                    break

                # Append token to sequence
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

                # Decode and print the new token
                new_token_text = tokenizer.decode(next_token, skip_special_tokens=True)
                print(new_token_text, end="", flush=True)

            print()  # New line after generation

            # Get full generated text for return
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        else:
            # Non-streaming generation (for compatibility)
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_text = generated_text[len(prompt):].strip()
            print(f"📝 Response: {new_text}")

    return generated_text


def run_comprehensive_tests(model, tokenizer):
    """Run a comprehensive set of tests"""
    print("\n" + "="*60)
    print("🧪 COMPREHENSIVE PURE 1-BIT MODEL TESTING")
    print("="*60)

    # Test prompts covering different capabilities
    test_prompts = [
        # Conversation (like Capybara training data)
        "Human: What is the capital of France?\nAssistant:",

        # Reasoning
        "Human: If I have 5 apples and eat 2, how many do I have left?\nAssistant:",

        # Creative writing
        "Human: Write a short story about a robot learning to paint.\nAssistant:",

        # Code generation
        "Human: Write a Python function to calculate factorial.\nAssistant:",

        # General knowledge
        "Human: Explain what machine learning is in simple terms.\nAssistant:",

        # Simple completion
        "The weather today is",

        # Question answering
        "Human: What are the benefits of renewable energy?\nAssistant:"
    ]

    print(f"\n🔍 Testing {len(test_prompts)} different prompts...")

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/{len(test_prompts)} ---")
        try:
            response = test_generation(model, tokenizer, prompt, max_length=150)

            # Basic quality checks
            if len(response.strip()) > len(prompt):
                print("✅ Generated new content")
            else:
                print("⚠️  No new content generated")

        except Exception as e:
            print(f"❌ Error in test {i}: {e}")
            continue

    print(f"\n" + "="*60)
    print("🎉 TESTING COMPLETED!")
    print("="*60)


def interactive_chat(model, tokenizer):
    """Interactive chat session with the model"""
    print("\n" + "="*60)
    print("💬 INTERACTIVE CHAT MODE")
    print("Type 'quit' to exit")
    print("="*60)

    conversation_history = ""

    while True:
        try:
            user_input = input("\nHuman: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            # Build prompt with conversation format
            prompt = f"{conversation_history}Human: {user_input}\nAssistant:"

            response = test_generation(model, tokenizer, prompt, max_length=200, temperature=0.8)

            # Extract just the assistant's response
            assistant_response = response[len(prompt):].strip()
            if assistant_response:
                print(f"Assistant: {assistant_response}")

                # Update conversation history (keep it reasonable length)
                conversation_history += f"Human: {user_input}\nAssistant: {assistant_response}\n"
                if len(conversation_history) > 1000:  # Truncate if too long
                    conversation_history = conversation_history[-800:]
            else:
                print("Assistant: [No response generated]")

        except KeyboardInterrupt:
            print("\n\n👋 Chat ended by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue


def main():
    # Default checkpoint path (adjust if needed)
    checkpoint_path = "./Falcon-E-Capybara-Pure1Bit/checkpoint-final"

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at: {checkpoint_path}")
        print("\n💡 Available checkpoints:")
        output_dir = "./Falcon-E-Capybara-Pure1Bit"
        if os.path.exists(output_dir):
            checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            for cp in sorted(checkpoints):
                print(f"   {output_dir}/{cp}")
            if checkpoints:
                checkpoint_path = f"{output_dir}/{checkpoints[-1]}"  # Use latest
                print(f"\n🔄 Using latest checkpoint: {checkpoint_path}")
        else:
            print("❌ No output directory found. Make sure training completed successfully.")
            return

    # Load model
    model, tokenizer = load_pure1bit_model(checkpoint_path)
    if model is None:
        return

    # Show menu
    print("\n" + "="*60)
    print("🎯 PURE 1-BIT MODEL TESTING MENU")
    print("="*60)
    print("1. Run comprehensive tests (automated)")
    print("2. Interactive chat mode")
    print("3. Custom single prompt test")
    print("4. Exit")

    while True:
        try:
            choice = input("\nSelect option (1-4): ").strip()

            if choice == "1":
                run_comprehensive_tests(model, tokenizer)

            elif choice == "2":
                interactive_chat(model, tokenizer)

            elif choice == "3":
                prompt = input("Enter your prompt: ")
                test_generation(model, tokenizer, prompt, max_length=200)

            elif choice == "4":
                print("👋 Goodbye!")
                break

            else:
                print("❌ Invalid choice. Please enter 1-4.")

        except KeyboardInterrupt:
            print("\n\n👋 Testing ended by user")
            break


if __name__ == "__main__":
    main()