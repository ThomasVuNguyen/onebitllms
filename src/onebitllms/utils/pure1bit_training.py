# Copyright 2025 The Falcon-LLM Team. All rights reserved.
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

import torch
import torch.nn as nn
from typing import Dict, Any, Iterator
from ..layers.bitnet import Pure1BitLinear


class Pure1BitTrainingHelper:
    """
    Helper class to manage gradient flow for Pure1BitLinear layers during training.

    This class sets up backward hooks to accumulate gradients in the layer's
    gradient buffer instead of letting them flow to the int8 parameters directly.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.pure1bit_layers = []

        # Find all Pure1BitLinear layers and register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register backward hooks for all Pure1BitLinear layers"""
        for name, module in self.model.named_modules():
            if isinstance(module, Pure1BitLinear):
                self.pure1bit_layers.append((name, module))
                # Register hook on weight parameter to capture gradients
                if module.weight.requires_grad:
                    hook = module.weight.register_hook(
                        lambda grad, layer=module: self._capture_gradient(grad, layer)
                    )
                    self.hooks.append(hook)

    def _capture_gradient(self, grad: torch.Tensor, layer: Pure1BitLinear):
        """Capture gradient and accumulate in layer's buffer"""
        if grad is not None:
            # Convert gradient to float32 for accumulation
            grad_float = grad.float() if grad.dtype != torch.float32 else grad
            layer.accumulate_gradients(grad_float)

        # Return zero gradient to prevent PyTorch from trying to update int8 weights
        return torch.zeros_like(grad) if grad is not None else None

    def apply_gradient_updates(self, lr: float = 1e-3):
        """Apply accumulated gradients to all Pure1BitLinear layers"""
        for name, layer in self.pure1bit_layers:
            layer.apply_accumulated_gradients(lr=lr)

    def zero_grad(self):
        """Clear gradient buffers for all Pure1BitLinear layers"""
        for name, layer in self.pure1bit_layers:
            layer.grad_buffer.zero_()

    def get_gradient_norms(self) -> Dict[str, float]:
        """Get gradient norms for each Pure1BitLinear layer"""
        norms = {}
        for name, layer in self.pure1bit_layers:
            if layer.grad_buffer.numel() > 0:
                norm = layer.grad_buffer.norm().item()
                norms[name] = norm
        return norms

    def cleanup(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def create_pure1bit_optimizer(model: nn.Module, lr: float = 1e-3, **kwargs):
    """
    Create optimizer that properly handles Pure1BitLinear layers.

    Returns:
        optimizer: Standard optimizer for regular parameters
        training_helper: Helper for Pure1BitLinear layers
    """
    from ..optimizers import Pure1BitOptimizer

    # Separate parameters: pure1bit vs regular
    pure1bit_params = []
    regular_params = []

    for name, module in model.named_modules():
        if isinstance(module, Pure1BitLinear):
            # Add bias parameters if they exist (these remain float)
            if module.bias is not None:
                regular_params.append(module.bias)
            # Weight parameters are handled by the training helper
        else:
            for param in module.parameters(recurse=False):
                regular_params.append(param)

    # Create optimizer for regular parameters only
    if regular_params:
        optimizer = torch.optim.AdamW(regular_params, lr=lr, **kwargs)
    else:
        # Create a dummy optimizer if no regular parameters
        dummy_param = nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.AdamW([dummy_param], lr=lr, **kwargs)

    # Create training helper for Pure1BitLinear layers
    training_helper = Pure1BitTrainingHelper(model)

    return optimizer, training_helper


def pure1bit_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    training_helper: Pure1BitTrainingHelper,
    loss_fn,
    inputs,
    targets,
    lr: float = 1e-3
):
    """
    Perform one training step with proper Pure1BitLinear handling.

    Args:
        model: Model containing Pure1BitLinear layers
        optimizer: Standard optimizer for regular parameters
        training_helper: Helper for Pure1BitLinear layers
        loss_fn: Loss function
        inputs: Input data
        targets: Target data
        lr: Learning rate for Pure1BitLinear layers

    Returns:
        loss: Training loss
    """
    # Zero gradients
    optimizer.zero_grad()
    training_helper.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)

    # Backward pass
    loss.backward()

    # Update regular parameters
    optimizer.step()

    # Update Pure1BitLinear parameters
    training_helper.apply_gradient_updates(lr=lr)

    return loss