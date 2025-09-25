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
import torch.nn.functional as F

from onebitllms import activation_quant_triton, weight_quant_triton

class Pure1BitLinear(nn.Module):
    """
    Pure 1-bit Linear layer that stores weights as discrete {-1, 0, 1} values
    without any full-precision shadow parameters during training.

    This eliminates the hybrid approach and uses true 1-bit weights throughout.
    """
    def __init__(self, in_features, out_features, bias: bool = False) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Store weights as int8 to represent {-1, 0, 1}
        # Initialize randomly to {-1, 0, 1}
        initial_weights = torch.randint(-1, 2, (out_features, in_features), dtype=torch.int8)
        self.weight = nn.Parameter(initial_weights, requires_grad=False)

        # Register gradient accumulation buffer (not a parameter)
        self.register_buffer('grad_buffer', torch.zeros(out_features, in_features, dtype=torch.float32))
        self.register_buffer('update_threshold', torch.tensor(0.1))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor):
        # Convert 1-bit weights to float for computation
        w_float = self.weight.float()

        # Apply activation quantization (simplified version without triton for testing)
        if x.device.type == 'cuda':
            try:
                x_quant = x + (activation_quant_triton(x) - x).detach()
            except Exception:
                # Fallback to simple quantization if triton is not available
                x_quant = self._simple_activation_quant(x)
        else:
            # CPU fallback
            x_quant = self._simple_activation_quant(x)

        # Use the 1-bit weights directly (no quantization needed)
        y = F.linear(x_quant, w_float, bias=self.bias)
        return y

    def _simple_activation_quant(self, x: torch.Tensor):
        """Simple activation quantization without triton"""
        # Quantize to int8 range [-128, 127] per token (row-wise)
        if x.dim() == 3:  # [batch, seq_len, hidden_dim]
            # Get row-wise absolute maximum
            x_max = x.abs().max(dim=-1, keepdim=True)[0]
            x_max = torch.clamp(x_max, min=1e-5)

            # Scale to [-127, 127] range
            scale = 127.0 / x_max
            x_scaled = x * scale
            x_quant_int = torch.clamp(torch.round(x_scaled), -128, 127)
            x_quant = x_quant_int / scale

            return x_quant
        else:
            # For other dimensions, just return original
            return x

    def apply_accumulated_gradients(self, lr=1e-3):
        """Apply accumulated gradients to update 1-bit weights"""
        with torch.no_grad():
            # Scale gradients by learning rate
            scaled_grads = self.grad_buffer * lr

            # Find weights to update based on accumulated gradient magnitude
            update_mask = scaled_grads.abs() > self.update_threshold

            if update_mask.any():
                # Update weights based on gradient sign
                grad_sign = torch.sign(scaled_grads)

                # Apply updates: move in direction opposite to gradient
                self.weight.data[update_mask] = torch.clamp(
                    self.weight.data[update_mask].float() - grad_sign[update_mask],
                    -1, 1
                ).to(torch.int8)

                # Reset gradient buffer for updated weights
                self.grad_buffer[update_mask] = 0.0

    def accumulate_gradients(self, grad):
        """Accumulate gradients in buffer for later application"""
        if grad is not None:
            self.grad_buffer += grad

    def __repr__(self):
        return 'Pure1BitLinear(in_features={0}, out_features={1})'.format(
            self.in_features, self.out_features)


class BitNetLinear(nn.Module):
    """
    Implementation of the BitNet Linear layer. The BitNet linear layer consists
    of having one linear layer with clamped weights during training and additional
    non learnable layer normalization layers

    Attributes:
        in_features (`int`):
            Number of input features of the linear layer
        out_features (`int`):
            Number of output features of the lienar layer
    """
    def __init__(self, in_features, out_features, bias: bool = False) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)


    def forward(self, x: torch.Tensor):
        w = self.weight

        with torch.cuda.device(w.device):
            x_quant = x + (activation_quant_triton(x) - x).detach()
            w_quant = w + (weight_quant_triton(w) - w).detach()

        y = F.linear(x_quant, w_quant, bias=self.bias)
        return y

    def __repr__(self):
        return 'BitnetLinear(in_features={0}, out_features={1})'.format(self.in_features, self.out_features)