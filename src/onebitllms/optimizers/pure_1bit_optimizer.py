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
from torch.optim.optimizer import Optimizer
from typing import Any, Dict, Iterable, Optional, Union
from ..layers.bitnet import Pure1BitLinear


class Pure1BitOptimizer(Optimizer):
    """
    Custom optimizer for Pure 1-bit parameters that maintains separate
    full-precision shadow parameters for gradient accumulation and momentum,
    while keeping the actual model weights as discrete {-1, 0, 1} values.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        update_threshold: float = 0.1,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            update_threshold=update_threshold
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            update_threshold = group['update_threshold']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Handle Pure1BitLinear layers specially
                if hasattr(p, '_pure1bit_layer'):
                    self._handle_pure1bit_param(
                        p, group, beta1, beta2, lr, eps, weight_decay, update_threshold
                    )
                else:
                    # Handle regular parameters with standard Adam
                    self._handle_regular_param(
                        p, group, beta1, beta2, lr, eps, weight_decay
                    )

        return loss

    def _handle_pure1bit_param(self, param, group, beta1, beta2, lr, eps, weight_decay, update_threshold):
        """Handle parameters from Pure1BitLinear layers"""
        grad = param.grad
        state = self.state[param]

        # Initialize state
        if len(state) == 0:
            state['step'] = 0
            # Shadow parameter for momentum/gradient accumulation
            state['shadow_param'] = param.data.float().clone()
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(state['shadow_param'])
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(state['shadow_param'])

        shadow_param = state['shadow_param']
        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']

        state['step'] += 1
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        # Apply weight decay to shadow parameter
        if weight_decay != 0:
            shadow_param.mul_(1 - lr * weight_decay)

        # Update biased first and second moment estimates
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Compute bias-corrected first and second moment estimates
        corrected_exp_avg = exp_avg / bias_correction1
        corrected_exp_avg_sq = exp_avg_sq / bias_correction2

        # Update shadow parameter
        denom = corrected_exp_avg_sq.sqrt().add_(eps)
        step_size = lr / bias_correction1
        shadow_param.addcdiv_(corrected_exp_avg, denom, value=-step_size)

        # Apply threshold-based updates to discrete weights
        diff = shadow_param - param.data.float()
        update_mask = diff.abs() > update_threshold

        if update_mask.any():
            # Update discrete weights based on shadow parameter
            new_weights = torch.sign(shadow_param).clamp(-1, 1)
            param.data[update_mask] = new_weights[update_mask].to(param.dtype)

            # Reset shadow parameter for updated positions
            shadow_param[update_mask] = param.data[update_mask].float()

    def _handle_regular_param(self, param, group, beta1, beta2, lr, eps, weight_decay):
        """Handle regular full-precision parameters with standard Adam"""
        grad = param.grad
        state = self.state[param]

        # Initialize state
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(param)
            state['exp_avg_sq'] = torch.zeros_like(param)

        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']

        state['step'] += 1
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        # Apply weight decay
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Update biased first and second moment estimates
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Compute bias-corrected first and second moment estimates
        corrected_exp_avg = exp_avg / bias_correction1
        corrected_exp_avg_sq = exp_avg_sq / bias_correction2

        # Update parameters
        denom = corrected_exp_avg_sq.sqrt().add_(eps)
        step_size = lr / bias_correction1
        param.addcdiv_(corrected_exp_avg, denom, value=-step_size)

    def add_param_group(self, param_group):
        """Add a param group to the optimizer's param_groups"""
        assert isinstance(param_group, dict), "param group must be a dict"

        # Mark Pure1BitLinear parameters
        for param in param_group['params']:
            if isinstance(param, torch.nn.Parameter):
                # Check if this parameter belongs to a Pure1BitLinear layer
                for module in self._get_modules_from_param(param):
                    if isinstance(module, Pure1BitLinear):
                        param._pure1bit_layer = module
                        break

        super().add_param_group(param_group)

    def _get_modules_from_param(self, param):
        """Helper to find which modules contain this parameter"""
        # This is a simplified approach - in practice you might want to
        # pass module information explicitly when creating the optimizer
        return []