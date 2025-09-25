# Pure 1-Bit Finetuning Research Notes

## Current Problem
The current `onebitllms` implementation uses a hybrid approach for finetuning:
- **Storage**: bf16 weights in `nn.Parameter`
- **Computation**: 1.58-bit quantized values during forward pass
- **Gradients**: Flow to full-precision bf16 parameters

## Goal
Implement pure 1-bit finetuning where:
- Weights are stored as 1-bit values {-1, 0, 1}
- No quantization/dequantization overhead during training
- Eliminate need for bf16 shadow parameters

## Technical Barriers

### 1. Gradient Computation Challenge
- **Problem**: 1-bit weights {-1, 0, 1} have undefined/zero gradients everywhere except transition points
- **Current Solution**: Straight-through estimation: `w + (quant(w) - w).detach()` allows gradients to flow to full-precision weights
- **Pure 1-bit Challenge**: No continuous parameter to receive gradients

### 2. Optimizer State Management  
- **Problem**: Optimizers (Adam, SGD) expect continuous values for:
  - Momentum tracking
  - Variance estimates  
  - Running averages
- **Current**: Optimizer operates on bf16 parameters
- **Pure 1-bit Challenge**: Need auxiliary storage for optimizer states

### 3. Numerical Precision Issues
- **Problem**: Weight updates are typically small (1e-4 to 1e-6 range)
- **Current**: Full precision accumulation of small updates
- **Pure 1-bit Challenge**: How to accumulate tiny updates on discrete values?

## Potential Solutions

### Approach 1: Shadow Parameter Architecture
```python
class Pure1BitLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Store actual 1-bit weights (non-learnable)
        self.weight_bits = nn.Parameter(torch.randint(-1, 2, (out_features, in_features)), 
                                       requires_grad=False)
        # Separate full-precision shadow weights for optimizer
        self.weight_shadow = nn.Parameter(torch.randn(out_features, in_features))
        
    def forward(self, x):
        # Quantize shadow weights to 1-bit for computation
        w_quant = torch.sign(self.weight_shadow).clamp(-1, 1)
        # Optionally sync stored bits
        with torch.no_grad():
            self.weight_bits.copy_(w_quant)
        return F.linear(x, w_quant)
```

**Pros**: 
- Maintains gradient flow through shadow parameters
- Optimizer works with continuous values

**Cons**: 
- Still requires bf16 storage (defeats the purpose)
- Memory overhead remains

### Approach 2: Custom Optimizer Integration
```python
class BitNetOptimizer:
    def __init__(self, model_params, lr=1e-3):
        self.lr = lr
        # Maintain separate full-precision states for 1-bit parameters
        self.shadow_params = {}
        self.momentum = {}
        
    def step(self, model):
        for name, param in model.named_parameters():
            if param.dtype == torch.int8:  # 1-bit stored as int8
                # Update shadow parameter with gradients
                shadow = self.shadow_params[name]
                shadow += self.lr * param.grad
                # Quantize shadow to update actual parameter
                param.data = torch.sign(shadow).clamp(-1, 1).to(torch.int8)
```

**Pros**:
- True 1-bit parameter storage
- Custom gradient handling

**Cons**:
- Complex integration with existing training frameworks
- Still needs auxiliary precision storage

### Approach 3: Gradient Accumulation Buffer
```python
class BufferedBitNetLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 1-bit weights
        self.weight = nn.Parameter(torch.randint(-1, 2, (out_features, in_features)).float())
        # Gradient accumulation buffer
        self.register_buffer('grad_buffer', torch.zeros_like(self.weight))
        
    def custom_backward_hook(self, grad):
        # Accumulate gradients in buffer
        self.grad_buffer += grad
        # Apply updates when buffer exceeds threshold
        mask = self.grad_buffer.abs() > self.threshold
        self.weight.data[mask] = torch.sign(self.grad_buffer[mask])
        self.grad_buffer[mask] = 0
        return grad
```

## Research Questions

1. **Gradient Flow**: Can we use techniques from binary neural networks (BNN) literature?
2. **Optimizer Compatibility**: Which optimizers work best with discrete parameters?
3. **Convergence**: Does pure 1-bit training converge as well as hybrid approaches?
4. **Memory Benefits**: What's the actual memory savings vs complexity trade-off?

## Next Steps

1. Implement Approach 1 as proof-of-concept
2. Compare memory usage: pure 1-bit vs current hybrid
3. Evaluate training convergence and final model quality
4. Benchmark training speed improvements
5. Test compatibility with existing frameworks (TRL, transformers)

## References
- BitNet paper: Straight-through estimation for gradient flow
- Binary Neural Networks: Discrete parameter optimization techniques
- Current implementation: `src/onebitllms/layers/bitnet.py`