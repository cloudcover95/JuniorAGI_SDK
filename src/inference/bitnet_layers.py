# src/inference/bitnet_layers.py
import mlx.core as mx
import mlx.nn as nn

@mx.compile
def compute_grouped_ternary(w: mx.array, group_size: int = 128) -> tuple:
    """Grouped AbsMean Quantization for Microsoft b1.58 parity."""
    eps = 1e-5
    orig_shape = w.shape
    pad_len = (group_size - (w.size % group_size)) % group_size
    w_flat = mx.concatenate([w.flatten(), mx.zeros((pad_len,))]) if pad_len > 0 else w.flatten()
    
    w_groups = w_flat.reshape(-1, group_size)
    gamma = mx.mean(mx.abs(w_groups), axis=-1, keepdims=True)
    w_q = mx.round(mx.clip(w_groups / (gamma + eps), -1.0, 1.0))
    return w_q.astype(mx.int8), gamma

@mx.compile
def fused_ternary_matmul(x: mx.array, w_q: mx.array, gamma: mx.array, orig_shape: tuple, q_max: float = 127.0) -> mx.array:
    """
    AMX-Optimized execution boundary. 
    Maintains int8 in UMA; unpacks to float16 inside the execution register.
    """
    eps = 1e-5
    # Per-Token Activation Quantization
    g_x = mx.max(mx.abs(x), axis=-1, keepdims=True)
    x_q = mx.clip(mx.round(x * (q_max / (g_x + eps))), -q_max, q_max)
    
    # Fused Dequantize & Matmul
    w_exec = (w_q.astype(x.dtype) * gamma).flatten()[:orig_shape[0]*orig_shape[1]].reshape(orig_shape)
    y = mx.matmul(x_q.astype(x.dtype), w_exec.T) * (g_x / q_max)
    
    return y

class DynamicBitLinear(nn.Module):
    """Drop-in replacement for mlx.nn.Linear"""
    def __init__(self, input_dims: int, output_dims: int, bias: bool = False):
        super().__init__()
        self.orig_shape = (output_dims, input_dims)
        self.w_q = mx.zeros((1,), dtype=mx.int8)
        self.gamma = mx.zeros((1,), dtype=mx.float16)
        self.bias = mx.zeros((output_dims,)) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        y = fused_ternary_matmul(x, self.w_q, self.gamma, self.orig_shape)
        if self.bias is not None:
            y += self.bias
        return y
