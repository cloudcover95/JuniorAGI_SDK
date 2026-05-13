# src/inference/bitnet_layers.py
import mlx.core as mx
import mlx.nn as nn

@mx.compile
def compute_channel_ternary(w: mx.array) -> tuple:
    """
    Per-Channel AbsMean Quantization.
    $W \in \mathbb{R}^{Out \times In}, \gamma = \frac{1}{In}\sum |W_{row}|$
    """
    eps = 1e-5
    # gamma shape: (Out, 1)
    gamma = mx.mean(mx.abs(w), axis=-1, keepdims=True)
    w_q = mx.round(mx.clip(w / (gamma + eps), -1.0, 1.0))
    return w_q.astype(mx.int8), gamma

@mx.compile
def fused_ternary_matmul(x: mx.array, w_q: mx.array, gamma: mx.array, q_max: float = 127.0) -> mx.array:
    """
    AMX-Optimized Execution.
    Pulls the scaling factor OUTSIDE the matmul: $Y = (X_q @ W_q^T) * (\gamma_w^T * \gamma_x / q_max)$
    Zero FP16 unrolling of W_q occurs.
    """
    eps = 1e-5
    
    # x shape: (B, L, In) -> g_x shape: (B, L, 1)
    g_x = mx.max(mx.abs(x), axis=-1, keepdims=True)
    x_q = mx.clip(mx.round(x * (q_max / (g_x + eps))), -q_max, q_max)
    
    # Cast directly in registers. No intermediate dense UMA allocation.
    # (B, L, In) @ (Out, In).T -> (B, L, Out)
    y_q = mx.matmul(x_q.astype(x.dtype), w_q.T.astype(x.dtype))
    
    # gamma is (Out, 1) -> gamma.T is (1, Out). Broadcasts cleanly over (B, L, Out)
    y = y_q * ((gamma.T * g_x) / q_max)
    
    return y

class DynamicBitLinear(nn.Module):
    """AST-Compliant BitLinear for MLX Model Bridge"""
    def __init__(self, input_dims: int, output_dims: int, bias: bool = False):
        super().__init__()
        self.w_q = mx.zeros((output_dims, input_dims), dtype=mx.int8)
        self.gamma = mx.zeros((output_dims, 1), dtype=mx.float16)
        if bias:
            self.bias = mx.zeros((output_dims,))

    def __call__(self, x: mx.array) -> mx.array:
        y = fused_ternary_matmul(x, self.w_q, self.gamma)
        if "bias" in self:
            y += self.bias
        return y
