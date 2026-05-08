# src/inference/bitnet_layers.py
import mlx.core as mx
import mlx.nn as nn

@mx.compile
def _fused_bitnet_forward(x: mx.array, weight: mx.array, r_u: mx.array, r_v: mx.array, tau: float) -> mx.array:
    """
    Fused Ternary MatMul + Gated Residual.
    $y = \text{Ternary}(x, W) + \sigma(\tau) \cdot \text{Residual}(x, R_u, R_v)$
    """
    epsilon = 1e-5
    
    # Ternary Weight Quantization: {-1, 0, 1}
    gamma_w = mx.mean(mx.abs(weight))
    w_q = mx.round(mx.clip(weight / (gamma_w + epsilon), -1.0, 1.0))
    
    # 8-bit Activation Quantization
    gamma_x = mx.max(mx.abs(x))
    x_q = mx.clip(x * (127.0 / (gamma_x + epsilon)), -128.0, 127.0)
    
    # Base Compute
    y_main = mx.matmul(x_q, w_q.T) * ((gamma_w * gamma_x) / 127.0)
    
    # Dynamic Residual Path
    y_res = mx.matmul(mx.matmul(x, r_v.T), r_u.T)
    y_res_gated = mx.where(tau > 0.05, y_res * mx.maximum(0.0, 1.0 - tau), mx.zeros_like(y_res))
    
    return y_main + y_res_gated

class DynamicBitLinear(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, max_rank: int = 16):
        super().__init__()
        self.weight = mx.random.normal((out_dims, in_dims)) * 0.02
        self.R_u = mx.random.normal((out_dims, max_rank)) * 0.01
        self.R_v = mx.random.normal((max_rank, in_dims)) * 0.01

    def __call__(self, x: mx.array, tau: float = 0.0) -> mx.array:
        return _fused_bitnet_forward(x, self.weight, self.R_u, self.R_v, tau)
