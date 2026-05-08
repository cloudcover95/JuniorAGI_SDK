# src/inference/bitnet_layers.py
import mlx.core as mx
import mlx.nn as nn

@mx.compile
def fused_bitnet(x: mx.array, w: mx.array, r_u: mx.array, r_v: mx.array, tau: float, power_budget: float) -> mx.array:
    """
    C2V-Proactive Fused BitNet Layer.
    Quantization bounds adapt continuously based on runtime power budget.
    """
    epsilon = 1e-5
    
    # Ternary Weight Quantization: {-1, 0, 1}
    gamma_w = mx.mean(mx.abs(w))
    w_q = mx.round(mx.clip(w / (gamma_w + epsilon), -1.0, 1.0))
    
    # Continuous Bit-Width interpolation mapping PB [0,1] -> q_max [3, 127]
    pb_array = mx.array(power_budget)
    # If PB high, near 8-bit (127). If low, near 3-bit (7).
    q_max = mx.clip(mx.round((pb_array * 120.0) + 7.0), 3.0, 127.0)
    
    # Per-Token Activation Quantization
    gamma_x = mx.max(mx.abs(x), axis=-1, keepdims=True)
    scale = q_max / (gamma_x + epsilon)
    x_q = mx.clip(mx.round(x * scale), -q_max, q_max)
    
    # Core Compute
    y_main = mx.matmul(x_q, w_q.T) * (gamma_w * gamma_x / q_max)
    
    # Residual projection
    y_res = mx.matmul(mx.matmul(x, r_v.T), r_u.T)
    res_gate = mx.where(tau > 0.05, y_res * mx.maximum(0.0, 1.0 - tau), mx.zeros_like(y_res))
    
    return y_main + res_gate

class DynamicBitLinear(nn.Module):
    def __init__(self, in_d: int, out_d: int, rank: int = 16):
        super().__init__()
        self.weight = mx.random.normal((out_d, in_d)) * 0.02
        self.R_u = mx.random.normal((out_d, rank)) * 0.01
        self.R_v = mx.random.normal((rank, in_d)) * 0.01
        
    def __call__(self, x: mx.array, tau: float = 0.0, power_budget: float = 1.0) -> mx.array:
        return fused_bitnet(x, self.weight, self.R_u, self.R_v, tau, power_budget)
