# src/inference/bitnet_layers.py
import mlx.core as mx
import mlx.nn as nn

@mx.compile
def fused_bitnet(x: mx.array, w: mx.array, r_u: mx.array, r_v: mx.array, tau: float, power_budget: float, depth_ratio: float) -> mx.array:
    """
    Depth-Aware Fused BitNet Layer.
    Early layers (depth_ratio ~ 0.0) maintain higher fidelity (near 127).
    Deep layers (depth_ratio ~ 1.0) compress aggressively to save UMA bandwidth.
    """
    epsilon = 1e-5
    
    # Ternary Weight Quantization
    gamma_w = mx.mean(mx.abs(w))
    w_q = mx.round(mx.clip(w / (gamma_w + epsilon), -1.0, 1.0))
    
    # Depth-Aware Activation Clipping
    # Effective power drops as we go deeper into the network
    effective_pb = mx.maximum(0.1, mx.array(power_budget) * (1.0 - (depth_ratio * 0.6)))
    q_max = mx.clip(mx.round((effective_pb * 120.0) + 7.0), 3.0, 127.0)
    
    # Per-Token Activation Quantization
    gamma_x = mx.max(mx.abs(x), axis=-1, keepdims=True)
    scale = q_max / (gamma_x + epsilon)
    x_q = mx.clip(mx.round(x * scale), -q_max, q_max)
    
    # MatMul
    y_main = mx.matmul(x_q, w_q.T) * (gamma_w * gamma_x / q_max)
    
    # Gated Residual
    y_res = mx.matmul(mx.matmul(x, r_v.T), r_u.T)
    res_gate = mx.where(tau > 0.05, y_res * mx.maximum(0.0, 1.0 - tau), mx.zeros_like(y_res))
    
    return y_main + res_gate

class DynamicBitLinear(nn.Module):
    def __init__(self, in_d: int, out_d: int, rank: int = 16):
        super().__init__()
        self.weight = mx.random.normal((out_d, in_d)) * 0.02
        self.R_u = mx.random.normal((out_d, rank)) * 0.01
        self.R_v = mx.random.normal((rank, in_d)) * 0.01
        
    def __call__(self, x: mx.array, tau: float = 0.0, power_budget: float = 1.0, depth_ratio: float = 0.0) -> mx.array:
        return fused_bitnet(x, self.weight, self.R_u, self.R_v, tau, power_budget, depth_ratio)
