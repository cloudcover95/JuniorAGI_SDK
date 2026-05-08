import mlx.core as mx
import mlx.nn as nn

@mx.compile
def fused_bitnet(x: mx.array, w: mx.array, r_u: mx.array, r_v: mx.array, tau: float) -> mx.array:
    # Ternary Weight Quantization: {-1, 0, 1}
    gamma_w = mx.mean(mx.abs(w))
    w_q = mx.round(mx.clip(w / (gamma_w + 1e-5), -1.0, 1.0))
    
    # 8-bit Activation Quantization
    gamma_x = mx.max(mx.abs(x))
    x_q = mx.clip(x * (127.0 / (gamma_x + 1e-5)), -128.0, 127.0)
    
    y_main = mx.matmul(x_q, w_q.T) * ((gamma_w * gamma_x) / 127.0)
    y_res = mx.matmul(mx.matmul(x, r_v.T), r_u.T)
    res_gate = mx.where(tau > 0.05, y_res * mx.maximum(0.0, 1.0 - tau), mx.zeros_like(y_res))
    
    return y_main + res_gate

class DynamicBitLinear(nn.Module):
    def __init__(self, in_d: int, out_d: int, rank: int = 16):
        super().__init__()
        self.weight = mx.random.normal((out_d, in_d)) * 0.02
        self.R_u = mx.random.normal((out_d, rank)) * 0.01
        self.R_v = mx.random.normal((rank, in_d)) * 0.01
        
    def __call__(self, x: mx.array, tau: float = 0.0) -> mx.array:
        return fused_bitnet(x, self.weight, self.R_u, self.R_v, tau)
