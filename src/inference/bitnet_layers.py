import mlx.core as mx
import mlx.nn as nn

@mx.compile
def fused_bitnet(x: mx.array, w: mx.array, r_u: mx.array, r_v: mx.array, tau: float, pb: float, depth_ratio: float) -> mx.array:
    eps = 1e-5
    g_w = mx.mean(mx.abs(w))
    w_q = mx.round(mx.clip(w / (g_w + eps), -1.0, 1.0))
    
    # Depth-Aware Q-Max
    eff_pb = mx.maximum(0.1, mx.array(pb) * (1.0 - (depth_ratio * 0.5)))
    q_max = mx.clip(mx.round((eff_pb * 120.0) + 7.0), 3.0, 127.0)
    
    g_x = mx.max(mx.abs(x), axis=-1, keepdims=True)
    x_q = mx.clip(mx.round(x * (q_max / (g_x + eps))), -q_max, q_max)
    
    y = mx.matmul(x_q, w_q.T) * (g_w * g_x / q_max)
    y_r = mx.matmul(mx.matmul(x, r_v.T), r_u.T)
    return y + mx.where(tau > 0.05, y_r * mx.maximum(0.0, 1.0 - tau), mx.zeros_like(y_r))

class DynamicBitLinear(nn.Module):
    def __init__(self, in_d: int, out_d: int, rank: int = 16):
        super().__init__()
        self.w = mx.random.normal((out_d, in_d)) * 0.02
        self.ru = mx.random.normal((out_d, rank)) * 0.01
        self.rv = mx.random.normal((rank, in_d)) * 0.01
    def __call__(self, x: mx.array, tau: float=0., pb: float=1., dr: float=0.) -> mx.array:
        return fused_bitnet(x, self.w, self.ru, self.rv, tau, pb, dr)
