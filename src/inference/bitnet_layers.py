# src/inference/bitnet_layers.py
import mlx.core as mx
import mlx.nn as nn

@mx.compile
def _compute_grouped_ternary(w: mx.array, group_size: int = 128) -> tuple:
    """
    Computes Grouped AbsMean Quantization.
    $W_g \in \mathbb{R}^{G}, \gamma_g = \frac{1}{G}\sum |W_g|$
    Returns packed int8 weights and the fp16 scaling factors.
    """
    eps = 1e-5
    orig_shape = w.shape
    
    # Pad to ensure divisibility by group_size
    pad_len = (group_size - (w.size % group_size)) % group_size
    if pad_len > 0:
        w_flat = mx.concatenate([w.flatten(), mx.zeros((pad_len,))])
    else:
        w_flat = w.flatten()
        
    w_groups = w_flat.reshape(-1, group_size)
    gamma = mx.mean(mx.abs(w_groups), axis=-1, keepdims=True)
    
    w_q = mx.round(mx.clip(w_groups / (gamma + eps), -1.0, 1.0))
    w_q_packed = w_q.astype(mx.int8)
    
    return w_q_packed, gamma

@mx.compile
def fused_bitnet_forward(x: mx.array, w_q: mx.array, gamma: mx.array, r_u: mx.array, r_v: mx.array, 
                         tau: float, pb: float, depth_ratio: float, orig_shape: tuple, group_size: int = 128) -> mx.array:
    """Depth-Aware execution using pre-packed int8 grouped weights."""
    eps = 1e-5
    
    # Dequantize weights on-the-fly inside the GPU cache
    w_unpacked = (w_q.astype(mx.float16) * gamma).flatten()[:orig_shape[0]*orig_shape[1]].reshape(orig_shape)
    
    # Depth-Aware Activation Q-Max
    eff_pb = mx.maximum(0.1, mx.array(pb) * (1.0 - (depth_ratio * 0.5)))
    q_max = mx.clip(mx.round((eff_pb * 120.0) + 7.0), 3.0, 127.0)
    
    g_x = mx.max(mx.abs(x), axis=-1, keepdims=True)
    x_q = mx.clip(mx.round(x * (q_max / (g_x + eps))), -q_max, q_max)
    
    # Base Compute
    y = mx.matmul(x_q, w_unpacked.T) * (g_x / q_max)
    
    # Residuals
    y_r = mx.matmul(mx.matmul(x, r_v.T), r_u.T)
    return y + mx.where(tau > 0.05, y_r * mx.maximum(0.0, 1.0 - tau), mx.zeros_like(y_r))

class DynamicBitLinear(nn.Module):
    def __init__(self, in_d: int, out_d: int, rank: int = 16):
        super().__init__()
        w_init = mx.random.normal((out_d, in_d)) * 0.02
        self.w_q, self.gamma = _compute_grouped_ternary(w_init)
        self.orig_shape = w_init.shape
        self.ru = mx.random.normal((out_d, rank)) * 0.01
        self.rv = mx.random.normal((rank, in_d)) * 0.01
        
    def __call__(self, x: mx.array, tau: float=0., pb: float=1., dr: float=0.) -> mx.array:
        return fused_bitnet_forward(x, self.w_q, self.gamma, self.ru, self.rv, tau, pb, dr, self.orig_shape)
