# src/inference/bitnet_layers.py
import mlx.core as mx
import mlx.nn as nn

@mx.compile
def _compute_grouped_ternary(w: mx.array, group_size: int = 128) -> tuple:
    eps = 1e-5
    orig_shape = w.shape
    pad_len = (group_size - (w.size % group_size)) % group_size
    w_flat = mx.concatenate([w.flatten(), mx.zeros((pad_len,))]) if pad_len > 0 else w.flatten()
    
    w_groups = w_flat.reshape(-1, group_size)
    gamma = mx.mean(mx.abs(w_groups), axis=-1, keepdims=True)
    
    w_q = mx.round(mx.clip(w_groups / (gamma + eps), -1.0, 1.0))
    return w_q.astype(mx.int8), gamma

@mx.compile
def fused_bitnet_forward(x: mx.array, w_q: mx.array, gamma: mx.array, r_u: mx.array, r_v: mx.array, 
                         tau: float, pb: float, depth_ratio: float, orig_shape: tuple) -> mx.array:
    """
    AMX-Optimized Ternary Graph.
    Weights remain int8 in UMA. They are cast to float16 and scaled instantly 
    at the execution boundary, preventing FP16 VRAM explosions.
    """
    eps = 1e-5
    eff_pb = mx.maximum(0.1, mx.array(pb) * (1.0 - (depth_ratio * 0.5)))
    q_max = mx.clip(mx.round((eff_pb * 120.0) + 7.0), 3.0, 127.0)
    
    g_x = mx.max(mx.abs(x), axis=-1, keepdims=True)
    x_q = mx.clip(mx.round(x * (q_max / (g_x + eps))), -q_max, q_max)
    
    # Unpack and scale in a single fused operation
    # MLX fuses the astype, multiply, and reshape into a single kernel pass
    w_exec = (w_q.astype(x.dtype) * gamma).flatten()[:orig_shape[0]*orig_shape[1]].reshape(orig_shape)
    
    # Core Matrix Multiplication
    y = mx.matmul(x_q.astype(x.dtype), w_exec.T) * (g_x / q_max)
    
    if tau > 0.05:
        y_r = mx.matmul(mx.matmul(x, r_v.T), r_u.T)
        y += y_r * mx.maximum(0.0, 1.0 - tau)
        
    return y

class DynamicBitLinear(nn.Module):
    def __init__(self, in_d: int, out_d: int, rank: int = 16):
        super().__init__()
        # Placeholders. These are overwritten by the SovereignLoader.
        self.w_q = mx.zeros((1,), dtype=mx.int8)
        self.gamma = mx.zeros((1,), dtype=mx.float16)
        self.orig_shape = (out_d, in_d)
        self.ru = mx.random.normal((out_d, rank)) * 0.01
        self.rv = mx.random.normal((rank, in_d)) * 0.01
        self._is_loaded = False

    def initialize_random(self):
        """Standard init for untrained local instances."""
        w_init = mx.random.normal(self.orig_shape) * 0.02
        self.w_q, self.gamma = _compute_grouped_ternary(w_init)
        self._is_loaded = True

    def load_ternary_state(self, w_q: mx.array, gamma: mx.array):
        """Direct injection from the SovereignLoader."""
        self.w_q = w_q
        self.gamma = gamma
        self._is_loaded = True

    def __call__(self, x: mx.array, tau: float=0., pb: float=1., dr: float=0.) -> mx.array:
        if not self._is_loaded: self.initialize_random()
        return fused_bitnet_forward(x, self.w_q, self.gamma, self.ru, self.rv, tau, pb, dr, self.orig_shape)
