import mlx.core as mx
import mlx.nn as nn
import math
from inference.bitnet_layers import DynamicBitLinear

class BitNetAttention(nn.Module):
    """
    Ternary Multi-Head Attention.
    Projections (Q,K,V) utilize BitNet b1.58 weights to eliminate FP16 MatMul bloat.
    $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d}})V$
    """
    def __init__(self, dims: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Ternary Projections
        self.q_proj = DynamicBitLinear(dims, dims)
        self.k_proj = DynamicBitLinear(dims, dims)
        self.v_proj = DynamicBitLinear(dims, dims)
        self.o_proj = DynamicBitLinear(dims, dims)

    def __call__(self, x: mx.array, mask: mx.array = None, tau: float = 0.0) -> mx.array:
        B, L, D = x.shape
        
        q = self.q_proj(x, tau).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x, tau).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x, tau).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask
            
        attn = mx.softmax(scores, axis=-1)
        out = mx.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        
        return self.o_proj(out, tau)

class BitNetTransformerBlock(nn.Module):
    """
    Complete 1.58-bit Transformer Block with RMSNorm.
    """
    def __init__(self, dims: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.attention = BitNetAttention(dims, num_heads)
        self.norm1 = nn.RMSNorm(dims)
        self.norm2 = nn.RMSNorm(dims)
        
        # SwiGLU MLP via Ternary
        self.gate_proj = DynamicBitLinear(dims, mlp_dim)
        self.up_proj = DynamicBitLinear(dims, mlp_dim)
        self.down_proj = DynamicBitLinear(mlp_dim, dims)

    def __call__(self, x: mx.array, mask: mx.array = None, tau: float = 0.0) -> mx.array:
        # Attention + Residual
        r = self.attention(self.norm1(x), mask, tau)
        h = x + r
        
        # MLP + Residual
        gate = nn.silu(self.gate_proj(self.norm2(h), tau))
        up = self.up_proj(self.norm2(h), tau)
        out = self.down_proj(gate * up, tau)
        
        return h + out
