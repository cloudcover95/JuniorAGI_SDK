import mlx.core as mx
import mlx.nn as nn
import math
from inference.bitnet_layers import DynamicBitLinear

class BitNetTransformerBlock(nn.Module):
    def __init__(self, d: int, h: int, mlp: int):
        super().__init__()
        self.h, self.hd, self.s = h, d//h, 1.0/math.sqrt(d//h)
        self.q, self.k, self.v, self.o = [DynamicBitLinear(d, d) for _ in range(4)]
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.g, self.u, self.dn = DynamicBitLinear(d, mlp), DynamicBitLinear(d, mlp), DynamicBitLinear(mlp, d)

    def __call__(self, x: mx.array, tau: float, pb: float, dr: float) -> mx.array:
        B, L, D = x.shape
        nx = self.n1(x)
        q = self.q(nx, tau, pb, dr).reshape(B, L, self.h, self.hd).transpose(0, 2, 1, 3)
        k = self.k(nx, tau, pb, dr).reshape(B, L, self.h, self.hd).transpose(0, 2, 1, 3)
        v = self.v(nx, tau, pb, dr).reshape(B, L, self.h, self.hd).transpose(0, 2, 1, 3)
        
        attn = mx.matmul(mx.softmax(mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.s, axis=-1), v)
        h_x = x + self.o(attn.transpose(0, 2, 1, 3).reshape(B, L, D), tau, pb, dr)
        
        nx2 = self.n2(h_x)
        return h_x + self.dn(nn.silu(self.g(nx2, tau, pb, dr)) * self.u(nx2, tau, pb, dr), tau, pb, dr)
