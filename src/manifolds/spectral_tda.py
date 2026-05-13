import mlx.core as mx
from typing import Tuple

@mx.compile
def compute_spectral_signature(activations: mx.array) -> Tuple[mx.array, mx.array]:
    mean = mx.mean(activations, axis=0, keepdims=True)
    centered = activations - mean
    gram = mx.matmul(centered, centered.T)
    eigenvalues, _ = mx.linalg.eigh(gram)
    singular_values = mx.sqrt(mx.maximum(eigenvalues[::-1], 0.0))
    gaps = singular_values[:-1] - singular_values[1:]
    return singular_values, gaps

class SpectralTDAManifold:
    def __init__(self, variance_threshold: float = 0.95):
        self.v_thresh = variance_threshold

    def extract_betti(self, x: mx.array) -> dict:
        sv, gaps = compute_spectral_signature(x)
        mx.eval(sv, gaps)
        tot = mx.sum(sv ** 2)
        exp = mx.cumsum(sv ** 2) / (tot + 1e-9)
        b0 = mx.sum(exp < self.v_thresh).item() + 1
        b1 = mx.sum(gaps > (mx.mean(gaps) + 2 * mx.std(gaps))).item()
        return {"betti_0": int(b0), "betti_1": int(b1), "entropy": float(mx.sum(mx.log(sv + 1e-5)).item())}
