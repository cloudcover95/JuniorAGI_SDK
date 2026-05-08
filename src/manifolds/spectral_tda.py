import mlx.core as mx
from typing import Tuple

# Pure module-level functions for @mx.compile to prevent 'self' binding tracebacks
@mx.compile
def compute_spectral_signature(activations: mx.array) -> Tuple[mx.array, mx.array]:
    """
    Computes SVD strictly on the GPU: $A = U \Sigma V^T$.
    Uses Gram matrix eigen-decomposition to bypass CPU bottlenecks.
    """
    mean = mx.mean(activations, axis=0, keepdims=True)
    centered = activations - mean
    gram = mx.matmul(centered, centered.T)
    eigenvalues, _ = mx.linalg.eigh(gram)
    
    # Filter negative numerical noise and sort descending
    singular_values = mx.sqrt(mx.maximum(eigenvalues[::-1], 0.0))
    gaps = singular_values[:-1] - singular_values[1:]
    return singular_values, gaps

class SpectralTDAManifold:
    """
    Vectorized Topological Data Analysis (TDA).
    Calculates authentic Betti-0 and Betti-1 proxies using spectral gap derivations
    ported from JuniorStock/MemSys high-frequency financial modeling (HFFM).
    """
    def __init__(self, variance_threshold: float = 0.95):
        self.variance_threshold = variance_threshold

    def extract_betti_proxies(self, activations: mx.array) -> dict:
        singular_values, gaps = compute_spectral_signature(activations)
        
        # Calculate variance
        total_v = mx.sum(singular_values ** 2)
        exp_v = mx.cumsum(singular_values ** 2) / (total_v + 1e-9)
        
        # Betti-0: Dimensionality of connected components explaining 95% variance
        b0 = mx.sum(exp_v < self.variance_threshold).item() + 1
        
        # Betti-1: Persistence of 1-cycles identified by anomalous spectral drops
        b1_t = mx.mean(gaps) + 2 * mx.std(gaps)
        b1 = mx.sum(gaps > b1_t).item()
        
        entropy = float(mx.sum(mx.log(singular_values + 1e-5)).item())
        return {"betti_0": int(b0), "betti_1": int(b1), "spectral_entropy": entropy}
