# src/manifolds/spectral_tda.py
import mlx.core as mx
from typing import Tuple

@mx.compile
def _generate_gram(activations: mx.array) -> mx.array:
    """GPU-accelerated Gram matrix generation."""
    mean = mx.mean(activations, axis=0, keepdims=True)
    centered = activations - mean
    return mx.matmul(centered, centered.T)

def compute_spectral_signature(activations: mx.array) -> Tuple[mx.array, mx.array]:
    """
    Hybrid Execution: $A = U \Sigma V^T$.
    Gram generation on GPU -> Eigen Decomposition on CPU (Zero-copy).
    """
    # 1. GPU Execution
    gram = _generate_gram(activations)
    mx.eval(gram) # Evaluate to flush GPU queue before CPU handoff
    
    # 2. CPU Execution (Bypasses linalg::eigh GPU constraint)
    eigenvalues, _ = mx.linalg.eigh(gram, stream=mx.cpu)
    
    # 3. Gap Derivation
    singular_values = mx.sqrt(mx.maximum(eigenvalues[::-1], 0.0))
    gaps = singular_values[:-1] - singular_values[1:]
    return singular_values, gaps

class SpectralTDAManifold:
    def __init__(self, variance_threshold: float = 0.95):
        self.v_thresh = variance_threshold

    def extract_betti(self, x: mx.array) -> dict:
        sv, gaps = compute_spectral_signature(x)
        mx.eval(sv, gaps) # Ensure sync
        
        tot = mx.sum(sv ** 2)
        exp = mx.cumsum(sv ** 2) / (tot + 1e-9)
        b0 = mx.sum(exp < self.v_thresh).item() + 1
        
        # Guard against zero-variance flatlines
        std_gaps = mx.std(gaps)
        threshold = mx.mean(gaps) + (2 * std_gaps) if std_gaps > 0 else 0.0
        b1 = mx.sum(gaps > threshold).item()
        
        return {"betti_0": int(b0), "betti_1": int(b1), "entropy": float(mx.sum(mx.log(sv + 1e-5)).item())}
