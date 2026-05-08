import mlx.core as mx
from typing import Tuple

# GPU Compiled Segment (Matrix Generation)
@mx.compile
def generate_gram_matrix(activations: mx.array) -> mx.array:
    mean = mx.mean(activations, axis=0, keepdims=True)
    centered = activations - mean
    return mx.matmul(centered, centered.T)

def compute_spectral_signature(activations: mx.array) -> Tuple[mx.array, mx.array]:
    """
    Hybrid Execution: $A = U \Sigma V^T$.
    Gram generation on GPU -> Eigen Decomposition on CPU (Zero-copy).
    """
    # 1. GPU Execution
    gram = generate_gram_matrix(activations)
    mx.eval(gram) # Lock buffer before stream swap
    
    # 2. CPU Execution (Bypasses linalg::eigh GPU constraint)
    eigenvalues, _ = mx.linalg.eigh(gram, stream=mx.cpu)
    
    # 3. Filter and Derive Gaps (Can remain on CPU or return to GPU context implicitly)
    singular_values = mx.sqrt(mx.maximum(eigenvalues[::-1], 0.0))
    gaps = singular_values[:-1] - singular_values[1:]
    return singular_values, gaps

class SpectralTDAManifold:
    def __init__(self, variance_threshold: float = 0.95):
        self.variance_threshold = variance_threshold

    def extract_betti_proxies(self, activations: mx.array) -> dict:
        singular_values, gaps = compute_spectral_signature(activations)
        
        total_v = mx.sum(singular_values ** 2)
        exp_v = mx.cumsum(singular_values ** 2) / (total_v + 1e-9)
        
        b0 = mx.sum(exp_v < self.variance_threshold).item() + 1
        b1_t = mx.mean(gaps) + 2 * mx.std(gaps)
        b1 = mx.sum(gaps > b1_t).item()
        
        entropy = float(mx.sum(mx.log(singular_values + 1e-5)).item())
        return {"betti_0": int(b0), "betti_1": int(b1), "spectral_entropy": entropy}
