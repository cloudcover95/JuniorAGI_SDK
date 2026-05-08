# src/manifolds/spectral_tda.py
import mlx.core as mx
import logging
from typing import Tuple

logger = logging.getLogger("JuniorAGI.TDA")

class SpectralTDAManifold:
    """
    Vectorized Topological Data Analysis (TDA).
    Approximates Betti-0 and Betti-1 persistence via Spectral Gap Analysis.
    Operates strictly in UMA to avoid CPU serialization bottlenecks.
    """
    def __init__(self, variance_threshold: float = 0.95):
        self.variance_threshold = variance_threshold

    @mx.compile
    def _compute_spectral_signature(self, activation_matrix: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Computes SVD: $A = U \Sigma V^T$.
        Returns Singular Values ($\Sigma$) and the spectral gap derivatives.
        """
        # Center the manifold
        mean = mx.mean(activation_matrix, axis=0, keepdims=True)
        centered = activation_matrix - mean
        
        # Calculate covariance proxy (Gram matrix) for efficiency if features > batch
        gram = mx.matmul(centered, centered.T)
        
        # Eigen decomposition of Gram matrix -> Singular values of centered matrix
        eigenvalues, _ = mx.linalg.eigh(gram)
        # Sort descending and filter negative numerical noise
        singular_values = mx.sqrt(mx.maximum(eigenvalues[::-1], 0.0))
        
        # Spectral gaps (derivative of singular values)
        gaps = singular_values[:-1] - singular_values[1:]
        return singular_values, gaps

    def extract_betti_proxies(self, activation_matrix: mx.array) -> dict:
        """
        Derives topological features from the spectral signature.
        Betti-0 proxy: Number of principal components explaining variance threshold.
        Betti-1 proxy: Irregularities in the spectral gap (holes in the manifold).
        """
        mx.eval(activation_matrix)
        singular_values, gaps = self._compute_spectral_signature(activation_matrix)
        
        # Compute cumulative explained variance
        total_variance = mx.sum(singular_values ** 2)
        explained_variance = mx.cumsum(singular_values ** 2) / (total_variance + 1e-9)
        
        # Betti-0 proxy (Dimensionality of connected components)
        betti_0 = mx.sum(explained_variance < self.variance_threshold).item() + 1
        
        # Betti-1 proxy (Manifold holes indicated by sudden spectral drops)
        betti_1_threshold = mx.mean(gaps) + 2 * mx.std(gaps)
        betti_1 = mx.sum(gaps > betti_1_threshold).item()
        
        return {
            "betti_0_proxy": int(betti_0),
            "betti_1_proxy": int(betti_1),
            "spectral_entropy": float(mx.sum(mx.log(singular_values + 1e-5)).item())
        }
