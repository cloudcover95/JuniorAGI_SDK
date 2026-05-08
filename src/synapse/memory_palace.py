# src/synapse/memory_palace.py
import mlx.core as mx
from manifolds.spectral_tda import SpectralTDAManifold

class MemoryPalace:
    """
    Sub-Linear TDA Retrieval.
    Concatenates homologous state vectors across the Sequence Length axis (Axis 1).
    """
    def __init__(self):
        self.tda = SpectralTDAManifold()
        self.signatures = []
        self.tensors = []

    def commit_state(self, x: mx.array):
        # We compute topology over the mean sequence state
        x_mean = mx.mean(x, axis=1) 
        topo = self.tda.extract_betti_proxies(x_mean)
        self.signatures.append(topo['spectral_entropy'])
        self.tensors.append(x)

    def retrieve_homologous_context(self, q: mx.array, top_k: int = 1) -> mx.array:
        if not self.signatures:
            return q
            
        q_mean = mx.mean(q, axis=1)
        q_entropy = self.tda.extract_betti_proxies(q_mean)['spectral_entropy']
        
        distances = [abs(sig - q_entropy) for sig in self.signatures]
        best_idx = sorted(range(len(distances)), key=lambda i: distances[i])[:top_k]
        
        context_tensors = [self.tensors[i] for i in best_idx]
        # Concatenate along the Sequence length axis (1)
        return mx.concatenate(context_tensors + [q], axis=1)
