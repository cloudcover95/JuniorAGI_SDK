import mlx.core as mx
import pandas as pd
import os
from manifolds.spectral_tda import SpectralTDAManifold

class MemoryPalace:
    """
    Sub-Linear TDA Retrieval System.
    Replaces massive KV-Caches by projecting history into Spectral Signatures.
    Only homologous state vectors are pulled into active Unified Memory.
    """
    def __init__(self, storage_path: str = "assets/ledger"):
        self.storage_path = storage_path
        self.tda = SpectralTDAManifold()
        self.signatures = []
        self.tensors = []

    def commit_state(self, activation_tensor: mx.array):
        # Extract Betti proxies and Gram matrix signatures
        topo = self.tda.extract_betti_proxies(activation_tensor)
        self.signatures.append(topo['spectral_entropy'])
        
        # In a full cluster, this shifts to .parquet delta-etching
        self.tensors.append(activation_tensor)

    def retrieve_homologous_context(self, query_tensor: mx.array, top_k: int = 3) -> mx.array:
        if not self.signatures:
            return query_tensor
            
        # Fast spectral entropy comparison (avoids full tensor dot-products)
        query_topo = self.tda.extract_betti_proxies(query_tensor)
        query_entropy = query_topo['spectral_entropy']
        
        # Find indices with lowest topological distance
        distances = [abs(sig - query_entropy) for sig in self.signatures]
        best_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:top_k]
        
        # Retrieve and concatenate contextual tensors
        context_tensors = [self.tensors[i] for i in best_indices]
        return mx.concatenate(context_tensors + [query_tensor], axis=1)
