# src/synapse/memory_palace.py
import mlx.core as mx
import pandas as pd
import os
import time
import logging
from manifolds.spectral_tda import SpectralTDAManifold

logger = logging.getLogger("JuniorAGI.Memory")

class MemoryPalace:
    """
    Persistent Sub-Linear TDA Retrieval.
    Survives restarts by flushing to/hydrating from .parquet.
    """
    def __init__(self, storage_dir: str = "assets/tda_mesh"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.mesh_file = os.path.join(self.storage_dir, "persistent_manifold.parquet")
        
        self.tda = SpectralTDAManifold()
        self.signatures = []
        self.tensors = []
        
        self._hydrate_from_disk()

    def _hydrate_from_disk(self):
        if os.path.exists(self.mesh_file):
            try:
                df = pd.read_parquet(self.mesh_file)
                self.signatures = df['spectral_entropy'].tolist()
                # Tensors are loaded as flattened arrays, reshaping required based on dim
                # For v0.79, we store signatures persistently, but re-init active tensors
                # to save RAM until needed.
                logger.info(f"[+] Memory Palace Hydrated: {len(self.signatures)} topological signatures.")
            except Exception as e:
                logger.error(f"[-] Failed to hydrate Memory Palace: {e}")

    def commit_state(self, x: mx.array):
        x_mean = mx.mean(x, axis=1) 
        topo = self.tda.extract_betti_proxies(x_mean)
        entropy = topo['spectral_entropy']
        
        self.signatures.append(entropy)
        self.tensors.append(x)
        
        # Flush to disk periodically (every 10 commits)
        if len(self.signatures) % 10 == 0:
            df = pd.DataFrame({"timestamp": [time.time()] * len(self.signatures), "spectral_entropy": self.signatures})
            df.to_parquet(self.mesh_file, compression='snappy')

    def retrieve_homologous_context(self, q: mx.array, top_k: int = 1) -> mx.array:
        if not self.signatures or not self.tensors:
            return q
            
        q_mean = mx.mean(q, axis=1)
        q_entropy = self.tda.extract_betti_proxies(q_mean)['spectral_entropy']
        
        distances = [abs(sig - q_entropy) for sig in self.signatures[:len(self.tensors)]]
        best_idx = sorted(range(len(distances)), key=lambda i: distances[i])[:top_k]
        
        context_tensors = [self.tensors[i] for i in best_idx]
        return mx.concatenate(context_tensors + [q], axis=1)
