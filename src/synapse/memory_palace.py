import mlx.core as mx
import pandas as pd
import os, time, logging
from manifolds.spectral_tda import SpectralTDAManifold

logger = logging.getLogger("JuniorAGI.Memory")

class MemoryPalace:
    """Persistent TDA. Survives node restarts via .parquet logging."""
    def __init__(self, dir: str = "assets/tda_mesh"):
        self.dir = dir
        os.makedirs(dir, exist_ok=True)
        self.path = os.path.join(dir, "persistent_manifold.parquet")
        self.tda = SpectralTDAManifold()
        self.signatures, self.tensors = [], []
        self._hydrate()

    def _hydrate(self):
        if os.path.exists(self.path):
            try:
                df = pd.read_parquet(self.path)
                self.signatures = df['spectral_entropy'].tolist()
                logger.info(f"[+] MemoryPalace Hydrated: {len(self.signatures)} Topological Signatures.")
            except: pass

    def commit(self, x: mx.array):
        ent = self.tda.extract_betti(mx.mean(x, axis=1))['entropy']
        self.signatures.append(ent)
        self.tensors.append(x)
        if len(self.signatures) % 5 == 0:
            pd.DataFrame({"ts": [time.time()]*len(self.signatures), "spectral_entropy": self.signatures}).to_parquet(self.path, compression='snappy')

    def retrieve(self, q: mx.array, top_k: int = 1) -> mx.array:
        if not self.signatures or not self.tensors: return q
        q_ent = self.tda.extract_betti(mx.mean(q, axis=1))['entropy']
        dists = [abs(s - q_ent) for s in self.signatures[:len(self.tensors)]]
        best = sorted(range(len(dists)), key=lambda i: dists[i])[:top_k]
        return mx.concatenate([self.tensors[i] for i in best] + [q], axis=1)
