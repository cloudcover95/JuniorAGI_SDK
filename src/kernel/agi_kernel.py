import sys, os, mlx.core as mx
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from inference.bitnet_layers import DynamicBitLinear
from manifolds.spectral_tda import SpectralTDAManifold
from fetch.data_pipeline import UnifiedFetchPipeline

class JuniorAGI:
    def __init__(self):
        self.layer = DynamicBitLinear(1024, 1024)
        self.tda = SpectralTDAManifold()
        self.fetch = UnifiedFetchPipeline()
        self.version = "0.68.0"

    def forward(self, x: mx.array) -> dict:
        y = self.layer(x, tau=0.08)
        topo = self.tda.extract_betti_proxies(y)
        return {"y": y, "topology": topo}
