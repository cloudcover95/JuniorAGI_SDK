import mlx.core as mx, os, logging
from core.compute_economy import InternalAttentionEconomy
logger = logging.getLogger("JuniorAGI.Injection")

class HybridLLMInjector:
    def __init__(self, kernel):
        self.eco = InternalAttentionEconomy()
        self.uma_limit = self.eco.specs['uma_gb'] * 0.8 

    def inject_safetensors(self, path: str):
        if not os.path.exists(path): return {"error": "Void path"}
        stream = mx.gpu if self.eco.specs['mps'] else mx.cpu
        cgb, gpu, cpu = 0.0, 0, 0
        weights = mx.load(path, stream=stream)
        for k, t in weights.items():
            cgb += (t.size * t.itemsize) / 1024**3
            if cgb > self.uma_limit and self.eco.specs['mps'] and stream != mx.cpu:
                logger.warning("UMA Saturated. Routing to mx.cpu.")
                stream = mx.cpu
            mx.eval(t)
            if stream == mx.cpu: cpu += 1
            else: gpu += 1
        return {"gpu_layers": gpu, "cpu_layers": cpu, "status": "Injected"}
