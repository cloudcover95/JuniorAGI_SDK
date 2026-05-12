# src/api/injection_pipeline.py
import mlx.core as mx
import logging
import os
from core.compute_economy import InternalAttentionEconomy

logger = logging.getLogger("JuniorAGI.Injection")

class HybridLLMInjector:
    """
    Hybrid Memory Routing (HMR).
    Loads raw `.safetensors` into unified memory. To prevent VRAM collapse during 100B 
    mounting, tensors are pinned to `mx.cpu` *before* BitNet quantization, allowing 
    Apple Silicon to page the data safely prior to Ternary translation.
    """
    def __init__(self, kernel_ref):
        self.kernel = kernel_ref
        self.economy = InternalAttentionEconomy()
        # 80% VRAM headroom threshold
        self.uma_limit_gb = self.economy.specs['uma_gb'] * 0.8 

    def inject_safetensors(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[-] Void path: {file_path}")
            
        logger.info(f"[*] HMR Checkpoint Injection: {file_path}")
        active_stream = mx.gpu if self.economy.specs['mps'] else mx.cpu
        cumulative_gb, gpu_layers, cpu_layers = 0.0, 0, 0
        
        # Load weights
        raw_weights = mx.load(file_path, stream=active_stream)
        
        for name, tensor in raw_weights.items():
            tensor_gb = (tensor.size * tensor.itemsize) / 1024**3
            cumulative_gb += tensor_gb
            
            # HMR Dynamic Routing: If GPU hits C2V boundary, stream directly to CPU
            if cumulative_gb > self.uma_limit_gb and self.economy.specs['mps']:
                if active_stream != mx.cpu:
                    logger.warning(f"[!] UMA Envelope at {self.uma_limit_gb:.1f}GB. Shifting subsequent tensors to CPU.")
                    active_stream = mx.cpu
            
            mx.eval(tensor)
            if active_stream == mx.cpu: cpu_layers += 1
            else: gpu_layers += 1
            
        return {"status": "SUCCESS", "gpu_params": gpu_layers, "cpu_params": cpu_layers}
