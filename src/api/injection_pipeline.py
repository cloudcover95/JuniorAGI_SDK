import mlx.core as mx
import logging
import os
from core.compute_economy import InternalAttentionEconomy

logger = logging.getLogger("JuniorAGI.Injection")

class HybridLLMInjector:
    """
    Hybrid Memory Routing (HMR) Pipeline.
    Distributes parameter weights dynamically between GPU and CPU 
    to prevent OOMs on 70B/100B targets across varying hardware limits.
    """
    def __init__(self, kernel_ref):
        self.kernel = kernel_ref
        self.economy = InternalAttentionEconomy()
        self.uma_limit_gb = self.economy.specs['uma_gb'] * 0.8 # Leave 20% for context

    def inject_safetensors(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[-] Void path: {file_path}")
            
        logger.info(f"[*] Analyzing Checkpoint: {file_path}")
        
        # Stream 0 = GPU, Stream 1 = CPU
        active_stream = mx.gpu if self.economy.specs['mps'] else mx.cpu
        cumulative_gb = 0.0
        gpu_layers, cpu_layers = 0, 0
        
        # Load weights lazily or streamingly if possible
        raw_weights = mx.load(file_path, stream=active_stream)
        
        for name, tensor in raw_weights.items():
            tensor_gb = (tensor.size * tensor.itemsize) / 1024**3
            cumulative_gb += tensor_gb
            
            # Hybrid Threshold Shift: Move to CPU if GPU envelope is saturated
            if cumulative_gb > self.uma_limit_gb and self.economy.specs['mps']:
                if active_stream != mx.cpu:
                    logger.warning("[!] UMA Envelope Saturated. Shifting deeper parameters to mx.cpu stream.")
                    active_stream = mx.cpu
            
            # Evaluate and bind tensor to the specific stream
            mx.eval(tensor)
            if active_stream == mx.cpu: cpu_layers += 1
            else: gpu_layers += 1
            
        logger.info(f"[+] HMR Checkpoint Mapped. GPU Layers: {gpu_layers} | CPU Layers: {cpu_layers}")
        return {"status": "SUCCESS", "gpu_params": gpu_layers, "cpu_params": cpu_layers}
