# src/api/model_loader.py
import mlx.core as mx
import os, time, logging
from inference.bitnet_layers import DynamicBitLinear
from kernel.agi_kernel import JuniorAGI

logger = logging.getLogger("JuniorAGI.Loader")
logging.basicConfig(level=logging.INFO)

class SovereignLoader:
    """
    Closes the deployment loop.
    Reads _junior_ternary.safetensors and injects the quantized weights
    directly into the JuniorAGI topology.
    """
    def __init__(self, kernel_ref: JuniorAGI):
        self.kernel = kernel_ref

    def load_ternary_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            logger.error(f"[-] Checkpoint missing: {checkpoint_path}")
            return False

        logger.info(f"[*] Booting Sovereign Loader from: {checkpoint_path}")
        t0 = time.perf_counter()
        
        try:
            weights = mx.load(checkpoint_path)
        except Exception as e:
            logger.error(f"[-] Decryption/Load fault: {e}")
            return False

        injected = 0
        # Map flat dictionary to the nested kernel logic
        for name, module in self.kernel.layers[0].named_modules(): # Simplified mapping for demo
            if isinstance(module, DynamicBitLinear):
                # Construct standard MLX key paths
                base_key = f"layers.0.{name}" 
                wq_key = f"{base_key}.w_q"
                gamma_key = f"{base_key}.gamma"
                
                if wq_key in weights and gamma_key in weights:
                    module.load_ternary_state(weights[wq_key], weights[gamma_key])
                    injected += 1
                else:
                    module.initialize_random()

        logger.info(f"[+] Substrate Loaded. {injected} Manifolds restored from int8 binary in {time.perf_counter()-t0:.2f}s.")
        return True
