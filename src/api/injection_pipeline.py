# src/api/injection_pipeline.py
import mlx.core as mx
import os, time, logging
from inference.bitnet_layers import _compute_grouped_ternary

logger = logging.getLogger("JuniorAGI.Converter")
logging.basicConfig(level=logging.INFO)

class SovereignConverterEngine:
    """
    Offline Ternary Conversion Engine.
    Ingests standard .safetensors (Llama, Mistral, JuniorCloud Base).
    Executes Grouped AbsMean Quantization and packs to int8, compressing
    disk and memory footprints by ~8.4x before execution.
    """
    def __init__(self, group_size: int = 128):
        self.group_size = group_size

    def convert_and_save(self, input_path: str, output_path: str = None):
        if not os.path.exists(input_path):
            logger.error(f"[-] Void path: {input_path}")
            return False
            
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_junior_ternary.safetensors"

        logger.info(f"[*] Igniting Sovereign Conversion Engine on {input_path}")
        t0 = time.perf_counter()
        
        # Lazy loading via MLX to prevent UMA explosion on 100B variants
        try:
            weights = mx.load(input_path)
        except Exception as e:
            logger.error(f"[-] Load fault: {e}")
            return False

        ternary_dict = {}
        processed_layers = 0
        total_layers = len(weights.keys())

        for name, tensor in weights.items():
            # Only quantize 2D linear weight matrices. Biases/Norms stay fp16.
            if "weight" in name and len(tensor.shape) == 2:
                w_q, gamma = _compute_grouped_ternary(tensor, self.group_size)
                # Force evaluation to manage VRAM graph size
                mx.eval(w_q, gamma)
                
                ternary_dict[f"{name}.w_q"] = w_q
                ternary_dict[f"{name}.gamma"] = gamma
                ternary_dict[f"{name}.shape"] = mx.array(tensor.shape)
            else:
                ternary_dict[name] = tensor
                
            processed_layers += 1
            if processed_layers % 50 == 0:
                logger.info(f"    -> Converted {processed_layers}/{total_layers} manifolds...")
                if hasattr(mx, 'clear_cache'): mx.clear_cache()

        logger.info("[*] Compressing and writing int8 binary mapping to disk...")
        mx.save_safetensors(output_path, ternary_dict)
        
        t1 = time.perf_counter()
        size_mb = os.path.getsize(output_path) / 1024**2
        logger.info(f"[+] Conversion Complete ({t1-t0:.2f}s). Substrate size: {size_mb:.1f} MB -> {output_path}")
        return True
