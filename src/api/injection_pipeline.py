# src/api/injection_pipeline.py
import mlx.core as mx
import os, time, logging, json
from pathlib import Path
from inference.bitnet_layers import _compute_grouped_ternary

logger = logging.getLogger("JuniorAGI.Converter")

class SovereignConverterEngine:
    """
    Production Offline Converter.
    Handles single .safetensors or sharded directories via huggingface index.
    Aggressive mx.clear_cache() protects UMA during 100B conversions.
    """
    def __init__(self, group_size: int = 128):
        self.group_size = group_size

    def convert_and_save(self, model_path: str, output_dir: str = "assets/ternary_models"):
        path = Path(model_path)
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f"{path.stem}_junior_ternary.safetensors")
        
        logger.info(f"[*] Igniting Conversion Engine on: {model_path}")
        t0 = time.perf_counter()
        
        # Shard detection
        files_to_process = []
        if path.is_dir() and (path / "model.safetensors.index.json").exists():
            with open(path / "model.safetensors.index.json", 'r') as f:
                index = json.load(f)
            shards = set(index['weight_map'].values())
            files_to_process = [path / s for s in shards]
        elif path.is_file():
            files_to_process = [path]
        else:
            logger.error("[-] Invalid model path.")
            return False

        ternary_dict = {}
        for shard in files_to_process:
            logger.info(f"[*] Processing Shard: {shard.name}")
            weights = mx.load(str(shard))
            
            for name, tensor in weights.items():
                if "weight" in name and len(tensor.shape) == 2 and "embed" not in name and "lm_head" not in name:
                    w_q, gamma = _compute_grouped_ternary(tensor, self.group_size)
                    mx.eval(w_q, gamma)
                    ternary_dict[f"{name}.w_q"] = w_q
                    ternary_dict[f"{name}.gamma"] = gamma
                    ternary_dict[f"{name}.shape"] = mx.array(tensor.shape)
                else:
                    ternary_dict[name] = tensor
                    
            if hasattr(mx, 'clear_cache'): mx.clear_cache()

        logger.info("[*] Compressing to binary int8 safetensors...")
        mx.save_safetensors(out_file, ternary_dict)
        
        size_gb = os.path.getsize(out_file) / 1024**3
        logger.info(f"[+] Conversion Complete ({time.perf_counter()-t0:.2f}s). Out: {size_gb:.2f} GB at {out_file}")
        return True
