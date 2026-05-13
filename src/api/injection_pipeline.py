# src/api/injection_pipeline.py
import mlx.core as mx
import os, time, logging, shutil, glob
from pathlib import Path
from inference.bitnet_layers import compute_grouped_ternary

logger = logging.getLogger("JuniorAGI.Converter")
logging.basicConfig(level=logging.INFO)

class SovereignConverterEngine:
    def __init__(self, group_size: int = 128):
        self.group_size = group_size

    def _copy_assets(self, src: Path, dest: Path):
        """Copies tokenizer and config configurations for mlx_lm compatibility."""
        for ext in ["*.json", "*.model", "*.txt"]:
            for file in src.glob(ext):
                # Ignore old safetensors indexes
                if "safetensors.index" in file.name: continue 
                shutil.copy(file, dest / file.name)

    def convert_and_save(self, model_path: str, output_dir: str):
        src_path = Path(model_path)
        dest_path = Path(output_dir)
        os.makedirs(dest_path, exist_ok=True)
        
        logger.info(f"[*] Igniting Conversion Engine on: {src_path}")
        t0 = time.perf_counter()
        
        self._copy_assets(src_path, dest_path)

        st_files = list(src_path.glob("*.safetensors"))
        if not st_files:
            logger.error("[-] No .safetensors found in directory.")
            return

        ternary_dict = {}
        for shard in st_files:
            logger.info(f"[*] Processing Shard: {shard.name}")
            weights = mx.load(str(shard))
            
            for name, tensor in weights.items():
                if "weight" in name and len(tensor.shape) == 2 and not any(k in name for k in ["embed", "lm_head"]):
                    w_q, gamma = compute_grouped_ternary(tensor, self.group_size)
                    mx.eval(w_q, gamma)
                    ternary_dict[f"{name}.w_q"] = w_q
                    ternary_dict[f"{name}.gamma"] = gamma
                    ternary_dict[f"{name}.orig_shape"] = mx.array(tensor.shape)
                else:
                    ternary_dict[name] = tensor
                    
            if hasattr(mx, 'clear_cache'): mx.clear_cache()

        out_file = dest_path / "model.safetensors"
        logger.info(f"[*] Compressing to packed int8 binary -> {out_file}")
        mx.save_safetensors(str(out_file), ternary_dict)
        
        size_gb = os.path.getsize(out_file) / 1024**3
        logger.info(f"[+] Conversion Complete ({time.perf_counter()-t0:.2f}s). Out: {size_gb:.2f} GB")
