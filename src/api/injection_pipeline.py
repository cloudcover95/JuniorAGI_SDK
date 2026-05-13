# src/api/injection_pipeline.py
import mlx.core as mx
import os, time, logging, shutil
from pathlib import Path
from inference.bitnet_layers import compute_channel_ternary

logger = logging.getLogger("JuniorAGI.Converter")

class SovereignConverterEngine:
    def _clone_assets(self, src: Path, dest: Path):
        for ext in ["*.json", "*.model", "*.txt", "*.jinja"]:
            for file in src.glob(ext):
                if "safetensors.index" in file.name: continue 
                shutil.copy(file, dest / file.name)

    def convert_and_save(self, model_path: str, output_dir: str):
        src_path = Path(model_path)
        dest_path = Path(output_dir)
        os.makedirs(dest_path, exist_ok=True)
        
        logger.info(f"[*] Igniting Native b1.58 Converter on: {src_path}")
        t0 = time.perf_counter()
        
        self._clone_assets(src_path, dest_path)
        st_files = list(src_path.glob("*.safetensors"))

        ternary_dict = {}
        for shard in st_files:
            logger.info(f"[*] Compressing Shard: {shard.name}")
            weights = mx.load(str(shard))
            
            for name, tensor in weights.items():
                if "weight" in name and len(tensor.shape) == 2 and not any(k in name for k in ["embed", "lm_head", "gate"]):
                    # Strict key mapping: 'model.layers.0.mlp.up_proj.weight' -> 'model.layers.0.mlp.up_proj'
                    base_name = name.rsplit('.weight', 1)[0]
                    w_q, gamma = compute_channel_ternary(tensor)
                    mx.eval(w_q, gamma)
                    
                    ternary_dict[f"{base_name}.w_q"] = w_q
                    ternary_dict[f"{base_name}.gamma"] = gamma
                else:
                    ternary_dict[name] = tensor
                    
            if hasattr(mx, 'clear_cache'): mx.clear_cache()

        out_file = dest_path / "model.safetensors"
        mx.save_safetensors(str(out_file), ternary_dict)
        
        size_gb = os.path.getsize(out_file) / 1024**3
        logger.info(f"[+] Conversion Complete ({time.perf_counter()-t0:.2f}s). VRAM Footprint: {size_gb:.2f} GB")
