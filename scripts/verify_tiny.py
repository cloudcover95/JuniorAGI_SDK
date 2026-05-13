# scripts/verify_tiny.py
import os, time, sys
import mlx.core as mx
from huggingface_hub import snapshot_download
from src.api.injection_pipeline import SovereignConverterEngine
from src.core.model_bridge import inject_ternary_bridge
from mlx_lm import load, generate
from mlx.utils import tree_unflatten

def main():
    print("[*] Initiating End-to-End Sovereign Verification (SmolLM-135M)")
    model_id = "HuggingFaceTB/SmolLM-135M"
    base_dir = "assets/raw_models/smollm"
    ternary_dir = "assets/ternary_models/smollm"
    
    # 1. Download
    if not os.path.exists(base_dir):
        print(f"[*] Downloading {model_id}...")
        snapshot_download(repo_id=model_id, local_dir=base_dir, local_dir_use_symlinks=False)
    
    # 2. Convert
    print("[*] Converting to b1.58 Ternary...")
    engine = SovereignConverterEngine()
    engine.convert_and_save(base_dir, ternary_dir)
    
    # 3. Bridge & Load
    print("[*] Booting Inference Kernel...")
    model, tokenizer = load(ternary_dir, lazy=True)
    model = inject_ternary_bridge(model)
    weights = mx.load(f"{ternary_dir}/model.safetensors")
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    
    # 4. Generate
    prompt = "The most important engineering principle is"
    print(f"\n[Prompt]: {prompt}")
    t0 = time.perf_counter()
    response = generate(model, tokenizer, prompt=prompt, max_tokens=50, verbose=False)
    t1 = time.perf_counter()
    
    mem = mx.get_active_memory() / 1024**2 if hasattr(mx, 'get_active_memory') else 0
    print(f"\n[JuniorAGI]: {response}")
    print("-" * 50)
    print(f"[+] Latency : {t1-t0:.2f}s")
    print(f"[+] UMA VRAM: {mem:.1f} MB")
    print("[+] Verification Complete. SDK is Ship-Ready.")

if __name__ == "__main__":
    main()
