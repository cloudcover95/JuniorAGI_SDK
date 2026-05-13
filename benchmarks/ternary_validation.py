# benchmarks/ternary_validation.py
import time, sys, os, mlx.core as mx
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from kernel.agi_kernel import JuniorAGI
from api.injection_pipeline import SovereignConverterEngine

def run_validation():
    print("[*] Initiating Closed-Loop Ternary Validation (v0.85.0)")
    
    # 1. Synthesize a dummy network and convert it
    os.makedirs("assets/ternary_models", exist_ok=True)
    dummy_weights = {
        "layers.0.q_proj.weight": mx.random.normal((4096, 4096)),
        "layers.0.k_proj.weight": mx.random.normal((4096, 4096))
    }
    dummy_path = "assets/ternary_models/dummy_dense.safetensors"
    mx.save_safetensors(dummy_path, dummy_weights)
    
    engine = SovereignConverterEngine()
    engine.convert_and_save(dummy_path, "assets/ternary_models")
    
    # 2. Boot Kernel and Load Ternary
    print("\n[*] Booting AGI Substrate...")
    agi = JuniorAGI("7B")
    agi.load_from_disk("assets/ternary_models/dummy_dense_junior_ternary.safetensors")
    
    # 3. Execution Verification
    x = mx.random.normal((1, 32, 4096))
    out = agi.forward(x)
    mx.eval(out["y"])
    
    print(f"\n[+] Validation Complete. Pipeline executed successfully.")
    print(f"    -> Output Shape: {out['y'].shape}")
    print(f"    -> End-to-End Latency: {out['latency']:.4f}s")
    
    # Cleanup
    os.remove(dummy_path)
    os.remove("assets/ternary_models/dummy_dense_junior_ternary.safetensors")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    run_validation()
