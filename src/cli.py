# src/cli.py
import sys, argparse, time
import mlx.core as mx
from mlx_lm import load, generate

def run_converter():
    from api.injection_pipeline import SovereignConverterEngine
    parser = argparse.ArgumentParser(description="JuniorAGI b1.58 Ternary Converter")
    parser.add_argument("model_path", type=str, help="Source path to HF/MLX model directory")
    parser.add_argument("output_dir", type=str, help="Target path to save ternary model")
    args = parser.parse_args()
    
    SovereignConverterEngine().convert_and_save(args.model_path, args.output_dir)

def run_chat():
    from core.model_bridge import inject_ternary_bridge
    parser = argparse.ArgumentParser(description="JuniorAGI Sovereign Node Generator")
    parser.add_argument("model_path", type=str, help="Path to converted ternary model directory")
    parser.add_argument("--prompt", type=str, default="Explain quantum topology.", help="Inference prompt")
    parser.add_argument("--tokens", type=int, default=128, help="Max output tokens")
    args = parser.parse_args()
    
    print(f"[*] Booting Kernel from: {args.model_path}")
    model, tokenizer = load(args.model_path)
    
    print("[*] Bridging Topology to DynamicBitLinear...")
    model = inject_ternary_bridge(model)
    
    print("[*] Hydrating Packed int8 Ternary Weights...")
    weights = mx.load(f"{args.model_path}/model.safetensors")
    for name, module in model.named_modules():
        from inference.bitnet_layers import DynamicBitLinear
        if isinstance(module, DynamicBitLinear):
            if f"{name}.weight.w_q" in weights:
                module.w_q = weights[f"{name}.weight.w_q"]
                module.gamma = weights[f"{name}.weight.gamma"]
            if f"{name}.bias" in weights:
                module.bias = weights[f"{name}.bias"]

    print(f"\n[Prompt]: {args.prompt}")
    t0 = time.perf_counter()
    response = generate(model, tokenizer, prompt=args.prompt, max_tokens=args.tokens, verbose=True)
    
    print(f"\n[+] Generation Latency: {time.perf_counter()-t0:.2f}s")
