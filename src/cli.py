# src/cli.py
import sys, argparse, time
import mlx.core as mx
from mlx.utils import tree_unflatten
from mlx_lm import load, generate
from core.model_bridge import inject_ternary_bridge

def run_converter():
    from api.injection_pipeline import SovereignConverterEngine
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()
    SovereignConverterEngine().convert_and_save(args.model_path, args.output_dir)

def run_chat():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--prompt", type=str, default="Explain quantum topology.")
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()
    
    print(f"[*] Booting Kernel from: {args.model_path}")
    
    # Load raw topology without weights
    model, tokenizer = load(args.model_path, lazy=True)
    
    print("[*] Bridging Topology to DynamicBitLinear...")
    model = inject_ternary_bridge(model)
    
    print("[*] Hydrating Packed int8 Ternary Weights via Native AST Mapping...")
    weights = mx.load(f"{args.model_path}/model.safetensors")
    
    # Native MLX Hydration. No hacky loops.
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())

    print(f"\n[Prompt]: {args.prompt}")
    t0 = time.perf_counter()
    response = generate(model, tokenizer, prompt=args.prompt, max_tokens=args.tokens, verbose=True)
    print(f"\n[+] Generation Latency: {time.perf_counter()-t0:.2f}s")
