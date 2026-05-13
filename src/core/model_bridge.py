# src/core/model_bridge.py
import mlx.nn as nn
import logging
from inference.bitnet_layers import DynamicBitLinear

logger = logging.getLogger("JuniorAGI.Bridge")

def inject_ternary_bridge(model: nn.Module, rank: int = 16) -> nn.Module:
    """
    Robust Recursive Hot-Swapping.
    Traverses the MLX AST natively. Bypasses fragile setattr string hacks.
    """
    replaced_count = 0
    total_layers = 0

    def _traverse_and_replace(module: nn.Module):
        nonlocal replaced_count, total_layers
        
        # Iterate through immediate children
        for name, child in list(module.children().items()):
            if isinstance(child, nn.Linear):
                total_layers += 1
                # Preserve embedding and output heads
                if any(x in name.lower() for x in ["lm_head", "embed", "vocab"]):
                    continue
                
                # Extract dimensions and swap
                in_d, out_d = child.weight.shape[1], child.weight.shape[0]
                bit_layer = DynamicBitLinear(in_d, out_d, rank=rank)
                bit_layer.initialize_random() # Or defer to loader
                
                setattr(module, name, bit_layer)
                replaced_count += 1
            else:
                # Recurse deeper into the manifold
                _traverse_and_replace(child)

    _traverse_and_replace(model)
    logger.info(f"[+] Model Bridge execution complete. {replaced_count}/{total_layers} Linear paths Ternarized.")
    return model
