# src/core/model_bridge.py
import mlx.nn as nn
import logging
from inference.bitnet_layers import DynamicBitLinear

logger = logging.getLogger("JuniorAGI.Bridge")

def inject_ternary_bridge(model: nn.Module, rank: int = 16) -> nn.Module:
    """
    Traverses any instantiated MLX model (e.g. from mlx_lm) and atomically 
    replaces standard nn.Linear layers with JuniorAGI DynamicBitLinear layers.
    Preserves dimensions while upgrading the architecture to C2V-routed b1.58.
    """
    replaced_count = 0
    total_layers = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_layers += 1
            # Skip output/embedding layers typically
            if "lm_head" in name or "embed" in name:
                continue
                
            in_d, out_d = module.weight.shape[1], module.weight.shape[0]
            bit_layer = DynamicBitLinear(in_d, out_d, rank=rank)
            
            # Hot-swap the layer in the parent object
            path = name.split('.')
            parent = model
            for p in path[:-1]: parent = getattr(parent, p)
            setattr(parent, path[-1], bit_layer)
            
            replaced_count += 1

    logger.info(f"[+] Model Bridge Complete. Replaced {replaced_count}/{total_layers} Linear layers with Ternary BitNet.")
    return model
