# src/core/model_bridge.py
import mlx.nn as nn
import logging
from inference.bitnet_layers import DynamicBitLinear

logger = logging.getLogger("JuniorAGI.Bridge")

def inject_ternary_bridge(model: nn.Module) -> nn.Module:
    """
    Recursively traverses an active mlx_lm model.
    Swaps nn.Linear manifolds for JuniorAGI DynamicBitLinear graphs,
    avoiding embedding spaces and vocab heads.
    """
    replaced_count = 0

    def _traverse_and_replace(module: nn.Module):
        nonlocal replaced_count
        for name, child in module.children().items():
            if isinstance(child, nn.Linear):
                # Guard critical boundaries
                if any(k in name.lower() for k in ["lm_head", "embed", "vocab", "gate"]):
                    continue
                
                in_d, out_d = child.weight.shape[1], child.weight.shape[0]
                has_bias = "bias" in child
                
                bit_layer = DynamicBitLinear(in_d, out_d, bias=has_bias)
                setattr(module, name, bit_layer)
                replaced_count += 1
            else:
                _traverse_and_replace(child)

    _traverse_and_replace(model)
    logger.info(f"[+] Model Bridge execution complete. {replaced_count} Linear matrices Ternarized.")
    return model
