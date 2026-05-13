# src/core/model_bridge.py
import mlx.nn as nn
import logging
from inference.bitnet_layers import DynamicBitLinear

logger = logging.getLogger("JuniorAGI.Bridge")
logging.basicConfig(level=logging.INFO)

def inject_ternary_bridge(model: nn.Module) -> nn.Module:
    """Recursively replaces nn.Linear with DynamicBitLinear, ignoring heads."""
    replaced_count = 0

    def _traverse_and_replace(module: nn.Module):
        nonlocal replaced_count
        for name, child in list(module.children().items()):
            if isinstance(child, nn.Linear):
                # Guard embeddings and vocab output manifolds
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
    logger.info(f"[+] Model Bridge complete. {replaced_count} Linear manifolds Ternarized.")
    return model
