# src/inference/agentic_router.py
import mlx.core as mx
import mlx.nn as nn
from core.agentic_tools import SovereignAgentTools
import logging

logger = logging.getLogger("JuniorAGI.Agent")

class AgenticRouter(nn.Module):
    """
    Evaluates intermediate activation tensors to determine if real-world tooling is required.
    Acts as the bridge between mathematical manifolds and the local OS.
    """
    def __init__(self, dims: int):
        super().__init__()
        self.tools = SovereignAgentTools()
        # A simple linear projection to estimate "Tool Confidence"
        self.tool_proj = nn.Linear(dims, len(self.tools.registered_tools))
        self.tool_names = list(self.tools.registered_tools.keys())

    def __call__(self, h: mx.array, threshold: float = 0.85) -> tuple:
        # We evaluate the mean state of the sequence
        h_mean = mx.mean(h, axis=1)
        logits = self.tool_proj(h_mean)
        probs = mx.softmax(logits, axis=-1)
        mx.eval(probs)
        
        # Determine highest probability tool
        max_idx = mx.argmax(probs, axis=-1).item()
        confidence = probs[0, max_idx].item()
        
        tool_action = None
        if confidence > threshold:
            selected_tool = self.tool_names[max_idx]
            logger.info(f"[*] Autonomic Tool Trigger: {selected_tool} (Confidence: {confidence:.2f})")
            
            # In v0.80, params are stubbed. In a full LLM, they are parsed from the sequence generation.
            stub_params = {} 
            if selected_tool == "list_directory": stub_params = {"directory": "."}
            
            tool_action = {"name": selected_tool, "params": stub_params}
            result = self.tools.execute_tool(selected_tool, stub_params)
            logger.info(f"    -> Result Length: {len(result)} chars")
            
        return h, tool_action
