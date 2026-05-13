import mlx.core as mx
import mlx.nn as nn
from core.agentic_tools import SovereignAgentTools

class AgenticRouter(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.tools = SovereignAgentTools()
        self.tool_proj = nn.Linear(dims, len(self.tools.registered_tools))
        self.tool_names = list(self.tools.registered_tools.keys())

    def __call__(self, h: mx.array, threshold: float = 0.85) -> tuple:
        h_mean = mx.mean(h, axis=1)
        probs = mx.softmax(self.tool_proj(h_mean), axis=-1)
        max_idx = mx.argmax(probs, axis=-1).item()
        confidence = probs[0, max_idx].item()
        
        action = None
        if confidence > threshold:
            name = self.tool_names[max_idx]
            # Stub parameters for autonomic execution (in prod, parse from sequence)
            params = {"directory": "."} if name == "list_dir" else {}
            res = self.tools.execute(name, params)
            action = {"name": name, "confidence": confidence, "result": res}
        return h, action
