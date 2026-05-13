# src/kernel/agi_kernel.py
import sys, os, time, mlx.core as mx
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.compute_economy import InternalAttentionEconomy
from inference.transformer_block import BitNetTransformerBlock
from inference.agentic_router import AgenticRouter
from synapse.memory_palace import MemoryPalace
from infrastructure.distributed_mesh import DistributedMesh
from enterprise.audit_ledger import AuditLedger

class JuniorAGI:
    MODEL_PRESETS = {
        "7B":   {"dims": 4096, "heads": 32, "layers": 8},
        "70B":  {"dims": 8192, "heads": 64, "layers": 16},
        "100B": {"dims": 12288, "heads": 96, "layers": 24}
    }

    def __init__(self, target_scale: str = "7B"):
        self.economy = InternalAttentionEconomy()
        self.mesh = DistributedMesh()
        self.memory_palace = MemoryPalace()
        self.ledger = AuditLedger()
        
        if target_scale not in self.MODEL_PRESETS: target_scale = "7B"
        config = self.MODEL_PRESETS[target_scale]
        
        self.num_layers = config["layers"]
        local_heads = self.mesh.shard_dimension(config["heads"])
        local_dims = self.mesh.shard_dimension(config["dims"])
        mlp_dim = int(local_dims * 3.5)
        
        self.layers = [BitNetTransformerBlock(local_dims, local_heads, mlp_dim) for _ in range(self.num_layers)]
        self.router = AgenticRouter(local_dims)
        self.version = "0.80.0"
        self.target_scale = target_scale

    def forward(self, x: mx.array) -> dict:
        t0 = time.perf_counter()
        
        x_context = self.memory_palace.retrieve_homologous_context(x)
        c2v = self.economy.get_c2v_metrics()
        pb = c2v['power_budget']
        tau = 0.08 if c2v['thermal_pressure'] > 0.7 else 0.02
        
        h = x_context
        tool_invoked = None
        
        for idx, layer in enumerate(self.layers):
            depth_ratio = idx / max(1, self.num_layers - 1)
            h = layer(h, tau=tau, pb=pb, depth_ratio=depth_ratio)
            
            # Autonomic Tool Evaluation midway through the stack
            if idx == self.num_layers // 2 and self.mesh.rank == 0:
                h, tool_invoked = self.router(h)
            
        y = self.mesh.all_reduce_tensor(h)
        mx.eval(y)
        if hasattr(mx, 'metal'): mx.metal.clear_cache()
        
        latency = time.perf_counter() - t0
        jpi = self.economy.calculate_jpi(latency)
        
        self.memory_palace.commit_state(y)
        self.ledger.record("inference", {
            "scale": self.target_scale, "shape": y.shape, 
            "pb": pb, "jpi": jpi, "tool": tool_invoked['name'] if tool_invoked else "None"
        })
        
        return {"y": y, "metrics": c2v, "jpi": jpi, "latency": latency, "agentic_action": tool_invoked}
