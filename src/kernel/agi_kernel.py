# src/kernel/agi_kernel.py
import sys, os, time, gc, mlx.core as mx
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.compute_economy import InternalAttentionEconomy
from inference.transformer_block import BitNetTransformerBlock
from inference.agentic_router import AgenticRouter
from synapse.memory_palace import MemoryPalace
from infrastructure.distributed_mesh import DistributedMesh
from enterprise.audit_ledger import AuditLedger

class JuniorAGI:
    PRESETS = {"7B": (4096, 32, 8), "70B": (8192, 64, 16), "100B": (12288, 96, 24)}

    def __init__(self, scale: str = "7B"):
        self.eco, self.mesh, self.mem, self.ledger = InternalAttentionEconomy(), DistributedMesh(), MemoryPalace(), AuditLedger()
        if scale not in self.PRESETS: scale = "7B"
        d, h, self.nl = self.PRESETS[scale]
        ld, lh = self.mesh.shard_dimension(d), self.mesh.shard_dimension(h)
        
        self.layers = [BitNetTransformerBlock(ld, lh, int(ld*3.5)) for _ in range(self.nl)]
        self.router = AgenticRouter(ld)
        self.scale = scale

    def forward(self, x: mx.array, cmd: dict = None) -> dict:
        t0 = time.perf_counter()
        c2v = self.eco.get_c2v_metrics()
        pb, tau = c2v['power_budget'], 0.08 if c2v['thermal_pressure'] > 0.7 else 0.02
        
        h = self.mem.retrieve(x)
        action = None
        if cmd: action = self.router.tools.execute(cmd['name'], cmd.get('params', {}))
        
        for i, lyr in enumerate(self.layers):
            h = lyr(h, tau, pb, i / max(1, self.nl-1))
            # Agentic Check
            if i == self.nl//2 and self.mesh.rank == 0 and not action:
                h, a = self.router(h)
                if a: action = a
                
        y = self.mesh.all_reduce_tensor(h)
        mx.eval(y)
        
        # Absolute VRAM Lockdown
        gc.collect()
        if hasattr(mx, 'clear_cache'): mx.clear_cache()
        
        lat = time.perf_counter() - t0
        jpi = self.eco.calculate_jpi(lat)
        
        # Extract Betti proxies and store
        self.mem.commit(y)
        self.ledger.record("inference", {"scale": self.scale, "jpi": jpi, "tool": bool(action)})
        
        return {"y": y, "metrics": c2v, "jpi": jpi, "latency": lat, "action": action}
