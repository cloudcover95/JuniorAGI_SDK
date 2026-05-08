# src/kernel/agi_kernel.py
import sys
import os
import mlx.core as mx

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from infrastructure.hardware_matrix import HardwareMatrix
from core.compute_economy import InternalAttentionEconomy
from inference.mps_optimized.engine import MPSEngine
from inference.bitnet_layers import DynamicBitLinear
from inference.mps_optimized.orchestrator import ComputeOrchestrator
from synapse.plasticity import SynapticPlasticity
from kernel.autonomic_daemon import AutonomicDaemon

class JuniorAGI:
    def __init__(self):
        self.hw = HardwareMatrix()
        self.economy = InternalAttentionEconomy()
        self.engine = MPSEngine()
        self.orchestrator = ComputeOrchestrator()
        self.plasticity = SynapticPlasticity()
        
        # Core Cognitive Manifold
        self.layer_0 = DynamicBitLinear(1024, 1024)
        
        # Active Inference Daemon
        self.daemon = AutonomicDaemon(self)
        
        self.version = "0.65.0"

    async def boot_sequence(self):
        await self.daemon.ignite()

    async def halt_sequence(self):
        await self.daemon.shutdown()

    async def submit_cognitive_task(self, prompt: str) -> dict:
        simulated_tensor = mx.random.normal((1, 1024))
        y = self.layer_0(simulated_tensor, tau=self.orchestrator.get_execution_params()['tau'])
        mx.eval(y)
        
        return {
            "status": "PROCESSED",
            "c2v_cost": round(1024 / self.economy.calculate_c2v_ratio(), 4),
            "payload_dim": y.shape
        }

    def get_telemetry(self) -> dict:
        params = self.orchestrator.get_execution_params()
        return {
            "kernel": "JuniorAGI_Autonomic",
            "version": self.version,
            "gamma": f"{self.orchestrator.gamma_engine.gamma_t.item():.4f}",
            "daemon_active": self.daemon.running,
            "mps": self.engine.get_telemetry()
        }
