import re
import logging
from typing import Dict, Any, Tuple
import mlx.core as mx

from .base import PhysicalManifold

logger = logging.getLogger("Sovereign.Horizon")

class HorizonManifold(PhysicalManifold):
    """
    Hierarchical Task Network (HTN) Anchor with Attention Trigger.
    """
    def __init__(self, sovereign_node: Any):
        self._name = "Sovereign_Temporal_Horizon"
        self.node = sovereign_node
        
        self.set_pattern = re.compile(r"\[\[SET_HORIZON:\s*(.*?)\]\]", re.DOTALL)
        self.clear_pattern = re.compile(r"\[\[CLEAR_HORIZON\]\]")
        
        self.active_horizon = None
        self.horizon_epochs = 0
        self.buffer = ""
        self.buffering = False

    @property
    def name(self) -> str:
        return self._name

    def compute_surprise(self, input_ids: mx.array, logits: mx.array) -> Tuple[mx.array, float]:
        return mx.zeros_like(logits), 1.0

    def process_stream(self, chunk: str) -> str:
        if not self.buffering:
            if "[" in chunk:
                self.buffering = True
                self.buffer += chunk
                return ""
            return chunk

        self.buffer += chunk
        
        if self.clear_pattern.search(self.buffer):
            logger.info(f"[+] Temporal Horizon Achieved/Cleared: {self.active_horizon}")
            self.active_horizon = None
            self.horizon_epochs = 0
            
            # Trigger Memory Pruning
            self.node.reconcile_working_memory()
            
            remainder = self.buffer.replace("[[CLEAR_HORIZON]]", "")
            self.buffer = ""
            self.buffering = False
            return remainder
            
        match = self.set_pattern.search(self.buffer)
        if match:
            new_horizon = match.group(1).strip()
            self.active_horizon = new_horizon
            self.horizon_epochs = 0
            logger.warning(f"[!] New Temporal Horizon Established: {self.active_horizon}")
            
            # Trigger Attention Shift to load required skills
            self.node.reconcile_working_memory()
            
            remainder = self.buffer[match.end():]
            self.buffer = ""
            self.buffering = False
            return remainder

        if len(self.buffer) > 1024 or "\n" in chunk:
            dump = self.buffer
            self.buffer = ""
            self.buffering = False
            return dump

        return ""

    def get_telemetry(self) -> Dict[str, Any]:
        if self.active_horizon:
            self.horizon_epochs += 1
            
        return {
            "status": "horizon_active" if self.active_horizon else "idle",
            "global_objective": self.active_horizon,
            "active_epochs": self.horizon_epochs
        }
