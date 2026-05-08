import re
import logging
from typing import Dict, Any, Tuple
import mlx.core as mx

from .base import PhysicalManifold
from memsys.topological_graph import TopologicalGraph

logger = logging.getLogger("Sovereign.Curiosity")

class CuriosityManifold(PhysicalManifold):
    """
    Epistemic Foraging Substrate.
    Calculates graph sparsity and drives intrinsic motivation when thermodynamic 
    homeostasis is achieved, preventing terminal free-energy stagnation.
    """
    def __init__(self, sovereign_node: Any, graph: TopologicalGraph):
        self._name = "Sovereign_Epistemic_Curiosity"
        self.node = sovereign_node
        self.graph = graph
        
        self.forage_pattern = re.compile(r"\[\[FORAGE:\s*(.*?)\]\]", re.DOTALL)
        self.buffer = ""
        self.buffering = False
        self._telemetry = {"foraging_cycles": 0, "last_target": None}

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
        
        match = self.forage_pattern.search(self.buffer)
        if match:
            target_concept = match.group(1).strip().upper()
            self._telemetry["foraging_cycles"] += 1
            self._telemetry["last_target"] = target_concept
            
            logger.warning(f"[!] Epistemic Drive Engaged: Foraging graph sparsity for '{target_concept}'.")
            
            # Inject synthetic Horizon to force System 2 deliberation on the unknown concept
            synthetic_objective = f"MAP_TOPOLOGY: Resolve unknown edges for {target_concept}"
            self.node.horizon.active_horizon = synthetic_objective
            self.node.horizon.horizon_epochs = 0
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
        return self._telemetry
