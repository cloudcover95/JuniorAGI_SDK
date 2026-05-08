import re
import logging
from typing import Dict, Any, Tuple
import mlx.core as mx

from .base import PhysicalManifold
from memsys.topological_graph import TopologicalGraph

logger = logging.getLogger("Sovereign.Relational")

class RelationalManifold(PhysicalManifold):
    """
    Topological Kinematics Interceptor.
    Extracts causal and structural links identified by the AGI to build the knowledge graph.
    """
    def __init__(self, graph: TopologicalGraph):
        self._name = "Sovereign_Relational_Kinematics"
        self.graph = graph
        
        # Regex: [[LINK: SUBJECT, PREDICATE, OBJECT]]
        self.link_pattern = re.compile(r"\[\[LINK:\s*(.+?),\s*(.+?),\s*(.+?)\]\]", re.DOTALL)
        self.buffer = ""
        self.buffering = False
        self.links_formed = 0

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
        
        match = self.link_pattern.search(self.buffer)
        if match:
            subject, predicate, obj = match.groups()
            self.graph.insert_edge(subject, predicate, obj)
            self.links_formed += 1
            
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
        return {"status": "active", "links_formed_session": self.links_formed}
