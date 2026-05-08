# manifolds/omega_structural.py
import re
import logging
import asyncio
from typing import Dict, Any, Tuple
import mlx.core as mx

from .base import PhysicalManifold
from core.omega_cad import ProceduralCADEngine

logger = logging.getLogger("JuniorOmega.Structural")

class OmegaStructuralManifold(PhysicalManifold):
    """
    Procedural CAD Interceptor.
    Routes geometric synthesis commands to the Omega CAD Engine.
    """
    def __init__(self, agi_node: Any, cad_engine: ProceduralCADEngine):
        self._name = "JuniorOmega_Structural_Synthesis"
        self.node = agi_node
        self.cad = cad_engine
        
        self.cad_pattern = re.compile(r"\[\[GENERATE_CAD:\s*([^,]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([^\]]+)\]\]", re.DOTALL)
        self.buffer = ""
        self.buffering = False
        self._telemetry = {"meshes_synthesized": 0, "total_vertices_generated": 0}

    @property
    def name(self) -> str:
        return self._name

    def compute_surprise(self, input_ids: mx.array, logits: mx.array) -> Tuple[mx.array, float]:
        return mx.zeros_like(logits), 1.0

    async def _execute_synthesis(self, path: str, l: float, w: float, h: float, geo_type: str):
        result = await asyncio.to_thread(self.cad.synthesize_geometry, path, l, w, h, geo_type)
        if result["status"] == "SUCCESS":
            self._telemetry["meshes_synthesized"] += 1
            self._telemetry["total_vertices_generated"] += result["vertex_count"]
            payload = f"[CAD_SUCCESS: {geo_type} mesh generated at {result['output_path']}. Vertices: {result['vertex_count']}]"
        else:
            payload = f"[CAD_ERROR: {result['message']}]"
            
        await self.node.hive_feedback.put({"agent": "OMEGA_CAD_ENGINE", "result": payload})

    def process_stream(self, chunk: str) -> str:
        if not self.buffering:
            if "[" in chunk:
                self.buffering = True
                self.buffer += chunk
                return ""
            return chunk

        self.buffer += chunk
        
        match = self.cad_pattern.search(self.buffer)
        if match:
            path = match.group(1).strip()
            l = float(match.group(2).strip())
            w = float(match.group(3).strip())
            h = float(match.group(4).strip())
            geo_type = match.group(5).strip()
            
            asyncio.create_task(self._execute_synthesis(path, l, w, h, geo_type))
            
            remainder = self.buffer[match.end():]
            self.buffer = ""
            self.buffering = False
            return remainder

        if len(self.buffer) > 512 or "\n" in chunk:
            dump = self.buffer
            self.buffer = ""
            self.buffering = False
            return dump

        return ""

    def get_telemetry(self) -> Dict[str, Any]:
        return self._telemetry
