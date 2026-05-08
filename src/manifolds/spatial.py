# manifolds/spatial.py
import re
import logging
import asyncio
from typing import Dict, Any, Tuple
import mlx.core as mx

from .base import PhysicalManifold
from core.spatial_kinematics import SpatialKinematicsEngine

logger = logging.getLogger("Sovereign.Spatial")

class SpatialManifold(PhysicalManifold):
    """
    Euclidean Interceptor.
    Routes LiDAR point clouds through the Metal acceleration matrix to map 
    physical voids into the relational working memory.
    """
    def __init__(self, sovereign_node: Any):
        self._name = "Sovereign_Euclidean_Kinematics"
        self.node = sovereign_node
        self.engine = SpatialKinematicsEngine()
        
        self.lidar_pattern = re.compile(r"\[\[PROCESS_LIDAR:\s*(.*?)\]\]", re.DOTALL)
        self.buffer = ""
        self.buffering = False
        self._telemetry = {"clouds_processed": 0, "last_void_ratio": 0.0}

    @property
    def name(self) -> str:
        return self._name

    def compute_surprise(self, input_ids: mx.array, logits: mx.array) -> Tuple[mx.array, float]:
        return mx.zeros_like(logits), 1.0

    async def _process_euclidean_data(self, file_path: str):
        logger.warning(f"[!] Euclidean Scan Initiated: Processing {file_path} via Metal API.")
        
        # Offload file parsing to thread pool to prevent async loop blocking
        result = await asyncio.to_thread(self.engine.detect_topological_voids, file_path)
        
        if result["status"] == "SUCCESS":
            self._telemetry["last_void_ratio"] = result["void_ratio"]
            payload = f"[EUCLIDEAN_VOID_DETECTION: {file_path}]\nVolume: {result['total_volume_m3']}m³\nVoid Ratio: {result['void_ratio']}\nCenter of Mass: {result['center_of_mass']}"
        else:
            payload = f"[EUCLIDEAN_ERROR: {result['message']}]"
            
        await self.node.hive_feedback.put({"agent": "SPATIAL_KINEMATICS", "result": payload})

    def process_stream(self, chunk: str) -> str:
        if not self.buffering:
            if "[" in chunk:
                self.buffering = True
                self.buffer += chunk
                return ""
            return chunk

        self.buffer += chunk
        
        match = self.lidar_pattern.search(self.buffer)
        if match:
            file_path = match.group(1).strip()
            self._telemetry["clouds_processed"] += 1
            
            asyncio.create_task(self._process_euclidean_data(file_path))
            
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
