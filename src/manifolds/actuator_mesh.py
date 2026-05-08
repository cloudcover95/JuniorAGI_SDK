# manifolds/actuator_mesh.py
import re
import logging
import asyncio
from typing import Dict, Any, Tuple
import mlx.core as mx

from .base import PhysicalManifold
from core.cyber_physical import IoTActuationMesh

logger = logging.getLogger("Sovereign.ActuatorMesh")

class CyberPhysicalManifold(PhysicalManifold):
    """
    Physical Actuation Interceptor.
    Routes Daemon commands natively to the 48V hardware bus, bridging logical
    inference to physical state changes.
    """
    def __init__(self, sovereign_node: Any, iot_mesh: IoTActuationMesh):
        self._name = "Sovereign_CyberPhysical_Actuation"
        self.node = sovereign_node
        self.iot_mesh = iot_mesh
        
        self.actuate_pattern = re.compile(r"\[\[ACTUATE:\s*([^,]+),\s*(ON|OFF)\]\]", re.DOTALL | re.IGNORECASE)
        self.buffer = ""
        self.buffering = False
        self._telemetry = {"commands_executed": 0}

    @property
    def name(self) -> str:
        return self._name

    def compute_surprise(self, input_ids: mx.array, logits: mx.array) -> Tuple[mx.array, float]:
        return mx.zeros_like(logits), 1.0

    async def _execute_actuation(self, relay_id: str, state: str):
        success = await self.iot_mesh.actuate_relay(relay_id, state)
        if success:
            self._telemetry["commands_executed"] += 1
            payload = f"[CYBER_PHYSICAL: Hardware node {relay_id} successfully locked to {state}]"
        else:
            payload = f"[CYBER_PHYSICAL_ERROR: Failed to actuate {relay_id}]"
            
        await self.node.hive_feedback.put({"agent": "ACTUATOR_MESH", "result": payload})

    def process_stream(self, chunk: str) -> str:
        if not self.buffering:
            if "[" in chunk:
                self.buffering = True
                self.buffer += chunk
                return ""
            return chunk

        self.buffer += chunk
        
        match = self.actuate_pattern.search(self.buffer)
        if match:
            relay_id = match.group(1).strip().upper()
            state = match.group(2).strip().upper()
            
            asyncio.create_task(self._execute_actuation(relay_id, state))
            
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
