import re
import asyncio
from typing import Dict, Any, Tuple, List, Callable
import mlx.core as mx
from .base import PhysicalManifold
from ..core.logger import log

class SovereignActuator(PhysicalManifold):
    """
    Deterministic FSM-driven hardware bridge with dynamic action binding.
    """
    def __init__(self):
        self._name = "Sovereign_Hardware_Actuator"
        self.action_pattern = re.compile(r"\[\[ACTION:\s*([A-Z0-9_]+)\]\]")
        
        self.buffer = ""
        self.buffering = False
        self.buffer_token_count = 0
        self.MAX_BUFFER_TOKENS = 20 
        
        self.action_queue: List[str] = []
        self._action_registry: Dict[str, Callable] = {}
        
        self._telemetry = {"status": "monitoring", "actions_executed": 0, "last_action": None}

    @property
    def name(self) -> str:
        return self._name

    def bind_action(self, command: str, callback: Callable):
        """
        Registers an external Python function to a specific token sequence.
        Example: actuator.bind_action("PRINTER_HALT", disable_stepper_motors)
        """
        self._action_registry[command] = callback
        log.info(f"[*] Hardware action bound: {command}")

    def compute_surprise(self, input_ids: mx.array, logits: mx.array) -> Tuple[mx.array, float]:
        return mx.zeros_like(logits), 1.0

    async def _execute_hardware_command(self, command: str):
        if command in self._action_registry:
            log.warning(f"[!] Actuator Fired: Executing bound logic for {command}")
            try:
                # Dispatch the user-defined callback
                if asyncio.iscoroutinefunction(self._action_registry[command]):
                    await self._action_registry[command]()
                else:
                    self._action_registry[command]()
                    
                self._telemetry["actions_executed"] += 1
                self._telemetry["last_action"] = command
                self.action_queue.append(command)
            except Exception as e:
                log.error(f"[-] Actuator callback failure on {command}: {e}")
        else:
            log.warning(f"[-] Unrecognized or unbound physical action: {command}")

    def process_stream(self, chunk: str) -> str:
        if not self.buffering:
            if "[" in chunk:
                self.buffering = True
                self.buffer += chunk
                self.buffer_token_count = 1
                return ""
            return chunk

        self.buffer += chunk
        self.buffer_token_count += 1
        
        match = self.action_pattern.search(self.buffer)
        if match:
            command = match.group(1)
            asyncio.create_task(self._execute_hardware_command(command))
            
            remainder = self.buffer[match.end():]
            self._reset_fsm()
            return remainder

        if self.buffer_token_count >= self.MAX_BUFFER_TOKENS or "\n" in chunk:
            dump = self.buffer
            self._reset_fsm()
            return dump

        return ""

    def _reset_fsm(self):
        self.buffer = ""
        self.buffering = False
        self.buffer_token_count = 0

    def get_telemetry(self) -> Dict[str, Any]:
        current_actions = list(self.action_queue)
        self.action_queue.clear()
        
        payload = self._telemetry.copy()
        payload["triggered_events"] = current_actions
        return payload