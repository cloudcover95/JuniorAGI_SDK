# manifolds/hive.py
import re
import logging
import asyncio
from typing import Dict, Any, Tuple
import mlx.core as mx

from .base import PhysicalManifold
from core.compute_economy import InternalAttentionEconomy
from core.value_tensor import SVDValueManifold

logger = logging.getLogger("Sovereign.Hive")

class HiveManifold(PhysicalManifold):
    """
    Sub-Agent Orchestrator with C2V Economic Routing.
    Intercepts SPAWN commands and evaluates their compute cost against their SVD alignment value.
    """
    def __init__(self, sovereign_node: Any, economy: InternalAttentionEconomy, value_engine: SVDValueManifold):
        self._name = "Sovereign_Hive_Orchestrator"
        self.node = sovereign_node
        self.economy = economy
        self.value_engine = value_engine
        
        # Updated Pattern to parse requested budget weight [0.0 - 1.0]
        self.spawn_pattern = re.compile(r"\[\[SPAWN:\s*([^,]+),\s*([^,]+),\s*([0-9.]+)\]\]", re.DOTALL)
        self.buffer = ""
        self.buffering = False
        self._telemetry = {"agents_spawned": 0, "active_subtasks": 0, "agents_pruned_c2v": 0}

    @property
    def name(self) -> str:
        return self._name

    def compute_surprise(self, input_ids: mx.array, logits: mx.array) -> Tuple[mx.array, float]:
        return mx.zeros_like(logits), 1.0

    async def _evaluate_and_spawn(self, agent_name: str, task_desc: str, budget_weight: float):
        # Determine alignment score of the proposed task
        alignment_score = 0.5 # Default neutral
        if self.value_engine.is_calibrated:
            try:
                tokens = mx.array([self.node.tokenizer.encode(task_desc)])
                embeds = self.node.model.model.embed_tokens(tokens)
                p_vector = mx.mean(embeds, axis=1)[0]
                alignment_score = self.value_engine.compute_alignment_score(p_vector)
            except Exception:
                pass

        if self.economy.evaluate_c2v(alignment_score, budget_weight):
            logger.warning(f"[!] Hive Mind Engaged: Spawning '{agent_name}' (Cost: {budget_weight:.2f}, Value: {alignment_score:.2f})")
            task_payload = {"agent": agent_name, "task": task_desc}
            self.node.hive_queue.put_nowait(task_payload)
            self._telemetry["agents_spawned"] += 1
            self._telemetry["active_subtasks"] += 1
        else:
            self._telemetry["agents_pruned_c2v"] += 1
            await self.node.internal_feedback.put(f"C2V_PRUNED: Sub-agent {agent_name} rejected. Expected value ({alignment_score:.2f}) does not justify the compute cost ({budget_weight:.2f}) under current thermodynamic scarcity.")

    def process_stream(self, chunk: str) -> str:
        if not self.buffering:
            if "[" in chunk:
                self.buffering = True
                self.buffer += chunk
                return ""
            return chunk

        self.buffer += chunk
        
        match = self.spawn_pattern.search(self.buffer)
        if match:
            agent_name = match.group(1).strip()
            task_desc = match.group(2).strip()
            try:
                budget_weight = float(match.group(3).strip())
            except ValueError:
                budget_weight = 1.0 # Default max cost if syntax fails
                
            asyncio.create_task(self._evaluate_and_spawn(agent_name, task_desc, budget_weight))
            
            remainder = self.buffer[match.end():]
            self.buffer = ""
            self.buffering = False
            return remainder

        if len(self.buffer) > 2048 or "\n" in chunk:
            dump = self.buffer
            self.buffer = ""
            self.buffering = False
            return dump

        return ""

    def get_telemetry(self) -> Dict[str, Any]:
        self._telemetry["active_subtasks"] = self.node.hive_queue.qsize()
        return self._telemetry
