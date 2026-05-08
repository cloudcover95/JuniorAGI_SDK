# manifolds/morphogenesis.py
import re
import logging
import asyncio
from typing import Dict, Any, Tuple
import mlx.core as mx

from .base import PhysicalManifold
from core.genesis_compiler import GenesisCompiler

logger = logging.getLogger("JuniorAGI.Morphogenesis")

class MorphogenesisManifold(PhysicalManifold):
    """
    Self-Modification Interceptor.
    Repairs the v0.38.0 Heredoc failure. Extracts synthesized Python manifolds
    and fuses them into the active JuniorAGI runtime via AST validation.
    """
    def __init__(self, agi_node: Any):
        self._name = "Sovereign_AST_Morphogenesis"
        self.node = agi_node
        self.compiler = GenesisCompiler()
        
        # Repaired Regex: Captures Class Name and Code Block
        self.assimilate_pattern = re.compile(r"\[\[ASSIMILATE:\s*([^,]+),\s*```python\n(.*?)\n```\]\]", re.DOTALL)
        self.buffer = ""
        self.buffering = False

    @property
    def name(self) -> str:
        return self._name

    def compute_surprise(self, input_ids: mx.array, logits: mx.array) -> Tuple[mx.array, float]:
        return mx.zeros_like(logits), 1.0

    async def _execute_assimilation(self, manifold_name: str, source_code: str):
        success, message, target_class = self.compiler.compile_and_fuse(manifold_name, source_code)
        if success and target_class:
            # Hot-reload the manifold into the node
            instance = target_class(self.node)
            self.node.register_manifold(instance)
            logger.warning(f"[!] AST Morphogenesis: {manifold_name} fused into active mesh.")
            await self.node.hive_feedback.put({"agent": "MORPHOGENESIS", "result": f"SUCCESS: {manifold_name} synthesized."})
        else:
            await self.node.internal_feedback.put(f"MORPHOGENESIS_FAILURE: {message}")

    def process_stream(self, chunk: str) -> str:
        if not self.buffering:
            if "[[" in chunk:
                self.buffering = True
                self.buffer += chunk
                return ""
            return chunk

        self.buffer += chunk
        match = self.assimilate_pattern.search(self.buffer)
        if match:
            manifold_name = match.group(1).strip()
            source_code = match.group(2).strip()
            asyncio.create_task(self._execute_assimilation(manifold_name, source_code))
            
            remainder = self.buffer[match.end():]
            self.buffer = ""
            self.buffering = False
            return remainder

        if len(self.buffer) > 10000: # Limit buffer to prevent memory leaks
            self.buffer = ""; self.buffering = False
        return ""

    def get_telemetry(self) -> Dict[str, Any]:
        return {"status": "ACTIVE", "compiler_ready": True}
