import re
import asyncio
import logging
from typing import Dict, Any, Tuple
import mlx.core as mx

from .base import PhysicalManifold
from .sandbox import ManifoldSandbox

logger = logging.getLogger("Sovereign.Genesis")

class GenesisManifold(PhysicalManifold):
    """
    Autonomous Skill Acquisition Substrate with Runtime Epistemic Simulation.
    """
    def __init__(self, sovereign_node: Any):
        self._name = "Sovereign_Genesis_Compiler"
        self.node = sovereign_node
        self.sandbox = ManifoldSandbox()
        
        self.compile_pattern = re.compile(r"\[\[COMPILE:\s*(.*?)\]\]", re.DOTALL)
        self.buffer = ""
        self.buffering = False
        self._telemetry = {"status": "monitoring", "skills_acquired": 0, "failed_compilations": 0}

    @property
    def name(self) -> str:
        return self._name

    def compute_surprise(self, input_ids: mx.array, logits: mx.array) -> Tuple[mx.array, float]:
        return mx.zeros_like(logits), 1.0

    async def _execute_compilation(self, source_code: str):
        logger.warning("[!] Genesis Protocol Initiated: AGI compiling and simulating self-modification.")
        
        manifold_class, error_trace = self.sandbox.compile_and_simulate(source_code)
        
        if manifold_class and not error_trace:
            try:
                new_manifold = manifold_class()
                self.node.register_manifold(new_manifold)
                self._telemetry["skills_acquired"] += 1
                logger.warning(f"[+] Genesis Successful: New capability '{new_manifold.name}' bound to substrate.")
                self._commit_to_memory(source_code, success=True)
            except Exception as bind_error:
                error_trace = str(bind_error)
                
        if error_trace:
            self._telemetry["failed_compilations"] += 1
            logger.error("[-] Genesis Failed: Epistemic Simulation rejected the authored logic.")
            self._commit_to_memory(source_code, success=False)
            
            # Route failure trace back to AGI Daemon for Recursive Correction
            error_payload = f"CODE_AUTHORED:\n{source_code}\n\nSIMULATION_FAILURE:\n{error_trace}"
            await self.node.internal_feedback.put(error_payload)

    def _commit_to_memory(self, source_code: str, success: bool):
        status = "SUCCESSFUL" if success else "FAILED_EPISTEMIC_SIMULATION"
        record = f"GENESIS_EPISODE: Attempted to compile skill. Status: {status}. Code: {source_code}"
        
        tokens = mx.array([self.node.tokenizer.encode(record)])
        if hasattr(self.node.model, "model") and hasattr(self.node.model.model, "embed_tokens"):
            embeddings = self.node.model.model.embed_tokens(tokens)
            pooled_embedding = mx.mean(embeddings, axis=1)[0]
            
            self.node.synapse.insert_landmark(
                vector=pooled_embedding,
                content=record,
                metadata={"source": "genesis_manifold"}
            )

    def process_stream(self, chunk: str) -> str:
        if not self.buffering:
            if "[" in chunk:
                self.buffering = True
                self.buffer += chunk
                return ""
            return chunk

        self.buffer += chunk
        
        match = self.compile_pattern.search(self.buffer)
        if match:
            source_code = match.group(1).strip()
            asyncio.create_task(self._execute_compilation(source_code))
            
            remainder = self.buffer[match.end():]
            self.buffer = ""
            self.buffering = False
            return remainder

        if len(self.buffer) > 4096: 
            dump = self.buffer
            self.buffer = ""
            self.buffering = False
            return dump

        return ""

    def get_telemetry(self) -> Dict[str, Any]:
        return self._telemetry
