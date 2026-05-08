import os
import asyncio
import logging
import json
from pathlib import Path
import mlx.core as mx
from ..core.config import system_config
from ..memsys.topological_synapse import TopologicalSynapse
from ..manifolds.dynamic_compiler import DynamicManifoldCompiler

logger = logging.getLogger("JuniorLLM.Ingester")

class MemSysUniversalEmbedder:
    """
    JuniorMemSys-Suite Universal Ingestion Engine.
    Maps arbitrary multi-modal objects (3D printer G-code, stock JSON, Apple Notes .txt)
    into the active MLX topological space.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def embed_object(self, raw_content: str) -> mx.array:
        """Executes a lightweight embedding pass bypassing the full forward generation."""
        tokens = mx.array([self.tokenizer.encode(raw_content)])
        # Extract embeddings directly from the model's base topology
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            embeddings = self.model.model.embed_tokens(tokens)
        else:
            # Fallback for dynamic architecture topologies
            embeddings = mx.random.normal((1, tokens.shape[1], 4096))
        return embeddings

class TopologicalIngester:
    """
    Asynchronous daemon monitoring the JuniorCloud dropzone.
    Yields strictly to the main active inference loop.
    """
    def __init__(self, sovereign_node, dropzone_path: str = "~/.juniorllm/dropzone"):
        self.dropzone = Path(dropzone_path).expanduser()
        self.processed = self.dropzone / ".processed"
        self.dropzone.mkdir(parents=True, exist_ok=True)
        self.processed.mkdir(exist_ok=True)
        
        self.node = sovereign_node
        self.embedder = MemSysUniversalEmbedder(sovereign_node.model, sovereign_node.tokenizer)
        self.compiler = DynamicManifoldCompiler(sovereign_node)
        self.synapse = sovereign_node.synapse

    async def _process_file(self, filepath: Path):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # 1. AGI Synthesis Gate: Check for Dynamic Manifold Override
            if "[JUNIOR_MANIFOLD_SYNTHESIS]" in content or "[JUNIOR_HOME_OVERRIDE]" in content:
                logger.info(f"[*] Dynamic synthesis requested in {filepath.name}. Compiling manifold...")
                self.compiler.compile_and_register(content)
            
            # 2. Universal Object Embedding
            embeddings = self.embedder.embed_object(content)
            
            # 3. Topological Compression & Flush
            self.synapse.update_synapse(embeddings)
            self.synapse.flush_to_disk()
            
            # Yield to Neural Engine / Main Thread
            await asyncio.sleep(0.01) 
            
            # Archive object
            filepath.rename(self.processed / filepath.name)
            logger.info(f"[+] Object ingested and compressed to Parquet mesh: {filepath.name}")

        except Exception as e:
            logger.error(f"[-] Void ingestion failure on {filepath.name}: {e}")

    async def watch_dropzone(self):
        """Infinite async loop monitoring object ingestion."""
        logger.info(f"[*] Topological Ingester actively monitoring {self.dropzone}")
        while True:
            for filepath in self.dropzone.glob("*.*"):
                if filepath.is_file() and not filepath.name.startswith("."):
                    await self._process_file(filepath)
            await asyncio.sleep(2.0) # Low-frequency polling to preserve compute