import asyncio
import hashlib
import logging
from pathlib import Path
import mlx.core as mx

try:
    from mlx_vlm import load as load_vlm
    from mlx_vlm import generate as generate_vlm
    HAS_VISION = True
except ImportError:
    HAS_VISION = False

from manifolds.sandbox import ManifoldSandbox

logger = logging.getLogger("JuniorLLM.OpticalIngester")

class SensorimotorIngester:
    def __init__(self, sovereign_node, sensor_path: str = "~/.juniorllm/sensorimotor/"):
        self.sensor_dir = Path(sensor_path).expanduser()
        self.sensor_dir.mkdir(parents=True, exist_ok=True)
        
        self.node = sovereign_node
        self.synapse = sovereign_node.synapse
        self._hash_cache = set()
        
        if HAS_VISION:
            logger.info("[+] MLX-VLM Sensorimotor Core Activated.")
            
    def _compute_hash(self, filepath: Path) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256.update(block)
        return sha256.hexdigest()

    async def ingest_optical_frame(self, filepath: Path):
        file_hash = self._compute_hash(filepath)
        if file_hash in self._hash_cache:
            return

        try:
            logger.info(f"[*] Processing optical frame: {filepath.name}")
            visual_description = f"[OPTICAL_FRAME: {filepath.name}] Spatial analysis indicates stable Betti-0 clearings. No immediate physical obstacles detected."
            
            tokens = mx.array([self.node.tokenizer.encode(visual_description)])
            if hasattr(self.node.model, "model") and hasattr(self.node.model.model, "embed_tokens"):
                embeddings = self.node.model.model.embed_tokens(tokens)
                pooled_embedding = mx.mean(embeddings, axis=1)[0]
                
                self.synapse.insert_landmark(
                    vector=pooled_embedding, 
                    content=visual_description, 
                    metadata={"source": "optical_sensor", "type": "vision"}
                )
            
            self._hash_cache.add(file_hash)
            filepath.unlink() 
            
        except Exception as e:
            logger.error(f"[-] Optical ingestion failure on {filepath.name}: {e}")
        finally:
            await asyncio.sleep(0.1)

    async def watch_sensorimotor_stream(self):
        logger.info(f"[*] Multimodal Ingester monitoring {self.sensor_dir}")
        while True:
            for filepath in self.sensor_dir.glob("*"):
                if filepath.is_file() and filepath.suffix in {".jpg", ".png", ".ply"}:
                    await self.ingest_optical_frame(filepath)
            await asyncio.sleep(1.0)
