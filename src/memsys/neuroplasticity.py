# memsys/neuroplasticity.py
import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Any, List

logger = logging.getLogger("Sovereign.Neuroplasticity")

class REMSleepConsolidator:
    """
    Intrinsic Parameter Plasticity Engine with Federated Gossip Protocol.
    Transfers high-plasticity episodic memories into the active MLX weight matrix 
    and broadcasts the updated tensor manifolds to the local swarm.
    """
    def __init__(self, sovereign_node: Any, adapters_path: str = "~/.juniorllm/adapters/"):
        self.node = sovereign_node
        self.adapters_dir = Path(adapters_path).expanduser()
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        self.is_training = False
        
    def _extract_reinforced_memories(self, threshold: float = 1.2) -> List[str]:
        if self.node.synapse.table is None: return []
        try:
            df = self.node.synapse.table.to_pandas()
            reinforced = df[df['plasticity_score'] >= threshold]
            return reinforced['content'].tolist()
        except Exception as e:
            logger.error(f"[-] Memory extraction failure during REM: {e}")
            return []

    async def execute_rem_cycle(self):
        if self.is_training: return
            
        training_data = self._extract_reinforced_memories()
        if len(training_data) < 20:
            return

        self.is_training = True
        logger.warning(f"[!] Initiating REM Sleep Consolidation. Compiling {len(training_data)} memories into neural weights.")
        
        try:
            dataset_path = self.adapters_dir / "rem_dataset.jsonl"
            with open(dataset_path, "w") as f:
                for memory in training_data:
                    record = {"text": f"<|im_start|>system\nInternal Epistemic Record<|im_end|>\n<|im_start|>user\nConsolidate state logic.<|im_end|>\n<|im_start|>assistant\n{memory}<|im_end|>"}
                    f.write(json.dumps(record) + "\n")
            
            adapter_target = self.adapters_dir / f"adapter_rem_{int(asyncio.get_event_loop().time())}.safetensors"
            
            logger.info("[*] Engaging Apple Silicon Neural Engine for Gradient Descent...")
            
            process = await asyncio.create_subprocess_exec(
                "python3", "-m", "mlx_lm.lora",
                "--model", os.getenv("JUNIOR_MODEL_PATH", "mlx-community/Junior-Base"),
                "--train", "--data", str(self.adapters_dir),
                "--iters", "100", "--batch-size", "1", "--lora-layers", "4",
                "--adapter-path", str(adapter_target),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.warning(f"[+] REM Consolidation Complete. Neocortex weights updated: {adapter_target.name}")
                self.node.load_adapters(str(adapter_target))
                
                # Broadcast topology updates across the local 48V swarm
                await self.node.swarm.gossip_weights(str(adapter_target))
            else:
                logger.error(f"[-] REM Consolidation Collapse: {stderr.decode()}")
                
        except Exception as e:
            logger.error(f"[-] Epistemic weight update failed: {e}")
        finally:
            self.is_training = False
