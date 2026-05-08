# memsys/sovereign_identity.py
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger("Sovereign.Identity")

class SovereignIdentity:
    """
    User configuration manifold.
    Maps enterprise directives, local hardware constraints, and collaborative local LLMs 
    to dictate MLX resource allocation and UI telemetry payloads.
    """
    def __init__(self, config_path: str = "~/.juniorllm/identity.json"):
        self.config_path = Path(config_path).expanduser()
        self.profile: Dict[str, Any] = {}
        self.collaborative_llms: List[str] = []
        self._bootstrap_identity()
        self._discover_local_llms()

    def _bootstrap_identity(self):
        if not self.config_path.exists():
            # Default initialization for Lead Architect / Edge Environment
            default_identity = {
                "user": "Nicolas John Regoli",
                "role": "Lead Architect",
                "organization": "JuniorCloud LLC",
                "hardware": {
                    "primary_node": "MacBook Air M4",
                    "unified_memory_gb": 16,
                    "ui_target": "iPad Pro M1 WebSocket Dashboard"
                },
                "active_workspaces": [
                    "JuniorStock", 
                    "JuniorHome",
                    "JC-SDK-CORE"
                ]
            }
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(default_identity, f, indent=4)
            self.profile = default_identity
            logger.info(f"[+] Sovereign Identity Bootstrapped. Node locked to {self.profile['hardware']['primary_node']}.")
        else:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.profile = json.load(f)

    def _discover_local_llms(self):
        """Scans local disk manifolds to register auxiliary MLX models for Hive Mind allocation."""
        search_paths = [
            Path("~/.cache/huggingface/hub").expanduser(), 
            Path("~/.juniorllm/models").expanduser()
        ]
        
        for directory in search_paths:
            if directory.exists():
                for entry in directory.iterdir():
                    if entry.is_dir() and "models--" in entry.name:
                        model_id = entry.name.replace("models--", "").replace("--", "/")
                        self.collaborative_llms.append(model_id)
        
        if not self.collaborative_llms:
            self.collaborative_llms = ["mlx-community/Junior-Base"]
            
        logger.info(f"[*] Local LLM Collaboration Mesh Registered: {len(self.collaborative_llms)} models available.")

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "profile": self.profile,
            "collaborative_llms": self.collaborative_llms
        }
