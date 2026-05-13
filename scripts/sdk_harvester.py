# scripts/sdk_harvester.py
import os
import json
import hashlib
import time
import pandas as pd
from pathlib import Path

FORBIDDEN_PATHS = {"01_Legal", "02_Assets"}
EXCLUDE_DIRS = {".git", "__pycache__", "env", "venv", ".venv", "build", "dist"}

def generate_hash(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def extract_ledger_state(root_path: Path) -> dict:
    """Attempts to read the latest hash from the sovereign audit parquet."""
    ledger_path = root_path / "assets" / "tda_mesh" / "persistent_manifold.parquet"
    if ledger_path.exists():
        try:
            df = pd.read_parquet(ledger_path)
            return {"active_signatures": len(df), "last_entropy": float(df['spectral_entropy'].iloc[-1])}
        except Exception: pass
    return {"active_signatures": 0, "last_entropy": None}

def harvest_suite(root_dir: str = "."):
    root_path = Path(root_dir).resolve()
    
    snapshot = {
        "substrate": "JuniorAGI_SDK",
        "version": "0.82.0",
        "audit_timestamp": time.time(),
        "binary_gate": "SOVEREIGN_EDGE_PROTECTED",
        "topology": {"L0": "kernel", "L1": "inference", "L2": "synapse", "L3": "api"},
        "tda_ledger_state": extract_ledger_state(root_path),
        "cryptographic_manifest": {}
    }
    
    for current_root, dirs, files in os.walk(root_path):
        # Enforce Logic Gates
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not any(f in d for f in FORBIDDEN_PATHS)]
        
        for file in files:
            if file.endswith(('.py', '.so', '.md', '.toml', '.yml', '.json', '.sh')):
                full_path = Path(current_root) / file
                rel_path = str(full_path.relative_to(root_path))
                snapshot["cryptographic_manifest"][rel_path] = generate_hash(full_path)
                
    output_path = root_path / "sovereign_manifest.json"
    with open(output_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"[+] Sovereign Manifest Generated: {output_path}")

if __name__ == "__main__":
    harvest_suite()
