# src/enterprise/audit_ledger.py
import os
import time
import hashlib
import pandas as pd
import logging

logger = logging.getLogger("JuniorAGI.Audit")

class AuditLedger:
    """
    Immutable Execution Tracking.
    Logs telemetry, C2V costs, and topological shifts to a hashed .parquet chain.
    """
    def __init__(self, ledger_dir: str = "assets/ledger"):
        self.ledger_dir = ledger_dir
        os.makedirs(self.ledger_dir, exist_ok=True)
        self.ledger_file = os.path.join(self.ledger_dir, "sovereign_audit.parquet")
        self.buffer = []
        self._last_hash = "GENESIS_BLOCK"

    def record(self, event_type: str, metadata: dict):
        """Unified logging method for the Sovereign Kernel."""
        timestamp = time.time()
        payload = f"{self._last_hash}{timestamp}{event_type}{metadata}".encode('utf-8')
        current_hash = hashlib.sha256(payload).hexdigest()
        self._last_hash = current_hash
        
        self.buffer.append({
            "timestamp": timestamp,
            "event_type": event_type,
            "metadata": str(metadata),
            "chain_hash": current_hash
        })
        
        if len(self.buffer) >= 10:
            self._flush_to_disk()

    def _flush_to_disk(self):
        df_new = pd.DataFrame(self.buffer)
        if os.path.exists(self.ledger_file):
            df_existing = pd.read_parquet(self.ledger_file)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_final = df_new
            
        df_final.to_parquet(self.ledger_file, compression='snappy')
        self.buffer.clear()
