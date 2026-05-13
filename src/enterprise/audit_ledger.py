import os, time, hashlib, pandas as pd
class AuditLedger:
    def __init__(self, d: str = "assets/ledger"):
        self.d = d
        os.makedirs(d, exist_ok=True)
        self.p = os.path.join(d, "audit.parquet")
        self.b, self.h = [], "GENESIS"
    def record(self, e: str, m: dict):
        ts = time.time()
        self.h = hashlib.sha256(f"{self.h}{ts}{e}{m}".encode()).hexdigest()
        self.b.append({"ts": ts, "event": e, "meta": str(m), "hash": self.h})
        if len(self.b) >= 10:
            df = pd.DataFrame(self.b)
            if os.path.exists(self.p): df = pd.concat([pd.read_parquet(self.p), df])
            df.to_parquet(self.p, compression='snappy')
            self.b.clear()
