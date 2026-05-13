# src/cli.py
import sys, uvicorn
from api.injection_pipeline import SovereignConverterEngine

def run_converter():
    if len(sys.argv) < 2:
        print("Usage: junioragi-convert <path_to_model_or_directory>")
        sys.exit(1)
    engine = SovereignConverterEngine()
    engine.convert_and_save(sys.argv[1])

def run_server():
    print("[*] Igniting Sovereign Node Server on 0.0.0.0:8000...")
    uvicorn.run("api.node_server:app", host="0.0.0.0", port=8000, reload=False)
