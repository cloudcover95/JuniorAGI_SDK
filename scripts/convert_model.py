#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from api.injection_pipeline import SovereignConverterEngine

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 convert_model.py <path_to_safetensors>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    engine = SovereignConverterEngine(group_size=128)
    engine.convert_and_save(input_file)

if __name__ == "__main__":
    main()
