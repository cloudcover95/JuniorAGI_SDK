# JuniorAGI SDK // Sovereign Intelligence OS

**JuniorCloud LLC | Lead Architect Node**
**Deployment Target:** Apple Silicon (M4/M1) | **Power Envelope:** 45W (48V/LiFePO4)
**Network Topology:** Sovereign Localism (Starlink / Slate AX)

## Architecture Overview
Unlike standard heuristic NLP models, JuniorAGI operates on **deterministic geometric regressions** and **Topological Data Analysis (TDA)**. It fuses a hardware-accelerated mathematical engine with a self-modulating Generative UI sandbox.

### Core Pipelines
1. **Mathematical Kernel (`fsd_math`)**: High-frequency telemetry is collapsed via explicit Singular Value Decomposition ($A = U \Sigma V^T$). Optimizes for ~14.2 bits of information per dimension, bypassing transformer bloat. Adaptive routing falls back to Numpy CPU RAM hashing if MLX streams are saturated.
2. **Generative UI Sandbox**: An adaptive `mlx-lm` (Llama-3) bridge that translates topological entropy into dynamic, TradingView-style dark theme frontends (e.g., Rigid Calendar Orchestrators) via WebSockets to an iPad M1 Audit API.
3. **Hardware Lock (`crispy-mouse`)**: Sovereign execution via ATmega32u4 passthrough, isolating logic gates and trade execution from the host OS kernel.
4. **Data Lakes**: High-density Time Series (TS) Parquet storage.

## Security & Directory Isolation Protocol
The system enforces absolute traversal bans on the following directories to protect proprietary logic and entity structuring:
* `01_Legal/`
* `02_Assets/`

## Execution Protocol
The environment is self-contained. No AWS. No Oracle. No external cloud dependencies.

```zsh
# 1. Activate the M4/Metal virtual environment
source scripts/junior_venv/bin/activate

# 2. Boot the Dual-Layer Pipeline (SVD Core + Sandbox Bridge)
python src/main.py
