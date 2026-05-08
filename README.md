# JuniorAGI SDK 
**Edge-Native Sovereign Substrate | Apple Silicon (M1-M5) Optimized**

[![Substrate Version](https://img.shields.io/badge/version-0.63.0-blue.svg)]()
[![Hardware](https://img.shields.io/badge/hardware-UMA_Metal-gray.svg)]()
[![Architecture](https://img.shields.io/badge/architecture-BitNet_b1.58-black.svg)]()

JuniorAGI is a localized, edge-native Artificial General Intelligence framework engineered exclusively for Apple's Unified Memory Architecture (UMA) and Metal Performance Shaders (MPS). It rejects cloud dependency in favor of discrete, logic-dense compute nodes capable of autonomous hardware-level scaling.

## Architecture: The Spatial Hierarchy

The SDK strictly maps software topology to hardware memory caches:

* **`src/kernel/` (L0 - Executive):** The Sovereign orchestrator. Manages Autonomous Manifold Routing (AMR) and the Compute-to-Value (C2V) economy.
* **`src/inference/` (L1 - Compute):** MPS-optimized BitNet layers ($1.58$-bit Ternary Weights $\{-1, 0, 1\}$) fused with low-rank FP16 Residuals to maximize arithmetic density.
* **`src/memsys/` & `src/synapse/` (L2 - Memory):** Vectorized episodic and spatial routing using Metal-accelerated dot-product similarity (cosine metric).
* **`src/manifolds/` (L2.5 - Specialty):** SVD-aligned value routing, topological spatial detection, and domain-specific gating logic.
* **`src/logistics/`, `src/fetch/`, `src/enterprise/` (L3 - I/O):** Real-world interfacing, SMTP meshes, Temporal CRON, and high-density data ETL.
* **`src/api/` & `src/ui/` (Gateway):** Containerized FastAPI routing and websockets for C2 (Command & Control) preemption.

## Core Features

* **Dynamic Manifold Rank Adaptation (MRA):** Adjusts residual memory footprint in real-time based on the $\Gamma_t$ (Temporal Surprise) signal to prevent UMA bottlenecks.
* **Active Buffer Injection (ABI):** Micro-optimization logic allowing the Synaptic Mesh to continuously update local residual tensors without full-weight unrolling.
* **Absolute C2V Economy:** Hardware profiling strictly dictates inference offloading, adapting instantaneously across Base, Pro, Max, and Ultra topologies (up to multi-die UltraFusion mapping).

## Deployment Protocol

**1. Virtual Environment & Dependencies**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

**2. Standard Node Execution**
```bash
uvicorn src.api.node_server:app --host 0.0.0.0 --port 8000
```

**3. Containerized Logistics (CPU components only)**
```bash
docker-compose up --build -d
```

---
*Developed by JuniorCloud LLC.*
