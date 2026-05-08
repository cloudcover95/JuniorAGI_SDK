# JuniorAGI_SDK
**Sovereign Edge-Native AGI Substrate | Apple Silicon (M1-M5) Optimized**

[![Substrate Version](https://img.shields.io/badge/version-0.66.0-blue.svg)]()
[![Hardware Base](https://img.shields.io/badge/hardware-UMA_Metal_MPS-gray.svg)]()
[![Architecture](https://img.shields.io/badge/architecture-BitNet_b1.58_TDA-black.svg)]()

JuniorAGI is a heavily engineered, localized Artificial General Intelligence framework. Designed exclusively for Apple's Unified Memory Architecture (UMA), it rejects cloud reliance in favor of logic-dense, edge-native compute nodes capable of autonomous hardware-level scaling.

## 🧠 Core Compute Substrates

### 1. BitNet b1.58 Ternary Execution & Residuals (RBN)
Replaces expensive FP16 matrix multiplications with additions by quantizing base weights to $\{-1, 0, 1\}$. High-fidelity reasoning is preserved via sparse, low-rank FP16 Residual tensors ($R = R_u \Sigma R_v^T$), evaluated via `@mx.compile` fused MLX kernels.

### 2. Spectral Topological Data Analysis (TDA) Manifold
GPU-native homology approximation. Extracts Betti-0 and Betti-1 proxies by analyzing the singular value spectrum ($A = U \Sigma V^T$) and spectral gap derivatives of the activation manifold, identifying geometric "holes" in reasoning spaces without CPU serialization.

### 3. Local LLM Weight Injection Pipeline
Direct ingestion of localized `.safetensors`. Intercepts standard FP16/BF16 weights and securely routes them into `DynamicBitLinear` structures, converting open-weight models (Llama 3, Mistral) into highly compressed ternary AGI nodes on the fly.

### 4. Immutable Enterprise Audit Ledger
All topological state shifts, Active Buffer Injections (ABI), and C2V economic decisions are cryptographically chained and flushed to an immutable `.parquet` datastore for enterprise-grade deterministic verification.

---

## 🚀 Deployment & Pipelines

### Standard Sovereign Node (Bare Metal)
Requires Apple Silicon. Connects direct to the Metal Performance Shaders (MPS).
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn src.api.node_server:app --host 0.0.0.0 --port 8000
Docker Containerized Logistics Gateway
Deploys the FastAPI routing, Websocket (C2) preemption, and Temporal CRON interfacing. (Note: Runs on CPU; routes intensive MLX calls to bare-metal workers).

Bash
docker-compose up --build -d
📊 Deterministic Benchmarking
Evaluate the Apple Neural Engine's capability on the TDA and BitNet manifolds natively:

Bash
python3 benchmarks/run_mlx_benchmark.py
Architected by JuniorCloud LLC. Zero-Trust. Zero-Bloat. Absolute Sovereignty.
