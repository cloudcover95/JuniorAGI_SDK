#!/usr/bin/env bash
# scripts/ignite_swarm.sh
# JuniorAGI Multi-Node Deployment Protocol

WORLD_SIZE=${1:-2}
TARGET_SCALE=${2:-"70B"}

echo "[*] JuniorAGI Swarm Ignition Protocol"
echo "    -> Nodes: $WORLD_SIZE"
echo "    -> Scale: $TARGET_SCALE"
echo "    -> Interconnect: Thunderbolt / IP Mesh"

if ! command -v mlx_run &> /dev/null; then
    echo "[-] MLX Distributed backend missing. Install via: pip install mlx"
    exit 1
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Launch via MLX Distributed wrapper
mlx_run --num-nodes $WORLD_SIZE \
        python3 -c "
import mlx.core as mx
from kernel.agi_kernel import JuniorAGI
print(f'[*] Initializing Swarm Node (Target: $TARGET_SCALE)')
agi = JuniorAGI(target_scale='$TARGET_SCALE')
x = mx.random.normal((1, 32, agi.mesh.shard_dimension(agi.MODEL_PRESETS['$TARGET_SCALE']['dims'])))
out = agi.forward(x)
print(f'[+] Swarm Sync Complete. Payload Shape: {out[\"y\"].shape}')
"
