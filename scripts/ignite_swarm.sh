#!/bin/bash
W=${1:-2}
S=${2:-"70B"}
echo "[*] Ignite Swarm: $W Nodes, Scale $S"
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
mlx_run --num-nodes $W python3 -c "
import mlx.core as mx; from kernel.agi_kernel import JuniorAGI
a = JuniorAGI('$S'); x = mx.random.normal((1, 32, a.mesh.shard_dimension(a.PRESETS['$S'][0])))
print(f'Swarm Sync Complete. Shape: {a.forward(x)[\"y\"].shape}')"
