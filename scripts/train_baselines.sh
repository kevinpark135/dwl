#!/usr/bin/env bash
set -euo pipefail

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1000}"

TASKS=(
  "Isaac-Velocity-DWL-StockProprio-G1-v0"
  "Isaac-Velocity-DWL-PrivilegedActor-G1-v0"
)

for task in "${TASKS[@]}"; do
  echo "Training ${task}"
  ./isaaclab.sh train --rl_library rsl_rl \
    --task "${task}" \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${MAX_ITERATIONS}" \
    --headless
done
