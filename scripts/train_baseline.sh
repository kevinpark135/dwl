#!/usr/bin/env bash
set -euo pipefail

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1000}"
TASK="${TASK:-Isaac-Velocity-DWL-StockProprio-G1-v0}"

./isaaclab.sh train --rl_library rsl_rl \
  --task "${TASK}" \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --headless
