#!/usr/bin/env bash
set -euo pipefail

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3000}"

./isaaclab.sh train --rl_library rsl_rl \
  --task Isaac-Velocity-ProprioBaseline-G1-v0 \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --headless
