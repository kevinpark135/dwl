# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Proprioception-only stock PPO baseline configuration for G1 locomotion."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1RoughEnvCfg,
    G1RoughEnvCfg_PLAY,
)


@configclass
class G1ProprioceptiveBaselineEnvCfg(G1RoughEnvCfg):
    """Stock G1 rough-locomotion task without exteroceptive policy observations."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.height_scan = None
        self.scene.height_scanner = None


@configclass
class G1ProprioceptiveBaselineEnvCfg_PLAY(G1RoughEnvCfg_PLAY):
    """Play configuration for the proprioception-only stock PPO baseline."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.height_scan = None
        self.scene.height_scanner = None
