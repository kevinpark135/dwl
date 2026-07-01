# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Baseline environment configurations for G1 locomotion."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1RoughEnvCfg,
    G1RoughEnvCfg_PLAY,
)

try:
    from .dwl_env_cfg import G1DwlEnvCfg, G1DwlEnvCfg_PLAY
except ImportError:
    from dwl_env_cfg import G1DwlEnvCfg, G1DwlEnvCfg_PLAY


@configclass
class G1DwlPrivilegedActorEnvCfg(G1DwlEnvCfg):
    """DWL task for the oracle actor that consumes privileged state directly."""


@configclass
class G1DwlPrivilegedActorEnvCfg_PLAY(G1DwlEnvCfg_PLAY):
    """Play configuration for the privileged actor oracle ablation."""


@configclass
class G1ProprioceptiveBaselineEnvCfg(G1RoughEnvCfg):
    """Stock G1 rough-locomotion PPO with only proprioceptive actor observations.

    The stock Isaac Lab rough locomotion policy includes both base linear
    velocity and height scan. Both are removed here so this baseline cannot use
    privileged velocity or exteroceptive terrain information.
    """

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.scene.height_scanner = None


@configclass
class G1ProprioceptiveBaselineEnvCfg_PLAY(G1RoughEnvCfg_PLAY):
    """Play configuration for the true proprioception-only stock PPO baseline."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.scene.height_scanner = None
