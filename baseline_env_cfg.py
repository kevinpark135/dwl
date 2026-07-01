# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Baseline and ablation environment configurations for G1 locomotion."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1RoughEnvCfg,
    G1RoughEnvCfg_PLAY,
)

try:
    from . import observations as dwl_obs
    from .dwl_env_cfg import G1DwlEnvCfg, G1DwlEnvCfg_PLAY
except ImportError:
    import observations as dwl_obs
    from dwl_env_cfg import G1DwlEnvCfg, G1DwlEnvCfg_PLAY


@configclass
class G1DwlPpoNoDenoisingEnvCfg(G1DwlEnvCfg):
    """DWL task trained with plain asymmetric PPO and no denoising decoder."""


@configclass
class G1DwlPpoNoDenoisingEnvCfg_PLAY(G1DwlEnvCfg_PLAY):
    """Play configuration for the plain PPO/no-denoising ablation."""


@configclass
class G1DwlHeightScanActorEnvCfg(G1DwlEnvCfg):
    """DWL task where the actor directly receives the terrain height scan."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.height_scan = ObsTerm(
            func=dwl_obs.state_height_scan,
            params={"sensor_cfg": dwl_obs.DEFAULT_HEIGHT_SCAN_CFG},
            clip=(-1.0, 1.0),
        )


@configclass
class G1DwlHeightScanActorEnvCfg_PLAY(G1DwlEnvCfg_PLAY):
    """Play configuration for the height-scan actor ablation."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.height_scan = ObsTerm(
            func=dwl_obs.state_height_scan,
            params={"sensor_cfg": dwl_obs.DEFAULT_HEIGHT_SCAN_CFG},
            clip=(-1.0, 1.0),
        )


@configclass
class G1DwlPrivilegedActorEnvCfg(G1DwlEnvCfg):
    """DWL task for the oracle actor that consumes privileged state directly."""


@configclass
class G1DwlPrivilegedActorEnvCfg_PLAY(G1DwlEnvCfg_PLAY):
    """Play configuration for the privileged actor oracle ablation."""


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
