# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the Isaac Lab DWL G1 locomotion tasks."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-DWL-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dwl_env_cfg:G1DwlEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DwlPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-DWL-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dwl_env_cfg:G1DwlEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DwlPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-DWL-PPO-NoDenoising-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline_env_cfg:G1DwlPpoNoDenoisingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DwlPpoNoDenoisingPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-DWL-PPO-NoDenoising-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline_env_cfg:G1DwlPpoNoDenoisingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DwlPpoNoDenoisingPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-DWL-HeightScanActor-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline_env_cfg:G1DwlHeightScanActorEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DwlHeightScanActorPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-DWL-HeightScanActor-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline_env_cfg:G1DwlHeightScanActorEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DwlHeightScanActorPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-DWL-PrivilegedActor-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline_env_cfg:G1DwlPrivilegedActorEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DwlPrivilegedActorPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-DWL-PrivilegedActor-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline_env_cfg:G1DwlPrivilegedActorEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DwlPrivilegedActorPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-DWL-StockProprio-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline_env_cfg:G1ProprioceptiveBaselineEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ProprioceptiveBaselinePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-DWL-StockProprio-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline_env_cfg:G1ProprioceptiveBaselineEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ProprioceptiveBaselinePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-DwlBaseline-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline_env_cfg:G1ProprioceptiveBaselineEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ProprioceptiveBaselinePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-DwlBaseline-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline_env_cfg:G1ProprioceptiveBaselineEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ProprioceptiveBaselinePPORunnerCfg",
    },
)
