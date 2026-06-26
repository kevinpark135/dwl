# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the DWL task.

DWL separates observations into two conceptual groups:

- policy observations: onboard/proprioceptive signals available on the real robot
- privileged/state observations: simulation-only targets used by the critic and
  decoder reconstruction loss

The policy group intentionally excludes base linear velocity and height scans.
Those are privileged/state terms in the DWL paper.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

try:
    from .gait import DwlGaitCfg, clock_input, stance_mask
except ImportError:
    from gait import DwlGaitCfg, clock_input, stance_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


CONTROLLED_LEG_JOINT_NAMES = [
    ".*_hip_yaw_joint",
    ".*_hip_roll_joint",
    ".*_hip_pitch_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
]
"""Regex patterns for the 12 leg joints controlled by the DWL policy."""

FOOT_BODY_NAMES = [".*_ankle_roll_link"]
"""Regex patterns for the left/right foot bodies used by foot state terms."""

DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
DEFAULT_CONTROLLED_JOINT_CFG = SceneEntityCfg("robot", joint_names=CONTROLLED_LEG_JOINT_NAMES)
DEFAULT_FOOT_BODY_CFG = SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES)
DEFAULT_CONTACT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES)
DEFAULT_HEIGHT_SCAN_CFG = SceneEntityCfg("height_scanner")


def _episode_time_s(env: "ManagerBasedEnv") -> torch.Tensor:
    """Return episode time as a flat tensor with shape ``(num_envs,)``."""

    return env.episode_length_buf.to(dtype=torch.float32) * env.step_dt


def _num_envs(env: "ManagerBasedEnv") -> int:
    """Return number of vectorized environments."""

    if hasattr(env, "num_envs"):
        return int(env.num_envs)
    return int(env.episode_length_buf.shape[0])


def _device(env: "ManagerBasedEnv") -> torch.device:
    """Return the torch device used by the environment."""

    if hasattr(env, "device"):
        return torch.device(env.device)
    return env.episode_length_buf.device


def _optional_env_tensor(
    env: "ManagerBasedEnv",
    attr_name: str,
    width: int,
    default_value: float = 0.0,
) -> torch.Tensor:
    """Read an optional per-env tensor stored on the env, or return a default."""

    value = getattr(env, attr_name, None)
    if value is None:
        return torch.full((_num_envs(env), width), default_value, device=_device(env), dtype=torch.float32)

    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=_device(env), dtype=torch.float32)
    tensor = tensor.to(device=_device(env), dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    return tensor.reshape(_num_envs(env), width)


def quat_to_rpy(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert XYZW quaternions to roll-pitch-yaw Euler angles."""

    roll, pitch, yaw = euler_xyz_from_quat(quat_xyzw)
    return torch.stack((roll, pitch, yaw), dim=-1)


def base_orientation_rpy(
    env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg = DEFAULT_ROBOT_CFG
) -> torch.Tensor:
    """Return root orientation as roll-pitch-yaw Euler angles."""

    asset = env.scene[asset_cfg.name]
    return quat_to_rpy(asset.data.root_quat_w.torch)


def base_orientation_projected_gravity(
    env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg = DEFAULT_ROBOT_CFG
) -> torch.Tensor:
    """Return projected gravity as an Isaac Lab-stable orientation proxy."""

    return mdp.projected_gravity(env, asset_cfg)


def policy_clock(env: "ManagerBasedEnv", gait_cfg: DwlGaitCfg = DwlGaitCfg()) -> torch.Tensor:
    """Return policy clock input ``[sin(phase), cos(phase)]``."""

    return clock_input(_episode_time_s(env), gait_cfg.cycle_time_s, gait_cfg.phase_offset)


def policy_velocity_commands(env: "ManagerBasedEnv", command_name: str = "base_velocity") -> torch.Tensor:
    """Return commanded linear/yaw velocity for the policy."""

    return mdp.generated_commands(env, command_name)


def policy_joint_pos(
    env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg = DEFAULT_CONTROLLED_JOINT_CFG
) -> torch.Tensor:
    """Return controlled joint positions relative to default pose."""

    return mdp.joint_pos_rel(env, asset_cfg)


def policy_joint_vel(
    env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg = DEFAULT_CONTROLLED_JOINT_CFG
) -> torch.Tensor:
    """Return controlled joint velocities relative to default velocity."""

    return mdp.joint_vel_rel(env, asset_cfg)


def policy_base_ang_vel(
    env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg = DEFAULT_ROBOT_CFG
) -> torch.Tensor:
    """Return base angular velocity from onboard IMU-like state."""

    return mdp.base_ang_vel(env, asset_cfg)


def policy_base_orientation(
    env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg = DEFAULT_ROBOT_CFG
) -> torch.Tensor:
    """Return paper-aligned policy orientation as roll-pitch-yaw."""

    return base_orientation_rpy(env, asset_cfg)


def policy_last_action(env: "ManagerBasedEnv", action_name: str | None = None) -> torch.Tensor:
    """Return previous policy action."""

    return mdp.last_action(env, action_name)


def state_base_lin_vel(
    env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg = DEFAULT_ROBOT_CFG
) -> torch.Tensor:
    """Return privileged base linear velocity."""

    return mdp.base_lin_vel(env, asset_cfg)


def state_friction(
    env: "ManagerBasedEnv",
    attr_name: str = "dwl_friction",
    default_value: float = 1.0,
) -> torch.Tensor:
    """Return terrain/contact friction scalar for each env.

    The exact storage is created by domain randomization. Until `events.py` owns
    that storage, this term reads `env.<attr_name>` if present and otherwise
    returns a neutral friction value.
    """

    return _optional_env_tensor(env, attr_name, width=1, default_value=default_value)


def state_push_force_torques(
    env: "ManagerBasedEnv",
    attr_name: str = "dwl_push_force_torques",
) -> torch.Tensor:
    """Return privileged external push force/torque vector with shape ``(N, 6)``."""

    return _optional_env_tensor(env, attr_name, width=6, default_value=0.0)


def state_cycle_time(env: "ManagerBasedEnv", gait_cfg: DwlGaitCfg = DwlGaitCfg()) -> torch.Tensor:
    """Return gait cycle time as a privileged scalar."""

    return torch.full((_num_envs(env), 1), gait_cfg.cycle_time_s, device=_device(env), dtype=torch.float32)


def state_stance_mask(env: "ManagerBasedEnv", gait_cfg: DwlGaitCfg = DwlGaitCfg()) -> torch.Tensor:
    """Return periodic stance mask ``[left, right]``."""

    return stance_mask(_episode_time_s(env), gait_cfg.cycle_time_s, gait_cfg.phase_offset)


def state_feet_movement(
    env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg = DEFAULT_FOOT_BODY_CFG
) -> torch.Tensor:
    """Return foot positions and linear velocities flattened as ``[pos, vel]``.

    With two feet this is 12D: left/right position XYZ plus left/right linear
    velocity XYZ.
    """

    asset = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_link_pose_w.torch[:, asset_cfg.body_ids, :3]
    foot_vel_w = asset.data.body_link_vel_w.torch[:, asset_cfg.body_ids, :3]
    return torch.cat((foot_pos_w.flatten(start_dim=1), foot_vel_w.flatten(start_dim=1)), dim=-1)


def state_feet_contact(
    env: "ManagerBasedEnv",
    sensor_cfg: SceneEntityCfg = DEFAULT_CONTACT_SENSOR_CFG,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Return binary foot contact mask from contact force magnitudes."""

    sensor = env.scene.sensors[sensor_cfg.name]
    forces_w = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]
    return (torch.linalg.norm(forces_w, dim=-1) > contact_threshold).to(dtype=torch.float32)


def state_body_mass(
    env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg = DEFAULT_ROBOT_CFG
) -> torch.Tensor:
    """Return total selected body mass as a privileged scalar."""

    asset = env.scene[asset_cfg.name]
    return asset.data.body_mass.torch[:, asset_cfg.body_ids].sum(dim=-1, keepdim=True)


def state_current_reward(env: "ManagerBasedEnv") -> torch.Tensor:
    """Return current reward buffer as a privileged scalar."""

    reward = getattr(env, "reward_buf", None)
    if reward is None:
        return torch.zeros((_num_envs(env), 1), device=_device(env), dtype=torch.float32)
    return reward.to(device=_device(env), dtype=torch.float32).reshape(_num_envs(env), 1)


def state_joint_torques(
    env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg = DEFAULT_CONTROLLED_JOINT_CFG
) -> torch.Tensor:
    """Return applied torques for controlled joints."""

    asset = env.scene[asset_cfg.name]
    return asset.data.applied_torque.torch[:, asset_cfg.joint_ids]


def state_height_scan(
    env: "ManagerBasedEnv", sensor_cfg: SceneEntityCfg = DEFAULT_HEIGHT_SCAN_CFG, offset: float = 0.5
) -> torch.Tensor:
    """Return privileged terrain height scan."""

    return mdp.height_scan(env, sensor_cfg, offset)


def make_policy_observation_terms(gait_cfg: DwlGaitCfg = DwlGaitCfg()) -> Mapping[str, ObsTerm]:
    """Create the policy observation term map for `dwl_env_cfg.py`."""

    return {
        "clock": ObsTerm(func=policy_clock, params={"gait_cfg": gait_cfg}),
        "velocity_commands": ObsTerm(func=policy_velocity_commands, params={"command_name": "base_velocity"}),
        "joint_pos": ObsTerm(func=policy_joint_pos, params={"asset_cfg": DEFAULT_CONTROLLED_JOINT_CFG}),
        "joint_vel": ObsTerm(func=policy_joint_vel, params={"asset_cfg": DEFAULT_CONTROLLED_JOINT_CFG}),
        "base_ang_vel": ObsTerm(func=policy_base_ang_vel, params={"asset_cfg": DEFAULT_ROBOT_CFG}),
        "base_orientation": ObsTerm(func=policy_base_orientation, params={"asset_cfg": DEFAULT_ROBOT_CFG}),
        "last_action": ObsTerm(func=policy_last_action),
    }


def make_privileged_observation_terms(gait_cfg: DwlGaitCfg = DwlGaitCfg()) -> Mapping[str, ObsTerm]:
    """Create the privileged/state observation term map for `dwl_env_cfg.py`."""

    return {
        "base_lin_vel": ObsTerm(func=state_base_lin_vel, params={"asset_cfg": DEFAULT_ROBOT_CFG}),
        "friction": ObsTerm(func=state_friction),
        "push_force_torques": ObsTerm(func=state_push_force_torques),
        "cycle_time": ObsTerm(func=state_cycle_time, params={"gait_cfg": gait_cfg}),
        "stance_mask": ObsTerm(func=state_stance_mask, params={"gait_cfg": gait_cfg}),
        "feet_movement": ObsTerm(func=state_feet_movement, params={"asset_cfg": DEFAULT_FOOT_BODY_CFG}),
        "feet_contact": ObsTerm(func=state_feet_contact, params={"sensor_cfg": DEFAULT_CONTACT_SENSOR_CFG}),
        "body_mass": ObsTerm(func=state_body_mass, params={"asset_cfg": DEFAULT_ROBOT_CFG}),
        "current_reward": ObsTerm(func=state_current_reward),
        "joint_torques": ObsTerm(func=state_joint_torques, params={"asset_cfg": DEFAULT_CONTROLLED_JOINT_CFG}),
        "height_scan": ObsTerm(func=state_height_scan, params={"sensor_cfg": DEFAULT_HEIGHT_SCAN_CFG}),
    }
