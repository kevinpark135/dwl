# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the DWL task.

This module implements the reward components from the DWL paper reward table.
The phase-aware terms use `gait.py` so rewards and observations share the same
clock, stance mask, and quintic foot trajectory convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards import (
    track_lin_vel_xy_yaw_frame_exp as isaac_track_lin_vel_xy_yaw_frame_exp,
)

try:
    from .gait import DwlGaitCfg, foot_height_reference, foot_velocity_reference, stance_mask
    from .observations import DEFAULT_CONTACT_SENSOR_CFG, DEFAULT_CONTROLLED_JOINT_CFG, DEFAULT_FOOT_BODY_CFG
except ImportError:
    from gait import DwlGaitCfg, foot_height_reference, foot_velocity_reference, stance_mask
    from observations import DEFAULT_CONTACT_SENSOR_CFG, DEFAULT_CONTROLLED_JOINT_CFG, DEFAULT_FOOT_BODY_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def tracking_exp(error: torch.Tensor, tolerance: float) -> torch.Tensor:
    """Return the paper's tracking kernel ``exp(-w * ||error||^2)``."""

    return torch.exp(-tolerance * torch.sum(torch.square(error), dim=-1))


def _episode_time_s(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return env.episode_length_buf.to(dtype=torch.float32) * env.step_dt


def _num_envs(env: "ManagerBasedRLEnv") -> int:
    if hasattr(env, "num_envs"):
        return int(env.num_envs)
    return int(env.episode_length_buf.shape[0])


def _device(env: "ManagerBasedRLEnv") -> torch.device:
    if hasattr(env, "device"):
        return torch.device(env.device)
    return env.episode_length_buf.device


def _command_xyz(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """Return commanded XYZ linear velocity, with Z command fixed to zero."""

    command = env.command_manager.get_command(command_name)
    command_xyz = torch.zeros((_num_envs(env), 3), device=command.device, dtype=command.dtype)
    command_xyz[:, :2] = command[:, :2]
    return command_xyz


def _command_rpy_rate(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """Return commanded angular velocity, with roll/pitch commands fixed to zero."""

    command = env.command_manager.get_command(command_name)
    command_rpy = torch.zeros((_num_envs(env), 3), device=command.device, dtype=command.dtype)
    command_rpy[:, 2] = command[:, 2]
    return command_rpy


def _foot_force_norm(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    forces_w = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]
    return torch.linalg.norm(forces_w, dim=-1)


def _foot_pos_vel(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    asset = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_link_pose_w.torch[:, asset_cfg.body_ids, :3]
    foot_vel_w = asset.data.body_link_vel_w.torch[:, asset_cfg.body_ids, :3]
    return foot_pos_w, foot_vel_w


def _action_history(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    prev_prev_action = getattr(env, "dwl_prev_prev_action", None)
    if prev_prev_action is None:
        prev_prev_action = torch.zeros_like(action)
    return action, prev_action, prev_prev_action


def alive(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward each non-terminated step equally."""

    return torch.ones(_num_envs(env), device=_device(env), dtype=torch.float32)


def base_motion_penalty(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize uncommanded vertical/roll/pitch base motion.

    Horizontal base velocity is handled by the command tracking rewards. Penalizing
    it here creates a standing local optimum that directly fights locomotion.
    """

    asset = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_lin_vel_b.torch
    ang_vel = asset.data.root_ang_vel_b.torch
    return torch.square(lin_vel[:, 2]) + torch.sum(torch.square(ang_vel[:, :2]), dim=-1)


def double_support(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = DEFAULT_CONTACT_SENSOR_CFG,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward both feet being in contact for stand-first training."""

    foot_contact = _foot_force_norm(env, sensor_cfg) > contact_threshold
    return torch.all(foot_contact, dim=-1).to(dtype=torch.float32)


def lin_velocity_tracking(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    tolerance: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track commanded base linear velocity with zero vertical command."""

    asset = env.scene[asset_cfg.name]
    return tracking_exp(asset.data.root_lin_vel_b.torch[:, :3] - _command_xyz(env, command_name), tolerance)


def lin_velocity_tracking_yaw_frame(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Compatibility wrapper for Isaac Lab's yaw-frame XY velocity tracking."""

    return isaac_track_lin_vel_xy_yaw_frame_exp(env, std=std, command_name=command_name, asset_cfg=asset_cfg)


def ang_velocity_tracking(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    tolerance: float = 7.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track commanded yaw velocity while keeping roll/pitch angular velocity near zero."""

    asset = env.scene[asset_cfg.name]
    return tracking_exp(asset.data.root_ang_vel_b.torch[:, :3] - _command_rpy_rate(env, command_name), tolerance)


def orientation_tracking(
    env: "ManagerBasedRLEnv", tolerance: float = 5.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track upright base orientation using roll/pitch error."""

    asset = env.scene[asset_cfg.name]
    projected_gravity_xy = asset.data.projected_gravity_b.torch[:, :2]
    return tracking_exp(projected_gravity_xy, tolerance)


def base_height_tracking(
    env: "ManagerBasedRLEnv",
    target_height: float = 0.7,
    tolerance: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track target base height."""

    asset = env.scene[asset_cfg.name]
    height_error = asset.data.root_pos_w.torch[:, 2:3] - target_height
    return tracking_exp(height_error, tolerance)


def periodic_force(
    env: "ManagerBasedRLEnv",
    gait_cfg: DwlGaitCfg = DwlGaitCfg(),
    sensor_cfg: SceneEntityCfg = DEFAULT_CONTACT_SENSOR_CFG,
    force_scale: float = 400.0,
) -> torch.Tensor:
    """Reward stance-foot contact force according to the periodic stance mask."""

    mask = stance_mask(_episode_time_s(env), gait_cfg.cycle_time_s, gait_cfg.phase_offset)
    force = torch.clamp(_foot_force_norm(env, sensor_cfg) / force_scale, min=0.0, max=1.0)
    return torch.sum(mask * force, dim=-1)


def periodic_velocity(
    env: "ManagerBasedRLEnv",
    gait_cfg: DwlGaitCfg = DwlGaitCfg(),
    asset_cfg: SceneEntityCfg = DEFAULT_FOOT_BODY_CFG,
    velocity_scale: float = 1.0,
) -> torch.Tensor:
    """Reward swing-foot movement according to the periodic stance mask."""

    mask = stance_mask(_episode_time_s(env), gait_cfg.cycle_time_s, gait_cfg.phase_offset)
    _, foot_vel_w = _foot_pos_vel(env, asset_cfg)
    foot_speed = torch.clamp(torch.linalg.norm(foot_vel_w, dim=-1) / velocity_scale, min=0.0, max=1.0)
    return torch.sum((1.0 - mask) * foot_speed, dim=-1)


def foot_height_tracking(
    env: "ManagerBasedRLEnv",
    gait_cfg: DwlGaitCfg = DwlGaitCfg(),
    tolerance: float = 5.0,
    asset_cfg: SceneEntityCfg = DEFAULT_FOOT_BODY_CFG,
    baseline_attr: str = "dwl_foot_height_baseline",
) -> torch.Tensor:
    """Track the quintic swing-foot height reference."""

    foot_pos_w, _ = _foot_pos_vel(env, asset_cfg)
    baseline = getattr(env, baseline_attr, None)
    if baseline is None:
        baseline = torch.zeros((_num_envs(env), foot_pos_w.shape[1]), device=foot_pos_w.device, dtype=foot_pos_w.dtype)
    actual_height = foot_pos_w[..., 2] - baseline
    target_height = foot_height_reference(_episode_time_s(env), gait_cfg)
    mask = 1.0 - stance_mask(_episode_time_s(env), gait_cfg.cycle_time_s, gait_cfg.phase_offset)
    error = (actual_height - target_height) * mask
    return tracking_exp(error, tolerance)


def foot_velocity_tracking(
    env: "ManagerBasedRLEnv",
    gait_cfg: DwlGaitCfg = DwlGaitCfg(),
    tolerance: float = 3.0,
    asset_cfg: SceneEntityCfg = DEFAULT_FOOT_BODY_CFG,
) -> torch.Tensor:
    """Track the quintic swing-foot vertical velocity reference."""

    _, foot_vel_w = _foot_pos_vel(env, asset_cfg)
    actual_velocity = foot_vel_w[..., 2]
    target_velocity = foot_velocity_reference(_episode_time_s(env), gait_cfg)
    mask = 1.0 - stance_mask(_episode_time_s(env), gait_cfg.cycle_time_s, gait_cfg.phase_offset)
    error = (actual_velocity - target_velocity) * mask
    return tracking_exp(error, tolerance)


def default_joint_tracking(
    env: "ManagerBasedRLEnv",
    tolerance: float = 2.0,
    asset_cfg: SceneEntityCfg = DEFAULT_CONTROLLED_JOINT_CFG,
) -> torch.Tensor:
    """Reward staying near the default controlled-joint posture."""

    asset = env.scene[asset_cfg.name]
    error = asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    return tracking_exp(error, tolerance)


def energy_cost(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = DEFAULT_CONTROLLED_JOINT_CFG
) -> torch.Tensor:
    """Return mechanical energy proxy ``sum(|tau| * |qdot|)``."""

    asset = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque.torch[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(torques) * torch.abs(joint_vel), dim=-1)


def action_smoothness(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return second-order action difference penalty term."""

    action, prev_action, prev_prev_action = _action_history(env)
    return torch.sum(torch.square(action - 2.0 * prev_action + prev_prev_action), dim=-1)


def feet_movement(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = DEFAULT_FOOT_BODY_CFG,
    velocity_scale: float = 1.0,
    acceleration_scale: float = 10.0,
) -> torch.Tensor:
    """Return scaled vertical foot velocity/acceleration regularization term."""

    asset = env.scene[asset_cfg.name]
    foot_vel_z = asset.data.body_link_vel_w.torch[:, asset_cfg.body_ids, 2] / velocity_scale
    foot_acc_z = asset.data.body_lin_acc_w.torch[:, asset_cfg.body_ids, 2] / acceleration_scale
    return torch.sum(torch.square(foot_vel_z) + torch.square(foot_acc_z), dim=-1)


def large_contact(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = DEFAULT_CONTACT_SENSOR_CFG,
    threshold: float = 400.0,
    clip_max: float = 100.0,
) -> torch.Tensor:
    """Penalize excessive foot contact forces."""

    force = _foot_force_norm(env, sensor_cfg)
    return torch.sum(torch.clamp(force - threshold, min=0.0, max=clip_max), dim=-1)
