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


def _yaw_warmup_scale(env: "ManagerBasedRLEnv", warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    common_step = float(getattr(env, "common_step_counter", 0))
    return min(max(common_step / float(warmup_steps), 0.0), 1.0)


def _yaw_curriculum_scale(env: "ManagerBasedRLEnv", curriculum_steps: tuple[int, int, int, int] | None) -> float:
    if curriculum_steps is None:
        return 1.0
    start_25, start_50, start_75, start_full = curriculum_steps
    common_step = int(getattr(env, "common_step_counter", 0))
    if common_step < start_25:
        return 0.0
    if common_step < start_50:
        return 0.25
    if common_step < start_75:
        return 0.5
    if common_step < start_full:
        return 0.75
    return 1.0


def _effective_yaw_command(
    env: "ManagerBasedRLEnv",
    command_name: str,
    yaw_warmup_steps: int,
    yaw_curriculum_steps: tuple[int, int, int, int] | None,
) -> torch.Tensor:
    command_yaw = env.command_manager.get_command(command_name)[:, 2]
    scale = _yaw_curriculum_scale(env, yaw_curriculum_steps)
    if yaw_curriculum_steps is None:
        scale = _yaw_warmup_scale(env, yaw_warmup_steps)
    return command_yaw * scale


def _command_rpy_rate_with_yaw_warmup(
    env: "ManagerBasedRLEnv",
    command_name: str,
    warmup_steps: int,
    yaw_curriculum_steps: tuple[int, int, int, int] | None = None,
) -> torch.Tensor:
    command_rpy = _command_rpy_rate(env, command_name)
    command_rpy[:, 2] = _effective_yaw_command(env, command_name, warmup_steps, yaw_curriculum_steps)
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


def _yaw_rotate_inverse(asset, vector_w: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vectors into the base yaw frame when root orientation is available."""

    root_quat = getattr(asset.data, "root_quat_w", None)
    if root_quat is None:
        return vector_w

    quat = root_quat.torch
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cos_yaw = torch.cos(yaw).unsqueeze(-1)
    sin_yaw = torch.sin(yaw).unsqueeze(-1)

    vector_b = vector_w.clone()
    vector_b[..., 0] = cos_yaw * vector_w[..., 0] + sin_yaw * vector_w[..., 1]
    vector_b[..., 1] = -sin_yaw * vector_w[..., 0] + cos_yaw * vector_w[..., 1]
    return vector_b


def _action_history(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    state = getattr(env, "dwl_action_smoothness_state", None)
    common_step = int(getattr(env, "common_step_counter", -1))
    if (
        state is None
        or state["prev_action_snapshot"].shape != prev_action.shape
        or state["prev_action_snapshot"].device != prev_action.device
    ):
        state = {
            "prev_action_snapshot": prev_action.clone(),
            "current_prev_prev_action": prev_action.clone(),
            "last_step": common_step,
        }
        setattr(env, "dwl_action_smoothness_state", state)
    elif state["last_step"] != common_step:
        state["current_prev_prev_action"] = state["prev_action_snapshot"].clone()
        state["prev_action_snapshot"] = prev_action.clone()
        state["last_step"] = common_step

    prev_prev_action = state["current_prev_prev_action"]
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


def lin_velocity_tracking(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    tolerance: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track commanded base linear velocity with zero vertical command."""

    asset = env.scene[asset_cfg.name]
    return tracking_exp(asset.data.root_lin_vel_b.torch[:, :3] - _command_xyz(env, command_name), tolerance)


def forward_progress(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    min_command_x: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward positive body-frame forward velocity when forward motion is commanded."""

    asset = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0].clamp_min(0.0)
    gate = (command_x > min_command_x).to(dtype=asset.data.root_lin_vel_b.torch.dtype)
    forward_vel = asset.data.root_lin_vel_b.torch[:, 0]
    capped_forward_vel = torch.minimum(torch.clamp(forward_vel, min=0.0), command_x)
    return gate * capped_forward_vel / torch.clamp(command_x, min=min_command_x)


def low_forward_speed_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    min_command_x: float = 0.2,
    min_forward_speed: float = 0.25,
    command_speed_fraction: float = 0.6,
    grace_period_s: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize commanded forward episodes that settle into slow or still behavior."""

    asset = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0].clamp_min(0.0)
    command_gate = command_x > min_command_x
    time_gate = _episode_time_s(env) > grace_period_s
    forward_speed = asset.data.root_lin_vel_b.torch[:, 0]
    speed_floor = torch.maximum(
        torch.full_like(command_x, min_forward_speed),
        command_x * command_speed_fraction,
    )
    shortfall = torch.clamp(speed_floor - forward_speed, min=0.0) / torch.clamp(speed_floor, min=1.0e-6)
    return (command_gate & time_gate).to(dtype=forward_speed.dtype) * torch.square(shortfall)


def ang_velocity_tracking(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    tolerance: float = 7.0,
    yaw_warmup_steps: int = 7200,
    yaw_curriculum_steps: tuple[int, int, int, int] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track commanded yaw velocity while keeping roll/pitch angular velocity near zero."""

    asset = env.scene[asset_cfg.name]
    command_rpy = _command_rpy_rate_with_yaw_warmup(env, command_name, yaw_warmup_steps, yaw_curriculum_steps)
    return tracking_exp(asset.data.root_ang_vel_b.torch[:, :3] - command_rpy, tolerance)


def yaw_drift_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    yaw_warmup_steps: int = 7200,
    yaw_curriculum_steps: tuple[int, int, int, int] | None = None,
    full_yaw_rate: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize unintended yaw drift most strongly while the yaw target is small."""

    asset = env.scene[asset_cfg.name]
    yaw_rate = asset.data.root_ang_vel_b.torch[:, 2]
    effective_yaw = torch.abs(_effective_yaw_command(env, command_name, yaw_warmup_steps, yaw_curriculum_steps))
    straight_gate = 1.0 - torch.clamp(effective_yaw / full_yaw_rate, min=0.0, max=1.0)
    return torch.square(yaw_rate) * straight_gate


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


def commanded_swing_air_time(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    gait_cfg: DwlGaitCfg = DwlGaitCfg(),
    asset_cfg: SceneEntityCfg = DEFAULT_FOOT_BODY_CFG,
    sensor_cfg: SceneEntityCfg = DEFAULT_CONTACT_SENSOR_CFG,
    min_command_x: float = 0.2,
    min_forward_speed: float = 0.0,
    max_tilt: float | None = None,
    contact_threshold: float = 1.0,
    clearance_height: float = 0.06,
    target_air_time: float = 0.25,
    max_air_time: float = 0.6,
    clearance_reward_scale: float = 0.25,
    long_air_penalty_scale: float = 0.5,
    baseline_attr: str = "dwl_foot_height_baseline",
    air_time_attr: str = "dwl_foot_air_time",
    prev_contact_attr: str = "dwl_prev_foot_contact",
) -> torch.Tensor:
    """Reward commanded swing clearance and useful air-time, penalizing overlong flight."""

    device = _device(env)
    num_envs = _num_envs(env)
    command_x = env.command_manager.get_command(command_name)[:, 0].clamp_min(0.0)
    gate = (command_x > min_command_x).to(device=device, dtype=torch.float32)
    asset = env.scene[asset_cfg.name]
    forward_speed = asset.data.root_lin_vel_b.torch[:, 0]
    gate = gate * (forward_speed > min_forward_speed).to(device=device, dtype=torch.float32)
    if max_tilt is not None:
        tilt = torch.linalg.norm(asset.data.projected_gravity_b.torch[:, :2], dim=-1)
        gate = gate * (tilt < max_tilt).to(device=device, dtype=torch.float32)

    foot_pos_w, _ = _foot_pos_vel(env, asset_cfg)
    baseline = getattr(env, baseline_attr, None)
    if baseline is None:
        baseline = torch.zeros((num_envs, foot_pos_w.shape[1]), device=foot_pos_w.device, dtype=foot_pos_w.dtype)
    foot_height = foot_pos_w[..., 2] - baseline

    expected_swing = 1.0 - stance_mask(_episode_time_s(env), gait_cfg.cycle_time_s, gait_cfg.phase_offset)
    clearance = torch.clamp(foot_height / clearance_height, min=0.0, max=1.0) * expected_swing
    clearance_reward = clearance_reward_scale * torch.sum(clearance, dim=-1)

    contact = _foot_force_norm(env, sensor_cfg) > contact_threshold
    air_time = getattr(env, air_time_attr, None)
    if air_time is None or air_time.shape != contact.shape:
        air_time = torch.zeros(contact.shape, device=contact.device, dtype=torch.float32)
        setattr(env, air_time_attr, air_time)

    prev_contact = getattr(env, prev_contact_attr, None)
    if prev_contact is None or prev_contact.shape != contact.shape:
        prev_contact = contact.clone()
        setattr(env, prev_contact_attr, prev_contact)

    reset_ids = torch.nonzero(env.episode_length_buf.to(device=contact.device) == 0, as_tuple=False).flatten()
    if reset_ids.numel() > 0:
        air_time[reset_ids] = 0.0
        prev_contact[reset_ids] = contact[reset_ids]

    airborne = ~contact
    first_contact = contact & (~prev_contact)
    useful_air_time = torch.clamp(air_time / target_air_time, min=0.0, max=1.0)
    touchdown_reward = torch.sum(first_contact.to(dtype=torch.float32) * useful_air_time, dim=-1)
    long_air_penalty = long_air_penalty_scale * torch.sum(
        torch.clamp(air_time - max_air_time, min=0.0) / max_air_time,
        dim=-1,
    )

    air_time[:] = torch.where(airborne, air_time + float(env.step_dt), torch.zeros_like(air_time))
    prev_contact[:] = contact
    return gate * (clearance_reward + touchdown_reward - long_air_penalty)


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


def foot_lateral_tracking(
    env: "ManagerBasedRLEnv",
    target_width: float = 0.16,
    tolerance: float = 70.0,
    asset_cfg: SceneEntityCfg = DEFAULT_FOOT_BODY_CFG,
) -> torch.Tensor:
    """Reward a natural left/right foot corridor instead of a wide A-frame stance."""

    asset = env.scene[asset_cfg.name]
    foot_pos_w, _ = _foot_pos_vel(env, asset_cfg)
    rel_foot_pos_w = foot_pos_w - asset.data.root_pos_w.torch[:, None, :3]
    foot_pos_b = _yaw_rotate_inverse(asset, rel_foot_pos_w)

    half_width = 0.5 * target_width
    target_y = torch.tensor([half_width, -half_width], device=foot_pos_b.device, dtype=foot_pos_b.dtype)
    target_y = target_y[: foot_pos_b.shape[1]].unsqueeze(0)
    lateral_error = foot_pos_b[..., 1] - target_y

    crossing_error = torch.zeros(_num_envs(env), device=foot_pos_b.device, dtype=foot_pos_b.dtype)
    if foot_pos_b.shape[1] >= 2:
        crossing_error = torch.clamp(foot_pos_b[:, 1, 1] - foot_pos_b[:, 0, 1], min=0.0).unsqueeze(-1)
        lateral_error = torch.cat((lateral_error, crossing_error), dim=-1)

    return tracking_exp(lateral_error, tolerance)


def foot_lateral_velocity(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = DEFAULT_FOOT_BODY_CFG,
    velocity_scale: float = 1.0,
) -> torch.Tensor:
    """Penalize side-shuffling foot velocity while leaving forward swing free."""

    asset = env.scene[asset_cfg.name]
    _, foot_vel_w = _foot_pos_vel(env, asset_cfg)
    foot_vel_b = _yaw_rotate_inverse(asset, foot_vel_w[..., :3])
    return torch.sum(torch.square(foot_vel_b[..., 1] / velocity_scale), dim=-1)


def foot_sagittal_tracking(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    gait_cfg: DwlGaitCfg = DwlGaitCfg(),
    asset_cfg: SceneEntityCfg = DEFAULT_FOOT_BODY_CFG,
    sensor_cfg: SceneEntityCfg = DEFAULT_CONTACT_SENSOR_CFG,
    min_command_x: float = 0.2,
    swing_speed_scale: float = 0.8,
    stance_speed_scale: float = 1.0,
    tolerance: float = 4.0,
    contact_threshold: float = 1.0,
    clearance_height: float = 0.07,
    swing_contact_penalty_scale: float = 0.5,
    baseline_attr: str = "dwl_foot_height_baseline",
) -> torch.Tensor:
    """Reward alternating forward swing instead of pivoting around one planted foot."""

    asset = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0].clamp_min(0.0)
    gate = (command_x > min_command_x).to(dtype=asset.data.root_lin_vel_b.torch.dtype)

    foot_pos_w, foot_vel_w = _foot_pos_vel(env, asset_cfg)
    foot_vel_b = _yaw_rotate_inverse(asset, foot_vel_w[..., :3])
    root_vel_b = asset.data.root_lin_vel_b.torch[:, None, :3]
    foot_rel_vel_b = foot_vel_b - root_vel_b

    stance = stance_mask(_episode_time_s(env), gait_cfg.cycle_time_s, gait_cfg.phase_offset)
    swing = 1.0 - stance
    target_forward = command_x[:, None] * swing_speed_scale
    target_backward = -command_x[:, None] * stance_speed_scale
    target_rel_x = swing * target_forward + stance * target_backward
    error = foot_rel_vel_b[..., 0] - target_rel_x

    baseline = getattr(env, baseline_attr, None)
    if baseline is None:
        baseline = torch.zeros((_num_envs(env), foot_pos_w.shape[1]), device=foot_pos_w.device, dtype=foot_pos_w.dtype)
    foot_height = foot_pos_w[..., 2] - baseline
    clearance = torch.clamp(foot_height / clearance_height, min=0.0, max=1.0)
    contact = (_foot_force_norm(env, sensor_cfg) > contact_threshold).to(dtype=foot_vel_b.dtype)
    swing_quality = swing * clearance * (1.0 - contact)
    stance_quality = stance * contact
    quality = swing_quality + stance_quality

    per_foot_reward = torch.exp(-tolerance * torch.square(error))
    quality_sum = torch.sum(quality, dim=-1)
    weighted_reward = torch.sum(quality * per_foot_reward, dim=-1) / torch.clamp(quality_sum, min=1.0)
    swing_contact_penalty = swing_contact_penalty_scale * torch.clamp(torch.sum(swing * contact, dim=-1), min=0.0, max=1.0)
    weighted_reward = weighted_reward * torch.clamp(1.0 - swing_contact_penalty, min=0.0)
    return gate * torch.where(quality_sum > 0.0, weighted_reward, torch.zeros_like(weighted_reward))


def foot_sagittal_symmetry(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = DEFAULT_FOOT_BODY_CFG,
    min_command_x: float = 0.2,
    tolerance: float = 16.0,
) -> torch.Tensor:
    """Reward left/right feet staying balanced around the base in the sagittal axis."""

    asset = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0].clamp_min(0.0)
    gate = (command_x > min_command_x).to(dtype=asset.data.root_lin_vel_b.torch.dtype)

    foot_pos_w, _ = _foot_pos_vel(env, asset_cfg)
    rel_foot_pos_w = foot_pos_w - asset.data.root_pos_w.torch[:, None, :3]
    foot_pos_b = _yaw_rotate_inverse(asset, rel_foot_pos_w)
    if foot_pos_b.shape[1] < 2:
        return torch.zeros(_num_envs(env), device=foot_pos_b.device, dtype=foot_pos_b.dtype)

    center_error = foot_pos_b[:, 0, 0] + foot_pos_b[:, 1, 0]
    return gate * torch.exp(-tolerance * torch.square(center_error))


def hip_deviation(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = DEFAULT_CONTROLLED_JOINT_CFG,
    joint_weight: float = 1.0,
    velocity_weight: float = 0.05,
) -> torch.Tensor:
    """Penalize excessive hip yaw/roll use that often creates bow-legged gaits."""

    asset = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return joint_weight * torch.sum(torch.square(joint_error), dim=-1) + velocity_weight * torch.sum(
        torch.square(joint_vel), dim=-1
    )


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


def body_contact(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize non-foot body contact before it becomes a stable failure mode."""

    force = _foot_force_norm(env, sensor_cfg)
    return torch.sum((force > threshold).to(dtype=torch.float32), dim=-1)
