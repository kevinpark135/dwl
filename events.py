# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Event and domain-randomization terms for the DWL task.

This module names the domain-randomization terms from the DWL paper and provides
the pieces that can be implemented without custom actuator/action-delay plumbing.
It also owns the privileged buffers consumed by `observations.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

try:
    from .observations import DEFAULT_CONTROLLED_JOINT_CFG, DEFAULT_FOOT_BODY_CFG
except ImportError:
    from observations import DEFAULT_CONTROLLED_JOINT_CFG, DEFAULT_FOOT_BODY_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


DWL_FRICTION_ATTR = "dwl_friction"
DWL_PUSH_FORCE_TORQUES_ATTR = "dwl_push_force_torques"
DWL_MOTOR_OFFSET_ATTR = "dwl_motor_offset"
DWL_MOTOR_STRENGTH_ATTR = "dwl_motor_strength"
DWL_PD_FACTORS_ATTR = "dwl_pd_factors"
DWL_SYSTEM_DELAY_ATTR = "dwl_system_delay_s"
DWL_OBSERVATION_NOISE_RANGES_ATTR = "dwl_observation_noise_ranges"
DWL_FOOT_HEIGHT_BASELINE_ATTR = "dwl_foot_height_baseline"


def _num_envs(env: "ManagerBasedEnv") -> int:
    if hasattr(env, "num_envs"):
        return int(env.num_envs)
    if hasattr(env, "scene") and hasattr(env.scene, "num_envs"):
        return int(env.scene.num_envs)
    return int(env.episode_length_buf.shape[0])


def _device(env: "ManagerBasedEnv") -> torch.device:
    if hasattr(env, "device"):
        return torch.device(env.device)
    if hasattr(env, "episode_length_buf"):
        return env.episode_length_buf.device
    return torch.device("cpu")


def _resolve_env_ids(env: "ManagerBasedEnv", env_ids: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(_num_envs(env), device=device, dtype=torch.long)
    return env_ids.to(device=device, dtype=torch.long)


def _resolve_body_ids(asset, asset_cfg: SceneEntityCfg, device: torch.device) -> torch.Tensor:
    if asset_cfg.body_ids == slice(None):
        return torch.arange(asset.num_bodies, dtype=torch.long, device=device)
    return torch.as_tensor(asset_cfg.body_ids, dtype=torch.long, device=device)


def _resolve_joint_ids(asset, asset_cfg: SceneEntityCfg, device: torch.device) -> torch.Tensor | slice:
    if asset_cfg.joint_ids == slice(None):
        return slice(None)
    return torch.as_tensor(asset_cfg.joint_ids, dtype=torch.long, device=device)


def _joint_ids_tensor(asset, asset_cfg: SceneEntityCfg, device: torch.device) -> torch.Tensor:
    joint_ids = _resolve_joint_ids(asset, asset_cfg, device)
    if isinstance(joint_ids, slice):
        return torch.arange(asset.num_joints, dtype=torch.long, device=device)
    return joint_ids


def _int32_index(index: torch.Tensor | slice) -> torch.Tensor | slice:
    if isinstance(index, slice):
        return index
    return index.to(dtype=torch.int32)


def _joint_positions(requested_joint_ids: torch.Tensor, sim_joint_ids: torch.Tensor) -> torch.Tensor:
    positions = []
    for joint_id in sim_joint_ids:
        match = torch.nonzero(requested_joint_ids == joint_id, as_tuple=False).flatten()
        if match.numel() == 0:
            raise RuntimeError(f"Joint id {int(joint_id)} was not found in requested joint ids.")
        positions.append(match[0])
    return torch.stack(positions)


def init_dwl_event_buffers(env: "ManagerBasedEnv", env_ids: torch.Tensor | None = None) -> None:
    """Initialize privileged buffers used by DWL observations."""

    num_envs = _num_envs(env)
    device = _device(env)
    setattr(env, DWL_FRICTION_ATTR, torch.ones(num_envs, 1, device=device))
    setattr(env, DWL_PUSH_FORCE_TORQUES_ATTR, torch.zeros(num_envs, 6, device=device))
    setattr(env, DWL_SYSTEM_DELAY_ATTR, torch.zeros(num_envs, 1, device=device))
    setattr(env, DWL_MOTOR_OFFSET_ATTR, torch.zeros(num_envs, 0, device=device))
    setattr(env, DWL_MOTOR_STRENGTH_ATTR, torch.ones(num_envs, 0, device=device))
    setattr(env, DWL_PD_FACTORS_ATTR, torch.ones(num_envs, 0, 2, device=device))
    setattr(
        env,
        DWL_OBSERVATION_NOISE_RANGES_ATTR,
        {
            "joint_position": (-0.3, 0.3),
            "joint_velocity": (-1.0, 1.0),
            "angular_velocity": (-0.1, 0.1),
            "orientation": (-0.1, 0.1),
        },
    )


def clear_push_force_torques(env: "ManagerBasedEnv", env_ids: torch.Tensor | None = None) -> None:
    """Clear the stored DWL push force/torque privileged buffer."""

    device = _device(env)
    ids = _resolve_env_ids(env, env_ids, device)
    if not hasattr(env, DWL_PUSH_FORCE_TORQUES_ATTR):
        init_dwl_event_buffers(env)
    getattr(env, DWL_PUSH_FORCE_TORQUES_ATTR)[ids] = 0.0


def sample_push_force_torques(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Sample external force/torque, apply it to the asset, and store a 6D privileged buffer."""

    asset = env.scene[asset_cfg.name]
    ids = _resolve_env_ids(env, env_ids, asset.device)
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    forces = sample_uniform(*force_range, (ids.numel(), num_bodies, 3), asset.device)
    torques = sample_uniform(*torque_range, (ids.numel(), num_bodies, 3), asset.device)
    asset.permanent_wrench_composer.set_forces_and_torques_index(
        forces=forces,
        torques=torques,
        body_ids=asset_cfg.body_ids,
        env_ids=ids.to(dtype=torch.int32),
    )

    if not hasattr(env, DWL_PUSH_FORCE_TORQUES_ATTR):
        init_dwl_event_buffers(env)
    wrench = torch.cat((forces.mean(dim=1), torques.mean(dim=1)), dim=-1).to(_device(env))
    getattr(env, DWL_PUSH_FORCE_TORQUES_ATTR)[ids.to(_device(env))] = wrench


def store_friction(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    friction_range: tuple[float, float] = (0.2, 2.0),
) -> torch.Tensor:
    """Sample/store per-env friction values for the DWL privileged state.

    Material randomization should still be configured through Isaac Lab's
    `randomize_rigid_body_material`; this function records the sampled values so
    `state_friction` has a source of truth.
    """

    device = _device(env)
    ids = _resolve_env_ids(env, env_ids, device)
    if not hasattr(env, DWL_FRICTION_ATTR):
        init_dwl_event_buffers(env)
    friction = sample_uniform(*friction_range, (ids.numel(), 1), device)
    getattr(env, DWL_FRICTION_ATTR)[ids] = friction
    return friction


def randomize_body_mass(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    mass_distribution_params: tuple[float, float] = (-5.0, 20.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_mass: float = 1.0,
) -> None:
    """Add bounded payload mass offsets to selected bodies."""

    asset = env.scene[asset_cfg.name]
    ids = _resolve_env_ids(env, env_ids, asset.device)
    body_ids = _resolve_body_ids(asset, asset_cfg, asset.device)

    default_key = "_dwl_default_body_mass"
    if not hasattr(asset, default_key):
        setattr(asset, default_key, asset.data.body_mass.torch.clone())
    default_mass = getattr(asset, default_key)

    low, high = mass_distribution_params
    payload = torch.empty((ids.numel(), body_ids.numel()), device=asset.device).uniform_(low, high)
    masses = default_mass[ids[:, None], body_ids].clone() + payload
    masses = torch.nan_to_num(masses, nan=min_mass, posinf=min_mass, neginf=min_mass).clamp_min(min_mass).contiguous()

    asset.set_masses_index(
        masses=masses,
        body_ids=body_ids.to(dtype=torch.int32),
        env_ids=ids.to(dtype=torch.int32),
    )


def randomize_joint_reset_noise(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    position_range: tuple[float, float] = (-0.3, 0.3),
    velocity_range: tuple[float, float] = (-1.0, 1.0),
    asset_cfg: SceneEntityCfg = DEFAULT_CONTROLLED_JOINT_CFG,
) -> None:
    """Reset selected joints around default state with additive DWL noise."""

    asset = env.scene[asset_cfg.name]
    ids = _resolve_env_ids(env, env_ids, asset.device)
    joint_ids = _resolve_joint_ids(asset, asset_cfg, asset.device)
    iter_env_ids = ids[:, None] if joint_ids != slice(None) else ids

    joint_pos = asset.data.default_joint_pos.torch[iter_env_ids, joint_ids].clone()
    joint_vel = asset.data.default_joint_vel.torch[iter_env_ids, joint_ids].clone()
    joint_pos += sample_uniform(*position_range, joint_pos.shape, asset.device)
    joint_vel += sample_uniform(*velocity_range, joint_vel.shape, asset.device)

    joint_pos_limits = asset.data.soft_joint_pos_limits.torch[iter_env_ids, joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = asset.data.soft_joint_vel_limits.torch[iter_env_ids, joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    asset.write_joint_position_to_sim_index(
        position=joint_pos, joint_ids=_int32_index(joint_ids), env_ids=ids.to(dtype=torch.int32)
    )
    asset.write_joint_velocity_to_sim_index(
        velocity=joint_vel, joint_ids=_int32_index(joint_ids), env_ids=ids.to(dtype=torch.int32)
    )


def store_foot_height_baseline(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = DEFAULT_FOOT_BODY_CFG,
    baseline_attr: str = DWL_FOOT_HEIGHT_BASELINE_ATTR,
) -> torch.Tensor:
    """Store per-env foot-height baselines for terrain-relative foot rewards."""

    asset = env.scene[asset_cfg.name]
    ids = _resolve_env_ids(env, env_ids, asset.device)
    body_ids = _resolve_body_ids(asset, asset_cfg, asset.device)
    foot_z = asset.data.body_link_pose_w.torch[ids[:, None], body_ids, 2].clone()

    buffer = getattr(env, baseline_attr, None)
    expected_shape = (_num_envs(env), body_ids.numel())
    if buffer is None or buffer.shape != expected_shape:
        buffer = torch.zeros(expected_shape, device=_device(env), dtype=foot_z.dtype)
        setattr(env, baseline_attr, buffer)
    getattr(env, baseline_attr)[ids.to(_device(env))] = foot_z.to(_device(env))
    return foot_z


def randomize_joint_position_observation_noise(
    env: "ManagerBasedEnv", env_ids: torch.Tensor | None, noise_range: tuple[float, float] = (-0.3, 0.3)
) -> None:
    """Record the DWL joint-position observation noise range.

    The actual noise is applied by `ObservationTermCfg.noise` in `dwl_env_cfg.py`;
    this event keeps the paper term named and inspectable.
    """

    if not hasattr(env, DWL_OBSERVATION_NOISE_RANGES_ATTR):
        init_dwl_event_buffers(env)
    getattr(env, DWL_OBSERVATION_NOISE_RANGES_ATTR)["joint_position"] = noise_range


def randomize_joint_velocity_observation_noise(
    env: "ManagerBasedEnv", env_ids: torch.Tensor | None, noise_range: tuple[float, float] = (-1.0, 1.0)
) -> None:
    """Record the DWL joint-velocity observation noise range."""

    if not hasattr(env, DWL_OBSERVATION_NOISE_RANGES_ATTR):
        init_dwl_event_buffers(env)
    getattr(env, DWL_OBSERVATION_NOISE_RANGES_ATTR)["joint_velocity"] = noise_range


def randomize_angular_velocity_observation_noise(
    env: "ManagerBasedEnv", env_ids: torch.Tensor | None, noise_range: tuple[float, float] = (-0.1, 0.1)
) -> None:
    """Record the DWL angular-velocity observation noise range."""

    if not hasattr(env, DWL_OBSERVATION_NOISE_RANGES_ATTR):
        init_dwl_event_buffers(env)
    getattr(env, DWL_OBSERVATION_NOISE_RANGES_ATTR)["angular_velocity"] = noise_range


def randomize_orientation_observation_noise(
    env: "ManagerBasedEnv", env_ids: torch.Tensor | None, noise_range: tuple[float, float] = (-0.1, 0.1)
) -> None:
    """Record the DWL orientation observation noise range."""

    if not hasattr(env, DWL_OBSERVATION_NOISE_RANGES_ATTR):
        init_dwl_event_buffers(env)
    getattr(env, DWL_OBSERVATION_NOISE_RANGES_ATTR)["orientation"] = noise_range


def randomize_system_delay(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    delay_range_s: tuple[float, float] = (0.0, 0.01),
) -> torch.Tensor:
    """Sample/store system delay values in seconds.

    The event stores `env.dwl_system_delay_s`. Action/observation delay buffers
    should consume this value when the DWL runner or wrapper is implemented.
    """

    device = _device(env)
    ids = _resolve_env_ids(env, env_ids, device)
    if not hasattr(env, DWL_SYSTEM_DELAY_ATTR):
        init_dwl_event_buffers(env)
    delay = sample_uniform(*delay_range_s, (ids.numel(), 1), device)
    getattr(env, DWL_SYSTEM_DELAY_ATTR)[ids] = delay
    return delay


def randomize_motor_offset(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    offset_range: tuple[float, float] = (-0.05, 0.05),
    asset_cfg: SceneEntityCfg = DEFAULT_CONTROLLED_JOINT_CFG,
) -> torch.Tensor:
    """Sample/store motor target offsets for selected joints.

    The sampled offsets are stored in `env.dwl_motor_offset`. The action target
    processing path should add these offsets when action integration is wired.
    """

    asset = env.scene[asset_cfg.name]
    ids = _resolve_env_ids(env, env_ids, asset.device)
    joint_ids = _joint_ids_tensor(asset, asset_cfg, asset.device)
    offsets = sample_uniform(*offset_range, (ids.numel(), joint_ids.numel()), asset.device)

    buffer = getattr(env, DWL_MOTOR_OFFSET_ATTR, None)
    if buffer is None or buffer.shape != (_num_envs(env), joint_ids.numel()):
        buffer = torch.zeros((_num_envs(env), joint_ids.numel()), device=asset.device)
        setattr(env, DWL_MOTOR_OFFSET_ATTR, buffer.to(_device(env)))
        buffer = getattr(env, DWL_MOTOR_OFFSET_ATTR)
    buffer[ids.to(_device(env))] = offsets.to(_device(env))
    return offsets


def randomize_motor_strength(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    strength_distribution_params: tuple[float, float] = (0.9, 1.1),
    asset_cfg: SceneEntityCfg = DEFAULT_CONTROLLED_JOINT_CFG,
) -> torch.Tensor:
    """Scale actuator effort limits to emulate motor strength variation."""

    from isaaclab.actuators import ImplicitActuator

    asset = env.scene[asset_cfg.name]
    ids = _resolve_env_ids(env, env_ids, asset.device)
    requested_joint_ids = _joint_ids_tensor(asset, asset_cfg, asset.device)
    low, high = strength_distribution_params
    stored_strength = torch.ones((ids.numel(), requested_joint_ids.numel()), device=asset.device)

    default_key = "_dwl_default_motor_strength"
    if not hasattr(asset, default_key):
        defaults = {}
        for name, actuator in getattr(asset, "actuators", {}).items():
            defaults[name] = {"effort_limit": actuator.effort_limit.clone()}
            if hasattr(actuator, "_saturation_effort"):
                saturation = actuator._saturation_effort
                if not isinstance(saturation, torch.Tensor):
                    saturation = torch.full_like(actuator.effort_limit, float(saturation))
                defaults[name]["saturation_effort"] = saturation.clone()
        setattr(asset, default_key, defaults)
    defaults = getattr(asset, default_key)

    for name, actuator in getattr(asset, "actuators", {}).items():
        actuator_joint_ids = actuator.joint_indices
        if isinstance(actuator_joint_ids, slice):
            global_joint_ids = torch.arange(asset.num_joints, dtype=torch.long, device=asset.device)
        else:
            global_joint_ids = actuator_joint_ids.to(asset.device)

        mask = torch.isin(global_joint_ids, requested_joint_ids)
        if not torch.any(mask):
            continue
        actuator_indices = torch.nonzero(mask, as_tuple=False).flatten()
        sim_joint_ids = global_joint_ids[actuator_indices]
        strength = sample_uniform(low, high, (ids.numel(), actuator_indices.numel()), asset.device)

        effort_limit = actuator.effort_limit[ids].clone()
        effort_limit[:, actuator_indices] = defaults[name]["effort_limit"][ids][:, actuator_indices] * strength
        actuator.effort_limit[ids] = effort_limit

        if "saturation_effort" in defaults[name]:
            saturation_effort = getattr(actuator, "_saturation_effort")
            if not isinstance(saturation_effort, torch.Tensor):
                saturation_effort = defaults[name]["saturation_effort"].clone()
            else:
                saturation_effort = saturation_effort.clone()
            saturation_effort[ids[:, None], actuator_indices] = (
                defaults[name]["saturation_effort"][ids[:, None], actuator_indices] * strength
            )
            actuator._saturation_effort = saturation_effort
            if hasattr(actuator, "_vel_at_effort_lim"):
                actuator._vel_at_effort_lim = actuator.velocity_limit * (
                    1.0 + actuator.effort_limit / actuator._saturation_effort
                )

        if isinstance(actuator, ImplicitActuator):
            asset.write_joint_effort_limit_to_sim_index(
                limits=effort_limit[:, actuator_indices],
                joint_ids=sim_joint_ids.to(dtype=torch.int32),
                env_ids=ids.to(dtype=torch.int32),
            )

        global_pos = _joint_positions(requested_joint_ids, sim_joint_ids)
        stored_strength[:, global_pos] = strength

    buffer = torch.ones((_num_envs(env), requested_joint_ids.numel()), device=_device(env))
    if hasattr(env, DWL_MOTOR_STRENGTH_ATTR) and getattr(env, DWL_MOTOR_STRENGTH_ATTR).shape == buffer.shape:
        buffer = getattr(env, DWL_MOTOR_STRENGTH_ATTR)
    else:
        setattr(env, DWL_MOTOR_STRENGTH_ATTR, buffer)
    getattr(env, DWL_MOTOR_STRENGTH_ATTR)[ids.to(_device(env))] = stored_strength.to(_device(env))
    return stored_strength


def randomize_pd_factors(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    pd_factor_distribution_params: tuple[float, float] = (0.8, 1.2),
    asset_cfg: SceneEntityCfg = DEFAULT_CONTROLLED_JOINT_CFG,
) -> torch.Tensor:
    """Scale actuator stiffness/damping factors for selected joints."""

    from isaaclab.actuators import ImplicitActuator

    asset = env.scene[asset_cfg.name]
    ids = _resolve_env_ids(env, env_ids, asset.device)
    requested_joint_ids = _joint_ids_tensor(asset, asset_cfg, asset.device)
    low, high = pd_factor_distribution_params
    stored_factors = torch.ones((ids.numel(), requested_joint_ids.numel(), 2), device=asset.device)

    default_key = "_dwl_default_pd_factors"
    if not hasattr(asset, default_key):
        defaults = {}
        for name, actuator in getattr(asset, "actuators", {}).items():
            defaults[name] = {
                "stiffness": actuator.stiffness.clone(),
                "damping": actuator.damping.clone(),
            }
        setattr(asset, default_key, defaults)
    defaults = getattr(asset, default_key)

    for name, actuator in getattr(asset, "actuators", {}).items():
        actuator_joint_ids = actuator.joint_indices
        if isinstance(actuator_joint_ids, slice):
            global_joint_ids = torch.arange(asset.num_joints, dtype=torch.long, device=asset.device)
        else:
            global_joint_ids = actuator_joint_ids.to(asset.device)

        mask = torch.isin(global_joint_ids, requested_joint_ids)
        if not torch.any(mask):
            continue
        actuator_indices = torch.nonzero(mask, as_tuple=False).flatten()
        sim_joint_ids = global_joint_ids[actuator_indices]
        factors = sample_uniform(low, high, (ids.numel(), actuator_indices.numel(), 2), asset.device)

        stiffness = actuator.stiffness[ids].clone()
        damping = actuator.damping[ids].clone()
        stiffness[:, actuator_indices] = defaults[name]["stiffness"][ids][:, actuator_indices] * factors[..., 0]
        damping[:, actuator_indices] = defaults[name]["damping"][ids][:, actuator_indices] * factors[..., 1]
        actuator.stiffness[ids] = stiffness
        actuator.damping[ids] = damping

        if isinstance(actuator, ImplicitActuator):
            asset.write_joint_stiffness_to_sim_index(
                stiffness=stiffness[:, actuator_indices],
                joint_ids=sim_joint_ids.to(dtype=torch.int32),
                env_ids=ids.to(dtype=torch.int32),
            )
            asset.write_joint_damping_to_sim_index(
                damping=damping[:, actuator_indices],
                joint_ids=sim_joint_ids.to(dtype=torch.int32),
                env_ids=ids.to(dtype=torch.int32),
            )

        global_pos = _joint_positions(requested_joint_ids, sim_joint_ids)
        stored_factors[:, global_pos, :] = factors

    buffer = torch.ones((_num_envs(env), requested_joint_ids.numel(), 2), device=_device(env))
    if hasattr(env, DWL_PD_FACTORS_ATTR) and getattr(env, DWL_PD_FACTORS_ATTR).shape == buffer.shape:
        buffer = getattr(env, DWL_PD_FACTORS_ATTR)
    else:
        setattr(env, DWL_PD_FACTORS_ATTR, buffer)
    getattr(env, DWL_PD_FACTORS_ATTR)[ids.to(_device(env))] = stored_factors.to(_device(env))
    return stored_factors
