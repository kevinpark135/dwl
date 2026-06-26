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
    from .observations import DEFAULT_CONTROLLED_JOINT_CFG
except ImportError:
    from observations import DEFAULT_CONTROLLED_JOINT_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


DWL_FRICTION_ATTR = "dwl_friction"
DWL_PUSH_FORCE_TORQUES_ATTR = "dwl_push_force_torques"
DWL_MOTOR_OFFSET_ATTR = "dwl_motor_offset"
DWL_MOTOR_STRENGTH_ATTR = "dwl_motor_strength"
DWL_PD_FACTORS_ATTR = "dwl_pd_factors"
DWL_SYSTEM_DELAY_ATTR = "dwl_system_delay_s"


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


def init_dwl_event_buffers(env: "ManagerBasedEnv", env_ids: torch.Tensor | None = None) -> None:
    """Initialize privileged buffers used by DWL observations."""

    num_envs = _num_envs(env)
    device = _device(env)
    setattr(env, DWL_FRICTION_ATTR, torch.ones(num_envs, 1, device=device))
    setattr(env, DWL_PUSH_FORCE_TORQUES_ATTR, torch.zeros(num_envs, 6, device=device))
    setattr(env, DWL_SYSTEM_DELAY_ATTR, torch.zeros(num_envs, 1, device=device))


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

    asset.write_joint_position_to_sim_index(position=joint_pos, joint_ids=joint_ids, env_ids=ids)
    asset.write_joint_velocity_to_sim_index(velocity=joint_vel, joint_ids=joint_ids, env_ids=ids)


def randomize_joint_position_observation_noise(*args, **kwargs):
    """Stub for DWL joint position observation noise `[-0.3, 0.3]`.

    TODO: wire this through ObservationTerm noise configs rather than event state.
    """

    raise NotImplementedError("Joint position observation noise belongs in observation noise config.")


def randomize_joint_velocity_observation_noise(*args, **kwargs):
    """Stub for DWL joint velocity observation noise `[-1, 1]`.

    TODO: wire this through ObservationTerm noise configs rather than event state.
    """

    raise NotImplementedError("Joint velocity observation noise belongs in observation noise config.")


def randomize_angular_velocity_observation_noise(*args, **kwargs):
    """Stub for DWL angular velocity observation noise `[-0.1, 0.1]`.

    TODO: wire this through ObservationTerm noise configs rather than event state.
    """

    raise NotImplementedError("Angular velocity observation noise belongs in observation noise config.")


def randomize_orientation_observation_noise(*args, **kwargs):
    """Stub for DWL orientation observation noise `[-0.1, 0.1]`.

    TODO: wire this through ObservationTerm noise configs rather than event state.
    """

    raise NotImplementedError("Orientation observation noise belongs in observation noise config.")


def randomize_system_delay(*args, **kwargs):
    """Stub for DWL system delay randomization `[0, 10] ms`.

    TODO: implement with an action/observation delay buffer in the env or runner.
    """

    raise NotImplementedError("System delay requires action/observation delay buffers.")


def randomize_motor_offset(*args, **kwargs):
    """Stub for DWL motor offset randomization `[-0.05, 0.05]` rad.

    TODO: connect sampled offsets to the action target processing path.
    """

    raise NotImplementedError("Motor offset requires action target processing integration.")


def randomize_motor_strength(*args, **kwargs):
    """Stub for DWL motor strength scaling `[0.9, 1.1]`.

    TODO: connect to actuator effort limits/saturation fields for the selected actuator type.
    """

    raise NotImplementedError("Motor strength requires actuator API integration.")


def randomize_pd_factors(*args, **kwargs):
    """Stub for DWL PD gain scaling `[0.8, 1.2]`.

    TODO: connect to actuator stiffness/damping fields for the selected actuator type.
    """

    raise NotImplementedError("PD factor randomization requires actuator gain integration.")
