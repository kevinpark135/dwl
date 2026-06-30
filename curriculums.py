# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for the DWL task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _as_env_ids(env: "ManagerBasedRLEnv", env_ids: Sequence[int] | torch.Tensor | slice) -> torch.Tensor:
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _termination_term(env: "ManagerBasedRLEnv", term_name: str, env_ids: torch.Tensor) -> torch.Tensor:
    termination_manager = getattr(env, "termination_manager", None)
    if termination_manager is None:
        return torch.zeros(env_ids.numel(), device=env.device, dtype=torch.bool)

    if hasattr(termination_manager, "get_term"):
        try:
            return termination_manager.get_term(term_name)[env_ids].to(device=env.device, dtype=torch.bool)
        except Exception:
            pass

    term_dones = getattr(termination_manager, "_term_dones", None)
    if isinstance(term_dones, dict) and term_name in term_dones:
        return term_dones[term_name][env_ids].to(device=env.device, dtype=torch.bool)

    return torch.zeros(env_ids.numel(), device=env.device, dtype=torch.bool)


def _terrain_level_cap(common_step: int, level_cap_steps: tuple[int, ...] | None) -> int | None:
    if level_cap_steps is None:
        return None
    for max_level, start_step in enumerate(level_cap_steps, start=1):
        if common_step < int(start_step):
            return max_level
    return None


def _sync_env_origins(env: "ManagerBasedRLEnv", env_ids: torch.Tensor) -> None:
    terrain = env.scene.terrain
    if not all(hasattr(terrain, attr) for attr in ("terrain_origins", "terrain_levels", "terrain_types")):
        return

    origins = terrain.terrain_origins[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]]
    if hasattr(terrain, "env_origins"):
        terrain.env_origins[env_ids] = origins
    if hasattr(env.scene, "env_origins"):
        env.scene.env_origins[env_ids] = origins


def dwl_terrain_levels(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int] | torch.Tensor | slice,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    up_distance_ratio: float = 0.75,
    down_command_distance_ratio: float = 0.15,
    min_episode_progress_for_up: float = 0.35,
    min_episode_progress_for_down: float = 0.15,
    level_cap_steps: tuple[int, ...] | None = (12000, 24000, 36000, 48000),
) -> torch.Tensor:
    """Conservative terrain curriculum for early DWL humanoid walking.

    The stock velocity curriculum can promote terrains as soon as distance looks
    good. DWL needs a little more proof of stable walking, so promotion requires
    both distance and episode progress, while demotion is reserved for short
    base-contact failures with very low commanded-distance progress.
    """

    env_ids = _as_env_ids(env, env_ids)
    terrain = env.scene.terrain
    if env_ids.numel() == 0:
        return torch.mean(terrain.terrain_levels.float())

    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    distance = torch.linalg.norm(asset.data.root_pos_w.torch[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)

    terrain_length = float(terrain.cfg.terrain_generator.size[0])
    commanded_distance = torch.linalg.norm(command[env_ids, :2], dim=1) * float(env.max_episode_length_s)
    commanded_distance = commanded_distance.clamp_min(terrain_length * 0.25)

    progress = env.episode_length_buf[env_ids].float() / float(env.max_episode_length)
    base_contact = _termination_term(env, "base_contact", env_ids)

    move_up = (distance > terrain_length * up_distance_ratio) & (progress > min_episode_progress_for_up)
    move_up &= ~base_contact

    low_progress = distance < commanded_distance * down_command_distance_ratio
    move_down = base_contact & low_progress & (progress > min_episode_progress_for_down)
    move_down &= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)

    level_cap = _terrain_level_cap(int(getattr(env, "common_step_counter", 0)), level_cap_steps)
    if level_cap is not None:
        capped_levels = terrain.terrain_levels[env_ids].clamp_max(level_cap)
        if not torch.equal(capped_levels, terrain.terrain_levels[env_ids]):
            terrain.terrain_levels[env_ids] = capped_levels
            _sync_env_origins(env, env_ids)

    return torch.mean(terrain.terrain_levels.float())
