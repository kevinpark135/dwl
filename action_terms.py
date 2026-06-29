# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime action terms for the DWL task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

try:
    from .actions import DwlJointPositionActionCfg, delay_steps_from_env
except ImportError:
    from actions import DwlJointPositionActionCfg, delay_steps_from_env

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DwlJointPositionAction(JointPositionAction):
    """Joint position action term with delayed commands and motor target offsets."""

    cfg: DwlJointPositionActionCfg

    def __init__(self, cfg: DwlJointPositionActionCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._max_delay_steps = int(cfg.max_delay_steps)
        self._action_history = torch.zeros(
            self.num_envs,
            self._max_delay_steps + 1,
            self.action_dim,
            device=self.device,
            dtype=self._raw_actions.dtype,
        )
        self._history_index = 0

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        self._history_index = (self._history_index + 1) % self._action_history.shape[1]
        self._action_history[:, self._history_index] = actions
        delayed_actions = self._delayed_actions()

        self._processed_actions = delayed_actions * self._scale + self._offset
        motor_offset = self._motor_offset()
        if motor_offset is not None:
            self._processed_actions = self._processed_actions + motor_offset

        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._action_history[env_ids] = 0.0

    def _delayed_actions(self) -> torch.Tensor:
        delay_steps = delay_steps_from_env(self._env, self._max_delay_steps, self.cfg.delay_attr_name)
        history_ids = (self._history_index - delay_steps) % self._action_history.shape[1]
        env_ids = torch.arange(self.num_envs, device=self.device)
        return self._action_history[env_ids, history_ids]

    def _motor_offset(self) -> torch.Tensor | None:
        offset = getattr(self._env, self.cfg.motor_offset_attr_name, None)
        if offset is None:
            return None

        offset = torch.as_tensor(offset, device=self.device, dtype=self._processed_actions.dtype)
        if offset.numel() == 0:
            return None
        offset = offset.reshape(self.num_envs, -1)
        if offset.shape[1] != self.action_dim:
            raise RuntimeError(
                f"Expected {self.cfg.motor_offset_attr_name} width {self.action_dim}, got {offset.shape[1]}."
            )
        return offset
