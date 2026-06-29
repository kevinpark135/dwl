# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""DWL action terms.

The paper randomizes motor target offsets and system delay.  Isaac Lab's stock
joint-position action term already handles the normal scale/default-pose offset,
so this module adds only the DWL-specific pieces around that existing behavior.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

DWL_MOTOR_OFFSET_ATTR = "dwl_motor_offset"
DWL_SYSTEM_DELAY_ATTR = "dwl_system_delay_s"


def delay_steps_from_env(
    env: "ManagerBasedEnv",
    max_delay_steps: int,
    delay_attr_name: str = DWL_SYSTEM_DELAY_ATTR,
) -> torch.Tensor:
    """Convert per-env delay seconds into bounded integer control-step delays."""

    device = torch.device(env.device)
    delay_s = getattr(env, delay_attr_name, None)
    if delay_s is None:
        return torch.zeros(env.num_envs, device=device, dtype=torch.long)

    delay_s = torch.as_tensor(delay_s, device=device, dtype=torch.float32).reshape(env.num_envs, -1)[:, 0]
    step_dt = float(getattr(env, "step_dt", 0.0))
    if step_dt <= 0.0:
        return torch.zeros(env.num_envs, device=device, dtype=torch.long)
    return torch.ceil(delay_s / step_dt).to(dtype=torch.long).clamp_(0, max_delay_steps)


@configclass
class DwlJointPositionActionCfg(JointPositionActionCfg):
    """Joint-position action with DWL motor offset and system-delay consumption."""

    class_type: type[DwlJointPositionAction] | str = "{DIR}:DwlJointPositionAction"

    motor_offset_attr_name: str = DWL_MOTOR_OFFSET_ATTR
    """Name of the env buffer containing per-env/per-joint motor target offsets."""

    delay_attr_name: str = DWL_SYSTEM_DELAY_ATTR
    """Name of the env buffer containing per-env system delay in seconds."""

    max_delay_steps: int = 4
    """Maximum number of control steps represented by the action delay buffer."""


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
